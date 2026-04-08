"""
trainer_utils.py

Shared infrastructure for all SSL downscaling models:
    FrozenSSLDownscaler  (frozen_ssl.py)
    SSLDownscaler        (ssl_downscaler.py)
    CASD                 (casd.py)
    FGD                  (fgd.py)

Public API
----------
# Architecture helpers (call in model __init__)
    compute_ssl_size(lr_shape, patch_size, min_patches=16)
        -> (ssl_size, n_patches_h, n_patches_w, n_patches)

    build_encoder(model_id, mode, lora_r, lora_alpha, lora_dropout)
        -> nn.Module  (frozen base model, or PEFT-wrapped LoRA model)

    init_conv_decoder(module)
        Kaiming-normal on all Conv2d, zeros on final Conv2d weight+bias.

# Forward helper (call in model _encode / _encode_multiscale)
    encoder_forward(encoder, mode, pixel_values, output_hidden_states=False)
        -> transformers ModelOutput

# Optimizer (replaces model.make_optimizer())
    make_optimizer(model, lr, decoder_lr, weight_decay)
        -> torch.optim.AdamW

# Scheduler
    make_scheduler(optimizer, warmup_epochs, max_epochs)
        -> LRScheduler
    fast_forward_scheduler(scheduler, start_epoch)

# Checkpoint
    save_checkpoint(path, epoch, model, model_key, cfg,
                    optimizer, scheduler, scaler,
                    best_val_rmse, epochs_no_improve, wandb_run_id)
    load_checkpoint(path, model, optimizer, scheduler, scaler)
        -> (start_epoch, best_val_rmse, epochs_no_improve, wandb_run_id)

# Train / eval loop
    train_one_epoch(model, loader, optimizer, scaler, epoch,
                    grad_accum_steps=1, loss_fn=None)
        -> dict
    evaluate(model, loader, hr_shape, split="val")
        -> dict

# Misc
    set_seed(seed)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import AutoModel

try:
    from peft import LoraConfig, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Architecture helpers ───────────────────────────────────────────────────────


def compute_ssl_size(
    lr_shape: tuple,
    patch_size: int,
    min_patches: int = 16,
) -> tuple:
    """
    Compute SSL encoder input size and patch grid dimensions.

    Targets min_patches per side, preserves the LR aspect ratio,
    and rounds both dims up to the nearest multiple of patch_size.

    Args:
        lr_shape    : (H_lr, W_lr) of the LR input grid
        patch_size  : ViT patch size (14 for DINOv2, 16 for DINOv3-SAT)
        min_patches : minimum patches per side (default 16)

    Returns:
        ssl_size     : (ssl_H, ssl_W)
        n_patches_h  : ssl_H // patch_size
        n_patches_w  : ssl_W // patch_size
        n_patches    : n_patches_h * n_patches_w
    """
    H_lr, W_lr = lr_shape
    ssl_H = min_patches * patch_size
    ssl_W = ssl_H * W_lr // H_lr
    ssl_W = ((ssl_W + patch_size - 1) // patch_size) * patch_size

    n_patches_h = ssl_H // patch_size
    n_patches_w = ssl_W // patch_size
    n_patches = n_patches_h * n_patches_w

    return (ssl_H, ssl_W), n_patches_h, n_patches_w, n_patches


def build_encoder(
    model_id: str,
    mode: str,
    lora_r: int = 8,
    lora_alpha: int = None,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """
    Load a pretrained SSL encoder and apply frozen or LoRA mode.

    Frozen: all parameters frozen, encoder set to eval().
    LoRA:   all base params frozen, LoRA adapters injected on q/v projections.
            Encoder left in train() so BatchNorm/Dropout behave correctly.

    DINOv2 attention projections are named query/value;
    DINOv3-SAT projections are named q_proj/v_proj.
    Detection is automatic via the presence of model.encoder attribute.

    Args:
        model_id     : HuggingFace model identifier
        mode         : "frozen" | "lora"
        lora_r       : LoRA rank (ignored when mode="frozen")
        lora_alpha   : LoRA alpha; defaults to lora_r (standard convention)
        lora_dropout : dropout on LoRA paths (default 0.0)

    Returns:
        encoder : nn.Module (base model or PEFT-wrapped model)
    """
    assert mode in ("frozen", "lora"), f"mode must be 'frozen' or 'lora', got {mode!r}"
    if mode == "lora" and not _PEFT_AVAILABLE:
        raise ImportError("pip install peft>=0.10.0 to use mode='lora'")

    encoder = AutoModel.from_pretrained(model_id)

    # Freeze all base parameters in both modes
    for param in encoder.parameters():
        param.requires_grad = False

    if mode == "frozen":
        encoder.eval()
        return encoder

    # LoRA mode — inject adapters then selectively unfreeze them
    _alpha = lora_alpha if lora_alpha is not None else lora_r
    # DINOv2 has a nested .encoder attribute; DINOv3-SAT does not
    _is_dinov2 = hasattr(encoder, "encoder")
    _target_modules = ["query", "value"] if _is_dinov2 else ["q_proj", "v_proj"]

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=_alpha,
        target_modules=_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )
    encoder = get_peft_model(encoder, lora_cfg)
    encoder.gradient_checkpointing_enable()
    encoder.train()
    return encoder


def init_conv_decoder(module: nn.Module) -> None:
    """
    Initialise convolutional decoder weights in-place.

    All Conv2d layers receive kaiming-normal weights and zero biases.
    The final Conv2d layer (last-encountered in module.modules()) receives
    additional zero-init on its weights so predictions start near zero
    at epoch 0, giving an initial RMSE ≈ HR std ≈ 1.0 rather than exploding.

    Works for nn.Sequential and arbitrary nested modules.

    Args:
        module : nn.Module containing Conv2d layers (decoder or output head)
    """
    last_conv = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            last_conv = m

    if last_conv is not None:
        nn.init.zeros_(last_conv.weight)
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)


# ── Forward helper ─────────────────────────────────────────────────────────────


def encoder_forward(
    encoder: nn.Module,
    mode: str,
    pixel_values: torch.Tensor,
    output_hidden_states: bool = False,
):
    """
    Run the SSL encoder, managing grad context based on mode.

    Frozen mode: wraps the call in torch.no_grad() — LoRA adapters produce
    no gradients and the encoder is already in eval().
    LoRA mode:   runs normally so gradients flow through LoRA parameters.

    Args:
        encoder              : the encoder module (base or PEFT-wrapped)
        mode                 : "frozen" | "lora"
        pixel_values         : [B, 3, ssl_H, ssl_W] bfloat16 tensor
        output_hidden_states : whether to return all hidden states

    Returns:
        transformers ModelOutput with .last_hidden_state and
        optionally .hidden_states
    """
    kwargs = dict(
        pixel_values=pixel_values,
        output_hidden_states=output_hidden_states,
    )
    if mode == "frozen":
        with torch.no_grad():
            return encoder(**kwargs)
    return encoder(**kwargs)


# ── Optimizer ──────────────────────────────────────────────────────────────────


def make_optimizer(
    model: nn.Module,
    lr: float = 5e-5,
    decoder_lr: float = 2e-4,
    weight_decay: float = 1e-2,
) -> torch.optim.AdamW:
    """
    Build an AdamW optimizer with one or two parameter groups.

    Frozen mode: single group — all trainable params at decoder_lr with
                 lighter weight decay (1e-4). Appropriate for decoder +
                 feat_norm only.

    LoRA mode:   two groups —
                   "lora_*" params  : lr (conservative), weight_decay
                   all other params : decoder_lr (higher), weight_decay=1e-4

    Mode is read from model.mode (default "frozen" if attribute absent).
    Works for FrozenSSLDownscaler, SSLDownscaler, CASD, FGD without
    modification.

    Args:
        model        : any downscaling model with model.mode attribute
        lr           : LoRA parameter learning rate (LoRA mode only)
        decoder_lr   : decoder / non-LoRA parameter learning rate
        weight_decay : weight decay for LoRA group (decoder group uses 1e-4)

    Returns:
        torch.optim.AdamW
    """
    mode = getattr(model, "mode", "frozen")

    if mode == "frozen":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=decoder_lr,
            weight_decay=1e-4,
        )

    # LoRA mode — split into two groups by name
    lora_params, decoder_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            decoder_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr, "weight_decay": weight_decay},
            {"params": decoder_params, "lr": decoder_lr, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.999),
    )


# ── Scheduler ─────────────────────────────────────────────────────────────────


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
):
    """
    warmup_epochs=0 → pure CosineAnnealingLR over max_epochs
    warmup_epochs>0 → LinearLR warmup (0.1→1.0) then CosineAnnealingLR
    """
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=max_epochs)

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max_epochs - warmup_epochs,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def fast_forward_scheduler(scheduler, start_epoch: int) -> None:
    """
    Advance scheduler internal counter to start_epoch without changing
    optimizer param groups. Used after resuming from a checkpoint.
    """
    if start_epoch <= 1:
        return
    print(f"Fast-forwarding scheduler to epoch {start_epoch}...")
    for _ in range(start_epoch - 1):
        scheduler.step()


# ── Checkpoint ────────────────────────────────────────────────────────────────


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    model_key: str,
    cfg,
    optimizer,
    scheduler,
    scaler,
    best_val_rmse: float,
    epochs_no_improve: int,
    wandb_run_id: str,
) -> None:
    """
    Save full training state to disk.

    Uses model.state_dict() — works for FrozenSSLDownscaler, SSLDownscaler,
    CASD, and FGD without modification. The full OmegaConf config is stored
    so load_checkpoint can validate architecture consistency on resume.

    Args:
        path             : destination file path
        epoch            : current epoch (resumed run starts at epoch+1)
        model            : any downscaling model
        model_key        : registry key string (e.g. "dinov3-sat")
        cfg              : OmegaConf config object
        optimizer        : optimizer instance
        scheduler        : LR scheduler instance
        scaler           : GradScaler instance
        best_val_rmse    : best validation RMSE seen so far
        epochs_no_improve: early-stopping counter
        wandb_run_id     : WandB run ID for resume
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_rmse": best_val_rmse,
            "epochs_no_improve": epochs_no_improve,
            "wandb_run_id": wandb_run_id,
            "model_key": model_key,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
) -> tuple:
    """
    Load a checkpoint and restore all training state.

    Supports two checkpoint formats:
        New format : "model_state" key  → model.load_state_dict()
        Old format : "decoder" + "feat_norm" keys (Pilot 2/3 only)

    Architecture consistency (mode, lora_r, tap_layers, etc.) is validated
    by comparing the saved config against model attributes when available.

    Args:
        path      : checkpoint file path
        model     : instantiated model (must already be on target device)
        optimizer : optimizer to restore
        scheduler : scheduler to restore
        scaler    : GradScaler to restore

    Returns:
        start_epoch        : epoch to start from (saved epoch + 1)
        best_val_rmse      : best validation RMSE from the saved run
        epochs_no_improve  : early-stopping counter from the saved run
        wandb_run_id       : WandB run ID for resuming the same run
    """
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=DEVICE)

    # ── Model weights ──────────────────────────────────────────────────────
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    elif "decoder" in ckpt and "feat_norm" in ckpt:
        # Legacy Pilot 2/3 format — decoder + feat_norm submodules only
        print("  Detected old checkpoint format (decoder + feat_norm). Loading...")
        model.decoder.load_state_dict(ckpt["decoder"])
        model.feat_norm.load_state_dict(ckpt["feat_norm"])
    else:
        raise KeyError(
            f"Unrecognised checkpoint format. Keys found: {list(ckpt.keys())}"
        )

    # ── Architecture consistency checks ────────────────────────────────────
    # Compare saved config fields against model attributes where available.
    # Warn rather than hard-crash so the user can override if intentional.
    saved_cfg = ckpt.get("config", {})
    saved_model_cfg = saved_cfg.get("model", {})

    def _warn_mismatch(field, saved_val, current_val):
        if saved_val != current_val:
            print(
                f"  WARNING: checkpoint {field}={saved_val!r} "
                f"!= current {field}={current_val!r}"
            )

    for attr in (
        "mode",
        "lora_r",
        "tap_layers",
        "use_static",
        "use_multiscale",
        "use_film",
        "use_cls",
        "cls_layer_idx",
    ):
        if attr in saved_model_cfg and hasattr(model, attr):
            _warn_mismatch(attr, saved_model_cfg[attr], getattr(model, attr))

    # ── Optimizer / scheduler / scaler ────────────────────────────────────
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])

    start_epoch = ckpt["epoch"] + 1
    best_val_rmse = ckpt["best_val_rmse"]
    epochs_no_improve = ckpt["epochs_no_improve"]
    wandb_run_id = ckpt.get("wandb_run_id", None)

    print(
        f"Resumed from epoch {ckpt['epoch']} |  "
        f"best_val_rmse={best_val_rmse:.4f} |  "
        f"epochs_no_improve={epochs_no_improve}"
    )
    return start_epoch, best_val_rmse, epochs_no_improve, wandb_run_id


# ── Train / eval mode helpers ─────────────────────────────────────────────────


def _set_train_mode(model: nn.Module) -> None:
    """
    Set correct train/eval modes per model type.

    Frozen encoder:  encoder.eval(), everything else train()
    LoRA encoder:    encoder.train(), everything else train()

    Works for SSLDownscaler, CASD, FGD, and FrozenSSLDownscaler
    (which has no .mode attribute — defaults to "frozen").
    """
    mode = getattr(model, "mode", "frozen")
    if mode == "frozen":
        model.encoder.eval()
    else:
        model.encoder.train()

    for name, module in model.named_children():
        if name != "encoder":
            module.train()


def _set_eval_mode(model: nn.Module) -> None:
    """Full eval mode — used during validation and test."""
    model.eval()


# ── Train one epoch ───────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scaler,
    epoch: int,
    grad_accum_steps: int = 1,
    loss_fn=None,
) -> dict:
    """
    Run one training epoch with gradient accumulation and AMP.

    Args:
        model             : any downscaling model
        loader            : training DataLoader yielding
                            (lr_imagenet, lr_norm, hr_norm)
        optimizer         : optimizer instance
        scaler            : GradScaler instance
        epoch             : current epoch number (for tqdm label)
        grad_accum_steps  : number of batches to accumulate before stepping
        loss_fn           : callable(pred, hr_norm) -> scalar loss tensor.
                            Defaults to F.mse_loss.
                            Pass fgd_loss (with spectral_lambda bound via
                            functools.partial) for FGD spectral training.

    Returns:
        dict with keys "train/loss" and "train/rmse"
    """
    if loss_fn is None:
        loss_fn = F.mse_loss

    _set_train_mode(model)

    total_loss = 0.0
    sum_sq_err = 0.0
    n_samples = 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad(set_to_none=True)

    for i, (lr_imagenet, _, hr_norm) in enumerate(
        tqdm(loader, desc=f"Train {epoch}", leave=False)
    ):
        lr_imagenet = lr_imagenet.to(DEVICE, non_blocking=True)
        hr_norm = hr_norm.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(lr_imagenet)
            loss = loss_fn(pred, hr_norm) / grad_accum_steps

        scaler.scale(loss).backward()

        real_loss = loss.item() * grad_accum_steps
        batch_sq_err = F.mse_loss(
            pred.detach().float(), hr_norm.float(), reduction="sum"
        )
        sum_sq_err += batch_sq_err.item()
        n_samples += hr_norm.numel()
        total_loss += real_loss

        is_last_batch = i + 1 == len(loader)
        if (i + 1) % grad_accum_steps == 0 or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    n_steps = len(loader)
    return {
        "train/loss": total_loss / n_steps,
        "train/rmse": (sum_sq_err / n_samples) ** 0.5,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    bilinear_rmse: float,  # ← pre-computed, passed in
    split: str = "val",
) -> dict:
    _set_eval_mode(model)

    sum_sq_err = 0.0
    total_pearson = 0.0
    total_bias = 0.0
    n_batches = 0
    n_pixels = 0
    film = getattr(model, "film", None)

    for lr_imagenet, _, hr_norm in tqdm(loader, desc=f"Eval {split}", leave=False):
        lr_imagenet = lr_imagenet.to(DEVICE, non_blocking=True)
        hr_norm = hr_norm.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(lr_imagenet).float()

        hr_norm = hr_norm.float()
        B = hr_norm.shape[0]

        sum_sq_err += ((pred - hr_norm) ** 2).sum().item()
        total_bias += (pred - hr_norm).mean().item()
        n_pixels += hr_norm.numel()

        pred_flat = pred.view(B, -1)
        target_flat = hr_norm.view(B, -1)
        pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
        target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
        numer = (pred_centered * target_centered).sum(dim=1)
        denom = torch.sqrt((pred_centered**2).sum(dim=1)) * torch.sqrt(
            (target_centered**2).sum(dim=1)
        )
        batch_pearson = numer / (denom + 1e-8)
        total_pearson += batch_pearson.mean().item()

        n_batches += 1

    # lr_norm is no longer needed in the loop — drop it from unpacking above
    # (change the for loop to: for lr_imagenet, _, hr_norm in ...)

    rmse = (sum_sq_err / n_pixels) ** 0.5

    if film is not None:
        print("FiLM gamma range:", film.mlp[-1].weight.abs().max().item())
        print("FiLM beta range:", film.mlp[-1].bias.abs().max().item())

    return {
        f"{split}/loss": sum_sq_err / n_pixels,
        f"{split}/rmse": rmse,
        f"{split}/pearson": total_pearson / n_batches,
        f"{split}/bias": total_bias / n_batches,
        f"{split}/rmse_vs_bilinear": rmse - bilinear_rmse,
    }


@torch.no_grad()
def compute_bilinear_rmse(loader, hr_shape: tuple) -> float:
    """
    Pre-compute bilinear baseline RMSE once before training starts.
    Pass the result to evaluate() via the bilinear_rmse argument.
    """
    sum_sq = 0.0
    n_pixels = 0
    for batch in tqdm(loader, desc="Bilinear baseline", leave=False):
        if len(batch) == 3:
            _, lr_norm, hr_norm = batch
        else:
            lr_norm, hr_norm = batch
        lr_norm = lr_norm.to(DEVICE, non_blocking=True)
        hr_norm = hr_norm.to(DEVICE, non_blocking=True).float()
        bil = F.interpolate(
            lr_norm, size=hr_shape, mode="bilinear", align_corners=False
        )
        sum_sq += ((bil - hr_norm) ** 2).sum().item()
        n_pixels += hr_norm.numel()
    return (sum_sq / n_pixels) ** 0.5


# ── Misc ──────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed}")
