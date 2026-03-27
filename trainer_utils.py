import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from omegaconf import OmegaConf

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Scheduler ─────────────────────────────────────────────────────────────────
def make_scheduler(optimizer, warmup_epochs, max_epochs):
    """
    warmup_epochs=0 → pure CosineAnnealingLR
    warmup_epochs>0 → LinearLR warmup then CosineAnnealingLR
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


def fast_forward_scheduler(scheduler, start_epoch):
    """
    Advance scheduler to match a resumed epoch without changing
    the optimizer's param groups (no actual LR change until correct epoch).
    Silently steps the scheduler's internal counter to start_epoch.
    """
    if start_epoch <= 1:
        return
    print(f"Fast-forwarding scheduler to epoch {start_epoch}...")
    for _ in range(start_epoch - 1):
        scheduler.step()


# ── Checkpoint save / load ────────────────────────────────────────────────────
def save_checkpoint(
    path,
    epoch,
    model,
    model_key,
    cfg,
    optimizer,
    scheduler,
    scaler,
    best_val_rmse,
    epochs_no_improve,
    wandb_run_id,
):
    """
    Save full training state.

    Uses model.state_dict() rather than named submodule state dicts —
    works for all model types: FrozenSSLDownscaler, SSLDownscaler, CASD.
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


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    """
    Load checkpoint and return training state.

    Supports two checkpoint formats:
        New (CASD-compatible): "model_state" key → model.load_state_dict()
        Old (Pilot 2/3):       "decoder" + "feat_norm" keys → submodule load

    Old format is detected automatically — no manual flag needed.
    Returns (start_epoch, best_val_rmse, epochs_no_improve, wandb_run_id).
    """
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=DEVICE)

    if "model_state" in ckpt:
        # New format — works for FrozenSSLDownscaler, SSLDownscaler, CASD
        model.load_state_dict(ckpt["model_state"])
    elif "decoder" in ckpt and "feat_norm" in ckpt:
        # Old format — Pilot 2/3 checkpoints (decoder + feat_norm only)
        print("  Detected old checkpoint format (decoder + feat_norm). Loading...")
        model.decoder.load_state_dict(ckpt["decoder"])
        model.feat_norm.load_state_dict(ckpt["feat_norm"])
    else:
        raise KeyError(
            f"Unrecognised checkpoint format. Keys found: {list(ckpt.keys())}"
        )

    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    scheduler.load_state_dict(ckpt["scheduler"])

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


# ── Train/eval mode helpers ────────────────────────────────────────────────────
def _set_train_mode(model):
    """
    Set correct train/eval modes depending on model type.

    FrozenSSLDownscaler / SSLDownscaler (frozen):
        encoder.eval(), decoder.train(), feat_norm.train()

    SSLDownscaler (lora):
        encoder.train(), decoder.train(), feat_norm.train()

    CASD (frozen):
        encoder.eval(), all decoder submodules train()

    CASD (lora):
        encoder.train(), all decoder submodules train()
    """
    mode = getattr(model, "mode", "frozen")

    # Encoder
    if mode == "frozen":
        model.encoder.eval()
    else:
        model.encoder.train()

    # Decoder submodules — set everything that isn't the encoder to train()
    # Works for both SSLDownscaler (decoder + feat_norm) and CASD
    # (token_projs + coord_proj + ca_blocks + output_head)
    for name, module in model.named_children():
        if name != "encoder":
            module.train()


def _set_eval_mode(model):
    """Set full eval mode — used during evaluation."""
    model.eval()


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, epoch, grad_accum_steps=1):
    _set_train_mode(model)

    total_loss = 0.0
    n_samples = 0
    sum_sq_err = 0.0

    optimizer.zero_grad()

    for i, (lr_imagenet, _, hr_norm) in enumerate(
        tqdm(loader, desc=f"Train {epoch}", leave=False)
    ):
        lr_imagenet = lr_imagenet.to(DEVICE)
        hr_norm = hr_norm.to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(lr_imagenet)
            loss = F.mse_loss(pred, hr_norm) / grad_accum_steps

        scaler.scale(loss).backward()

        real_mse = loss.item() * grad_accum_steps
        sum_sq_err += real_mse * hr_norm.numel()
        n_samples += hr_norm.numel()
        total_loss += real_mse

        is_last_batch = i + 1 == len(loader)
        if (i + 1) % grad_accum_steps == 0 or is_last_batch:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    n_steps = len(loader)
    return {
        "train/loss": total_loss / n_steps,
        "train/rmse": (sum_sq_err / n_samples) ** 0.5,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, hr_shape, split="val"):
    _set_eval_mode(model)

    sum_sq_err = 0.0
    sum_bil_sq = 0.0
    total_pearson = 0.0
    total_bias = 0.0
    n_batches = 0
    n_pixels = 0

    for lr_imagenet, lr_norm, hr_norm in tqdm(
        loader, desc=f"Eval {split}", leave=False
    ):
        lr_imagenet = lr_imagenet.to(DEVICE)
        lr_norm = lr_norm.to(DEVICE)
        hr_norm = hr_norm.to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(lr_imagenet).float()

        hr_norm = hr_norm.float()
        B = hr_norm.shape[0]

        sq_err = ((pred - hr_norm) ** 2).sum().item()
        sum_sq_err += sq_err
        n_pixels += hr_norm.numel()

        total_bias += (pred - hr_norm).mean().item()

        batch_pearson = 0.0
        for i in range(B):
            p = pred[i].flatten()
            t = hr_norm[i].flatten()
            pc = p - p.mean()
            tc = t - t.mean()
            r = (pc * tc).sum() / (
                torch.sqrt((pc**2).sum()) * torch.sqrt((tc**2).sum()) + 1e-8
            )
            batch_pearson += r.item()
        total_pearson += batch_pearson / B

        bil = F.interpolate(
            lr_norm, size=hr_shape, mode="bilinear", align_corners=False
        )
        bil_sq = ((bil - hr_norm) ** 2).sum().item()
        sum_bil_sq += bil_sq

        n_batches += 1

    true_mse = sum_sq_err / n_pixels
    true_rmse = true_mse**0.5
    bil_mse = sum_bil_sq / n_pixels
    bil_rmse = bil_mse**0.5

    return {
        f"{split}/loss": true_mse,
        f"{split}/rmse": true_rmse,
        f"{split}/pearson": total_pearson / n_batches,
        f"{split}/bias": total_bias / n_batches,
        f"{split}/bilinear_rmse": bil_rmse,
        f"{split}/rmse_vs_bilinear": true_rmse - bil_rmse,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed}")
