"""
training.py

Unified training script for all SSL downscaling models:
    frozen      → FrozenSSLDownscaler  (Pilot 2)
    lora        → SSLDownscaler        (Pilot 3)
    casd_frozen → CASD frozen          (Pilot 4)
    casd_lora   → CASD LoRA            (Pilot 4)
    fgd_frozen  → FGD frozen           (Pilot 5)
    fgd_lora    → FGD LoRA             (Pilot 5)

Usage:
    python training.py --config configs/lora.yaml
    python training.py --config configs/fgd_lora.yaml --resume checkpoints/latest.pt

Expected config fields:
    global_vars: seed
    model:       model_id, mode, upscale, hidden_dim,
                 lora_r, lora_alpha, lora_dropout,      # LoRA models
                 tap_layers, proj_dim, n_heads,          # CASD
                 n_ca_layers, ca_dropout, use_static,   # CASD
                 use_multiscale, use_film, film_hidden,  # FGD
                 use_cls, cls_hidden, cls_layer_idx,     # FGD
    data:        lr_dir, hr_dir, stride, batch_size,
                 num_workers, pin_memory, persistent_workers, prefetch_factor
    training:    lr, decoder_lr, weight_decay,
                 warmup_epochs, max_epochs, grad_accum_steps, patience,
                 spectral_lambda,                        # FGD only, default 0.0
                 ckpt_dir
    wandb:       entity, project, name
"""

import os
import functools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from omegaconf import OmegaConf
import wandb
from dotenv import load_dotenv

from dataset import DownscalingDataset, BilinearBaselineView
from model import FrozenSSLDownscaler, SSLDownscaler, CASD, FGD, fgd_loss
from trainer_utils import (
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    load_checkpoint,
    train_one_epoch,
    evaluate,
    compute_bilinear_rmse,
    set_seed,
)

# ── Token config ──────────────────────────────────────────────────────────────

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
wandb_api_key = os.getenv("WANDB_API_KEY")
if not hf_token or not wandb_api_key:
    raise ValueError("HF_TOKEN and WANDB_API_KEY must be set in environment / .env")
os.environ["HF_TOKEN"] = hf_token
wandb.login(wandb_api_key)

# ── CLI + config ──────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
parser.add_argument(
    "--resume", type=str, default=None, help="Path to checkpoint to resume from"
)
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

set_seed(cfg.global_vars.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "dinov2-l": {"model_id": "facebook/dinov2-large", "patch_size": 14},
    "dinov2-reg": {"model_id": "facebook/dinov2-large-registers", "patch_size": 14},
    "dinov3-sat": {
        "model_id": "facebook/dinov3-vitl16-pretrain-sat493m",
        "patch_size": 16,
    },
}

model_key = cfg.model.model_id
assert model_key in MODEL_REGISTRY, (
    f"Unknown model_id '{model_key}'. Valid keys: {list(MODEL_REGISTRY)}"
)
model_id = MODEL_REGISTRY[model_key]["model_id"]
patch_size = MODEL_REGISTRY[model_key]["patch_size"]

# ── Checkpoint paths ──────────────────────────────────────────────────────────

CKPT_DIR = cfg.training.get("ckpt_dir", "./checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_LATEST = os.path.join(CKPT_DIR, f"{cfg.model.upscale}x_g2g_latest.pt")
CKPT_BEST = os.path.join(CKPT_DIR, f"{cfg.model.upscale}x_g2g_best.pt")


def load_static_vars(hr_dir: str, hr_shape: tuple):
    """
    Load orography and land-sea mask from ERA5 constants file.

    Tries common WeatherBench filenames: constants.npz, invariant.npz, static.npz
    Orography keys tried: orography, z, oro
    LSM keys tried:       lsm, land_sea_mask, lsm_mask

    Returns (oro_t, lsm_t) — float32 CPU tensors of shape hr_shape.
    Orography is z-score normalised; LSM is clamped to [0, 1].
    """
    for fname in ("constants.npz", "invariant.npz", "static.npz"):
        fpath = os.path.join(hr_dir, fname)
        if os.path.exists(fpath):
            data = np.load(fpath)
            print(f"Loaded static variables from {fpath}")
            print(f"  Available keys: {list(data.keys())}")
            break
    else:
        raise FileNotFoundError(
            f"No constants file found in {hr_dir}. "
            "Set use_static=false in config to skip static variables."
        )

    for key in ("orography", "z", "oro"):
        if key in data:
            oro_np = data[key].astype(np.float32)
            break
    else:
        raise KeyError(f"No orography key found. Available: {list(data.keys())}")

    for key in ("lsm", "land_sea_mask", "lsm_mask"):
        if key in data:
            lsm_np = data[key].astype(np.float32)
            break
    else:
        raise KeyError(f"No LSM key found. Available: {list(data.keys())}")

    oro_t = torch.tensor(oro_np)
    lsm_t = torch.tensor(lsm_np)

    while oro_t.dim() > 2:
        oro_t = oro_t.squeeze(0)
    while lsm_t.dim() > 2:
        lsm_t = lsm_t.squeeze(0)

    H_hr, W_hr = hr_shape
    if tuple(oro_t.shape) != (H_hr, W_hr):
        print(f"  Resizing orography {tuple(oro_t.shape)} → {hr_shape}")
        oro_t = F.interpolate(
            oro_t[None, None], size=(H_hr, W_hr), mode="bilinear", align_corners=False
        ).squeeze()
    if tuple(lsm_t.shape) != (H_hr, W_hr):
        print(f"  Resizing LSM {tuple(lsm_t.shape)} → {hr_shape}")
        lsm_t = F.interpolate(
            lsm_t[None, None], size=(H_hr, W_hr), mode="nearest"
        ).squeeze()

    oro_t = (oro_t - oro_t.mean()) / (oro_t.std() + 1e-8)
    lsm_t = lsm_t.clamp(0.0, 1.0)

    print(
        f"  Orography : mean={oro_t.mean():.3f}, std={oro_t.std():.3f}, shape={tuple(oro_t.shape)}"
    )
    print(
        f"  LSM       : min={lsm_t.min():.1f},  max={lsm_t.max():.1f},  shape={tuple(lsm_t.shape)}"
    )
    return oro_t, lsm_t


def _build_model(mode, model_id, patch_size, lr_shape, hr_shape, cfg):
    """
    Construct the correct model class from config mode string.
    Returns model on CPU — caller moves to DEVICE.
    """
    m = cfg.model

    base_kwargs = dict(
        model_id=model_id,
        patch_size=patch_size,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        upscale=m.upscale,
        hidden_dim=m.hidden_dim,
    )
    lora_kwargs = dict(
        lora_r=m.get("lora_r", 8),
        lora_alpha=m.get("lora_alpha", None),
        lora_dropout=m.get("lora_dropout", 0.0),
    )

    if mode == "frozen":
        return FrozenSSLDownscaler(**base_kwargs)

    if mode == "lora":
        return SSLDownscaler(**base_kwargs, mode="lora", **lora_kwargs)

    if mode == "casd":
        use_static = m.get("use_static", False)
        oro_t, lsm_t = (
            load_static_vars(cfg.data.hr_dir, hr_shape) if use_static else (None, None)
        )
        return CASD(
            **base_kwargs,
            mode=m.encoder_mode,
            **lora_kwargs,
            tap_layers=list(m.get("tap_layers", [4, 7, 14])),
            proj_dim=m.get("proj_dim", 256),
            n_heads=m.get("n_heads", 8),
            n_ca_layers=m.get("n_ca_layers", 2),
            ca_dropout=m.get("ca_dropout", 0.0),
            use_static=use_static,
            oro_hr=oro_t,
            lsm_hr=lsm_t,
        )

    if mode == "fgd":
        return FGD(
            **base_kwargs,
            mode=m.encoder_mode,
            **lora_kwargs,
            use_multiscale=m.get("use_multiscale", True),
            tap_layers=list(m.get("tap_layers", [4, 7, 14])),
            use_film=m.get("use_film", False),
            film_hidden=m.get("film_hidden", 128),
            use_cls=m.get("use_cls", False),
            cls_hidden=m.get("cls_hidden", 128),
            cls_layer_idx=m.get("cls_layer_idx", -1),
        )

    raise ValueError(f"Unknown mode '{mode}'. Expected: frozen | lora | casd | fgd")


def _watched_module(model, mode):
    """Return the decoder submodule to watch in WandB."""
    if mode in ("casd_frozen", "casd_lora"):
        return model.output_head
    return model.decoder


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    mode = cfg.model.mode
    LR_DIR = cfg.data.lr_dir
    HR_DIR = cfg.data.hr_dir

    print(f"LR directory : {LR_DIR}")
    print(f"HR directory : {HR_DIR}")

    # ── Datasets + loaders ────────────────────────────────────────────────────
    train_ds = DownscalingDataset(
        LR_DIR,
        HR_DIR,
        "train",
        stride=cfg.data.stride,
        lr_preload=True,
        hr_preload=False,
    )
    val_ds = DownscalingDataset(
        LR_DIR, HR_DIR, "val", stride=cfg.data.stride, lr_preload=True, hr_preload=True
    )
    test_ds = DownscalingDataset(
        LR_DIR, HR_DIR, "test", stride=cfg.data.stride, lr_preload=True, hr_preload=True
    )

    lr_shape = train_ds.lr_shape
    hr_shape = train_ds.hr_shape
    val_baseline_ds = BilinearBaselineView(val_ds)
    test_baseline_ds = BilinearBaselineView(test_ds)

    pw = cfg.data.persistent_workers if cfg.data.num_workers > 0 else False
    pf = cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None

    def make_loader(ds, shuffle):
        kwargs = dict(
            batch_size=cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=pw,
            worker_init_fn=ds.worker_init_fn if cfg.data.num_workers > 0 else None,
        )
        if pf is not None:
            kwargs["prefetch_factor"] = pf
        return DataLoader(ds, **kwargs)

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)
    val_baseline_loader = make_loader(val_baseline_ds, shuffle=False)
    test_baseline_loader = make_loader(test_baseline_ds, shuffle=False)

    # ── Bilinear baselines (computed once, never changes) ─────────────────────
    print("Computing bilinear baselines...")
    val_bilinear_rmse = compute_bilinear_rmse(val_baseline_loader, hr_shape)
    test_bilinear_rmse = compute_bilinear_rmse(test_baseline_loader, hr_shape)
    print(f"  Val  bilinear RMSE : {val_bilinear_rmse:.4f}")
    print(f"  Test bilinear RMSE : {test_bilinear_rmse:.4f}")

    # ── Model + optimizer ──────────────────────────────────────────────────────
    model = _build_model(mode, model_id, patch_size, lr_shape, hr_shape, cfg).to(DEVICE)

    optimizer = make_optimizer(
        model,
        lr=cfg.training.get("lr", 5e-5),
        decoder_lr=cfg.training.decoder_lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = make_scheduler(
        optimizer,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Loss function ──────────────────────────────────────────────────────────
    spectral_lambda = cfg.training.get("spectral_lambda", 0.0)
    loss_fn = (
        functools.partial(fgd_loss, spectral_lambda=spectral_lambda)
        if spectral_lambda > 0.0
        else None
    )

    # ── Resume or fresh start ──────────────────────────────────────────────────
    start_epoch = 1
    best_val_rmse = float("inf")
    epochs_no_improve = 0
    wandb_run_id = None

    resume_path = args.resume or (CKPT_LATEST if os.path.exists(CKPT_LATEST) else None)
    if resume_path and os.path.exists(resume_path):
        start_epoch, best_val_rmse, epochs_no_improve, wandb_run_id = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )
    else:
        print("Starting fresh training run.")

    # ── WandB ──────────────────────────────────────────────────────────────────
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        id=wandb_run_id,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_run_id = run.id
    print(f"WandB run: {run.url}")
    if cfg.wandb.get("watch", False):
        run.watch(
            _watched_module(model, mode),
            log=cfg.wandb.get("watch_log", "gradients"),
            log_freq=cfg.wandb.get("watch_log_freq", 200),
        )

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch,
            grad_accum_steps=cfg.training.grad_accum_steps,
            loss_fn=loss_fn,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            bilinear_rmse=val_bilinear_rmse,
            split="val",
        )
        scheduler.step()

        is_new_best = val_metrics["val/rmse"] < best_val_rmse
        if is_new_best:
            best_val_rmse = val_metrics["val/rmse"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        ckpt_kwargs = dict(
            epoch=epoch,
            model=model,
            model_key=model_key,
            cfg=cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epochs_no_improve=epochs_no_improve,
            wandb_run_id=wandb_run_id,
        )

        save_checkpoint(CKPT_LATEST, best_val_rmse=best_val_rmse, **ckpt_kwargs)

        current_lr = float(scheduler.get_last_lr()[0])
        log_dict = {
            "epoch": epoch,
            "lr": current_lr,
            **{k: float(v) for k, v in train_metrics.items()},
            **{k: float(v) for k, v in val_metrics.items()},
            "best/val_rmse": float(best_val_rmse),
        }
        run.log(log_dict, step=epoch, commit=True)

        print(
            f"Epoch {epoch:3d}/{cfg.training.max_epochs} |  "
            f"LR: {current_lr:.2e} |  "
            f"Loss: {train_metrics['train/loss']:.4f} |  "
            f"Val RMSE: {val_metrics['val/rmse']:.4f} |  "
            f"Pearson: {val_metrics['val/pearson']:.4f} |  "
            f"vs Bilinear: {val_metrics['val/rmse_vs_bilinear']:+.4f}"
        )

        if is_new_best:
            save_checkpoint(CKPT_BEST, best_val_rmse=best_val_rmse, **ckpt_kwargs)
            print(f"  → New best: {best_val_rmse:.4f} (saved to {CKPT_BEST})")
        else:
            if epochs_no_improve >= cfg.training.patience:
                print(f"  → Early stopping at epoch {epoch}")
                break

    # ── Test — load best checkpoint ────────────────────────────────────────────
    print(f"\nLoading best checkpoint for test evaluation: {CKPT_BEST}")
    best_ckpt = torch.load(CKPT_BEST, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])

    test_metrics = evaluate(
        model,
        test_loader,
        bilinear_rmse=test_bilinear_rmse,
        split="test",
    )

    run.log(
        {k: float(v) for k, v in test_metrics.items()},
        commit=True,
    )
    run.summary["test/rmse"] = float(test_metrics["test/rmse"])
    run.summary["test/pearson"] = float(test_metrics["test/pearson"])

    print(f"\n── Test Results ({model_key} / {mode}) ──────────────────────")
    for k, v in test_metrics.items():
        print(f"  {k:<30}: {v:.4f}")

    run.finish()


if __name__ == "__main__":
    main()
