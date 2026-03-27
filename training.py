import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from downscaling_dataset import DownscalingDataset
from frozen_ssl import FrozenSSLDownscaler
from ssl_downscaler import SSLDownscaler
from casd import CASD
from trainer_utils import (
    make_scheduler,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    set_seed,
)
from dotenv import load_dotenv
from argparse import ArgumentParser
from omegaconf import OmegaConf
import wandb

# ── Token config ──────────────────────────────────────────────────────────────
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
wandb_api_key = os.getenv("WANDB_API_KEY")
if not hf_token or not wandb_api_key:
    raise ValueError("HF_TOKEN and WANDB_API_KEY must be set")
else:
    os.environ["HF_TOKEN"] = hf_token
    wandb.login(wandb_api_key)

# ── Parser and config ─────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to config file")
parser.add_argument(
    "--resume", type=str, default=None, help="Path to checkpoint to resume from"
)
args = parser.parse_args()
cfg = OmegaConf.load(args.config)
set_seed(cfg.global_vars.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_REGISTRY = {
    "dinov2-l": {"model_id": "facebook/dinov2-large", "patch_size": 14},
    "dinov2-reg": {"model_id": "facebook/dinov2-large-registers", "patch_size": 14},
    "dinov3-sat": {
        "model_id": "facebook/dinov3-vitl16-pretrain-sat493m",
        "patch_size": 16,
    },
}

model_key = cfg.model.model_id
assert model_key in MODEL_REGISTRY, f"Unknown model_id '{model_key}'"
model_id = MODEL_REGISTRY[model_key]["model_id"]
patch_size = MODEL_REGISTRY[model_key]["patch_size"]

CKPT_DIR = cfg.training.get("ckpt_dir", "./checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_LATEST = os.path.join(CKPT_DIR, f"{cfg.model.upscale}x_g2g_latest.pt")
CKPT_BEST = os.path.join(CKPT_DIR, f"{cfg.model.upscale}x_g2g_best.pt")

LR_DIR = cfg.data.lr_dir
HR_DIR = cfg.data.hr_dir
print(f"Using LR directory: {LR_DIR}")
print(f"Using HR directory: {HR_DIR}")


# ── Static variable loader ────────────────────────────────────────────────────
def load_static_vars(hr_dir: str, hr_shape: tuple):
    """
    Load orography and land-sea mask from ERA5 constants file.
    Normalises orography to z-score. LSM kept as {0, 1}.
    Resizes to hr_shape if needed via bilinear / nearest interpolation.

    Returns (oro_t, lsm_t) as float32 CPU tensors of shape hr_shape.

    Tries common file names used by WeatherBench ERA5 datasets:
        constants.npz, invariant.npz, static.npz
    Keys tried: orography, z, oro; lsm, land_sea_mask, lsm_mask
    """
    candidates = ["constants.npz", "invariant.npz", "static.npz"]
    data = None
    for fname in candidates:
        fpath = os.path.join(hr_dir, fname)
        if os.path.exists(fpath):
            data = np.load(fpath)
            print(f"Loaded static variables from {fpath}")
            print(f"  Available keys: {list(data.keys())}")
            break

    if data is None:
        raise FileNotFoundError(
            f"No constants file found in {hr_dir}. "
            f"Tried: {candidates}. "
            f"Set use_static=false in config to skip static variables."
        )

    # Orography
    for key in ["orography", "z", "oro"]:
        if key in data:
            oro_np = data[key].astype(np.float32)
            break
    else:
        raise KeyError(f"No orography key found. Available: {list(data.keys())}")

    # Land-sea mask
    for key in ["lsm", "land_sea_mask", "lsm_mask"]:
        if key in data:
            lsm_np = data[key].astype(np.float32)
            break
    else:
        raise KeyError(f"No LSM key found. Available: {list(data.keys())}")

    oro_t = torch.tensor(oro_np, dtype=torch.float32)
    lsm_t = torch.tensor(lsm_np, dtype=torch.float32)

    # Squeeze to 2D if extra dims present (e.g. [1, H, W] → [H, W])
    while oro_t.dim() > 2:
        oro_t = oro_t.squeeze(0)
    while lsm_t.dim() > 2:
        lsm_t = lsm_t.squeeze(0)

    # Resize to hr_shape if not already correct
    H_hr, W_hr = hr_shape
    if tuple(oro_t.shape) != (H_hr, W_hr):
        print(f"  Resizing orography {tuple(oro_t.shape)} → {hr_shape}")
        oro_t = F.interpolate(
            oro_t[None, None],
            size=(H_hr, W_hr),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
    if tuple(lsm_t.shape) != (H_hr, W_hr):
        print(f"  Resizing LSM {tuple(lsm_t.shape)} → {hr_shape}")
        lsm_t = F.interpolate(
            lsm_t[None, None],
            size=(H_hr, W_hr),
            mode="nearest",
        ).squeeze()

    # Z-score normalise orography; clamp LSM to {0, 1}
    oro_t = (oro_t - oro_t.mean()) / (oro_t.std() + 1e-8)
    lsm_t = lsm_t.clamp(0.0, 1.0)

    print(
        f"  Orography: mean={oro_t.mean():.3f}, std={oro_t.std():.3f}, "
        f"shape={tuple(oro_t.shape)}"
    )
    print(
        f"  LSM      : min={lsm_t.min():.1f}, max={lsm_t.max():.1f}, "
        f"shape={tuple(lsm_t.shape)}"
    )

    return oro_t, lsm_t


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = DownscalingDataset(
        LR_DIR,
        HR_DIR,
        "train",
        stride=cfg.data.stride,
        lr_preload=True,  # always — LR is ~5GB total
        hr_preload=False,  # HR train is ~38GB — keep lazy
    )
    val_ds = DownscalingDataset(
        LR_DIR,
        HR_DIR,
        "val",
        stride=cfg.data.stride,
        lr_preload=True,
        hr_preload=True,  # HR val is small — preload
    )
    test_ds = DownscalingDataset(
        LR_DIR,
        HR_DIR,
        "test",
        stride=cfg.data.stride,
        lr_preload=True,
        hr_preload=True,  # HR test is small — preload
    )
    lr_shape = train_ds.lr_shape
    hr_shape = train_ds.hr_shape

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

    # ── Model ──────────────────────────────────────────────────────────────────
    mode = cfg.model.mode

    if mode == "frozen":
        model = FrozenSSLDownscaler(
            model_id=model_id,
            patch_size=patch_size,
            lr_shape=lr_shape,
            hr_shape=hr_shape,
            upscale=cfg.model.upscale,
            hidden_dim=cfg.model.hidden_dim,
        ).to(DEVICE)

        optimizer = model.make_optimizer(
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

    elif mode == "lora":
        model = SSLDownscaler(
            model_id=model_id,
            patch_size=patch_size,
            lr_shape=lr_shape,
            hr_shape=hr_shape,
            upscale=cfg.model.upscale,
            hidden_dim=cfg.model.hidden_dim,
            mode=mode,
            lora_r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
        ).to(DEVICE)

        optimizer = model.make_optimizer(
            lr=cfg.training.lr,
            decoder_lr=cfg.training.decoder_lr,
            weight_decay=cfg.training.weight_decay,
        )

    elif mode in ("casd_frozen", "casd_lora"):
        # ── Load static variables if requested ────────────────────────────
        use_static = cfg.model.get("use_static", False)
        oro_t, lsm_t = None, None
        if use_static:
            oro_t, lsm_t = load_static_vars(HR_DIR, hr_shape)

        # Encoder mode: casd_frozen → "frozen", casd_lora → "lora"
        encoder_mode = "frozen" if mode == "casd_frozen" else "lora"

        model = CASD(
            model_id=model_id,
            patch_size=patch_size,
            lr_shape=lr_shape,
            hr_shape=hr_shape,
            mode=encoder_mode,
            lora_r=cfg.model.get("lora_r", 8),
            lora_alpha=cfg.model.get("lora_alpha", None),
            lora_dropout=cfg.model.get("lora_dropout", 0.0),
            tap_layers=list(cfg.model.get("tap_layers", [4, 7, 14])),
            proj_dim=cfg.model.get("proj_dim", 256),
            n_heads=cfg.model.get("n_heads", 8),
            n_ca_layers=cfg.model.get("n_ca_layers", 2),
            ca_dropout=cfg.model.get("ca_dropout", 0.0),
            use_static=use_static,
            oro_hr=oro_t,
            lsm_hr=lsm_t,
        ).to(DEVICE)

        optimizer = model.make_optimizer(
            lr=cfg.training.get("lr", 5e-5),
            decoder_lr=cfg.training.decoder_lr,
            weight_decay=cfg.training.weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Expected: frozen | lora | casd_frozen | casd_lora"
        )

    scheduler = make_scheduler(
        optimizer,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )
    scaler = torch.amp.GradScaler("cuda")

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

    # Watch the appropriate decoder module for gradient logging
    if mode in ("casd_frozen", "casd_lora"):
        run.watch(model.output_head, log="all", log_freq=50)
    else:
        run.watch(model.decoder, log="all", log_freq=50)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            epoch,
            grad_accum_steps=cfg.training.grad_accum_steps,
        )
        val_metrics = evaluate(model, val_loader, hr_shape, split="val")
        scheduler.step()

        current_lr = float(scheduler.get_last_lr()[0])

        log_dict = {
            "epoch": epoch,
            "lr": current_lr,
            **{k: float(v) for k, v in train_metrics.items()},
            **{k: float(v) for k, v in val_metrics.items()},
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

        # ── Save latest checkpoint every epoch ────────────────────────────
        save_checkpoint(
            CKPT_LATEST,
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
        )

        # ── Save best checkpoint ───────────────────────────────────────────
        if val_metrics["val/rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["val/rmse"]
            epochs_no_improve = 0
            save_checkpoint(
                CKPT_BEST,
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
            )
            run.log({"best/val_rmse": float(best_val_rmse)}, step=epoch, commit=False)
            print(f"  → New best: {best_val_rmse:.4f} (saved to {CKPT_BEST})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.patience:
                print(f"  → Early stopping at epoch {epoch}")
                break

    # ── Test — load best checkpoint ────────────────────────────────────────────
    print(f"\nLoading best checkpoint for test evaluation: {CKPT_BEST}")
    best_ckpt = torch.load(CKPT_BEST, map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, hr_shape, split="test")
    improvement = (
        (test_metrics["test/bilinear_rmse"] - test_metrics["test/rmse"])
        / test_metrics["test/bilinear_rmse"]
        * 100
    )

    run.log(
        {k: float(v) for k, v in test_metrics.items()}
        | {"test/improvement_pct": float(improvement)},
        commit=True,
    )
    run.summary["test/rmse"] = float(test_metrics["test/rmse"])
    run.summary["test/pearson"] = float(test_metrics["test/pearson"])
    run.summary["test/bilinear_rmse"] = float(test_metrics["test/bilinear_rmse"])
    run.summary["test/improvement_pct"] = float(improvement)

    print(f"\n── Test Results ({model_key} / {mode}) ──────────────────────")
    for k, v in test_metrics.items():
        print(f"  {k:<30}: {v:.4f}")
    print(f"  {'test/improvement_pct':<30}: {improvement:.1f}%")

    run.finish()


if __name__ == "__main__":
    main()
