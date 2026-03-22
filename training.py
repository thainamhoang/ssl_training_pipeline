import os
import torch
from torch.utils.data import DataLoader
from downscaling_dataset import DownscalingDataset
from frozen_ssl import FrozenSSLDownscaler
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

# ── Token config ─────────────────────────────────────────────────────────
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
    "--resume", type=str, default=None, help="Path to checkpoint dir to resume from"
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

CKPT_DIR = os.path.join(cfg.training.get("ckpt_dir", "./checkpoints"), model_key)
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_LATEST = os.path.join(CKPT_DIR, f"{cfg.model.upscale}x_g2g_latest.pt")
CKPT_BEST = os.path.join(CKPT_DIR, f"{cfg.model.upscale}x_g2g_best.pt")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = DownscalingDataset(
        cfg.data.lr_dir,
        cfg.data.hr_dir,
        "train",
        stride=cfg.data.stride,
        preload=cfg.data.preload,
    )
    val_ds = DownscalingDataset(
        cfg.data.lr_dir,
        cfg.data.hr_dir,
        "val",
        stride=cfg.data.stride,
        preload=cfg.data.preload,
    )
    test_ds = DownscalingDataset(
        cfg.data.lr_dir,
        cfg.data.hr_dir,
        "test",
        stride=cfg.data.stride,
        preload=cfg.data.preload,
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

    # ── Model ──────────────────────────────────────────────────────────────
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
    scheduler = make_scheduler(
        optimizer,
        warmup_epochs=cfg.training.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume or fresh start ──────────────────────────────────────────────
    start_epoch = 1
    best_val_rmse = float("inf")
    epochs_no_improve = 0
    wandb_run_id = None

    resume_path = args.resume or (CKPT_LATEST if os.path.exists(CKPT_LATEST) else None)

    if resume_path and os.path.exists(resume_path):
        start_epoch, best_val_rmse, epochs_no_improve, wandb_run_id = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )
        # Scheduler state is restored from checkpoint — no fast-forward needed
    else:
        print("Starting fresh training run. ")

    # ── WandB — resume if run_id exists ────────────────────────────────────
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        id=wandb_run_id,  # None = new run, str = resume existing
        resume="allow",  # allow resume if id matches
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    wandb_run_id = run.id  # store for next checkpoint
    print(f"WandB run: {run.url} ")
    run.watch(model.decoder, log="all", log_freq=50)

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.training.max_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
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
            f"vs Bilinear: {val_metrics['val/rmse_vs_bilinear']:+.4f} "
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
            print(f"  → New best: {best_val_rmse:.4f} (saved to {CKPT_BEST}) ")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.patience:
                print(f"  → Early stopping at epoch {epoch} ")
                break

    # ── Test ───────────────────────────────────────────────────────────────
    best_ckpt = torch.load(CKPT_BEST, map_location=DEVICE)
    model.decoder.load_state_dict(best_ckpt["decoder"])
    model.feat_norm.load_state_dict(best_ckpt["feat_norm"])

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

    print(f"\n── Test Results ({model_key}) ────────────────────── ")
    for k, v in test_metrics.items():
        print(f"  {k: <30}: {v:.4f} ")
    print(f"  {'test/improvement_pct': <30}: {improvement:.1f}% ")

    run.finish()


if __name__ == "__main__":
    main()
