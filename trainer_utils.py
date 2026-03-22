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
    torch.save(
        {
            "epoch": epoch,
            "decoder": model.decoder.state_dict(),
            "feat_norm": model.feat_norm.state_dict(),
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
    Returns (start_epoch, best_val_rmse, epochs_no_improve, wandb_run_id).
    """
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=DEVICE)

    model.decoder.load_state_dict(ckpt["decoder"])
    model.feat_norm.load_state_dict(ckpt["feat_norm"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])

    # Restore scheduler state directly (preserves exact LR curve position)
    scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt["epoch"] + 1
    best_val_rmse = ckpt["best_val_rmse"]
    epochs_no_improve = ckpt["epochs_no_improve"]
    wandb_run_id = ckpt.get("wandb_run_id", None)

    print(
        f"Resumed from epoch {ckpt['epoch']} |  "
        f"best_val_rmse={best_val_rmse:.4f} |  "
        f"epochs_no_improve={epochs_no_improve} "
    )
    return start_epoch, best_val_rmse, epochs_no_improve, wandb_run_id


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.decoder.train()
    model.feat_norm.train()
    model.encoder.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0

    for lr_imagenet, _, hr_norm in tqdm(loader, desc=f"Train {epoch}", leave=False):
        lr_imagenet = lr_imagenet.to(DEVICE)
        hr_norm = hr_norm.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(lr_imagenet)
            loss = F.mse_loss(pred, hr_norm)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.decoder.parameters()) + list(model.feat_norm.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_rmse += loss.item() ** 0.5
        n_batches += 1

    return {
        "train/loss": total_loss / n_batches,
        "train/rmse": total_rmse / n_batches,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, hr_shape, split="val"):
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    total_pearson = 0.0
    total_bias = 0.0
    bilinear_loss = 0.0
    bilinear_rmse = 0.0
    n_batches = 0

    for lr_imagenet, lr_norm, hr_norm in tqdm(
        loader, desc=f"Eval {split}", leave=False
    ):
        lr_imagenet = lr_imagenet.to(DEVICE)
        lr_norm = lr_norm.to(DEVICE)
        hr_norm = hr_norm.to(DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(lr_imagenet).float()

        hr_norm = hr_norm.float()

        mse = F.mse_loss(pred, hr_norm).item()
        rmse = mse**0.5
        bias = (pred - hr_norm).mean().item()

        pred_f = pred.flatten()
        tgt_f = hr_norm.flatten()
        pred_c = pred_f - pred_f.mean()
        tgt_c = tgt_f - tgt_f.mean()
        pearson = (
            (pred_c * tgt_c).sum()
            / (torch.sqrt((pred_c**2).sum()) * torch.sqrt((tgt_c**2).sum()) + 1e-8)
        ).item()

        bil = F.interpolate(
            lr_norm, size=hr_shape, mode="bilinear", align_corners=False
        )
        bil_mse = F.mse_loss(bil, hr_norm).item()
        bil_rmse = bil_mse**0.5

        total_loss += mse
        total_rmse += rmse
        total_pearson += pearson
        total_bias += bias
        bilinear_loss += bil_mse
        bilinear_rmse += bil_rmse
        n_batches += 1

    avg_rmse = total_rmse / n_batches
    avg_bil_rmse = bilinear_rmse / n_batches

    return {
        f"{split}/loss": total_loss / n_batches,
        f"{split}/rmse": avg_rmse,
        f"{split}/pearson": total_pearson / n_batches,
        f"{split}/bias": total_bias / n_batches,
        f"{split}/bilinear_loss": bilinear_loss / n_batches,
        f"{split}/bilinear_rmse": avg_bil_rmse,
        f"{split}/rmse_vs_bilinear": avg_rmse - avg_bil_rmse,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
