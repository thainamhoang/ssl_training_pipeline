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

    # frozen: encoder stays eval (no grads, no state update needed)
    # lora:   encoder stays train (LoRA params are active)
    if model.mode == "frozen":
        model.encoder.eval()
    else:
        model.encoder.train()

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

        # Must unscale before clipping — otherwise clipping scaled gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
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

    sum_sq_err = 0.0  # accumulate MSE numerator for true RMSE
    sum_bil_sq = 0.0
    total_pearson = 0.0
    total_bias = 0.0
    n_batches = 0
    n_pixels = 0  # total pixels for true MSE denominator

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

        # True MSE accumulation — sum of squared errors, divide once at end
        sq_err = ((pred - hr_norm) ** 2).sum().item()
        sum_sq_err += sq_err
        n_pixels += hr_norm.numel()

        # Bias — mean over batch
        total_bias += (pred - hr_norm).mean().item()

        # Pearson — per sample, then average
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

        # Bilinear baseline
        bil = F.interpolate(
            lr_norm, size=hr_shape, mode="bilinear", align_corners=False
        )
        bil_sq = ((bil - hr_norm) ** 2).sum().item()
        sum_bil_sq += bil_sq

        n_batches += 1

    # True RMSE from accumulated squared errors
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
