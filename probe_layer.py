"""
probe_layers.py

Layer-wise feature analysis for DINOv3-SAT on ERA5 climate data.

For each of the 24 transformer layers, extracts patch tokens, runs PCA,
and computes Spearman correlation of each PC against:
    - Raw T2m field (flattened to patch resolution)
    - Latitude grid (proxy for large-scale temperature gradient)

Outputs:
    1. probe_results.json  — correlation values per layer per PC
    2. layer_probe.png     — plot of best T2m and lat correlation vs layer depth
    3. Console summary     — which layers to tap for CASD (early/mid/late)

Usage:
    python probe_layers.py \
        --lr_dir  /workspace/data/5.625deg/2m_temperature \
        --model_id facebook/dinov3-vitl16-pretrain-sat493m \
        --n_samples 30 \
        --out_dir ./probe_results

Runtime: ~10-20 min on a single GPU for 30 samples × 24 layers.
"""

import os
import argparse
import json
import glob

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from scipy.stats import spearmanr
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Constants ──────────────────────────────────────────────────────────────

PATCH_SIZE = 16  # DINOv3-SAT
N_LAYERS = 24  # DINOv3-SAT ViT-L
SSL_H, SSL_W = 256, 512  # encoder input size (16×32 = 512 patch tokens)
N_PATCHES_H = SSL_H // PATCH_SIZE  # 16
N_PATCHES_W = SSL_W // PATCH_SIZE  # 32
N_PATCHES = N_PATCHES_H * N_PATCHES_W  # 512
N_PCS = 5  # number of PCA components to analyse per layer

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

LR_MEAN = 278.45  # K
LR_STD = 21.25  # K


# ── Data loading ───────────────────────────────────────────────────────────


def load_samples(lr_dir: str, n_samples: int, device: torch.device):
    """
    Load n_samples ERA5 LR timesteps and return:
        x_ssl   : [N, 3, SSL_H, SSL_W]  ImageNet-normalised encoder input
        t2m_patch: [N, N_PATCHES]        T2m field downsampled to patch grid
        lat_patch: [N_PATCHES]           latitude value per patch (same for all N)
    """
    shards = sorted(glob.glob(os.path.join(lr_dir, "train", "*.npz")))
    shards = [f for f in shards if "climatology" not in f]
    assert len(shards) > 0, f"No shards found in {lr_dir}/train/"

    samples = []
    with tqdm(total=n_samples, desc="Loading samples", unit="sample") as pbar:
        for shard_path in shards:
            data = np.load(shard_path)["2m_temperature"]  # [T, 1, H_lr, W_lr]
            for t in range(data.shape[0]):
                samples.append(data[t, 0])  # [H_lr, W_lr] raw Kelvin
                pbar.update(1)
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break

    samples = samples[:n_samples]
    print(f"Loaded {len(samples)} samples from {len(shards)} shards")

    # Peek shape
    H_lr, W_lr = samples[0].shape
    print(f"LR grid shape: {H_lr}×{W_lr}")

    # Build encoder input: raw K → z-score → [0,1] → ImageNet → [N,3,SSL_H,SSL_W]
    x_list = []
    for s in tqdm(samples, desc="Preprocessing samples", unit="sample"):
        t = torch.tensor(s, dtype=torch.float32)  # [H_lr, W_lr]
        z = (t - LR_MEAN) / LR_STD
        t01 = torch.clamp(z / 6.0 + 0.5, 0.0, 1.0)
        x3 = t01.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)  # [1,3,H,W]
        x3 = (x3 - IMAGENET_MEAN) / IMAGENET_STD
        x_ssl = F.interpolate(
            x3, size=(SSL_H, SSL_W), mode="bilinear", align_corners=False
        )
        x_list.append(x_ssl)
    x_ssl = torch.cat(x_list, dim=0).to(device)  # [N,3,256,512]

    # T2m at patch resolution: downsample to [N_PATCHES_H, N_PATCHES_W]
    t2m_stack = (
        torch.tensor(np.stack(samples, axis=0), dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )  # [N,1,H_lr,W_lr]
    t2m_patch_2d = F.adaptive_avg_pool2d(
        t2m_stack, (N_PATCHES_H, N_PATCHES_W)
    )  # [N,1,16,32]
    t2m_patch = t2m_patch_2d.squeeze(1).reshape(len(samples), -1).cpu().numpy()
    # [N, N_PATCHES]

    # Latitude grid at patch resolution
    # ERA5 5.625° LR grid: 32 rows (lat) × 64 cols (lon)
    # Patch grid: 16 rows × 32 cols
    lats = np.linspace(90, -90, N_PATCHES_H)  # descending (north→south)
    lons = np.linspace(0, 360, N_PATCHES_W, endpoint=False)
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    lat_patch = lat_grid.flatten()  # [N_PATCHES]

    return x_ssl, t2m_patch, lat_patch


# ── PCA ───────────────────────────────────────────────────────────────────


def run_pca(tokens: np.ndarray, n_components: int = N_PCS):
    """
    Lightweight PCA via SVD on token matrix.

    Args:
        tokens : [N_samples * N_patches, enc_dim]  stacked patch tokens
        n_components : number of PCs to return

    Returns:
        scores : [N_samples * N_patches, n_components]  PC scores
        var_ratio : [n_components]  variance explained ratio
    """
    X = tokens - tokens.mean(axis=0, keepdims=True)
    # Use randomised SVD for speed — exact SVD is slow for 1024-dim
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]
    var_total = (S**2).sum()
    var_ratio = (S[:n_components] ** 2) / var_total
    return scores, var_ratio


# ── Spearman correlation ───────────────────────────────────────────────────


def best_spearman(pc_scores: np.ndarray, target: np.ndarray):
    """
    Compute Spearman |r| between each PC and target, return best.

    Args:
        pc_scores : [N_total, n_components]
        target    : [N_total]

    Returns:
        best_r    : float  best absolute Spearman r across PCs
        best_pc   : int    which PC index
        all_r     : list   r for each PC
    """
    all_r = []
    for pc_idx in tqdm(
        range(pc_scores.shape[1]), desc="  Spearman", leave=False, unit="PC"
    ):
        r, _ = spearmanr(pc_scores[:, pc_idx], target)
        all_r.append(abs(float(r)))
    best_pc = int(np.argmax(all_r))
    return all_r[best_pc], best_pc, all_r


# ── Main probe loop ────────────────────────────────────────────────────────


@torch.no_grad()
def probe_all_layers(
    model_id: str,
    lr_dir: str,
    n_samples: int,
    out_dir: str,
    device: torch.device,
):
    os.makedirs(out_dir, exist_ok=True)

    # ── Load encoder ──────────────────────────────────────────────────────
    print(f"\nLoading encoder: {model_id}")
    encoder = AutoModel.from_pretrained(model_id)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\nLoading {n_samples} samples from {lr_dir}")
    x_ssl, t2m_patch, lat_patch = load_samples(lr_dir, n_samples, device)
    N = x_ssl.shape[0]
    print(f"x_ssl shape   : {tuple(x_ssl.shape)}")
    print(f"t2m_patch shape: {t2m_patch.shape}")

    # ── Forward pass — get all hidden states ──────────────────────────────
    print(f"\nRunning encoder forward pass (output_hidden_states=True)...")
    with tqdm(total=1, desc="Encoder forward pass") as pbar:
        out = encoder(
            pixel_values=x_ssl.to(dtype=torch.bfloat16),
            output_hidden_states=True,
        )
        pbar.update(1)
    # hidden_states: tuple of (n_layers+1) tensors, each [N, seq_len, 1024]
    # Index 0 = embedding layer, 1-24 = transformer layer outputs
    hidden_states = out.hidden_states
    print(
        f"Hidden states available: {len(hidden_states)} "
        f"(index 0=embedding, 1-{len(hidden_states) - 1}=transformer layers)"
    )

    # ── Per-layer probe ───────────────────────────────────────────────────
    results = {}
    layer_indices = list(range(1, N_LAYERS + 1))  # layers 1–24

    print(
        f"\n{'Layer':>6} {'Var(PC1)':>10} "
        f"{'Best T2m r':>12} {'Best T2m PC':>12} "
        f"{'Best Lat r':>12} {'Best Lat PC':>12}"
    )
    print("-" * 70)

    for layer_idx in tqdm(layer_indices, desc="Probing layers", unit="layer"):
        hs = hidden_states[layer_idx]  # [N, seq_len, 1024]

        # Extract patch tokens — layout: [CLS | patch_0..patch_511 | reg_0..reg_3]
        patch_tokens = hs[:, 1 : 1 + N_PATCHES, :].float().cpu().numpy()
        # [N, N_PATCHES, 1024]

        # Stack into [N*N_PATCHES, 1024] for PCA
        N_total = N * N_PATCHES
        tokens_2d = patch_tokens.reshape(N_total, -1)

        # PCA
        scores, var_ratio = run_pca(tokens_2d, n_components=N_PCS)

        # Repeat per-patch targets to match stacked layout
        # t2m_patch: [N, N_PATCHES] → [N*N_PATCHES]
        t2m_flat = t2m_patch.flatten()
        # lat_patch: [N_PATCHES] → [N*N_PATCHES] (tile across samples)
        lat_flat = np.tile(lat_patch, N)

        # Spearman correlations
        t2m_r, t2m_pc, t2m_all = best_spearman(scores, t2m_flat)
        lat_r, lat_pc, lat_all = best_spearman(scores, lat_flat)

        print(
            f"{layer_idx:>6} {var_ratio[0] * 100:>9.2f}% "
            f"{t2m_r:>12.3f} {'PC' + str(t2m_pc + 1):>12} "
            f"{lat_r:>12.3f} {'PC' + str(lat_pc + 1):>12}"
        )

        results[layer_idx] = {
            "var_pc1": float(var_ratio[0]),
            "var_ratio": [float(v) for v in var_ratio],
            "t2m_r": t2m_r,
            "t2m_pc": t2m_pc + 1,  # 1-indexed for readability
            "t2m_all_r": t2m_all,
            "lat_r": lat_r,
            "lat_pc": lat_pc + 1,
            "lat_all_r": lat_all,
        }

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    layers = layer_indices
    t2m_rs = [results[l]["t2m_r"] for l in layers]
    lat_rs = [results[l]["lat_r"] for l in layers]
    var_pc1s = [results[l]["var_pc1"] * 100 for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Spearman r vs layer
    axes[0].plot(
        layers,
        t2m_rs,
        "o-",
        color="#1D9E75",
        linewidth=2,
        markersize=5,
        label="Best |r| vs T2m",
    )
    axes[0].plot(
        layers,
        lat_rs,
        "s--",
        color="#7F77DD",
        linewidth=2,
        markersize=5,
        label="Best |r| vs Latitude",
    )
    axes[0].set_ylabel("Spearman |r|", fontsize=12)
    axes[0].set_title("DINOv3-SAT layer-wise feature probe on ERA5 T2m", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0.8, color="gray", linewidth=0.8, linestyle=":")

    # Annotate peak T2m layer
    peak_layer = layers[int(np.argmax(t2m_rs))]
    peak_r = max(t2m_rs)
    axes[0].annotate(
        f"peak: layer {peak_layer}\nr={peak_r:.3f}",
        xy=(peak_layer, peak_r),
        xytext=(peak_layer + 1, peak_r - 0.08),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    # Bottom: Variance explained by PC1
    axes[1].bar(layers, var_pc1s, color="#5DCAA5", alpha=0.7)
    axes[1].set_xlabel("Transformer layer", fontsize=12)
    axes[1].set_ylabel("Var explained by PC1 (%)", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Vertical lines at recommended tap points (updated after viewing results)
    for ax in axes:
        ax.set_xticks(layers)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "layer_probe.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # ── CASD tap recommendation ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CASD layer tap recommendation")
    print("=" * 60)

    # Find peak T2m layer
    peak_t2m_layer = layers[int(np.argmax(t2m_rs))]
    peak_t2m_r = max(t2m_rs)

    # Early: layer with highest T2m r in first 8 layers
    early_layers = [l for l in layers if l <= 8]
    early_best = early_layers[int(np.argmax([t2m_rs[l - 1] for l in early_layers]))]

    # Mid: layer with highest T2m r in layers 9-16
    mid_layers = [l for l in layers if 9 <= l <= 16]
    mid_best = mid_layers[int(np.argmax([t2m_rs[l - 1] for l in mid_layers]))]

    # Late: layer with highest T2m r in layers 17-24
    late_layers = [l for l in layers if l >= 17]
    late_best = late_layers[int(np.argmax([t2m_rs[l - 1] for l in late_layers]))]

    print(f"Peak T2m correlation : layer {peak_t2m_layer} (r={peak_t2m_r:.3f})")
    print(f"Recommended taps:")
    print(f"  early : layer {early_best:2d}  (r={results[early_best]['t2m_r']:.3f})")
    print(f"  mid   : layer {mid_best:2d}  (r={results[mid_best]['t2m_r']:.3f})")
    print(f"  late  : layer {late_best:2d}  (r={results[late_best]['t2m_r']:.3f})")
    print()
    print("Add to CASD config:")
    print(f"  encoder_tap_layers: [{early_best}, {mid_best}, {late_best}]")
    print("=" * 60)

    # Save recommendation to JSON
    recommendation = {
        "peak_t2m_layer": peak_t2m_layer,
        "peak_t2m_r": peak_t2m_r,
        "tap_early": early_best,
        "tap_mid": mid_best,
        "tap_late": late_best,
        "tap_layers": [early_best, mid_best, late_best],
    }
    rec_path = os.path.join(out_dir, "casd_tap_recommendation.json")
    with open(rec_path, "w") as f:
        json.dump(recommendation, f, indent=2)
    print(f"Recommendation saved to {rec_path}")

    return results, recommendation


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="DINOv3-SAT layer probe for CASD")
    p.add_argument(
        "--lr_dir",
        required=True,
        help="Path to LR ERA5 directory (contains train/*.npz)",
    )
    p.add_argument(
        "--model_id",
        default="facebook/dinov3-vitl16-pretrain-sat493m",
        help="HuggingFace model ID",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=30,
        help="Number of timesteps to probe (default: 30)",
    )
    p.add_argument(
        "--out_dir",
        default="./probe_results",
        help="Output directory for results and plots",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device  : {device}")
    print(f"Model   : {args.model_id}")
    print(f"LR dir  : {args.lr_dir}")
    print(f"Samples : {args.n_samples}")
    print(f"Out dir : {args.out_dir}")

    results, recommendation = probe_all_layers(
        model_id=args.model_id,
        lr_dir=args.lr_dir,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
        device=device,
    )
