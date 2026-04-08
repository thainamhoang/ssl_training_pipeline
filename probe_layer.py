"""
probe_layers.py

Layer-wise feature analysis for DINOv3-SAT on ERA5 climate data.
Now includes geographic stratification to determine whether LayerWiseFiLM
is theoretically justified.

For each of the 24 transformer layers, extracts patch tokens, runs PCA,
and computes Spearman correlation of each PC against:
    - Raw T2m field (flattened to patch resolution)
    - Latitude grid (proxy for large-scale temperature gradient)

Additionally, splits patches into ocean / mountain / flat groups and
computes per-group T2m correlations per layer. If layer importance ranking
differs significantly across groups (std > 0.05), LayerWiseFiLM is justified.

Outputs:
    1. probe_results.json        — correlation values per layer per PC
    2. layer_probe.png           — plot of best T2m and lat correlation vs layer depth
    3. stratified_results.json   — per-group T2m correlation per layer
    4. stratified_probe.png      — per-group correlation plot
    5. Console summary           — LayerWiseFiLM justification verdict

Usage:
    python probe_layers.py \
        --lr_dir  /workspace/data/5.625deg/2m_temperature \
        --model_id facebook/dinov3-vitl16-pretrain-sat493m \
        --n_samples 100 \
        --out_dir ./probe_results_stratified \
        --constants_file /workspace/data/1.40625deg/2m_temperature/constants.npz
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

PATCH_SIZE = 16
N_LAYERS = 24
SSL_H, SSL_W = 256, 512
N_PATCHES_H = SSL_H // PATCH_SIZE  # 16
N_PATCHES_W = SSL_W // PATCH_SIZE  # 32
N_PATCHES = N_PATCHES_H * N_PATCHES_W  # 512
N_PCS = 5

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

LR_MEAN = 278.45  # K
LR_STD = 21.25  # K


# ── Data loading ───────────────────────────────────────────────────────────


def load_samples(
    lr_dir: str, n_samples: int, device: torch.device, constants_file: str = None
):
    """
    Load n_samples ERA5 LR timesteps.

    Returns:
        x_ssl        : [N, 3, SSL_H, SSL_W]  ImageNet-normalised encoder input
        t2m_patch    : [N, N_PATCHES]         T2m downsampled to patch grid
        lat_patch    : [N_PATCHES]            latitude per patch
        oro_patch_np : [N_PATCHES] or None    orography in metres at patch res
        lsm_patch_np : [N_PATCHES] or None    land-sea mask at patch res
    """
    shards = sorted(glob.glob(os.path.join(lr_dir, "train", "*.npz")))
    shards = [f for f in shards if "climatology" not in f]
    assert len(shards) > 0, f"No shards found in {lr_dir}/train/"

    samples = []
    with tqdm(total=n_samples, desc="Loading samples", unit="sample") as pbar:
        for shard_path in shards:
            data = np.load(shard_path)["2m_temperature"]  # [T, 1, H_lr, W_lr]
            for t in range(data.shape[0]):
                samples.append(data[t, 0])
                pbar.update(1)
                if len(samples) >= n_samples:
                    break
            if len(samples) >= n_samples:
                break

    samples = samples[:n_samples]
    print(f"Loaded {len(samples)} samples from {len(shards)} shards")

    H_lr, W_lr = samples[0].shape
    print(f"LR grid shape: {H_lr}×{W_lr}")

    x_list = []
    for s in tqdm(samples, desc="Preprocessing samples", unit="sample"):
        t = torch.tensor(s, dtype=torch.float32)
        z = (t - LR_MEAN) / LR_STD
        t01 = torch.clamp(z / 6.0 + 0.5, 0.0, 1.0)
        x3 = t01.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        x3 = (x3 - IMAGENET_MEAN) / IMAGENET_STD
        x_ssl = F.interpolate(
            x3, size=(SSL_H, SSL_W), mode="bilinear", align_corners=False
        )
        x_list.append(x_ssl)
    x_ssl = torch.cat(x_list, dim=0).to(device)

    t2m_stack = (
        torch.tensor(np.stack(samples, axis=0), dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )
    t2m_patch_2d = F.adaptive_avg_pool2d(t2m_stack, (N_PATCHES_H, N_PATCHES_W))
    t2m_patch = t2m_patch_2d.squeeze(1).reshape(len(samples), -1).cpu().numpy()

    lats = np.linspace(90, -90, N_PATCHES_H)
    lons = np.linspace(0, 360, N_PATCHES_W, endpoint=False)
    lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
    lat_patch = lat_grid.flatten()

    # ── Load constants for stratification ────────────────────────────────
    oro_patch_np = None
    lsm_patch_np = None

    # Resolve constants file — explicit arg or auto-discover in lr_dir
    candidate = constants_file
    if candidate is None:
        for fname in ["constants.npz", "invariant.npz", "static.npz"]:
            p = os.path.join(lr_dir, fname)
            if os.path.exists(p):
                candidate = p
                break

    if candidate and os.path.exists(candidate):
        print(f"Loading constants from {candidate}")
        data = np.load(candidate)
        print(f"  Available keys: {list(data.keys())}")

        # Orography
        oro_raw = None
        for key in ["orography", "z", "oro"]:
            if key in data:
                oro_raw = data[key].squeeze().astype(np.float32)
                if oro_raw.max() > 10000:
                    # Geopotential → metres
                    oro_raw = oro_raw / 9.80665
                    print(f"  Orography: key='{key}', converted from geopotential")
                else:
                    print(f"  Orography: key='{key}', already in metres")
                break
        if oro_raw is None:
            print("  Warning: no orography key found")

        # LSM
        lsm_raw = None
        for key in ["lsm", "land_sea_mask", "lsm_mask"]:
            if key in data:
                lsm_raw = data[key].squeeze().astype(np.float32)
                print(f"  LSM: key='{key}'")
                break
        if lsm_raw is None:
            print("  Warning: no LSM key found")

        if oro_raw is not None and lsm_raw is not None:
            oro_t = torch.tensor(oro_raw, dtype=torch.float32)
            lsm_t = torch.tensor(lsm_raw, dtype=torch.float32)

            # Squeeze extra dims
            while oro_t.dim() > 2:
                oro_t = oro_t.squeeze(0)
            while lsm_t.dim() > 2:
                lsm_t = lsm_t.squeeze(0)

            # Downsample to patch resolution (16×32)
            oro_patch = F.adaptive_avg_pool2d(
                oro_t[None, None], (N_PATCHES_H, N_PATCHES_W)
            ).squeeze()
            lsm_patch = F.adaptive_avg_pool2d(
                lsm_t[None, None], (N_PATCHES_H, N_PATCHES_W)
            ).squeeze()

            oro_patch_np = oro_patch.numpy().flatten()
            lsm_patch_np = lsm_patch.numpy().flatten()

            print(
                f"  Orography range: [{oro_patch_np.min():.0f}, "
                f"{oro_patch_np.max():.0f}] m at patch resolution"
            )
            print(f"  LSM range: [{lsm_patch_np.min():.2f}, {lsm_patch_np.max():.2f}]")
    else:
        print("No constants file found — skipping geographic stratification")

    return x_ssl, t2m_patch, lat_patch, oro_patch_np, lsm_patch_np


# ── PCA ───────────────────────────────────────────────────────────────────


def run_pca(tokens: np.ndarray, n_components: int = N_PCS):
    X = tokens - tokens.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]
    var_total = (S**2).sum()
    var_ratio = (S[:n_components] ** 2) / var_total
    return scores, var_ratio


# ── Spearman correlation ───────────────────────────────────────────────────


def best_spearman(pc_scores: np.ndarray, target: np.ndarray):
    all_r = []
    for pc_idx in tqdm(
        range(pc_scores.shape[1]), desc="  Spearman", leave=False, unit="PC"
    ):
        r, _ = spearmanr(pc_scores[:, pc_idx], target)
        all_r.append(abs(float(r)))
    best_pc = int(np.argmax(all_r))
    return all_r[best_pc], best_pc, all_r


# ── Geographic stratification ─────────────────────────────────────────────


def stratify_patches(oro_patch_np: np.ndarray, lsm_patch_np: np.ndarray):
    """
    Split 512 patch locations into three geographic groups.

    Returns dict of group_name → boolean mask [N_PATCHES]
    """
    ocean = lsm_patch_np < 0.5
    mountain = (lsm_patch_np >= 0.5) & (oro_patch_np > 500)
    flat = (lsm_patch_np >= 0.5) & (oro_patch_np <= 500)

    print(f"  Ocean   : {ocean.sum():3d} patches ({ocean.mean() * 100:.1f}%)")
    print(f"  Mountain: {mountain.sum():3d} patches ({mountain.mean() * 100:.1f}%)")
    print(f"  Flat    : {flat.sum():3d} patches ({flat.mean() * 100:.1f}%)")

    return {"ocean": ocean, "mountain": mountain, "flat": flat}


def stratified_spearman(
    pc_scores: np.ndarray, t2m_flat: np.ndarray, group_mask: np.ndarray, n_samples: int
):
    """
    Spearman r for a geographic subset of patches across all samples.

    group_mask : [N_PATCHES] boolean
    Returns best |r| vs T2m for this group.
    """
    mask_tiled = np.tile(group_mask, n_samples)  # [N*N_PATCHES]
    scores_sub = pc_scores[mask_tiled]
    t2m_sub = t2m_flat[mask_tiled]

    all_r = []
    for pc_idx in range(scores_sub.shape[1]):
        r, _ = spearmanr(scores_sub[:, pc_idx], t2m_sub)
        all_r.append(abs(float(r)))
    return max(all_r)


# ── Main probe loop ────────────────────────────────────────────────────────


@torch.no_grad()
def probe_all_layers(
    model_id: str,
    lr_dir: str,
    n_samples: int,
    out_dir: str,
    device: torch.device,
    constants_file: str = None,
):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading encoder: {model_id}")
    encoder = AutoModel.from_pretrained(model_id)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder = encoder.to(device)

    print(f"\nLoading {n_samples} samples from {lr_dir}")
    x_ssl, t2m_patch, lat_patch, oro_patch_np, lsm_patch_np = load_samples(
        lr_dir, n_samples, device, constants_file=constants_file
    )
    N = x_ssl.shape[0]
    print(f"x_ssl shape    : {tuple(x_ssl.shape)}")
    print(f"t2m_patch shape: {t2m_patch.shape}")

    print(f"\nRunning encoder forward pass (output_hidden_states=True)...")
    with tqdm(total=1, desc="Encoder forward pass") as pbar:
        out = encoder(
            pixel_values=x_ssl.to(dtype=torch.bfloat16),
            output_hidden_states=True,
        )
        pbar.update(1)
    hidden_states = out.hidden_states
    print(
        f"Hidden states available: {len(hidden_states)} "
        f"(index 0=embedding, 1-{len(hidden_states) - 1}=transformer layers)"
    )

    # ── Per-layer global probe ────────────────────────────────────────────
    results = {}
    layer_indices = list(range(1, N_LAYERS + 1))

    print(
        f"\n{'Layer':>6} {'Var(PC1)':>10} "
        f"{'Best T2m r':>12} {'Best T2m PC':>12} "
        f"{'Best Lat r':>12} {'Best Lat PC':>12}"
    )
    print("-" * 70)

    # Pre-compute flat targets once
    t2m_flat = t2m_patch.flatten()
    lat_flat = np.tile(lat_patch, N)

    # Cache per-layer scores for stratification reuse (avoids re-running PCA)
    layer_scores = {}

    for layer_idx in tqdm(layer_indices, desc="Probing layers", unit="layer"):
        hs = hidden_states[layer_idx]
        patch_tokens = hs[:, 1 : 1 + N_PATCHES, :].float().cpu().numpy()
        tokens_2d = patch_tokens.reshape(N * N_PATCHES, -1)

        scores, var_ratio = run_pca(tokens_2d, n_components=N_PCS)
        layer_scores[layer_idx] = scores  # cache for stratification

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
            "t2m_pc": t2m_pc + 1,
            "t2m_all_r": t2m_all,
            "lat_r": lat_r,
            "lat_pc": lat_pc + 1,
            "lat_all_r": lat_all,
        }

    json_path = os.path.join(out_dir, "probe_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── Plot global probe ─────────────────────────────────────────────────
    layers = layer_indices
    t2m_rs = [results[l]["t2m_r"] for l in layers]
    lat_rs = [results[l]["lat_r"] for l in layers]
    var_pc1s = [results[l]["var_pc1"] * 100 for l in layers]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
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
    peak_layer = layers[int(np.argmax(t2m_rs))]
    peak_r = max(t2m_rs)
    axes[0].annotate(
        f"peak: layer {peak_layer}\nr={peak_r:.3f}",
        xy=(peak_layer, peak_r),
        xytext=(peak_layer + 1, peak_r - 0.08),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    axes[1].bar(layers, var_pc1s, color="#5DCAA5", alpha=0.7)
    axes[1].set_xlabel("Transformer layer", fontsize=12)
    axes[1].set_ylabel("Var explained by PC1 (%)", fontsize=12)
    axes[1].grid(True, alpha=0.3, axis="y")
    for ax in axes:
        ax.set_xticks(layers)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "layer_probe.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # ── Tap recommendation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CASD layer tap recommendation")
    print("=" * 60)
    peak_t2m_layer = layers[int(np.argmax(t2m_rs))]
    peak_t2m_r = max(t2m_rs)
    early_layers = [l for l in layers if l <= 8]
    early_best = early_layers[int(np.argmax([t2m_rs[l - 1] for l in early_layers]))]
    mid_layers = [l for l in layers if 9 <= l <= 16]
    mid_best = mid_layers[int(np.argmax([t2m_rs[l - 1] for l in mid_layers]))]
    late_layers = [l for l in layers if l >= 17]
    late_best = late_layers[int(np.argmax([t2m_rs[l - 1] for l in late_layers]))]
    print(f"Peak T2m correlation : layer {peak_t2m_layer} (r={peak_t2m_r:.3f})")
    print(f"Recommended taps:")
    print(f"  early : layer {early_best:2d}  (r={results[early_best]['t2m_r']:.3f})")
    print(f"  mid   : layer {mid_best:2d}  (r={results[mid_best]['t2m_r']:.3f})")
    print(f"  late  : layer {late_best:2d}  (r={results[late_best]['t2m_r']:.3f})")
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

    # ── Geographic stratification ─────────────────────────────────────────
    if oro_patch_np is not None and lsm_patch_np is not None:
        print("\n" + "=" * 70)
        print("Geographic stratification — LayerWiseFiLM justification check")
        print("=" * 70)

        groups = stratify_patches(oro_patch_np, lsm_patch_np)

        tap_layers_check = [4, 7, 14, 24]

        print(
            f"\n{'Layer':>6} {'Global':>10} {'Ocean':>10} "
            f"{'Mountain':>10} {'Flat':>10} {'Std':>8}  ← T2m |r|"
        )
        print("-" * 60)

        strat_results = {g: {} for g in groups}

        for layer_idx in tqdm(layer_indices, desc="Stratified probe", unit="layer"):
            scores = layer_scores[layer_idx]
            global_r = results[layer_idx]["t2m_r"]

            group_rs = {}
            for gname, gmask in groups.items():
                r = stratified_spearman(scores, t2m_flat, gmask, N)
                group_rs[gname] = r
                strat_results[gname][layer_idx] = r

            std = np.std(list(group_rs.values()))
            print(
                f"{layer_idx:>6} {global_r:>10.3f} "
                f"{group_rs['ocean']:>10.3f} "
                f"{group_rs['mountain']:>10.3f} "
                f"{group_rs['flat']:>10.3f} "
                f"{std:>8.3f}" + (" ◄ high variance" if std > 0.05 else "")
            )

        # Save
        strat_path = os.path.join(out_dir, "stratified_results.json")
        with open(strat_path, "w") as f:
            json.dump(strat_results, f, indent=2)
        print(f"\nStratified results saved to {strat_path}")

        # ── LayerWiseFiLM verdict ─────────────────────────────────────────
        print("\n" + "=" * 70)
        print("LayerWiseFiLM justification verdict")
        print("=" * 70)

        print("\nPeak T2m layer per group:")
        peak_per_group = {}
        for gname in groups:
            layer_rs = {l: strat_results[gname][l] for l in layer_indices}
            peak_l = max(layer_rs, key=layer_rs.get)
            peak_per_group[gname] = peak_l
            print(f"  {gname:>10}: layer {peak_l:2d}  (r={layer_rs[peak_l]:.3f})")

        print(f"\nPer-layer variance across groups at tap layers {tap_layers_check}:")
        stds = []
        for layer_idx in tap_layers_check:
            rs = [strat_results[g][layer_idx] for g in groups]
            std = np.std(rs)
            stds.append(std)
            print(
                f"  Layer {layer_idx:2d}: "
                f"ocean={rs[0]:.3f}  mountain={rs[1]:.3f}  flat={rs[2]:.3f}  "
                f"std={std:.3f}" + (" ◄ high" if std > 0.05 else "")
            )

        mean_std = np.mean(stds)
        peaks_differ = len(set(peak_per_group.values())) > 1

        print(f"\nMean std across tap layers: {mean_std:.3f}")
        print(f"Peak layers differ across groups: {peaks_differ}")
        print()

        if mean_std > 0.05 or peaks_differ:
            print("VERDICT: LayerWiseFiLM IS justified.")
            print("  Layer importance differs meaningfully across geographic groups.")
            print("  Different locations benefit from different encoder layer weights.")
        else:
            print("VERDICT: LayerWiseFiLM is NOT clearly justified.")
            print("  Layer importance is similar across ocean / mountain / flat.")
            print("  Post-summation FiLM or global layer weights would suffice.")
            print("  Spectral loss (Addition C) is a better use of compute.")

        # ── Stratification plot ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = {"ocean": "#378ADD", "mountain": "#D85A30", "flat": "#639922"}
        global_rs = [results[l]["t2m_r"] for l in layer_indices]
        ax.plot(
            layer_indices, global_rs, "k--", linewidth=1.5, alpha=0.5, label="Global"
        )
        for gname, color in colors.items():
            rs = [strat_results[gname][l] for l in layer_indices]
            ax.plot(
                layer_indices,
                rs,
                "o-",
                color=color,
                linewidth=2,
                markersize=4,
                label=gname.capitalize(),
            )
        ax.set_xlabel("Transformer layer", fontsize=12)
        ax.set_ylabel("Best Spearman |r| vs T2m", fontsize=12)
        ax.set_title(
            "Layer importance by geographic group\n"
            "(divergence = LayerWiseFiLM justified)",
            fontsize=12,
        )
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(layer_indices)
        ax.axhline(0.8, color="gray", linewidth=0.8, linestyle=":")
        ax.grid(True, alpha=0.3)
        for layer_idx in tap_layers_check:
            ax.axvline(layer_idx, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)
        plt.tight_layout()
        strat_plot_path = os.path.join(out_dir, "stratified_probe.png")
        plt.savefig(strat_plot_path, dpi=150, bbox_inches="tight")
        print(f"\nStratification plot saved to {strat_plot_path}")

    else:
        print("\nSkipping geographic stratification — no constants file loaded.")

    return results, recommendation


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="DINOv3-SAT layer probe with stratification"
    )
    p.add_argument(
        "--lr_dir",
        required=True,
        help="Path to LR ERA5 directory (contains train/*.npz)",
    )
    p.add_argument("--model_id", default="facebook/dinov3-vitl16-pretrain-sat493m")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--out_dir", default="./probe_results_stratified")
    p.add_argument(
        "--constants_file",
        default=None,
        help="Path to constants.npz with orography and lsm. "
        "If not set, auto-discovers in lr_dir.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    print(f"Device           : {device}")
    print(f"Model            : {args.model_id}")
    print(f"LR dir           : {args.lr_dir}")
    print(f"Samples          : {args.n_samples}")
    print(f"Out dir          : {args.out_dir}")
    print(f"Constants file   : {args.constants_file or 'auto-discover'}")

    results, recommendation = probe_all_layers(
        model_id=args.model_id,
        lr_dir=args.lr_dir,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
        device=device,
        constants_file=args.constants_file,
    )
