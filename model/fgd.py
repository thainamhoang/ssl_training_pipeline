"""
fgd.py

FGD — Frequency-Geographic Decoder. Pilot 5.

Extends SSLDownscaler (LoRA encoder + pixel-shuffle decoder) with three
independently ablatable additions:

    A — Multi-scale token fusion (use_multiscale=True)
        Taps encoder hidden states at layers [4, 7, 14, 24].
        Each layer projected to hidden_dim and summed → same decoder shape.

    B — FiLM geographic conditioning (use_film=True, requires A)
        Per-location scale γ and shift β from sin/cos lat/lon.
        Applied at patch resolution before pixel-shuffle.

    C — Spectral loss (spectral_lambda > 0, passed to fgd_loss())
        Frequency-weighted L1 on 2D FFT of prediction vs target.
        High frequencies penalised more to counter SSL low-pass bias.

Ablation modes:
    use_multiscale=False, use_film=False → identical to SSLDownscaler
    use_multiscale=True,  use_film=False → addition A only
    use_multiscale=True,  use_film=True  → additions A + B
    spectral_lambda > 0                  → addition C (any mode)

Training integration:
    In training.py, pass to train_one_epoch:
        loss_fn=functools.partial(fgd_loss, spectral_lambda=cfg.training.spectral_lambda)
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from trainer_utils import (
    compute_ssl_size,
    build_encoder,
    init_conv_decoder,
    encoder_forward,
)


# ── Spectral loss (Addition C) ────────────────────────────────────────────────


def fgd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    spectral_lambda: float = 0.0,
    freq_ramp: float = 10.0,
) -> torch.Tensor:
    """
    FGD training loss: MSE + optional frequency-weighted spectral term.

    spectral_lambda=0.0 → pure MSE, identical to existing training loop.
    spectral_lambda>0   → MSE + λ * spectral_L1

    Args:
        pred            : [B, 1, H, W] predicted z-score field
        target          : [B, 1, H, W] ground-truth z-score field
        spectral_lambda : weight on spectral term (default 0.0 = MSE only)
        freq_ramp       : multiplier on frequency weight ramp (default 10.0)

    Returns:
        scalar loss tensor
    """
    mse = F.mse_loss(pred, target)
    if spectral_lambda == 0.0:
        return mse

    pred_fft = torch.fft.rfft2(pred.float())
    target_fft = torch.fft.rfft2(target.float())

    freqs_h = torch.fft.fftfreq(pred.shape[-2], device=pred.device)
    freqs_w = torch.fft.rfftfreq(pred.shape[-1], device=pred.device)
    freq_grid = torch.sqrt(freqs_h[:, None] ** 2 + freqs_w[None, :] ** 2)
    freq_weight = 1.0 + freq_ramp * freq_grid

    spectral = (freq_weight * (pred_fft - target_fft).abs()).mean()
    return mse + spectral_lambda * spectral


# ── FiLM geographic conditioning (Addition B) ─────────────────────────────────


class FiLMConditioner(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on sin/cos lat/lon.

    Learns per-location γ and β at patch resolution (n_patches_h × n_patches_w).
    Applied before pixel-shuffle — conditioning stays cheap (512 locations
    rather than 32768 at HR resolution).

    Initialised so γ≈0, β≈0 → FiLM is identity at epoch 0, preserving
    initial RMSE.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_patches_h: int = 16,
        n_patches_w: int = 32,
        coord_hidden: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w

        lats = torch.linspace(90.0, -90.0, n_patches_h)
        lons = torch.linspace(0.0, 358.59375, n_patches_w)
        lat_rad = lats * (math.pi / 180.0)
        lon_rad = lons * (math.pi / 180.0)
        lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")

        coords = torch.stack(
            [lat_grid.sin(), lat_grid.cos(), lon_grid.sin(), lon_grid.cos()],
            dim=-1,
        )  # [H_p, W_p, 4]
        self.register_buffer("coord_grid", coords)

        self.mlp = nn.Sequential(
            nn.Linear(4, coord_hidden),
            nn.GELU(),
            nn.Linear(coord_hidden, 2 * hidden_dim),
        )
        # Zero-init → identity at epoch 0
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat : [B, hidden_dim, H_p, W_p]"""
        B = feat.shape[0]
        film_params = self.mlp(self.coord_grid)  # [H_p, W_p, 2*hidden_dim]
        film_params = film_params.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        gamma, beta = film_params.chunk(2, dim=1)
        return (1.0 + gamma) * feat + beta


# ── Multi-scale token fusion (Addition A) ─────────────────────────────────────


class MultiScaleProjection(nn.Module):
    """
    Projects and sums tokens from multiple encoder layers into one
    spatial feature map [B, hidden_dim, H_p, W_p].

    Summation (not concatenation) keeps output channels = hidden_dim so
    the pixel-shuffle decoder input shape is unchanged vs the baseline.
    Each tap layer gets its own Linear + LayerNorm projection.
    """

    def __init__(
        self,
        enc_dim: int,
        hidden_dim: int,
        tap_layers: list,
        n_patches_h: int,
        n_patches_w: int,
    ):
        super().__init__()
        self.tap_layers = tap_layers
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w
        self.n_patches = n_patches_h * n_patches_w

        self.projs = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(enc_dim, hidden_dim), nn.LayerNorm(hidden_dim))
                for _ in tap_layers
            ]
        )

    def forward(self, hidden_states: tuple) -> torch.Tensor:
        fused = None
        for proj, layer_idx in zip(self.projs, self.tap_layers):
            tokens = hidden_states[layer_idx][:, 1 : 1 + self.n_patches, :].float()
            projected = proj(tokens)  # [B, N, hidden_dim]

            B, N, D = projected.shape
            spatial = projected.reshape(B, self.n_patches_h, self.n_patches_w, D)
            spatial = spatial.permute(0, 3, 1, 2)  # [B, hidden_dim, H_p, W_p]

            fused = spatial if fused is None else fused + spatial

        return fused


# ── Main model ────────────────────────────────────────────────────────────────


class FGD(nn.Module):
    """
    Frequency-Geographic Decoder.

    Parameters
    ----------
    model_id       : HuggingFace model ID
    patch_size     : ViT patch size (16 for DINOv3-SAT)
    lr_shape       : (H_lr, W_lr) e.g. (32, 64)
    hr_shape       : (H_hr, W_hr) e.g. (128, 256)
    upscale        : SR upscaling factor (default 4)
    hidden_dim     : decoder hidden channels (default 256)
    mode           : "frozen" | "lora"
    lora_r         : LoRA rank (default 8)
    lora_alpha     : LoRA alpha (defaults to lora_r)
    lora_dropout   : LoRA dropout (default 0.0)
    use_multiscale : enable multi-scale token fusion — Addition A
    tap_layers     : layers to tap when use_multiscale=True (default [4,7,14,24])
    use_film       : enable FiLM geographic conditioning — Addition B
                     requires use_multiscale=True
    film_hidden    : FiLM MLP hidden size (default 128)
    """

    def __init__(
        self,
        model_id: str,
        patch_size: int,
        lr_shape: tuple,
        hr_shape: tuple,
        upscale: int = 4,
        hidden_dim: int = 256,
        mode: str = "lora",
        lora_r: int = 8,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
        use_multiscale: bool = True,
        tap_layers: list = None,
        use_film: bool = False,
        film_hidden: int = 128,
    ):
        super().__init__()

        assert mode in ("frozen", "lora"), (
            f"mode must be 'frozen' or 'lora', got {mode!r}"
        )
        if use_film and not use_multiscale:
            raise ValueError("use_film=True requires use_multiscale=True")

        self.mode = mode
        self.patch_size = patch_size
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.upscale = upscale
        self.hidden_dim = hidden_dim
        self.lora_r = lora_r
        self.use_multiscale = use_multiscale
        self.use_film = use_film
        self.tap_layers = tap_layers if tap_layers is not None else [4, 7, 14, 24]

        self.ssl_size, self.n_patches_h, self.n_patches_w, self.n_patches = (
            compute_ssl_size(lr_shape, patch_size)
        )

        additions = (["A:multiscale"] if use_multiscale else []) + (
            ["B:FiLM"] if use_film else []
        )
        print(
            f"[FGD] mode={mode}, lora_r={lora_r}"
            + (f", additions=[{', '.join(additions)}]" if additions else ", baseline")
        )
        print(f"SSL input size : {self.ssl_size[0]}×{self.ssl_size[1]}")
        print(
            f"Patch grid     : {self.n_patches_h}×{self.n_patches_w} = {self.n_patches} tokens"
        )
        if use_multiscale:
            print(f"Tap layers     : {self.tap_layers}")

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder = build_encoder(model_id, mode, lora_r, lora_alpha, lora_dropout)

        enc_dim = 1024

        # ── Feature normalizer — single-layer path only ────────────────────
        self.feat_norm = nn.LayerNorm(enc_dim) if not use_multiscale else None

        # ── Addition A: multi-scale projection ────────────────────────────
        self.ms_proj = (
            MultiScaleProjection(
                enc_dim, hidden_dim, self.tap_layers, self.n_patches_h, self.n_patches_w
            )
            if use_multiscale
            else None
        )

        # ── Addition B: FiLM conditioning ─────────────────────────────────
        self.film = (
            FiLMConditioner(hidden_dim, self.n_patches_h, self.n_patches_w, film_hidden)
            if use_film
            else None
        )

        # ── Pixel-shuffle decoder ──────────────────────────────────────────
        # Single-layer path feeds enc_dim (1024) directly into the first Conv2d.
        # Multi-scale path feeds hidden_dim (256) — already projected by ms_proj.
        decoder_in = hidden_dim if use_multiscale else enc_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in, hidden_dim * upscale * upscale, kernel_size=1),
            nn.PixelShuffle(upscale),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        init_conv_decoder(self.decoder)

        n_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_dec = sum(p.numel() for p in self.decoder.parameters())
        n_aux = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.feat_norm, self.ms_proj, self.film]
            if m is not None
        )
        print(f"Trainable — encoder : {n_enc:,}")
        print(f"Trainable — decoder : {n_dec:,}")
        print(f"Trainable — aux     : {n_aux:,}  (feat_norm / ms_proj / film)")
        print(f"Trainable — total   : {n_enc + n_dec + n_aux:,}")

    def _encode(self, x: torch.Tensor):
        x_ssl = F.interpolate(
            x, size=self.ssl_size, mode="bilinear", align_corners=False
        )
        out = encoder_forward(
            self.encoder,
            self.mode,
            x_ssl.to(dtype=torch.bfloat16),
            output_hidden_states=self.use_multiscale,
        )
        if self.use_multiscale:
            return None, out.hidden_states
        patch_tokens = out.last_hidden_state[:, 1 : 1 + self.n_patches, :].float()
        return patch_tokens, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H_lr, W_lr] ImageNet-normalised float32
        Returns:
            [B, 1, H_hr, W_hr] predicted z-score normalised HR field
        """
        patch_tokens, hidden_states = self._encode(x)

        if self.use_multiscale:
            feat = self.ms_proj(hidden_states)  # [B, hidden_dim, H_p, W_p]
            if self.use_film:
                feat = self.film(feat)
        else:
            patch_tokens = self.feat_norm(patch_tokens)
            B, N, D = patch_tokens.shape
            feat = (
                patch_tokens.reshape(B, self.n_patches_h, self.n_patches_w, D)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # [B, enc_dim, H_p, W_p]

        out = self.decoder(feat)

        if out.shape[-2:] != self.hr_shape:
            out = F.interpolate(
                out, size=self.hr_shape, mode="bilinear", align_corners=False
            )

        return out
