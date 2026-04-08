"""
fgd.py

FGD — Frequency-Geographic Decoder

Pilot 5 architecture. Extends SSLDownscaler with three additions on top of
the existing LoRA r=8 encoder, each independently ablatable:

    A — Multi-scale token fusion
        Taps encoder hidden states at layers [4, 7, 14, 24] instead of only
        layer 24. Projects each to hidden_dim and concatenates into a single
        spatial feature map before pixel-shuffle. Motivated by probe results
        showing T2m correlation r>0.88 across layers 1-20.

    B — Layer-wise FiLM geographic conditioning
        Feature-wise Linear Modulation applied independently to each tap layer
        before summation. Each layer gets its own γ_l(coord) and β_l(coord),
        allowing location-dependent layer selection:
            F = Σ_l FiLM_l(proj_l(h_l), coord)
        vs post-summation FiLM which can only modulate the collapsed sum.
        Parameters: ~270K (n_tap × 2 × hidden_dim × coord_hidden).

    C — Spectral loss
        Frequency-weighted L1 loss on 2D FFT of prediction vs target.
        High-frequency components weighted by radial distance from DC.
        Counters SSL encoder's natural low-pass bias (motivated by probe
        showing SSL features degrade at layers 22-23).

Ablation modes (set via config):
    use_multiscale = False, use_film = False  → identical to SSLDownscaler
    use_multiscale = True,  use_film = False  → addition A only
    use_multiscale = True,  use_film = True   → additions A + B
    spectral_lambda > 0                       → addition C (loss term, any mode)

VRAM: identical to LoRA r=8 SSLDownscaler — no cross-attention, no new
large tensors. Multi-scale taps 4 layers instead of 1 but hidden_states
are already computed; the only overhead is 3 extra projection layers (~3M).

Usage:
    # Addition A only
    model = FGD(
        model_id       = "facebook/dinov3-vitl16-pretrain-sat493m",
        patch_size     = 16,
        lr_shape       = (32, 64),
        hr_shape       = (128, 256),
        mode           = "lora",
        lora_r         = 8,
        use_multiscale = True,
        use_film       = False,
    )

    # Additions A + B (layer-wise FiLM — location-dependent layer selection)
    model = FGD(..., use_multiscale=True, use_film=True)

    # Full FGD (A + B + C): pass spectral_lambda to the loss function
    loss = fgd_loss(pred, hr_norm, spectral_lambda=0.1)

Training script integration:
    In trainer_utils.py train_one_epoch, replace:
        loss = F.mse_loss(pred, hr_norm) / grad_accum_steps
    With:
        loss = fgd_loss(pred, hr_norm,
                        spectral_lambda=cfg.training.get("spectral_lambda", 0.0)
                       ) / grad_accum_steps

    In training.py, add "fgd" mode to the model construction block.
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


# ── Layer-wise FiLM geographic conditioning (Addition B) ──────────────────


class LayerWiseFiLM(nn.Module):
    """
    Layer-wise Feature-wise Linear Modulation conditioned on geographic coords.

    The key architectural difference from post-summation FiLM:
        OLD: F = FiLM(Σ_l proj_l(h_l), coord)      ← modulates collapsed sum
        NEW: F = Σ_l FiLM_l(proj_l(h_l), coord)    ← modulates each layer

    This allows location-dependent layer selection — a mountain pixel can
    learn to upweight layer 4 (high spatial coherence, r=0.895) while an
    ocean pixel upweights layer 14 (distributed features, lower noise).

    One MLP outputs γ and β for all tap layers simultaneously:
        output dim = 2 * n_tap * hidden_dim
        split into n_tap pairs of (γ_l, β_l), each [B, hidden_dim, H_p, W_p]

    Zero-init on final MLP layer → all FiLM ops are identity at epoch 0,
    so adding layer-wise FiLM doesn't change initial RMSE.

    Parameters
    ----------
    hidden_dim  : feature channels per layer (must match MultiScaleProjection)
    n_tap       : number of tap layers
    n_patches_h : patch grid height (default 16)
    n_patches_w : patch grid width  (default 32)
    coord_hidden: MLP hidden size (default 128)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_tap: int,
        n_patches_h: int = 16,
        n_patches_w: int = 32,
        coord_hidden: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_tap = n_tap
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w

        # Build coordinate grid at patch resolution — same as before
        lats = torch.linspace(90.0, -90.0, n_patches_h)
        lons = torch.linspace(0.0, 358.59375, n_patches_w)
        lat_rad = lats * (math.pi / 180.0)
        lon_rad = lons * (math.pi / 180.0)
        lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")
        coords = torch.stack(
            [
                lat_grid.sin(),
                lat_grid.cos(),
                lon_grid.sin(),
                lon_grid.cos(),
            ],
            dim=-1,
        )  # [H_p, W_p, 4]
        self.register_buffer("coord_grid", coords)

        # Single MLP outputs γ and β for ALL tap layers at once
        # Output: 2 * n_tap * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(4, coord_hidden),
            nn.GELU(),
            nn.Linear(coord_hidden, 2 * n_tap * hidden_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        # Zero-init final layer → identity at epoch 0
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, spatial_list: list) -> torch.Tensor:
        """
        Apply per-layer FiLM modulation then sum.

        Args:
            spatial_list : list of n_tap tensors, each [B, hidden_dim, H_p, W_p]
                           (projected spatial maps from MultiScaleProjection)

        Returns:
            fused : [B, hidden_dim, H_p, W_p]
        """
        B = spatial_list[0].shape[0]

        # [H_p, W_p, 2*n_tap*D] → [2*n_tap*D, H_p, W_p] → [B, 2*n_tap*D, H_p, W_p]
        all_params = self.mlp(self.coord_grid)  # [H_p, W_p, 2*n_tap*D]
        all_params = all_params.permute(2, 0, 1)  # [2*n_tap*D, H_p, W_p]
        all_params = all_params.unsqueeze(0).expand(B, -1, -1, -1)
        # [B, 2*n_tap*D, H_p, W_p]

        # Split into per-layer γ and β
        # all_params = [γ_0, γ_1, ..., γ_{n-1}, β_0, β_1, ..., β_{n-1}]
        gammas_all, betas_all = all_params.chunk(2, dim=1)
        # Each: [B, n_tap*hidden_dim, H_p, W_p]

        gammas = gammas_all.chunk(self.n_tap, dim=1)  # n_tap × [B, D, H_p, W_p]
        betas = betas_all.chunk(self.n_tap, dim=1)

        # Apply per-layer FiLM and sum
        fused = None
        for feat, gamma, beta in zip(spatial_list, gammas, betas):
            modulated = (1.0 + gamma) * feat + beta  # [B, D, H_p, W_p]
            fused = modulated if fused is None else fused + modulated

        return fused  # [B, hidden_dim, H_p, W_p]


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


# ── Multi-scale token projections (Addition A) ────────────────────────────


class MultiScaleProjection(nn.Module):
    """
    Projects and fuses tokens from multiple encoder layers into one
    spatial feature map of shape [B, hidden_dim, H_p, W_p].

    Each tap layer gets its own independent Linear projection (1024 → proj_dim).
    The projections are then summed (not concatenated) to keep the output
    channel count at hidden_dim regardless of how many layers are tapped.

    Summation rather than concatenation:
        - Keeps output channels = hidden_dim = 256 (same as single-layer baseline)
        - The pixel-shuffle decoder input shape is unchanged
        - Total added params: n_tap × Linear(1024, proj_dim) ≈ 3M for 3 extra taps
        - Concatenation would require changing the decoder's first Conv2d

    Parameters
    ----------
    enc_dim    : encoder hidden dim (1024 for DINOv3-SAT ViT-L)
    hidden_dim : output channel count = decoder input channels
    tap_layers : which hidden state layers to tap (e.g. [4, 7, 14, 24])
                 layer 24 = last_hidden_state (Pilot 2/3 baseline)
    n_patches_h: patch grid height
    n_patches_w: patch grid width
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

        # One projection + LayerNorm per tap layer
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(enc_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in tap_layers
            ]
        )

    def forward(self, hidden_states: tuple, return_list: bool = False):
        """
        Args:
            hidden_states : tuple of 25 tensors, each [B, 517, 1024]
                            (output of encoder with output_hidden_states=True)
            return_list   : if True, return list of per-layer spatial maps
                            (used by LayerWiseFiLM). If False, return summed map.

        Returns:
            if return_list=False : [B, hidden_dim, H_p, W_p]  summed feature map
            if return_list=True  : list of n_tap [B, hidden_dim, H_p, W_p] tensors
        """
        n_patches = self.n_patches_h * self.n_patches_w
        spatial_list = []
        fused = None

        for proj, layer_idx in zip(self.projs, self.tap_layers):
            hs = hidden_states[layer_idx]  # [B, 517, 1024]
            tokens = hs[:, 1 : 1 + n_patches, :].float()  # [B, 512, 1024]
            projected = proj(tokens)  # [B, 512, hidden_dim]

            B, N, D = projected.shape
            spatial = projected.reshape(
                B, self.n_patches_h, self.n_patches_w, D
            ).permute(0, 3, 1, 2)  # [B, hidden_dim, H_p, W_p]
            if return_list:
                spatial_list.append(spatial)
            else:
                fused = spatial if fused is None else fused + spatial

        if return_list:
            return spatial_list
        return fused  # [B, hidden_dim, H_p, W_p]


# ── Main model ────────────────────────────────────────────────────────────────


class FGD(nn.Module):
    """
    Frequency-Geographic Decoder.

    Extends SSLDownscaler (LoRA r=8, pixel-shuffle) with three ablatable additions:
        A — multi-scale token fusion      (use_multiscale=True)
        B — layer-wise FiLM conditioning  (use_film=True, requires A)
        C — spectral loss                 (spectral_lambda > 0, passed to fgd_loss)

    Addition B uses LayerWiseFiLM, not post-summation FiLM:
        OLD (post-sum):   F = FiLM(Σ_l proj_l(h_l), coord)
        NEW (layer-wise): F = Σ_l FiLM_l(proj_l(h_l), coord)
    This allows location-dependent layer selection — each location learns
    which encoder layers to upweight based on its geographic context.

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
    lora_alpha     : LoRA alpha (defaults to 2*lora_r)
    lora_dropout   : LoRA dropout (default 0.0)
    use_multiscale : enable multi-scale token fusion — Addition A
    tap_layers     : layers to tap when use_multiscale=True (default [4,7,14,24])
    use_film       : enable layer-wise FiLM conditioning — Addition B
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
            ["B:layer-wise-FiLM"] if use_film else []
        )
        print(
            f"[FGD] mode={mode}, lora_r={lora_r}"
            + (f", additions=[{', '.join(additions)}]" if additions else ", baseline")
        )
        print(f"SSL input size : {self.ssl_size[0]}×{self.ssl_size[1]}")
        print(
            f"Patch grid     : {self.n_patches_h}×{self.n_patches_w}"
            f" = {self.n_patches} tokens"
        )
        if use_multiscale:
            print(f"Tap layers     : {self.tap_layers}")

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder = build_encoder(model_id, mode, lora_r, lora_alpha, lora_dropout)

        enc_dim = 1024

        # ── Feature normalizer — single-layer path only ────────────────────
        self.feat_norm = nn.LayerNorm(enc_dim) if not use_multiscale else None

        # ── Addition A: multi-scale projection ─────────────────────────────
        self.ms_proj = (
            MultiScaleProjection(
                enc_dim,
                hidden_dim,
                self.tap_layers,
                self.n_patches_h,
                self.n_patches_w,
            )
            if use_multiscale
            else None
        )

        # ── Addition B: layer-wise FiLM conditioning ───────────────────────
        # LayerWiseFiLM applies independent γ_l / β_l per tap layer before
        # summation, enabling location-dependent layer weighting.
        self.film = (
            LayerWiseFiLM(
                hidden_dim=hidden_dim,
                n_tap=len(self.tap_layers),
                n_patches_h=self.n_patches_h,
                n_patches_w=self.n_patches_w,
                coord_hidden=film_hidden,
            )
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
            if self.use_film:
                # Addition A+B: get per-layer spatial maps, apply layer-wise
                # FiLM per layer, sum inside LayerWiseFiLM
                spatial_list = self.ms_proj(hidden_states, return_list=True)
                feat = self.film(spatial_list)  # [B, hidden_dim, H_p, W_p]
            else:
                # Addition A only: project and sum
                feat = self.ms_proj(hidden_states)  # [B, hidden_dim, H_p, W_p]
        else:
            # Baseline: single final layer
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
