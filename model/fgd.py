"""
fgd.py

FGD — Frequency-Geographic Decoder

Pilot 5 architecture. Extends SSLDownscaler with four additions on top of
the existing LoRA r=8 encoder, each independently ablatable:

    A — Multi-scale token fusion
        Taps encoder hidden states at layers [4, 7, 14] instead of only
        layer 24. Projects each to hidden_dim and sums into a single spatial
        feature map before pixel-shuffle. Motivated by probe results showing
        T2m correlation r>0.88 across layers 1-20, degrading at layer 24.

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

    D — CLS token atmospheric conditioning
        Uses CLS tokens from tap layers as dynamic FiLM conditioning on the
        fused spatial feature map. Unlike static lat/lon coordinates (B),
        this conditioning varies per timestep — it encodes the current
        global atmospheric state.

        check_cls_layout.py (n=100) T2m correlation advantage over patches:
            Layer  4: CLS r=0.258, patch r=0.218, advantage=+0.040
            Layer  7: CLS r=0.221, patch r=0.098, advantage=+0.123
            Layer 14: CLS r=0.266, patch r=0.011, advantage=+0.256
        Layer 14 CLS retains global T2m signal where patch tokens have lost it.

        Applied after multi-scale fusion (and B if active), before decoder:
            cls_vec = norm(CLS_14)                   [B, 1024]
            γ, β    = chunk(mlp(cls_vec), 2)         [B, hidden_dim] each
            feat    = (1 + γ[...,None,None]) * feat + β[...,None,None]

        Zero-init on final MLP layer → identity at epoch 0.
        ~130K parameters. VRAM cost: negligible.

Ablation modes (set via config):
    use_multiscale=False, use_film=False, use_cls=False → identical to SSLDownscaler
    use_multiscale=True,  use_film=False, use_cls=False → addition A only
    use_multiscale=True,  use_film=True,  use_cls=False → additions A + B
    use_multiscale=True,  use_film=False, use_cls=True  → additions A + D
    use_multiscale=True,  use_film=True,  use_cls=True  → additions A + B + D
    spectral_lambda > 0                                 → addition C (any mode)
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


# ── Layer-wise FiLM geographic conditioning (Addition B) ──────────────────────


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
            nn.Linear(coord_hidden, 2 * n_tap * hidden_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, spatial_list: list) -> torch.Tensor:
        """
        Apply per-layer FiLM modulation then sum.

        Args:
            spatial_list : list of n_tap tensors, each [B, hidden_dim, H_p, W_p]

        Returns:
            fused : [B, hidden_dim, H_p, W_p]
        """
        B = spatial_list[0].shape[0]

        all_params = self.mlp(self.coord_grid)  # [H_p, W_p, 2*n_tap*D]
        all_params = all_params.permute(2, 0, 1)  # [2*n_tap*D, H_p, W_p]
        all_params = all_params.unsqueeze(0).expand(B, -1, -1, -1)

        gammas_all, betas_all = all_params.chunk(2, dim=1)
        gammas = gammas_all.chunk(self.n_tap, dim=1)
        betas = betas_all.chunk(self.n_tap, dim=1)

        fused = None
        for feat, gamma, beta in zip(spatial_list, gammas, betas):
            modulated = (1.0 + gamma) * feat + beta
            fused = modulated if fused is None else fused + modulated

        return fused  # [B, hidden_dim, H_p, W_p]


# ── Post-summation FiLM conditioning (legacy Addition B) ─────────────────────


class FiLMConditioner(nn.Module):
    """
    Post-summation Feature-wise Linear Modulation conditioned on sin/cos lat/lon.

    Legacy variant — applies FiLM to the already-collapsed sum rather than
    per-layer. Kept for backward checkpoint compatibility.
    Prefer LayerWiseFiLM for new experiments.
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
        )
        self.register_buffer("coord_grid", coords)

        self.mlp = nn.Sequential(
            nn.Linear(4, coord_hidden),
            nn.GELU(),
            nn.Linear(coord_hidden, 2 * hidden_dim),
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat : [B, hidden_dim, H_p, W_p]"""
        B = feat.shape[0]
        film_params = self.mlp(self.coord_grid)
        film_params = film_params.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        gamma, beta = film_params.chunk(2, dim=1)
        return (1.0 + gamma) * feat + beta


# ── CLS token atmospheric conditioning (Addition D) ──────────────────────────


class CLSConditioner(nn.Module):
    """
    Dynamic atmospheric conditioning via CLS tokens.

    Applies FiLM conditioning to the fused spatial feature map using the CLS
    token from a chosen tap layer. Unlike LayerWiseFiLM (static lat/lon), this
    conditioning varies per timestep — it encodes the current atmospheric state.

    check_cls_layout.py (n=100) T2m correlation advantage over patch mean:
        Layer  4: CLS r=0.258, patch r=0.218, advantage=+0.040
        Layer  7: CLS r=0.221, patch r=0.098, advantage=+0.123
        Layer 14: CLS r=0.266, patch r=0.011, advantage=+0.256  <- largest

    Layer 14 CLS retains global T2m signal (r=0.266) where patch tokens have
    lost it entirely (r=0.011). Default cls_layer_idx=-1 selects the last
    tap layer, which is layer 14 when tap_layers=[4, 7, 14].

    Architecture:
        CLS [B, 1024] -> LayerNorm -> Linear(1024->cls_hidden) -> GELU
        -> Linear(cls_hidden->2*hidden_dim) -> split -> (gamma, beta)
        -> feat = (1 + gamma[...,None,None]) * feat + beta[...,None,None]

    Zero-init final linear -> identity at epoch 0, training-stable.
    ~130K parameters. VRAM cost: negligible.

    Parameters
    ----------
    hidden_dim    : feature map channels (must match fused feature map)
    enc_dim       : encoder hidden dim (1024 for DINOv3-SAT)
    cls_hidden    : MLP hidden size (default 128)
    cls_layer_idx : index into tap_layers list for CLS source.
                    Default -1 = last tap layer (layer 14 when taps=[4,7,14]).
                    Set None to average CLS across all tap layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        enc_dim: int = 1024,
        cls_hidden: int = 128,
        cls_layer_idx: int = -1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.cls_layer_idx = cls_layer_idx

        self.norm = nn.LayerNorm(enc_dim)
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, cls_hidden),
            nn.GELU(),
            nn.Linear(cls_hidden, 2 * hidden_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.mlp[0].weight)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].weight)  # zero-init → identity at epoch 0
        nn.init.zeros_(self.mlp[2].bias)

    def forward(
        self,
        feat: torch.Tensor,
        hidden_states: tuple,
        tap_layers: list,
    ) -> torch.Tensor:
        """
        Args:
            feat          : [B, hidden_dim, H_p, W_p]
            hidden_states : tuple of encoder hidden states (25 tensors)
            tap_layers    : list of tapped layer indices, e.g. [4, 7, 14]

        Returns:
            [B, hidden_dim, H_p, W_p] modulated feature map
        """
        if self.cls_layer_idx is None:
            cls_vecs = [hidden_states[l][:, 0, :].float() for l in tap_layers]
            cls_vec = torch.stack(cls_vecs, dim=1).mean(dim=1)  # [B, 1024]
        else:
            layer_idx = tap_layers[self.cls_layer_idx]  # e.g. 14
            cls_vec = hidden_states[layer_idx][:, 0, :].float()  # [B, 1024]

        cls_vec = self.norm(cls_vec)  # [B, 1024]
        film_params = self.mlp(cls_vec)  # [B, 2*D]
        gamma, beta = film_params.chunk(2, dim=-1)  # Each [B, D]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return (1.0 + gamma) * feat + beta  # [B, D, H_p, W_p]


# ── Multi-scale token projections (Addition A) ────────────────────────────────


class MultiScaleProjection(nn.Module):
    """
    Projects and fuses tokens from multiple encoder layers into one
    spatial feature map of shape [B, hidden_dim, H_p, W_p].

    Each tap layer gets its own independent Linear projection (1024 → hidden_dim)
    + LayerNorm. The projections are summed (not concatenated) to keep the
    output channel count at hidden_dim regardless of how many layers are tapped.

    Parameters
    ----------
    enc_dim    : encoder hidden dim (1024 for DINOv3-SAT ViT-L)
    hidden_dim : output channel count = decoder input channels
    tap_layers : which hidden state layers to tap, e.g. [4, 7, 14]
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
            return_list   : if True return list of per-layer maps (for LayerWiseFiLM)
                            if False return summed map (for A-only or CLSConditioner)

        Returns:
            list of n_tap [B, hidden_dim, H_p, W_p]  if return_list=True
            [B, hidden_dim, H_p, W_p]                if return_list=False
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
            ).permute(0, 3, 1, 2)  # [B, D, H_p, W_p]

            if return_list:
                spatial_list.append(spatial)
            else:
                fused = spatial if fused is None else fused + spatial

        return spatial_list if return_list else fused


# ── Main model ────────────────────────────────────────────────────────────────


class FGD(nn.Module):
    """
    Frequency-Geographic Decoder.

    Extends SSLDownscaler (LoRA r=8, pixel-shuffle) with four ablatable additions:
        A — multi-scale token fusion      (use_multiscale=True)
        B — layer-wise FiLM conditioning  (use_film=True, requires A)
        C — spectral loss                 (spectral_lambda > 0, via fgd_loss)
        D — CLS token conditioning        (use_cls=True, requires A)

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
    tap_layers     : layers to tap (default [4, 7, 14])
    use_film       : enable layer-wise FiLM conditioning — Addition B
                     requires use_multiscale=True
    film_hidden    : FiLM MLP hidden size (default 128)
    use_cls        : enable CLS token conditioning — Addition D
                     requires use_multiscale=True
    cls_hidden     : CLS MLP hidden size (default 128)
    cls_layer_idx  : index into tap_layers for CLS source, default -1 (last)
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
        use_cls: bool = False,
        cls_hidden: int = 128,
        cls_layer_idx: int = -1,
    ):
        super().__init__()

        assert mode in ("frozen", "lora"), (
            f"mode must be 'frozen' or 'lora', got {mode!r}"
        )
        if use_film and not use_multiscale:
            raise ValueError("use_film=True requires use_multiscale=True")
        if use_cls and not use_multiscale:
            raise ValueError(
                "use_cls=True requires use_multiscale=True "
                "(hidden_states needed for CLS extraction)"
            )

        self.mode = mode
        self.patch_size = patch_size
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.upscale = upscale
        self.hidden_dim = hidden_dim
        self.lora_r = lora_r
        self.use_multiscale = use_multiscale
        self.use_film = use_film
        self.use_cls = use_cls
        self.tap_layers = tap_layers if tap_layers is not None else [4, 7, 14]

        self.ssl_size, self.n_patches_h, self.n_patches_w, self.n_patches = (
            compute_ssl_size(lr_shape, patch_size)
        )

        additions = (
            (["A:multiscale"] if use_multiscale else [])
            + (["B:layer-wise-FiLM"] if use_film else [])
            + (["D:CLS"] if use_cls else [])
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

        # ── Addition D: CLS token atmospheric conditioning ─────────────────
        # Applied after fusion (and B if active), before pixel-shuffle decoder.
        # Uses hidden_states so requires use_multiscale=True (output_hidden_states).
        self.cls_cond = (
            CLSConditioner(
                hidden_dim=hidden_dim,
                enc_dim=enc_dim,
                cls_hidden=cls_hidden,
                cls_layer_idx=cls_layer_idx,
            )
            if use_cls
            else None
        )

        # ── Pixel-shuffle decoder ──────────────────────────────────────────
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
            for m in [self.feat_norm, self.ms_proj, self.film, self.cls_cond]
            if m is not None
        )
        print(f"Trainable — encoder : {n_enc:,}")
        print(f"Trainable — decoder : {n_dec:,}")
        print(
            f"Trainable — aux     : {n_aux:,}  (feat_norm / ms_proj / film / cls_cond)"
        )
        print(f"Trainable — total   : {n_enc + n_dec + n_aux:,}")

    def _encode(self, x: torch.Tensor):
        """
        Run SSL encoder.

        Requests output_hidden_states when use_multiscale OR use_cls is True —
        both need access to intermediate hidden states.
        """
        x_ssl = F.interpolate(
            x, size=self.ssl_size, mode="bilinear", align_corners=False
        )
        need_hidden = self.use_multiscale or self.use_cls
        out = encoder_forward(
            self.encoder,
            self.mode,
            x_ssl.to(dtype=torch.bfloat16),
            output_hidden_states=need_hidden,
        )
        if need_hidden:
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
                # A + B: per-layer spatial maps → LayerWiseFiLM → sum
                spatial_list = self.ms_proj(hidden_states, return_list=True)
                feat = self.film(spatial_list)
            else:
                # A only (or A + D): project and sum
                feat = self.ms_proj(hidden_states)
        else:
            # Baseline: single final layer, same as SSLDownscaler
            patch_tokens = self.feat_norm(patch_tokens)
            B, N, D = patch_tokens.shape
            feat = (
                patch_tokens.reshape(B, self.n_patches_h, self.n_patches_w, D)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        # Addition D: CLS atmospheric conditioning — after fusion, before decoder
        if self.use_cls:
            feat = self.cls_cond(feat, hidden_states, self.tap_layers)

        out = self.decoder(feat)

        if out.shape[-2:] != self.hr_shape:
            out = F.interpolate(
                out, size=self.hr_shape, mode="bilinear", align_corners=False
            )

        return out
