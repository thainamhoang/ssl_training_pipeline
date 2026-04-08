"""
casd.py

CASD — Climate-Aware SSL Decoder. Pilot 4.

Replaces the pixel-shuffle decoder with a multi-scale token extractor and
cross-attention decoder where the HR output grid queries SSL features using
geographic coordinates.

Architecture:
    1. SSL encoder (frozen or LoRA) → hidden_states at tap layers [4, 7, 14]
    2. Per-layer projection: 1024 → proj_dim (default 256)
    3. Token pool: concat projected tokens → [B, n_tap×512, proj_dim]
    4. HR coordinate grid: sin/cos(lat, lon) [+ orography + lsm] → queries
    5. Cross-attention: Q=coord_grid, K=V=token_pool → [B, H_hr×W_hr, proj_dim]
    6. Reshape + Conv2d → [B, 1, H_hr, W_hr]

Tap layer motivation (from layer probe, n=100 ERA5 samples):
    Layer  4 : T2m r=0.895 — primary temperature signal
    Layer  7 : T2m r=0.894 — geographic transition point
    Layer 14 : T2m r=0.890 — mid-range local maximum
    Layer 24 (Pilot 2-3 baseline): r=0.792 — worse than all tap layers

Static variables (optional):
    use_static=False → coord_dim=4, lat/lon only (ablation baseline)
    use_static=True  → coord_dim=6, lat/lon + orography + lsm (full method)
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


# ── Geographic coordinate encoding ────────────────────────────────────────────


def build_coord_grid(
    hr_shape: tuple,
    use_static: bool = False,
    oro_hr: torch.Tensor = None,
    lsm_hr: torch.Tensor = None,
) -> torch.Tensor:
    """
    Build coordinate encoding for every HR output pixel.

    use_static=False → [H_hr*W_hr, 4]  sin/cos lat/lon
    use_static=True  → [H_hr*W_hr, 6]  sin/cos lat/lon + oro + lsm

    ERA5 1.40625° grid:
        Latitudes  : 90° → -90°  (128 rows, north to south)
        Longitudes : 0°  → 358.59375°  (256 cols)
    """
    if use_static:
        assert oro_hr is not None and lsm_hr is not None
        assert tuple(oro_hr.shape) == tuple(hr_shape), (
            f"oro_hr shape {tuple(oro_hr.shape)} != hr_shape {hr_shape}"
        )
        assert tuple(lsm_hr.shape) == tuple(hr_shape), (
            f"lsm_hr shape {tuple(lsm_hr.shape)} != hr_shape {hr_shape}"
        )

    H_hr, W_hr = hr_shape
    lats = torch.linspace(90.0, -90.0, H_hr)
    lons = torch.linspace(0.0, 358.59375, W_hr)
    lat_rad = lats * (math.pi / 180.0)
    lon_rad = lons * (math.pi / 180.0)
    lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")

    channels = [lat_grid.sin(), lat_grid.cos(), lon_grid.sin(), lon_grid.cos()]
    if use_static:
        channels.append(oro_hr.float().cpu())
        channels.append(lsm_hr.float().cpu())

    coords = torch.stack(channels, dim=-1)  # [H, W, coord_dim]
    return coords.reshape(H_hr * W_hr, -1).float()  # [H*W, coord_dim]


# ── Submodules ────────────────────────────────────────────────────────────────


class CoordProjection(nn.Module):
    """Two-layer MLP + GELU + LayerNorm projecting coord_dim → proj_dim."""

    def __init__(self, proj_dim: int, coord_dim: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(coord_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.proj(coords)


class TokenProjection(nn.Module):
    """Linear + LayerNorm projecting enc_dim → proj_dim. One per tap layer."""

    def __init__(self, enc_dim: int = 1024, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(enc_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


class CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block with feed-forward sublayer.

        Q   = HR coordinate queries  [B, H_hr*W_hr, proj_dim]
        K,V = multi-scale token pool [B, n_tap*n_patches, proj_dim]

    Uses F.scaled_dot_product_attention — FlashAttention when available.
    """

    def __init__(
        self,
        proj_dim: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert proj_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = proj_dim // n_heads

        self.q_proj = nn.Linear(proj_dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(proj_dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(proj_dim, proj_dim, bias=False)
        self.o_proj = nn.Linear(proj_dim, proj_dim, bias=False)

        ff_dim = proj_dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(proj_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, proj_dim),
            nn.Dropout(dropout),
        )
        self.norm_q = nn.LayerNorm(proj_dim)
        self.norm_kv = nn.LayerNorm(proj_dim)
        self.norm_ff = nn.LayerNorm(proj_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys_values: torch.Tensor) -> torch.Tensor:
        B, N_q, D = queries.shape

        q = self.norm_q(queries)
        kv = self.norm_kv(keys_values)

        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        def split_heads(t):
            B_, N_, D_ = t.shape
            return t.reshape(B_, N_, self.n_heads, D_ // self.n_heads).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.drop.p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, N_q, D)
        attn_out = self.o_proj(attn_out)

        queries = queries + self.drop(attn_out)
        queries = queries + self.drop(self.ff(self.norm_ff(queries)))
        return queries


class OutputHead(nn.Module):
    """
    Reshape [B, H_hr*W_hr, proj_dim] → [B, 1, H_hr, W_hr].
    3×3 → 3×3 → 1×1 conv pipeline. Final conv zero-initialised.
    """

    def __init__(self, proj_dim: int, hr_shape: tuple):
        super().__init__()
        self.hr_shape = hr_shape
        self.refine = nn.Sequential(
            nn.Conv2d(proj_dim, proj_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(proj_dim, proj_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(proj_dim // 2, 1, kernel_size=1),
        )
        init_conv_decoder(self.refine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H, W = self.hr_shape
        feat = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return self.refine(feat)


# ── Main model ────────────────────────────────────────────────────────────────


class CASD(nn.Module):
    """
    Climate-Aware SSL Decoder.

    Parameters
    ----------
    model_id     : HuggingFace model ID
    patch_size   : ViT patch size (16 for DINOv3-SAT)
    lr_shape     : (H_lr, W_lr) e.g. (32, 64)
    hr_shape     : (H_hr, W_hr) e.g. (128, 256)
    mode         : "frozen" | "lora"
    lora_r       : LoRA rank (ignored when mode="frozen")
    lora_alpha   : LoRA alpha (defaults to lora_r)
    lora_dropout : LoRA dropout (default 0.0)
    tap_layers   : encoder layers to tap (default [4, 7, 14])
    proj_dim     : projection dimension (default 256)
    n_heads      : cross-attention heads (default 8)
    n_ca_layers  : stacked cross-attention blocks (default 2)
    ca_dropout   : cross-attention dropout (default 0.0)
    use_static   : inject orography + lsm into coord queries
    oro_hr       : [H_hr, W_hr] z-score normalised orography
    lsm_hr       : [H_hr, W_hr] land-sea mask {0, 1}
    """

    def __init__(
        self,
        model_id: str,
        patch_size: int,
        lr_shape: tuple,
        hr_shape: tuple,
        mode: str = "frozen",
        lora_r: int = 8,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
        tap_layers: list = None,
        proj_dim: int = 256,
        n_heads: int = 8,
        n_ca_layers: int = 2,
        ca_dropout: float = 0.0,
        use_static: bool = False,
        oro_hr: torch.Tensor = None,
        lsm_hr: torch.Tensor = None,
    ):
        super().__init__()

        assert mode in ("frozen", "lora"), (
            f"mode must be 'frozen' or 'lora', got {mode!r}"
        )
        if use_static and (oro_hr is None or lsm_hr is None):
            raise ValueError("oro_hr and lsm_hr required when use_static=True")

        self.mode = mode
        self.patch_size = patch_size
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.lora_r = lora_r
        self.tap_layers = tap_layers if tap_layers is not None else [4, 7, 14]
        self.proj_dim = proj_dim
        self.n_tap = len(self.tap_layers)
        self.use_static = use_static
        self.coord_dim = 6 if use_static else 4

        self.ssl_size, self.n_patches_h, self.n_patches_w, self.n_patches = (
            compute_ssl_size(lr_shape, patch_size)
        )

        print(
            f"[CASD] mode={mode}"
            + (f", lora_r={lora_r}" if mode == "lora" else "")
            + f", use_static={use_static}"
        )
        print(f"SSL input size : {self.ssl_size[0]}×{self.ssl_size[1]}")
        print(
            f"Patch grid     : {self.n_patches_h}×{self.n_patches_w} = {self.n_patches} tokens"
        )
        print(f"Tap layers     : {self.tap_layers}")
        print(
            f"Token pool     : {self.n_tap}×{self.n_patches} = {self.n_tap * self.n_patches} tokens"
        )
        print(
            f"HR queries     : {hr_shape[0] * hr_shape[1]} ({hr_shape[0]}×{hr_shape[1]})"
        )
        print(
            f"Coord dim      : {self.coord_dim} ({'lat/lon+oro+lsm' if use_static else 'lat/lon only'})"
        )

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder = build_encoder(model_id, mode, lora_r, lora_alpha, lora_dropout)

        enc_dim = 1024

        # ── Per-layer token projections ────────────────────────────────────
        self.token_projs = nn.ModuleList(
            [TokenProjection(enc_dim, proj_dim) for _ in self.tap_layers]
        )

        # ── HR coordinate grid (fixed buffer) ─────────────────────────────
        coords = build_coord_grid(
            hr_shape, use_static=use_static, oro_hr=oro_hr, lsm_hr=lsm_hr
        )
        self.register_buffer("coord_grid", coords)  # [H*W, coord_dim]

        # ── Coordinate projection ──────────────────────────────────────────
        self.coord_proj = CoordProjection(proj_dim, coord_dim=self.coord_dim)

        # ── Cross-attention decoder ────────────────────────────────────────
        self.ca_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(proj_dim, n_heads, ff_mult=4, dropout=ca_dropout)
                for _ in range(n_ca_layers)
            ]
        )

        # ── Output head ────────────────────────────────────────────────────
        self.output_head = OutputHead(proj_dim, hr_shape)

        n_enc = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_dec = (
            sum(p.numel() for p in self.token_projs.parameters())
            + sum(p.numel() for p in self.coord_proj.parameters())
            + sum(p.numel() for p in self.ca_blocks.parameters())
            + sum(p.numel() for p in self.output_head.parameters())
        )
        print(f"Trainable — encoder : {n_enc:,}")
        print(f"Trainable — decoder : {n_dec:,}")
        print(f"Trainable — total   : {n_enc + n_dec:,}")

    def _encode_multiscale(self, x: torch.Tensor) -> list:
        x_ssl = F.interpolate(
            x, size=self.ssl_size, mode="bilinear", align_corners=False
        )
        out = encoder_forward(
            self.encoder,
            self.mode,
            x_ssl.to(dtype=torch.bfloat16),
            output_hidden_states=True,
        )
        return [
            out.hidden_states[i][:, 1 : 1 + self.n_patches, :].float()
            for i in self.tap_layers
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H_lr, W_lr] ImageNet-normalised float32
        Returns:
            [B, 1, H_hr, W_hr] predicted z-score normalised HR field
        """
        B = x.shape[0]

        tokens_per_layer = self._encode_multiscale(x)

        projected = [
            proj(tokens) for proj, tokens in zip(self.token_projs, tokens_per_layer)
        ]
        token_pool = torch.cat(projected, dim=1)  # [B, n_tap*N, proj_dim]

        queries = self.coord_proj(self.coord_grid).unsqueeze(0).expand(B, -1, -1)
        # Coordinate queries are identical for every sample in the batch, so
        # project once and broadcast instead of re-running the MLP B times.

        for ca_block in self.ca_blocks:
            queries = ca_block(queries, token_pool)

        out = self.output_head(queries)  # [B, 1, H_hr, W_hr]

        if out.shape[-2:] != self.hr_shape:
            out = F.interpolate(
                out, size=self.hr_shape, mode="bilinear", align_corners=False
            )

        return out
