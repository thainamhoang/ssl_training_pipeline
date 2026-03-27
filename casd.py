"""
casd.py

CASD — Climate-Aware SSL Decoder

Pilot 4 architecture. Replaces the pixel-shuffle decoder from SSLDownscaler
with a multi-scale token extractor and cross-attention decoder where the HR
output grid queries SSL features using geographic coordinates.

Architecture overview:
    1. SSL encoder (frozen or LoRA) → hidden_states at tap layers [4, 7, 14]
    2. Per-layer projection: 1024 → proj_dim (default 256)
    3. Token pool: concat projected tokens → [B, 3×512, proj_dim]
    4. HR coordinate grid: sin/cos(lat, lon) [+ orography + lsm] → queries
    5. Cross-attention: Q=coord_grid, K=V=token_pool → [B, H_hr×W_hr, proj_dim]
    6. Reshape + Conv2d → [B, 1, H_hr, W_hr]

Static variables (optional):
    Orography and land-sea mask injected into coordinate queries, giving each
    HR pixel a terrain-aware identity in addition to lat/lon.
    use_static=False (default) → coord_dim=4, lat/lon only (ablation baseline)
    use_static=True            → coord_dim=6, lat/lon + oro + lsm (full method)

Tap layer motivation (from layer probe, n=100 ERA5 samples):
    Layer  4 : T2m r=0.895, PC1 var=88% — primary temperature signal
    Layer  7 : T2m r=0.894, Lat r=0.848 — geographic transition point
    Layer 14 : T2m r=0.890, PC1 var=21% — mid-range local maximum, distributed
    Layers 22-23 avoided: T2m r drops to 0.70-0.72 (signal degradation)
    Layer 24 (Pilot 2-3 baseline): r=0.792 — worse than all tap layers

Usage:
    # CASD frozen, lat/lon only (ablation baseline)
    model = CASD(
        model_id   = "facebook/dinov3-vitl16-pretrain-sat493m",
        patch_size = 16,
        lr_shape   = (32, 64),
        hr_shape   = (128, 256),
        mode       = "frozen",
        tap_layers = [4, 7, 14],
        use_static = False,
    )

    # CASD frozen + static variables (terrain-aware queries)
    model = CASD(
        model_id   = "facebook/dinov3-vitl16-pretrain-sat493m",
        patch_size = 16,
        lr_shape   = (32, 64),
        hr_shape   = (128, 256),
        mode       = "frozen",
        tap_layers = [4, 7, 14],
        use_static = True,
        oro_hr     = oro_tensor,   # [128, 256] z-score normalised float32
        lsm_hr     = lsm_tensor,   # [128, 256] values in {0, 1} float32
    )

    # CASD LoRA r=8 + static (full Pilot 4 method)
    model = CASD(
        model_id   = "facebook/dinov3-vitl16-pretrain-sat493m",
        patch_size = 16,
        lr_shape   = (32, 64),
        hr_shape   = (128, 256),
        mode       = "lora",
        lora_r     = 8,
        tap_layers = [4, 7, 14],
        use_static = True,
        oro_hr     = oro_tensor,
        lsm_hr     = lsm_tensor,
    )

Static variable preparation:
    import numpy as np, torch, torch.nn.functional as F

    data   = np.load("/workspace/data/1.40625deg/2m_temperature/constants.npz")
    oro_np = data["orography"]   # [H_hr, W_hr]
    lsm_np = data["lsm"]         # [H_hr, W_hr]

    oro_t  = torch.tensor(oro_np, dtype=torch.float32)
    oro_t  = (oro_t - oro_t.mean()) / (oro_t.std() + 1e-8)   # z-score
    lsm_t  = torch.tensor(lsm_np, dtype=torch.float32)        # keep {0,1}

    # Resize to HR shape if not already (128, 256)
    if tuple(oro_t.shape) != (128, 256):
        oro_t = F.interpolate(oro_t[None, None], size=(128, 256),
                              mode="bilinear", align_corners=False).squeeze()
        lsm_t = F.interpolate(lsm_t[None, None], size=(128, 256),
                              mode="nearest").squeeze()
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

try:
    from peft import LoraConfig, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# ── Geographic coordinate encoding ────────────────────────────────────────


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

    Args:
        hr_shape   : (H_hr, W_hr)
        use_static : append orography and lsm channels
        oro_hr     : [H_hr, W_hr] z-score normalised orography
        lsm_hr     : [H_hr, W_hr] land-sea mask in {0, 1}

    Returns:
        coords : [H_hr*W_hr, coord_dim] float32
    """
    if use_static:
        assert oro_hr is not None and lsm_hr is not None, (
            "oro_hr and lsm_hr must be provided when use_static=True"
        )
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

    channels = [
        lat_grid.sin(),
        lat_grid.cos(),
        lon_grid.sin(),
        lon_grid.cos(),
    ]

    if use_static:
        channels.append(oro_hr.float().cpu())
        channels.append(lsm_hr.float().cpu())

    coords = torch.stack(channels, dim=-1)  # [H_hr, W_hr, coord_dim]
    return coords.reshape(H_hr * W_hr, -1).float()  # [H_hr*W_hr, coord_dim]


class CoordProjection(nn.Module):
    """
    Projects coord_dim coordinates to proj_dim query vectors.

    coord_dim = 4  (use_static=False): sin/cos lat/lon
    coord_dim = 6  (use_static=True):  sin/cos lat/lon + oro + lsm

    Two-layer MLP + GELU + LayerNorm for stable cross-attention queries.
    """

    def __init__(self, proj_dim: int, coord_dim: int = 4):
        super().__init__()
        self.coord_dim = coord_dim
        self.proj = nn.Sequential(
            nn.Linear(coord_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.proj(coords)


# ── Multi-scale token projections ─────────────────────────────────────────


class TokenProjection(nn.Module):
    """
    Projects encoder tokens 1024 → proj_dim. One instance per tap layer.
    Weights are NOT shared across layers — each layer gets its own projection.
    """

    def __init__(self, enc_dim: int = 1024, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(enc_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(tokens)


# ── Cross-attention decoder ────────────────────────────────────────────────


class CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block with feed-forward sublayer.

        Q   = HR coordinate queries  [B, H_hr*W_hr, proj_dim]
        K,V = multi-scale token pool [B, n_tap*n_patches, proj_dim]

    Uses F.scaled_dot_product_attention — automatically uses FlashAttention
    when available (5090, A100, H100).
    """

    def __init__(
        self, proj_dim: int, n_heads: int = 8, ff_mult: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        assert proj_dim % n_heads == 0, (
            f"proj_dim={proj_dim} must be divisible by n_heads={n_heads}"
        )

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
        """
        Args:
            queries     : [B, N_q,  proj_dim]
            keys_values : [B, N_kv, proj_dim]
        Returns:
            [B, N_q, proj_dim]
        """
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
            Q,
            K,
            V,
            dropout_p=self.drop.p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, N_q, D)
        attn_out = self.o_proj(attn_out)

        queries = queries + self.drop(attn_out)
        queries = queries + self.drop(self.ff(self.norm_ff(queries)))

        return queries


# ── Output head ───────────────────────────────────────────────────────────


class OutputHead(nn.Module):
    """
    Reshape [B, H_hr*W_hr, proj_dim] → [B, 1, H_hr, W_hr].

    3×3 → 3×3 → 1×1 conv pipeline.
    Zero-init final layer → predictions start near zero at epoch 0.
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
        self._init_weights()

    def _init_weights(self):
        for m in self.refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        H, W = self.hr_shape
        feat = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return self.refine(feat)


# ── Main CASD model ───────────────────────────────────────────────────────


class CASD(nn.Module):
    """
    Climate-Aware SSL Decoder (CASD).

    Parameters
    ----------
    model_id     : HuggingFace model ID
    patch_size   : ViT patch size (16 for DINOv3-SAT)
    lr_shape     : (H_lr, W_lr) e.g. (32, 64)
    hr_shape     : (H_hr, W_hr) e.g. (128, 256)
    mode         : "frozen" | "lora"
    lora_r       : LoRA rank (ignored when mode="frozen")
    lora_alpha   : LoRA alpha (defaults to 2*lora_r)
    lora_dropout : LoRA dropout (default 0.0)
    tap_layers   : encoder layers to tap (default [4, 7, 14])
    proj_dim     : projection dimension (default 256)
    n_heads      : cross-attention heads (default 8)
    n_ca_layers  : stacked cross-attention blocks (default 2)
    ca_dropout   : cross-attention dropout (default 0.0)
    use_static   : inject orography + lsm into coord queries (default False)
    oro_hr       : [H_hr, W_hr] z-score normalised orography (required if use_static)
    lsm_hr       : [H_hr, W_hr] land-sea mask {0,1} (required if use_static)
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
        if mode == "lora" and not _PEFT_AVAILABLE:
            raise ImportError("pip install peft>=0.10.0 to use mode='lora'")
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

        # ── SSL input size ─────────────────────────────────────────────────
        H_lr, W_lr = lr_shape
        ssl_H = 16 * patch_size
        ssl_W = ssl_H * W_lr // H_lr
        ssl_W = ((ssl_W + patch_size - 1) // patch_size) * patch_size

        self.ssl_size = (ssl_H, ssl_W)
        self.n_patches_h = ssl_H // patch_size
        self.n_patches_w = ssl_W // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        print(
            f"[CASD] mode={mode}"
            + (f", lora_r={lora_r}" if mode == "lora" else "")
            + f", use_static={use_static}"
        )
        print(f"SSL input size : {ssl_H}×{ssl_W}")
        print(
            f"Patch grid     : {self.n_patches_h}×{self.n_patches_w}"
            f" = {self.n_patches} tokens"
        )
        print(f"Tap layers     : {self.tap_layers}")
        print(
            f"Token pool     : {self.n_tap}×{self.n_patches}"
            f" = {self.n_tap * self.n_patches} tokens"
        )
        print(
            f"HR queries     : {hr_shape[0] * hr_shape[1]}"
            f" ({hr_shape[0]}×{hr_shape[1]})"
        )
        print(
            f"Coord dim      : {self.coord_dim}"
            f" ({'lat/lon+oro+lsm' if use_static else 'lat/lon only'})"
        )

        # ── SSL Encoder ────────────────────────────────────────────────────
        enc_dim = 1024
        encoder = AutoModel.from_pretrained(model_id)

        if mode == "frozen":
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()
            self.encoder = encoder
        else:
            for p in encoder.parameters():
                p.requires_grad = False
            _alpha = lora_alpha if lora_alpha is not None else lora_r * 2
            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.encoder = get_peft_model(encoder, lora_cfg)
            self.encoder.train()

        # ── Per-layer token projections ────────────────────────────────────
        self.token_projs = nn.ModuleList(
            [TokenProjection(enc_dim, proj_dim) for _ in self.tap_layers]
        )

        # ── HR coordinate grid (fixed buffer) ─────────────────────────────
        # Shape: [H_hr*W_hr, coord_dim] — 4 or 6 depending on use_static
        # Registered as buffer: moves to GPU with .to(device),
        # saved in checkpoint, never touched by optimizer.
        coords = build_coord_grid(
            hr_shape,
            use_static=use_static,
            oro_hr=oro_hr,
            lsm_hr=lsm_hr,
        )
        self.register_buffer("coord_grid", coords)

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

        # ── Parameter summary ──────────────────────────────────────────────
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

    # ── Multi-scale encoding ───────────────────────────────────────────────

    def _encode_multiscale(self, x: torch.Tensor) -> list:
        """
        Forward through SSL encoder with output_hidden_states=True.
        Extract patch tokens from each tap layer.

        Args:
            x : [B, 3, H_lr, W_lr] ImageNet-normalised float32

        Returns:
            list of n_tap tensors, each [B, n_patches, 1024] float32
        """
        x_ssl = F.interpolate(
            x, size=self.ssl_size, mode="bilinear", align_corners=False
        )

        if self.mode == "frozen":
            with torch.no_grad():
                out = self.encoder(
                    pixel_values=x_ssl.to(dtype=torch.bfloat16),
                    output_hidden_states=True,
                )
        else:
            out = self.encoder(
                pixel_values=x_ssl.to(dtype=torch.bfloat16),
                output_hidden_states=True,
            )

        # Token layout: [CLS | patch_0..patch_511 | reg_0..reg_3]
        tokens_per_layer = []
        for layer_idx in self.tap_layers:
            hs = out.hidden_states[layer_idx]  # [B, 517, 1024]
            tokens = hs[:, 1 : 1 + self.n_patches, :].float()
            tokens_per_layer.append(tokens)  # [B, 512, 1024]

        return tokens_per_layer

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H_lr, W_lr] ImageNet-normalised float32

        Returns:
            [B, 1, H_hr, W_hr] predicted z-score normalised HR field
        """
        B = x.shape[0]

        # 1. Multi-scale token extraction
        tokens_per_layer = self._encode_multiscale(x)

        # 2. Project each tap layer to proj_dim
        projected = [
            proj(tokens) for proj, tokens in zip(self.token_projs, tokens_per_layer)
        ]

        # 3. Token pool [B, n_tap*n_patches, proj_dim]  e.g. [B, 1536, 256]
        token_pool = torch.cat(projected, dim=1)

        # 4. HR coordinate queries
        # coord_grid is [H*W, coord_dim] — expand to batch
        coords = self.coord_grid.unsqueeze(0).expand(B, -1, -1)
        queries = self.coord_proj(coords)  # [B, H*W, proj_dim]

        # 5. Cross-attention decoder
        for ca_block in self.ca_blocks:
            queries = ca_block(queries, token_pool)  # [B, H*W, proj_dim]

        # 6. Reshape + conv prediction
        out = self.output_head(queries)  # [B, 1, H_hr, W_hr]

        if out.shape[-2:] != self.hr_shape:
            out = F.interpolate(
                out, size=self.hr_shape, mode="bilinear", align_corners=False
            )

        return out

    # ── Optimiser ─────────────────────────────────────────────────────────

    def make_optimizer(
        self,
        lr: float = 5e-5,
        decoder_lr: float = 2e-4,
        weight_decay: float = 1e-2,
    ):
        """
        Two parameter groups:
            LoRA params    : lr (conservative)
            Decoder params : decoder_lr (higher — trained from scratch)

        Frozen mode: single group at decoder_lr.
        """
        if self.mode == "frozen":
            return torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad],
                lr=decoder_lr,
                weight_decay=1e-4,
            )

        lora_params, decoder_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(param)
            else:
                decoder_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": lora_params, "lr": lr, "weight_decay": weight_decay},
                {"params": decoder_params, "lr": decoder_lr, "weight_decay": 1e-4},
            ],
            betas=(0.9, 0.999),
        )

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer,
        scheduler,
        scaler,
        best_val_rmse: float,
        epochs_no_improve: int,
        wandb_run_id: str = None,
    ):
        """Save full training state including architecture config."""
        torch.save(
            {
                "epoch": epoch,
                "mode": self.mode,
                "lora_r": self.lora_r,
                "tap_layers": self.tap_layers,
                "proj_dim": self.proj_dim,
                "use_static": self.use_static,
                "coord_dim": self.coord_dim,
                "model_state": self.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "best_val_rmse": best_val_rmse,
                "epochs_no_improve": epochs_no_improve,
                "wandb_run_id": wandb_run_id,
            },
            path,
        )

    @classmethod
    def load_checkpoint(
        cls, path: str, model_kwargs: dict, optimizer=None, scheduler=None, scaler=None
    ):
        """
        Resume from checkpoint.

        model_kwargs must include use_static, oro_hr, lsm_hr if applicable.

        Example:
            model, ckpt = CASD.load_checkpoint(
                "latest.pt",
                model_kwargs=dict(
                    model_id="facebook/dinov3-vitl16-pretrain-sat493m",
                    patch_size=16, lr_shape=(32,64), hr_shape=(128,256),
                    mode="lora", lora_r=8, tap_layers=[4,7,14],
                    use_static=True, oro_hr=oro_t, lsm_hr=lsm_t,
                ),
            )
            start_epoch   = ckpt["epoch"] + 1
            best_val_rmse = ckpt["best_val_rmse"]
            wandb_run_id  = ckpt["wandb_run_id"]
        """
        ckpt = torch.load(path, map_location="cpu")

        assert ckpt["mode"] == model_kwargs.get("mode", "frozen"), (
            f"mode mismatch: ckpt={ckpt['mode']} vs {model_kwargs.get('mode')}"
        )
        assert ckpt["tap_layers"] == model_kwargs.get("tap_layers", [4, 7, 14]), (
            "tap_layers mismatch"
        )
        assert ckpt["use_static"] == model_kwargs.get("use_static", False), (
            f"use_static mismatch: ckpt={ckpt['use_static']} vs {model_kwargs.get('use_static')}"
        )

        model = cls(**model_kwargs)
        model.load_state_dict(ckpt["model_state"])

        if optimizer and ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])

        return model, ckpt
