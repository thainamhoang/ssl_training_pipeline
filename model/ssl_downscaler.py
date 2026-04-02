"""
ssl_downscaler.py

SSLDownscaler — Pilots 2 (frozen) and 3 (LoRA).

Unified SSL ViT-L encoder + pixel-shuffle SR decoder supporting two encoder
modes:
    mode="frozen"  — encoder fully frozen (identical to FrozenSSLDownscaler)
    mode="lora"    — LoRA adapters on q/v projections, all else frozen

DINOv3-SAT notes:
    - Attention projections: q_proj, v_proj (not query/value)
    - Uses RoPE — generalises to non-square inputs without modification
    - Top-level transformer blocks at model.layer (not model.encoder.layer)

Checkpoint compatibility:
    Frozen checkpoints from FrozenSSLDownscaler are NOT directly loadable
    here (module names differ due to PEFT wrapping). Use FrozenSSLDownscaler
    for those. LoRA checkpoints saved here load via trainer_utils.load_checkpoint.
"""

import torch
from torch import nn
from torch.nn import functional as F
from trainer_utils import (
    compute_ssl_size,
    build_encoder,
    init_conv_decoder,
    encoder_forward,
)


class SSLDownscaler(nn.Module):
    """
    Parameters
    ----------
    model_id     : HuggingFace model identifier
    patch_size   : ViT patch size (16 for DINOv3-SAT, 14 for DINOv2)
    lr_shape     : (H_lr, W_lr) of the LR input grid, e.g. (32, 64)
    hr_shape     : (H_hr, W_hr) of the HR target grid, e.g. (128, 256)
    upscale      : SR upscaling factor (2 or 4)
    hidden_dim   : decoder hidden channel count
    mode         : "frozen" | "lora"
    lora_r       : LoRA rank (ignored when mode="frozen"). Recommended: 8 or 16
    lora_alpha   : LoRA scaling factor. Defaults to lora_r
    lora_dropout : dropout on LoRA paths. 0.0 recommended for small datasets
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
        lora_r: int = 16,
        lora_alpha: int = None,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.mode = mode
        self.patch_size = patch_size
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.upscale = upscale
        self.lora_r = lora_r

        self.ssl_size, self.n_patches_h, self.n_patches_w, self.n_patches = (
            compute_ssl_size(lr_shape, patch_size)
        )

        print(
            f"[SSLDownscaler] mode={mode}"
            + (f", lora_r={lora_r}" if mode == "lora" else "")
        )
        print(f"SSL input size  : {self.ssl_size[0]}×{self.ssl_size[1]}")
        print(
            f"Patch grid      : {self.n_patches_h}×{self.n_patches_w} = {self.n_patches} tokens"
        )

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder = build_encoder(model_id, mode, lora_r, lora_alpha, lora_dropout)

        enc_dim = 1024

        # ── Feature normalizer (trained) ───────────────────────────────────
        self.feat_norm = nn.LayerNorm(enc_dim)

        # ── Pixel-shuffle decoder (trained) ────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Conv2d(enc_dim, hidden_dim * upscale * upscale, kernel_size=1),
            nn.PixelShuffle(upscale),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        init_conv_decoder(self.decoder)

        n_decoder = sum(p.numel() for p in self.decoder.parameters()) + sum(
            p.numel() for p in self.feat_norm.parameters()
        )
        if mode == "lora":
            n_lora = sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )
            print(
                f"Trainable params: {n_decoder + n_lora:,}  (LoRA: {n_lora:,}  decoder+norm: {n_decoder:,})"
            )
        else:
            print(f"Trainable params: {n_decoder:,}  (decoder+norm only)")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x_ssl = F.interpolate(
            x, size=self.ssl_size, mode="bilinear", align_corners=False
        )
        out = encoder_forward(self.encoder, self.mode, x_ssl.to(dtype=torch.bfloat16))
        return out.last_hidden_state[:, 1 : 1 + self.n_patches, :].float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H_lr, W_lr] ImageNet-normalised float32
        Returns:
            [B, 1, H_hr, W_hr] predicted z-score normalised HR field
        """
        patch_tokens = self._encode(x)  # [B, N, D]
        patch_tokens = self.feat_norm(patch_tokens)

        B, N, D = patch_tokens.shape
        feat = patch_tokens.reshape(B, self.n_patches_h, self.n_patches_w, D)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, D, H_p, W_p]

        out = self.decoder(feat)

        if out.shape[-2:] != self.hr_shape:
            out = F.interpolate(
                out, size=self.hr_shape, mode="bilinear", align_corners=False
            )

        return out
