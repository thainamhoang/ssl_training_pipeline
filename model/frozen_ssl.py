"""
frozen_ssl.py

FrozenSSLDownscaler — Pilot 2 baseline.

Frozen SSL encoder (DINOv2-L or DINOv3-SAT) + lightweight pixel-shuffle
SR decoder. Only feat_norm and decoder are trained.

Architecture:
    1. Resize LR input to SSL native size
    2. Extract patch tokens from frozen SSL encoder
    3. LayerNorm on patch tokens
    4. Reshape to 2D spatial feature map
    5. Pixel-shuffle upsample → single-channel HR prediction
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


class FrozenSSLDownscaler(nn.Module):
    def __init__(
        self,
        model_id: str,
        patch_size: int,
        lr_shape: tuple,
        hr_shape: tuple,
        upscale: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.mode = "frozen"
        self.patch_size = patch_size
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.upscale = upscale

        self.ssl_size, self.n_patches_h, self.n_patches_w, self.n_patches = (
            compute_ssl_size(lr_shape, patch_size)
        )

        print(f"[FrozenSSLDownscaler]")
        print(f"SSL input size  : {self.ssl_size[0]}×{self.ssl_size[1]}")
        print(
            f"Patch grid      : {self.n_patches_h}×{self.n_patches_w} = {self.n_patches} tokens"
        )

        # ── Encoder (frozen) ───────────────────────────────────────────────
        self.encoder = build_encoder(model_id, mode="frozen")

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

        n_trainable = sum(p.numel() for p in self.decoder.parameters()) + sum(
            p.numel() for p in self.feat_norm.parameters()
        )
        print(f"Trainable params: {n_trainable:,}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x_ssl = F.interpolate(
            x, size=self.ssl_size, mode="bilinear", align_corners=False
        )
        out = encoder_forward(self.encoder, "frozen", x_ssl.to(dtype=torch.bfloat16))
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
