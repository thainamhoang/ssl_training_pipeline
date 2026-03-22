import torch 
from torch import nn 
from torch.nn import functional as F 
from transformers import AutoModel

class FrozenSSLDownscaler(nn.Module):
    """
    Frozen SSL encoder (DINOv2-L or DINOv3-SAT) + lightweight SR decoder.

    Architecture:
        1. Resize LR input to SSL native size (preserving aspect ratio)
        2. Extract patch tokens from frozen SSL encoder
        3. LayerNorm on patch tokens
        4. Reshape tokens to 2D spatial feature map
        5. Pixel-shuffle upsample to HR resolution
        6. 1×1 + 3×3 convs to predict single-channel output (T2m)

    Only feat_norm and decoder are trained. Encoder is completely frozen.
    """

    def __init__(
        self,
        model_id:   str,
        patch_size: int,
        lr_shape:   tuple,    # (H_lr, W_lr) e.g. (32, 64)
        hr_shape:   tuple,    # (H_hr, W_hr) e.g. (128, 256)
        upscale:    int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.lr_shape   = lr_shape
        self.hr_shape   = hr_shape
        self.upscale    = upscale

        # ── Compute SSL input size ─────────────────────────────────────────
        # Target: 16 patches minimum per side, preserve LR aspect ratio,
        # both dims divisible by patch_size.
        H_lr, W_lr = lr_shape   # e.g. 32, 64

        min_patches = 16
        ssl_H = min_patches * patch_size          # 224 for patch/14, 256 for patch/16
        ssl_W = ssl_H * W_lr // H_lr              # preserve 1:2 ratio → 448 or 512
        ssl_W = ((ssl_W + patch_size - 1) // patch_size) * patch_size  # round up

        self.ssl_size = (ssl_H, ssl_W)

        self.n_patches_h = ssl_H // patch_size    # 16
        self.n_patches_w = ssl_W // patch_size    # 32
        self.n_patches   = self.n_patches_h * self.n_patches_w  # 512

        print(f"SSL input size  : {ssl_H}×{ssl_W}")
        print(f"Patch grid      : {self.n_patches_h}×{self.n_patches_w} = {self.n_patches} tokens")

        # ── Frozen SSL Encoder ─────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(model_id)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        enc_dim = 1024   # DINOv2-L and DINOv3-SAT-L both output 1024

        # ── Feature normalizer (trained) ───────────────────────────────────
        # Normalizes encoder patch tokens before the decoder.
        # Prevents exploding decoder inputs from high-norm SSL features.
        self.feat_norm = nn.LayerNorm(enc_dim)

        # ── SR Decoder (trained) ───────────────────────────────────────────
        # Input:  [B, enc_dim, n_patches_h, n_patches_w]  e.g. [B, 1024, 16, 32]
        # Output: [B, 1, H_hr, W_hr]                      e.g. [B, 1, 128, 256]
        #
        # Step 1: 1×1 conv projects enc_dim → hidden_dim * upscale²
        # Step 2: PixelShuffle(upscale) → [B, hidden_dim, H_patch*up, W_patch*up]
        #         e.g. [B, 256, 64, 128] for upscale=4
        # Step 3: 3×3 conv refines spatial features
        # Step 4: 3×3 conv outputs single channel
        # Step 5: bilinear resize to exact hr_shape if needed
        self.decoder = nn.Sequential(
            nn.Conv2d(enc_dim, hidden_dim * upscale * upscale, kernel_size=1),
            nn.PixelShuffle(upscale),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

        self._init_decoder()

        n_trainable = (
            sum(p.numel() for p in self.decoder.parameters()) +
            sum(p.numel() for p in self.feat_norm.parameters())
        )
        print(f"Trainable params: {n_trainable:,}")

    def _init_decoder(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init final conv so predictions start near zero at epoch 0.
        # This means initial RMSE ≈ hr_norm std ≈ 1.0 rather than exploding.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def make_optimizer(self, lr: float = 2e-4, weight_decay: float = 1e-4):
        """Returns optimizer over trainable parameters only (decoder + feat_norm)."""
        return torch.optim.AdamW(
            list(self.decoder.parameters()) + list(self.feat_norm.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    @torch.no_grad()
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and normalize patch tokens from frozen SSL encoder.

        Args:
            x: [B, 3, H_lr, W_lr] ImageNet-normalized float32

        Returns:
            feat: [B, enc_dim, n_patches_h, n_patches_w] float32
        """
        # Resize to SSL native size
        x_ssl = F.interpolate(
            x, size=self.ssl_size, mode="bilinear", align_corners=False
        )

        # Forward through frozen encoder in bfloat16
        out = self.encoder(pixel_values=x_ssl.to(dtype=torch.bfloat16))

        # Slice image patch tokens — layout: [CLS | patches | registers]
        patch_tokens = out.last_hidden_state[
            :, 1 : 1 + self.n_patches, :
        ].float()    # [B, n_patches, enc_dim]

        return patch_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H_lr, W_lr] ImageNet-normalized float32

        Returns:
            [B, 1, H_hr, W_hr] predicted z-score normalized HR field
        """
        # Encode (no grad — encoder is frozen)
        patch_tokens = self._encode(x)          # [B, n_patches, enc_dim]

        # Normalize encoder features (trained LayerNorm)
        patch_tokens = self.feat_norm(patch_tokens)   # [B, n_patches, enc_dim]

        # Reshape to 2D spatial feature map
        B, N, D = patch_tokens.shape
        feat = patch_tokens.reshape(B, self.n_patches_h, self.n_patches_w, D)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, enc_dim, H_p, W_p]

        # Decode to HR field
        out = self.decoder(feat)               # [B, 1, H_p*up, W_p*up]

        # Resize to exact HR shape if pixel-shuffle output size differs
        if out.shape[-2:] != self.hr_shape:
            out = F.interpolate(
                out, size=self.hr_shape, mode="bilinear", align_corners=False
            )

        return out   # [B, 1, H_hr, W_hr]