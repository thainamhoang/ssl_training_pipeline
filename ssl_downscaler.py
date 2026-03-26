"""
ssl_downscaler.py

Unified SSL downscaler supporting two encoder modes:
    mode="frozen"  — encoder fully frozen (Pilot 2)
    mode="lora"    — encoder with LoRA adapters on q_proj/v_proj (Pilot 3)

DINOv3-SAT specific notes:
    - Attention projections named q_proj, k_proj, v_proj (not query/key/value)
    - k_proj has bias=False and zero weights in the original checkpoint — skip it
    - Top-level transformer blocks at model.layer (not model.encoder.layer)
    - Uses RoPE positional encoding — generalizes to non-square inputs without modification

Usage:
    # Frozen (Pilot 2 — unchanged behaviour)
    model = SSLDownscaler(model_id="facebook/dinov3-vitl16-pretrain-sat493m",
                          patch_size=16, lr_shape=(32,64), hr_shape=(128,256),
                          mode="frozen")

    # LoRA r=8 (Pilot 3)
    model = SSLDownscaler(model_id="facebook/dinov3-vitl16-pretrain-sat493m",
                          patch_size=16, lr_shape=(32,64), hr_shape=(128,256),
                          mode="lora", lora_r=8)

    # LoRA r=16 (Pilot 3)
    model = SSLDownscaler(model_id="facebook/dinov3-vitl16-pretrain-sat493m",
                          patch_size=16, lr_shape=(32,64), hr_shape=(128,256),
                          mode="lora", lora_r=16)

Checkpoint compatibility:
    Frozen checkpoints saved from FrozenSSLDownscaler are NOT directly loadable
    here because the module names changed (encoder.* is now wrapped by PEFT).
    Load frozen checkpoints with the original FrozenSSLDownscaler class.
    LoRA checkpoints saved here load cleanly via torch.load + model.load_state_dict.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

# ---------------------------------------------------------------------------
# LoRA availability check — informative error if peft not installed
# ---------------------------------------------------------------------------
try:
    from peft import LoraConfig, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class SSLDownscaler(nn.Module):
    """
    SSL ViT-L encoder + pixel-shuffle SR decoder for climate downscaling.

    Encoder modes
    -------------
    frozen : all encoder parameters frozen, encoder runs under torch.no_grad().
             Identical behaviour to the original FrozenSSLDownscaler.

    lora   : LoRA adapters injected on q_proj and v_proj in every attention
             block. All other encoder parameters remain frozen. Encoder runs
             with grad enabled for LoRA parameters only.

    Decoder
    -------
    Always trained from scratch. Architecture unchanged from Pilot 2:
        Conv2d(enc_dim → hidden*up²) → PixelShuffle(up) → GELU →
        Conv2d(hidden → hidden, 3×3) → GELU → Conv2d(hidden → 1, 3×3)

    Parameters
    ----------
    model_id   : HuggingFace model identifier
    patch_size : ViT patch size (16 for DINOv3-SAT, 14 for DINOv2)
    lr_shape   : (H_lr, W_lr) of the LR input grid, e.g. (32, 64)
    hr_shape   : (H_hr, W_hr) of the HR target grid, e.g. (128, 256)
    upscale    : SR upscaling factor (2 or 4)
    hidden_dim : decoder hidden channel count
    mode       : "frozen" | "lora"
    lora_r     : LoRA rank (ignored when mode="frozen"). Recommended: 8 or 16
    lora_alpha : LoRA scaling factor. Defaults to lora_r (standard convention)
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

        assert mode in ("frozen", "lora"), (
            f"mode must be 'frozen' or 'lora', got {mode!r}"
        )
        if mode == "lora" and not _PEFT_AVAILABLE:
            raise ImportError("pip install peft>=0.10.0 to use mode='lora'")

        self.mode = mode
        self.patch_size = patch_size
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.upscale = upscale
        self.lora_r = lora_r

        # ── SSL input size ─────────────────────────────────────────────────
        H_lr, W_lr = lr_shape
        min_patches = 16
        ssl_H = min_patches * patch_size
        ssl_W = ssl_H * W_lr // H_lr
        ssl_W = ((ssl_W + patch_size - 1) // patch_size) * patch_size

        self.ssl_size = (ssl_H, ssl_W)
        self.n_patches_h = ssl_H // patch_size
        self.n_patches_w = ssl_W // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w

        print(
            f"[SSLDownscaler] mode={mode}"
            + (f", lora_r={lora_r}" if mode == "lora" else "")
        )
        print(f"SSL input size  : {ssl_H}×{ssl_W}")
        print(
            f"Patch grid      : {self.n_patches_h}×{self.n_patches_w}"
            f" = {self.n_patches} tokens"
        )

        # ── Load base encoder ──────────────────────────────────────────────
        encoder = AutoModel.from_pretrained(model_id)

        if mode == "frozen":
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
            self.encoder = encoder

        else:  # lora
            # First freeze everything — LoRA will selectively unfreeze
            for param in encoder.parameters():
                param.requires_grad = False

            _alpha = lora_alpha if lora_alpha is not None else lora_r
            _is_dinov2 = hasattr(
                encoder, "encoder"
            )  # Dinov2Model has .encoder; DINOv3 doesn't
            _target_modules = ["query", "value"] if _is_dinov2 else ["q_proj", "v_proj"]

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=_alpha,
                target_modules=_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
            )

            self.encoder = get_peft_model(encoder, lora_cfg)
            self.encoder.gradient_checkpointing_enable()
            # get_peft_model sets encoder to train() — keep it that way
            # so BatchNorm/Dropout behave correctly if present
            self.encoder.train()

        # ── Feature normalizer (always trained) ───────────────────────────
        enc_dim = 1024
        self.feat_norm = nn.LayerNorm(enc_dim)

        # ── SR Decoder (always trained) ────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Conv2d(enc_dim, hidden_dim * upscale * upscale, kernel_size=1),
            nn.PixelShuffle(upscale),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        self._init_decoder()

        # ── Parameter count summary ────────────────────────────────────────
        n_decoder = sum(p.numel() for p in self.decoder.parameters()) + sum(
            p.numel() for p in self.feat_norm.parameters()
        )
        if mode == "lora":
            n_lora = sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad
            )
            print(
                f"Trainable params: {n_decoder + n_lora:,}"
                f"  (LoRA: {n_lora:,}  decoder+norm: {n_decoder:,})"
            )
        else:
            print(f"Trainable params: {n_decoder:,}  (decoder+norm only)")

    # ── Decoder initialisation ─────────────────────────────────────────────

    def _init_decoder(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init final conv → predictions start near zero, RMSE ≈ 1.0 at epoch 0
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    # ── Optimiser ─────────────────────────────────────────────────────────

    def make_optimizer(
        self, lr: float = 5e-5, decoder_lr: float = 2e-4, weight_decay: float = 1e-2
    ):
        if self.mode == "frozen":
            # Single group — no LoRA params, decoder LR only
            return torch.optim.AdamW(
                list(self.decoder.parameters()) + list(self.feat_norm.parameters()),
                lr=decoder_lr,
                weight_decay=1e-4,  # lighter decay for decoder
            )

        # LoRA mode — two separate groups
        lora_params, decoder_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(param)
            else:
                decoder_params.append(param)  # decoder + feat_norm

        return torch.optim.AdamW(
            [
                {"params": lora_params, "lr": lr, "weight_decay": weight_decay},
                {"params": decoder_params, "lr": decoder_lr, "weight_decay": 1e-4},
            ],
            betas=(0.9, 0.999),
        )

    # ── Encode ────────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize input → SSL encoder → extract patch tokens.

        Args:
            x : [B, 3, H_lr, W_lr] ImageNet-normalised float32

        Returns:
            patch_tokens : [B, n_patches, enc_dim] float32
        """
        x_ssl = F.interpolate(
            x, size=self.ssl_size, mode="bilinear", align_corners=False
        )

        if self.mode == "frozen":
            with torch.no_grad():
                out = self.encoder(pixel_values=x_ssl.to(dtype=torch.bfloat16))
        else:
            # LoRA mode: grad flows through LoRA parameters
            out = self.encoder(pixel_values=x_ssl.to(dtype=torch.bfloat16))

        # Token layout: [CLS | patch_0 … patch_N | reg_0 … reg_3]
        patch_tokens = out.last_hidden_state[
            :, 1 : 1 + self.n_patches, :
        ].float()  # [B, n_patches, enc_dim]

        return patch_tokens

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H_lr, W_lr] ImageNet-normalised float32

        Returns:
            [B, 1, H_hr, W_hr] predicted z-score normalised HR field
        """
        patch_tokens = self._encode(x)  # [B, N, D]
        patch_tokens = self.feat_norm(patch_tokens)  # [B, N, D]

        B, N, D = patch_tokens.shape
        feat = patch_tokens.reshape(B, self.n_patches_h, self.n_patches_w, D)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, D, H_p, W_p]

        out = self.decoder(feat)  # [B, 1, H_p*up, W_p*up]

        if out.shape[-2:] != self.hr_shape:
            out = F.interpolate(
                out, size=self.hr_shape, mode="bilinear", align_corners=False
            )

        return out  # [B, 1, H_hr, W_hr]

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
        """Save full training state. Compatible with SLURM multi-step resume."""
        torch.save(
            {
                "epoch": epoch,
                "mode": self.mode,
                "lora_r": self.lora_r,
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
        Resume from checkpoint saved by save_checkpoint.

        Returns (model, checkpoint_dict) — caller unpacks epoch, wandb_run_id etc.

        Example:
            model, ckpt = SSLDownscaler.load_checkpoint(
                "latest.pt",
                model_kwargs=dict(model_id=..., patch_size=16,
                                  lr_shape=(32,64), hr_shape=(128,256),
                                  mode="lora", lora_r=16),
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            )
            start_epoch    = ckpt["epoch"] + 1
            best_val_rmse  = ckpt["best_val_rmse"]
            wandb_run_id   = ckpt["wandb_run_id"]
        """
        ckpt = torch.load(path, map_location="cpu")

        # Validate mode/rank consistency
        assert ckpt["mode"] == model_kwargs.get("mode", "frozen"), (
            f"Checkpoint mode={ckpt['mode']} ≠ requested mode={model_kwargs.get('mode')}"
        )
        assert ckpt["lora_r"] == model_kwargs.get("lora_r", 16), (
            f"Checkpoint lora_r={ckpt['lora_r']} ≠ requested lora_r={model_kwargs.get('lora_r')}"
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
