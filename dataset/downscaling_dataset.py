import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class DownscalingDataset(Dataset):
    """
    Paired (LR, HR) ERA5 downscaling dataset.

    Loading modes (controlled per resolution):
        lr_preload=True  — load all LR shards into RAM at init (recommended,
                           LR is only ~5GB total across all splits)
        hr_preload=True  — load all HR shards into RAM at init (only safe for
                           val/test splits; HR train is ~38GB)
        hr_preload=False — lazy per-worker shard cache for HR (safe for train)

    Recommended usage:
        train : lr_preload=True,  hr_preload=False  (~5GB RAM, HR read lazily)
        val   : lr_preload=True,  hr_preload=True   (~5GB LR + ~5GB HR val)
        test  : lr_preload=True,  hr_preload=True   (~5GB LR + ~5GB HR test)

    Args:
        lr_dir      : path to LR resolution directory (normalize_*.npz + partition/)
        hr_dir      : path to HR resolution directory
        partition   : "train" | "val" | "test"
        stride      : sample every Nth timestep (1=all, 6=6-hourly, 24=daily)
        lr_preload  : preload LR into RAM
        hr_preload  : preload HR into RAM
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    EPS = 1e-8

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        partition: str,
        stride: int = 6,
        lr_preload: bool = False,
        hr_preload: bool = False,
    ):
        self.stride = stride
        self.partition = partition

        # Resolve per-resolution preload flags
        # Explicit lr_preload / hr_preload take priority over legacy preload
        self.lr_preload = lr_preload if lr_preload is not None else False
        self.hr_preload = hr_preload if hr_preload is not None else False

        # ── Normalization stats ───────────────────────────────────────────
        lr_mean_f = np.load(os.path.join(lr_dir, "normalize_mean.npz"))
        lr_std_f = np.load(os.path.join(lr_dir, "normalize_std.npz"))
        hr_mean_f = np.load(os.path.join(hr_dir, "normalize_mean.npz"))
        hr_std_f = np.load(os.path.join(hr_dir, "normalize_std.npz"))

        self.lr_mean = torch.tensor(
            lr_mean_f["2m_temperature"], dtype=torch.float32
        ).view(1, 1, 1)
        self.lr_std = torch.tensor(
            lr_std_f["2m_temperature"], dtype=torch.float32
        ).view(1, 1, 1)
        self.hr_mean = torch.tensor(
            hr_mean_f["2m_temperature"], dtype=torch.float32
        ).view(1, 1, 1)
        self.hr_std = torch.tensor(
            hr_std_f["2m_temperature"], dtype=torch.float32
        ).view(1, 1, 1)
        self.lr_inv_std = 1.0 / (self.lr_std + self.EPS)
        self.hr_inv_std = 1.0 / (self.hr_std + self.EPS)

        # ── Shard paths ───────────────────────────────────────────────────
        lr_shards = sorted(glob.glob(os.path.join(lr_dir, partition, "*.npz")))
        hr_shards = sorted(glob.glob(os.path.join(hr_dir, partition, "*.npz")))
        lr_shards = [f for f in lr_shards if "climatology" not in f]
        hr_shards = [f for f in hr_shards if "climatology" not in f]

        assert len(lr_shards) == len(hr_shards), (
            f"Shard mismatch: LR={len(lr_shards)}, HR={len(hr_shards)}"
        )
        assert len(lr_shards) > 0, f"No shards in {lr_dir}/{partition}/"

        self.lr_shards = lr_shards
        self.hr_shards = hr_shards

        # Peek at first shard for shape info
        first_lr = np.load(lr_shards[0])["2m_temperature"]
        first_hr = np.load(hr_shards[0])["2m_temperature"]
        self.lr_shape = first_lr.shape[2:]  # [T, 1, H_lr, W_lr] → (H_lr, W_lr)
        self.hr_shape = first_hr.shape[2:]
        self.T_per_shard = first_lr.shape[0]
        total = self.T_per_shard * len(lr_shards)
        self.indices = list(range(0, total, stride))

        print(f"[{partition}] LR shape: {self.lr_shape}, HR shape: {self.hr_shape}")
        print(
            f"[{partition}] {len(lr_shards)} shards × "
            f"{self.T_per_shard} timesteps = {total} total"
        )
        print(
            f"[{partition}] {len(self.indices)} samples "
            f"(stride={stride}, lr_preload={self.lr_preload}, "
            f"hr_preload={self.hr_preload})"
        )

        # ── LR preload ────────────────────────────────────────────────────
        if self.lr_preload:
            print(f"[{partition}] Preloading LR shards into RAM...")
            lr_list = [np.load(f)["2m_temperature"] for f in lr_shards]
            self._lr_data = np.concatenate(lr_list, axis=0)  # [T_total, 1, H_lr, W_lr]
            print(f"[{partition}] LR preloaded: {self._lr_data.nbytes / 1e9:.2f} GB")
        else:
            self._lr_data = None
            self._cache_lr = None

        # ── HR preload ────────────────────────────────────────────────────
        if self.hr_preload:
            print(f"[{partition}] Preloading HR shards into RAM...")
            hr_list = [np.load(f)["2m_temperature"] for f in hr_shards]
            self._hr_data = np.concatenate(hr_list, axis=0)  # [T_total, 1, H_hr, W_hr]
            print(f"[{partition}] HR preloaded: {self._hr_data.nbytes / 1e9:.2f} GB")
        else:
            self._hr_data = None
            self._cache_hr = None
            self._cache_shard_idx = -1

    # ── Length ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.indices)

    # ── Raw array fetch ───────────────────────────────────────────────────

    def _get_raw(self, idx: int):
        """
        Return (lr_np, hr_np) numpy arrays for dataset index idx.

        Handles four combinations of lr_preload × hr_preload:
            (T, T) — both in RAM, direct index
            (T, F) — LR from RAM, HR from shard cache
            (F, T) — LR from shard cache, HR from RAM
            (F, F) — both from shard cache (original lazy behaviour)
        """
        real_idx = self.indices[idx]
        shard_idx = real_idx // self.T_per_shard
        t_idx = real_idx % self.T_per_shard

        # ── LR ────────────────────────────────────────────────────────────
        if self.lr_preload:
            lr = self._lr_data[real_idx]
        else:
            # Load shard if not cached — shard_idx tracked separately for LR
            if not hasattr(self, "_cache_lr_shard_idx"):
                self._cache_lr_shard_idx = -1
            if shard_idx != self._cache_lr_shard_idx:
                self._cache_lr = np.load(self.lr_shards[shard_idx])["2m_temperature"]
                self._cache_lr_shard_idx = shard_idx
            lr = self._cache_lr[t_idx]

        # ── HR ────────────────────────────────────────────────────────────
        if self.hr_preload:
            hr = self._hr_data[real_idx]
        else:
            if shard_idx != self._cache_shard_idx:
                self._cache_hr = np.load(self.hr_shards[shard_idx])["2m_temperature"]
                self._cache_shard_idx = shard_idx
            hr = self._cache_hr[t_idx]

        return lr, hr

    # ── Dataset item ──────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        lr_np, hr_np = self._get_raw(idx)

        lr_raw = torch.from_numpy(lr_np).to(dtype=torch.float32)  # [1, H_lr, W_lr]
        hr_raw = torch.from_numpy(hr_np).to(dtype=torch.float32)  # [1, H_hr, W_hr]

        # ── Z-score normalize ─────────────────────────────────────────────
        lr_norm = (lr_raw - self.lr_mean) * self.lr_inv_std
        hr_norm = (hr_raw - self.hr_mean) * self.hr_inv_std

        # ── SSL encoder input: z-score → [0,1] → ImageNet ─────────────────
        lr_01 = torch.clamp(lr_norm / 6.0 + 0.5, 0.0, 1.0)  # [-3σ,+3σ] → [0,1]
        lr_imagenet = lr_01.expand(3, -1, -1)
        lr_imagenet = (lr_imagenet - self.IMAGENET_MEAN) / self.IMAGENET_STD

        return lr_imagenet, lr_norm, hr_norm

    # ── Worker init ───────────────────────────────────────────────────────

    def worker_init_fn(self, worker_id: int):
        """
        Reset per-worker shard caches.
        Call as worker_init_fn in DataLoader when num_workers > 0.
        Only has effect for lazy (non-preloaded) resolutions.
        """
        if not self.lr_preload:
            self._cache_lr = None
            self._cache_lr_shard_idx = -1
        if not self.hr_preload:
            self._cache_hr = None
            self._cache_shard_idx = -1
