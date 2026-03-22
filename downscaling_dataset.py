import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class DownscalingDataset(Dataset):
    """
    Paired (LR, HR) ERA5 downscaling dataset.

    Two loading modes controlled by `preload`:
        preload=True  — loads all shards into RAM at init (fast, needs high RAM)
        preload=False — lazy per-worker shard cache (low RAM, multiprocessing-safe)

    Args:
        lr_dir    : path to LR resolution directory (contains normalize_*.npz + partition/)
        hr_dir    : path to HR resolution directory
        partition : "train" | "val" | "test"
        stride    : sample every Nth timestep (1=all, 6=6-hourly, 24=daily)
        preload   : True for high-RAM machines, False for low-RAM machines
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        partition: str,
        stride: int = 6,
        preload: bool = False,
    ):
        self.stride = stride
        self.preload = preload
        self.partition = partition

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
            f"(stride={stride}, preload={preload})"
        )

        # ── Preload mode: load everything into RAM now ─────────────────────
        if preload:
            print(f"[{partition}] Preloading all shards into RAM...")
            lr_list, hr_list = [], []
            for lf, hf in zip(lr_shards, hr_shards):
                lr_list.append(np.load(lf)["2m_temperature"])
                hr_list.append(np.load(hf)["2m_temperature"])
            # Shape: [T_total, 1, H, W]
            self._lr_data = np.concatenate(lr_list, axis=0)
            self._hr_data = np.concatenate(hr_list, axis=0)
            print(
                f"[{partition}] Preloaded: "
                f"LR {self._lr_data.nbytes / 1e9:.2f} GB, "
                f"HR {self._hr_data.nbytes / 1e9:.2f} GB"
            )
        else:
            # ── Lazy mode: per-worker shard cache ─────────────────────────
            self._lr_data = None
            self._hr_data = None
            self._cache_lr = None
            self._cache_hr = None
            self._cache_shard_idx = -1

    def __len__(self):
        return len(self.indices)

    def _get_raw(self, idx: int):
        """Get raw numpy arrays for LR and HR at global index idx."""
        real_idx = self.indices[idx]
        shard_idx = real_idx // self.T_per_shard
        t_idx = real_idx % self.T_per_shard

        if self.preload:
            # Preload mode: index directly into RAM array
            return (
                self._lr_data[real_idx],  # [1, H_lr, W_lr]
                self._hr_data[real_idx],  # [1, H_hr, W_hr]
            )
        else:
            # Lazy mode: load shard if not cached
            if shard_idx != self._cache_shard_idx:
                self._cache_lr = np.load(self.lr_shards[shard_idx])["2m_temperature"]
                self._cache_hr = np.load(self.hr_shards[shard_idx])["2m_temperature"]
                self._cache_shard_idx = shard_idx
            return (
                self._cache_lr[t_idx],
                self._cache_hr[t_idx],
            )

    def __getitem__(self, idx: int):
        lr_np, hr_np = self._get_raw(idx)

        lr_raw = torch.tensor(lr_np, dtype=torch.float32)  # [1, H_lr, W_lr]
        hr_raw = torch.tensor(hr_np, dtype=torch.float32)  # [1, H_hr, W_hr]

        # ── Z-score normalize ─────────────────────────────────────────────
        lr_norm = (lr_raw - self.lr_mean) / (self.lr_std + 1e-8)
        hr_norm = (hr_raw - self.hr_mean) / (self.hr_std + 1e-8)

        # ── SSL encoder input: raw Kelvin → [0,1] → ImageNet ──────────────
        lr_z = (lr_raw - self.lr_mean) / (self.lr_std + 1e-8)  # z-score (deterministic)
        lr_01 = torch.clamp(lr_z / 6.0 + 0.5, 0.0, 1.0)  # [-3σ,+3σ] → [0,1]
        lr_imagenet = lr_01.repeat(3, 1, 1)
        lr_imagenet = (lr_imagenet - self.IMAGENET_MEAN) / self.IMAGENET_STD

        return lr_imagenet, lr_norm, hr_norm

    def worker_init_fn(self, worker_id: int):
        """
        Call this as worker_init_fn in DataLoader.
        Resets lazy cache per worker so workers don't share state.
        Only needed when preload=False and num_workers > 0.
        """
        if not self.preload:
            self._cache_lr = None
            self._cache_hr = None
            self._cache_shard_idx = -1
