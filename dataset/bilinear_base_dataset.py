from torch.utils.data import Dataset
import torch
from downscaling_dataset import DownscalingDataset


class BilinearBaselineView(Dataset):
    """View over DownscalingDataset that skips SSL/ImageNet preprocessing."""

    def __init__(self, base_ds: DownscalingDataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        lr_np, hr_np = self.base_ds._get_raw(idx)
        lr_raw = torch.from_numpy(lr_np).to(dtype=torch.float32)
        hr_raw = torch.from_numpy(hr_np).to(dtype=torch.float32)

        lr_norm = (lr_raw - self.base_ds.lr_mean) * self.base_ds.lr_inv_std
        hr_norm = (hr_raw - self.base_ds.hr_mean) * self.base_ds.hr_inv_std
        return lr_norm, hr_norm

    def worker_init_fn(self, worker_id: int):
        self.base_ds.worker_init_fn(worker_id)
