import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from configs.dataset.imagedata_config import CFG, DEFAULT_DATA_DIR, LABEL_COLS, SHARED_LABELS
from .data_augmentation import SimCLRTransform, eval_transform

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def clean_labels(df, label_cols=LABEL_COLS):
    """Clean CheXpert labels (missing -> 0, uncertain (-1) -> 0)."""
    labels = df[label_cols].copy()
    labels = labels.fillna(0)
    labels = labels.replace(-1, 0)
    return labels.astype(np.float32)


class CheXpertDataset(Dataset):
    """CheXpert X-ray image dataset for both unsupervised multi-view and supervised tasks."""

    def __init__(self, dataframe, data_dir=DEFAULT_DATA_DIR, transform=None, label_cols=None, return_labels=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_labels = return_labels
        self.label_cols = label_cols

        if self.return_labels and self.label_cols is not None:
            labels = self.dataframe[self.label_cols].fillna(0).replace(-1, 0)
            self.labels = labels.astype(np.float32).values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        path_str = self.dataframe.iloc[idx]["Path"]
        if isinstance(path_str, str) and path_str.startswith("CheXpert-v1.0-small/"):
            path_str = path_str.replace("CheXpert-v1.0-small/", "")
        image_path = self.data_dir / path_str

        try:
            image = Image.open(image_path).convert("L")
        except (OSError, UnidentifiedImageError):
            # Fallback to a random index if image file is missing or corrupted
            fallback_idx = np.random.randint(len(self.dataframe))
            return self.__getitem__(fallback_idx)

        if self.transform is not None:
            image = self.transform(image)

        if not self.return_labels:
            return image

        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, labels


def wrap_device_loader(loader, device=None):
    if HAS_XLA and device is not None and str(device).startswith("xla"):
        return pl.MpDeviceLoader(loader, device)
    return loader


def create_image_dataloaders(train_df, valid_df, data_dir=DEFAULT_DATA_DIR, batch_size=None, num_workers=None, device=None):
    if batch_size is None:
        batch_size = CFG["batch_size"]
    if num_workers is None:
        num_workers = CFG["num_workers"]
    prefetch = CFG["prefetch_factor"] if num_workers > 0 else None

    if "Frontal/Lateral" in train_df.columns:
        train_df = train_df[train_df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    if "Frontal/Lateral" in valid_df.columns:
        valid_df = valid_df[valid_df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

    simclr_dataset = CheXpertDataset(train_df, data_dir, transform=SimCLRTransform(), return_labels=False)
    simclr_loader = DataLoader(simclr_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               prefetch_factor=prefetch, persistent_workers=(num_workers > 0), drop_last=True)

    supcon_dataset = CheXpertDataset(train_df, data_dir, transform=SimCLRTransform(), label_cols=SHARED_LABELS, return_labels=True)
    supcon_loader = DataLoader(supcon_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               prefetch_factor=prefetch, persistent_workers=(num_workers > 0), drop_last=True)

    clf_train_ds = CheXpertDataset(train_df, data_dir, transform=eval_transform, label_cols=LABEL_COLS, return_labels=True)
    clf_valid_ds = CheXpertDataset(valid_df, data_dir, transform=eval_transform, label_cols=LABEL_COLS, return_labels=True)

    clf_train_loader = DataLoader(clf_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  prefetch_factor=prefetch, persistent_workers=(num_workers > 0), drop_last=True)
    clf_valid_loader = DataLoader(clf_valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  prefetch_factor=prefetch, persistent_workers=(num_workers > 0), drop_last=False)

    return (
        wrap_device_loader(simclr_loader, device),
        wrap_device_loader(supcon_loader, device),
        wrap_device_loader(clf_train_loader, device),
        wrap_device_loader(clf_valid_loader, device),
    )
