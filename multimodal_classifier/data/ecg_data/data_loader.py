import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from .split_data import split_data
from configs.dataset.ecgdata_config import SUPERCLASSES, META_FEATURES, BATCH_SIZE


class ECGDataset(Dataset):
    """Multimodal dataset: ECG signal (12 leads) + demographics."""

    def __init__(self, X_signals, Y_meta_df, superclasses=SUPERCLASSES, meta_features=META_FEATURES):
        # Signal: (N, 1000, 12) → transposed to (N, 12, 1000) for Conv1D
        self.signals = X_signals.transpose(0, 2, 1).astype(np.float32)
        self.meta = Y_meta_df[meta_features].values.astype(np.float32)
        self.labels = Y_meta_df[superclasses].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx])
        meta = torch.from_numpy(self.meta[idx])
        label = torch.from_numpy(self.labels[idx])
        return signal, meta, label


def create_ecg_dataloaders(Y_clean, X, batch_size=BATCH_SIZE):
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(Y_clean, X)

    train_ds = ECGDataset(x_train, y_train)
    valid_ds = ECGDataset(x_valid, y_valid)
    test_ds = ECGDataset(x_test, y_test)

    train_loader = TorchDataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    valid_loader = TorchDataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )
    test_loader = TorchDataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    return train_loader, valid_loader, test_loader
