import os
import sys

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..', '..'))
DATA_DIR = os.path.join(REPO_ROOT, 'Multimodal classifier', 'data', 'Ecg data')
MODELS_DIR = os.path.join(REPO_ROOT, 'Multimodal classifier', 'models', 'ecg models')
LOSSES_DIR = os.path.join(REPO_ROOT, 'Multimodal classifier', 'losses')

for path in (REPO_ROOT, DATA_DIR, MODELS_DIR, LOSSES_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_augmentation import ECGSSLData
from ecg_encoder import ECGEncoder
from ECGSimCLR import ECGSimCLR
from nt_xent_loss import nt_xent_loss
from extractload_data import prepare_dataset
from split_data import split_data
from configs.dataset.ecgdata_config import DEVICE


def run_simclr_training(epochs=10, batch_size=128, lr=1e-3):
    X, Y_clean = prepare_dataset()
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(Y_clean, X)

    ssl_dataset = ECGSSLData(x_train)
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    model = ECGSimCLR(ECGEncoder()).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x1, x2 in tqdm(ssl_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(ssl_loader)
        print(f'Epoch {epoch + 1}/{epochs} avg loss: {avg_loss:.4f}')

    return model


if __name__ == '__main__':
    run_simclr_training()
