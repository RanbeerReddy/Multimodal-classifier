import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from multimodal_classifier.data.ecg_data.data_augmentation import ECGSSLData
from multimodal_classifier.models.ecg_models.ecg_encoder import ECGEncoder
from multimodal_classifier.models.ecg_models.ECGSimCLR import ECGSimCLR
from multimodal_classifier.losses.nt_xent_loss import nt_xent_loss
from multimodal_classifier.data.ecg_data.extractload_data import prepare_dataset
from multimodal_classifier.data.ecg_data.transform_data import preprocess_ecg_batch
from multimodal_classifier.data.ecg_data.split_data import split_data
from configs.dataset.ecgdata_config import DEVICE

def run_simclr_training(epochs=10, batch_size=128, lr=1e-3):
    X, Y_clean = prepare_dataset()
    X = preprocess_ecg_batch(X)
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
