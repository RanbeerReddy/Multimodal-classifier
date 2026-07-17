import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from multimodal_classifier.data.ecg_data.extractload_data import prepare_dataset
from multimodal_classifier.data.ecg_data.transform_data import preprocess_ecg_batch
from multimodal_classifier.data.ecg_data.data_loader import create_ecg_dataloaders
from multimodal_classifier.models.ecg_models.ecg_encoder import ECGEncoder
from multimodal_classifier.models.ecg_models.ECGClassifier import ECGClassifier
from configs.dataset.ecgdata_config import DEVICE

def run_classifier_training(epochs=10, batch_size=128, lr=1e-3, encoder_path="ecg_encoder_ssl.pth"):
    # 1. Load and preprocess data
    X, Y_clean = prepare_dataset()
    X = preprocess_ecg_batch(X)
    train_loader, valid_loader, test_loader = create_ecg_dataloaders(Y_clean, X, batch_size=batch_size)

    # 2. Load frozen encoder
    encoder = ECGEncoder()
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    else:
        print(f"Warning: {encoder_path} not found. Using untrained encoder weights.")
    
    encoder = encoder.to(DEVICE)
    for p in encoder.parameters():
        p.requires_grad = False

    # 3. Create classifier model
    model = ECGClassifier(encoder).to(DEVICE)
    
    # 4. Optimizer and Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for signal, meta, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            signal = signal.to(DEVICE)
            meta = meta.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(signal, meta)  # (B, 5)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}")

        # Validation Loop
        model.eval()
        val_loss = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for signal, meta, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                signal = signal.to(DEVICE)
                meta = meta.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(signal, meta)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                y_pred.append(probs)
                y_true.append(labels.cpu().numpy())

        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}/{epochs}: Val Loss = {val_loss:.4f}")

        # Compute AUC
        y_pred = np.vstack(y_pred)
        y_true = np.vstack(y_true)
        try:
            auc = roc_auc_score(y_true, y_pred, average='macro')
            print(f"Epoch {epoch+1}/{epochs}: Validation AUC = {auc:.4f}")
        except ValueError as e:
            print(f"Epoch {epoch+1}/{epochs}: Could not compute AUC ({e})")

    return model

if __name__ == '__main__':
    run_classifier_training()