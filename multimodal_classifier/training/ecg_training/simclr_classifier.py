import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from multimodal_classifier.data.ecg_data.extractload_data import prepare_dataset
from multimodal_classifier.data.ecg_data.data_loader import create_ecg_dataloaders
from multimodal_classifier.models.ecg_models.ecg_encoder import ECGEncoder
from multimodal_classifier.models.ecg_models.ECGClassifier import ECGClassifier
from configs.dataset.ecgdata_config import CFG, DEVICE, SUPERCLASSES


# ==========================================
# Classifier training step + evaluation
# ==========================================

def train_clf_epoch(model, loader, optimizer, criterion, device, log_every=50):
    """Train one epoch of the classifier (frozen encoder + trainable head)."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(loader)
    for signal, meta, labels in progress_bar:
        signal, meta, labels = signal.to(device), meta.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(signal, meta)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()
        num_batches += 1

        if num_batches % log_every == 0:
            progress_bar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

    return running_loss / num_batches


@torch.no_grad()
def evaluate_clf(model, loader, criterion, device, label_cols=SUPERCLASSES):
    """Evaluate classifier on a loader, return loss, macro AUC, per-label AUC."""
    model.eval()
    val_loss = 0.0
    num_batches = 0
    y_true, y_pred = [], []

    for signal, meta, labels in loader:
        signal, meta, labels = signal.to(device), meta.to(device), labels.to(device)
        outputs = model(signal, meta)
        loss = criterion(outputs, labels)

        val_loss += loss.detach().item()
        num_batches += 1

        probs = torch.sigmoid(outputs)
        y_pred.append(probs.detach().cpu().numpy())
        y_true.append(labels.detach().cpu().numpy())

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

    per_label_auc = {}
    for i, col in enumerate(label_cols):
        y_va = y_true[:, i]
        if len(np.unique(y_va)) < 2:
            continue  # AUC undefined with only one class present
        per_label_auc[col] = roc_auc_score(y_va, y_pred[:, i])

    macro_auc = float(np.mean(list(per_label_auc.values()))) if per_label_auc else float("nan")
    avg_val_loss = val_loss / max(num_batches, 1)

    return avg_val_loss, macro_auc, per_label_auc


# ==========================================
# Main classifier training function
# ==========================================

def run_classifier_training(
    epochs=None,
    batch_size=None,
    lr=None,
    encoder_path="checkpoints_simclr/encoder_best.pth",
    checkpoint_dir="checkpoints_classifier_simclr",
    device=DEVICE,
):
    """Train a classifier with a frozen SimCLR encoder, with early stopping
    on validation AUROC and best-checkpoint saving."""

    epochs = epochs or CFG["clf_epochs"]
    batch_size = batch_size or CFG["clf_batch_size"]
    lr = lr or CFG["clf_lr"]
    patience = CFG["clf_patience"]

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # ---- Data ----
    X, Y_clean = prepare_dataset()
    train_loader, valid_loader, test_loader = create_ecg_dataloaders(Y_clean, X, batch_size=batch_size)

    # ---- Load frozen encoder ----
    encoder = ECGEncoder()
    if os.path.exists(encoder_path):
        encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
        print(f"Loaded encoder from {encoder_path}")
    else:
        print(f"Warning: {encoder_path} not found. Using untrained encoder weights.")

    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # ---- Create classifier model ----
    model = ECGClassifier(encoder).to(device)

    # ---- Dynamic pos_weight ----
    # Compute from the training data instead of using a hardcoded snapshot
    from multimodal_classifier.data.ecg_data.split_data import split_data
    _, _, _, y_train, _, _ = split_data(Y_clean, X)

    pos_counts = y_train[SUPERCLASSES].sum(axis=0).values
    neg_counts = len(y_train) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.clip(pos_counts, 1, None), dtype=torch.float32)
    print(f"Dynamic pos_weight: {dict(zip(SUPERCLASSES, pos_weight.tolist()))}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    # ---- Training loop with early stopping ----
    best_val_auc = -1.0
    patience_counter = 0
    train_history = []
    val_history = []

    for epoch in range(epochs):
        train_loss = train_clf_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, per_label_auc = evaluate_clf(model, valid_loader, criterion, device)

        train_history.append(train_loss)
        val_history.append((val_loss, val_auc))

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "classifier_best.pth")
            print(f"      New best classifier saved (Macro AUC={best_val_auc:.4f})")
        else:
            patience_counter += 1
            print(f"      No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ---- Final evaluation with best checkpoint ----
    model.load_state_dict(torch.load(checkpoint_dir / "classifier_best.pth", map_location="cpu"))
    model = model.to(device)

    val_loss, val_auc, per_label_auc = evaluate_clf(model, valid_loader, criterion, device)
    print(f"\nFinal Val Loss  = {val_loss:.4f}")
    print(f"Final Macro AUC = {val_auc:.4f}")
    print("Per-label AUC:")
    for col, score in per_label_auc.items():
        print(f"  {col}: {score:.4f}")

    return model


if __name__ == '__main__':
    run_classifier_training()