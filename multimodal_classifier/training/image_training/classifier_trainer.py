import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

from configs.dataset.imagedata_config import LABEL_COLS
from .simclr_trainer import print_log, save_checkpoint, step_optimizer

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def train_clf_epoch(model, loader, optimizer, criterion, device, log_every=50):
    model.train()
    running_loss = torch.zeros((), device=device)
    num_batches = 0

    progress_bar = tqdm(loader, desc="Classifier Train")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        autocast_kwargs = {"device_type": "xla", "dtype": torch.bfloat16} if HAS_XLA else {"device_type": str(device).split(":")[0], "enabled": False}
        with torch.autocast(**autocast_kwargs):
            outputs = model(images)
            loss = criterion(outputs.float(), labels)

        loss.backward()
        step_optimizer(optimizer)

        running_loss += loss.detach()
        num_batches += 1

        if num_batches % log_every == 0:
            progress_bar.set_postfix(loss=f"{(running_loss / num_batches).item():.4f}")

    return (running_loss / max(num_batches, 1)).item()


@torch.no_grad()
def evaluate_clf(model, loader, criterion, device, label_cols=LABEL_COLS):
    model.eval()
    val_loss = torch.zeros((), device=device)
    num_batches = 0
    y_true, y_pred = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs.float(), labels)

        val_loss += loss.detach()
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
            continue
        per_label_auc[col] = roc_auc_score(y_va, y_pred[:, i])

    macro_auc = float(np.mean(list(per_label_auc.values()))) if per_label_auc else float("nan")
    avg_val_loss = (val_loss / max(num_batches, 1)).item()
    return avg_val_loss, macro_auc, per_label_auc


def run_classifier_training(model, train_loader, valid_loader, device=None, epochs=20, pos_weight=None, checkpoint_dir="checkpoints_classifier"):
    if device is None:
        device = xm.xla_device() if HAS_XLA else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if pos_weight is None:
        pos_weight = torch.ones(len(LABEL_COLS), dtype=torch.float32, device=device)
    else:
        pos_weight = pos_weight.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    best_val_auc = -1.0
    patience_counter, patience_limit = 0, 3
    history = []

    for epoch in range(epochs):
        train_loss = train_clf_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, per_label_auc = evaluate_clf(model, valid_loader, criterion, device)
        history.append((train_loss, val_loss, val_auc))

        print_log(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_checkpoint(model.state_dict(), checkpoint_dir / "classifier_best.pth")
            print_log(f"      New best classifier saved (Macro AUC={best_val_auc:.4f})")
        else:
            patience_counter += 1
            print_log(f"      No improvement ({patience_counter}/{patience_limit})")
            if patience_counter >= patience_limit:
                print_log(f"Early stopping at epoch {epoch + 1}")
                break

    return model, history
