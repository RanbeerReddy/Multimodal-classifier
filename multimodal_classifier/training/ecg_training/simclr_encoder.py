import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from multimodal_classifier.data.ecg_data.data_augmentation import ECGSSLData
from multimodal_classifier.data.ecg_data.data_loader import ECGDataset
from multimodal_classifier.models.ecg_models.ecg_encoder import ECGEncoder
from multimodal_classifier.models.ecg_models.ECGSimCLR import ECGSimCLR
from multimodal_classifier.losses.nt_xent_loss import NTXentLoss
from multimodal_classifier.data.ecg_data.extractload_data import prepare_dataset
from multimodal_classifier.data.ecg_data.split_data import split_data
from configs.dataset.ecgdata_config import CFG, DEVICE, SUPERCLASSES


# ==========================================
# Training step
# ==========================================

def train_one_epoch(model, loader, optimizer, criterion, device, log_every=50):
    """Train one epoch of SimCLR pretraining."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(loader)
    for x1, x2 in progress_bar:
        x1, x2 = x1.to(device), x2.to(device)
        optimizer.zero_grad(set_to_none=True)

        _, z1 = model(x1)
        _, z2 = model(x2)
        loss = criterion(z1, z2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.detach().item()
        num_batches += 1

        if num_batches % log_every == 0:
            progress_bar.set_postfix(loss=f"{running_loss / num_batches:.4f}")

    return running_loss / num_batches


# ==========================================
# Linear-probe AUROC harness
# ==========================================

@torch.no_grad()
def extract_features(encoder, loader, device):
    """Extract frozen encoder features for linear probe evaluation."""
    encoder.eval()
    feats, labels = [], []
    for signal, meta, y in loader:
        signal = signal.to(device)
        f = encoder(signal)
        feats.append(f.detach().cpu())
        labels.append(y.detach().cpu())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def linear_probe_auroc(encoder, probe_train_loader, probe_valid_loader,
                       device, label_cols=SUPERCLASSES, verbose=True):
    """Freeze encoder, fit a linear head per label on frozen features,
    return mean AUROC on the held-out valid split."""
    X_train_feat, y_train_feat = extract_features(encoder, probe_train_loader, device)
    X_valid_feat, y_valid_feat = extract_features(encoder, probe_valid_loader, device)

    per_label = {}
    for i, col in enumerate(label_cols):
        y_tr, y_va = y_train_feat[:, i], y_valid_feat[:, i]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X_train_feat, y_tr)
        preds = clf.predict_proba(X_valid_feat)[:, 1]
        per_label[col] = roc_auc_score(y_va, preds)

    if verbose:
        for col, score in per_label.items():
            print(f"      {col}: {score:.4f}")
        skipped = [c for c in label_cols if c not in per_label]
        if skipped:
            print(f"      (skipped, insufficient class variance: {skipped})")

    encoder.train()
    return float(np.mean(list(per_label.values()))) if per_label else float("nan")


# ==========================================
# Main training function
# ==========================================

def run_simclr_training(
    epochs=None,
    batch_size=None,
    lr=None,
    weight_decay=None,
    temperature=None,
    warmup_epochs=None,
    checkpoint_dir="checkpoints_simclr",
    device=DEVICE,
    seed=42,
):
    """Run SimCLR pretraining with warmup+cosine LR schedule, linear-probe
    AUROC evaluation, and early stopping."""

    # Resolve defaults from CFG
    epochs = epochs or CFG["simclr_epochs"]
    batch_size = batch_size or CFG["simclr_batch_size"]
    lr = lr or CFG["simclr_lr"]
    weight_decay = weight_decay or CFG["simclr_weight_decay"]
    temperature = temperature or CFG["simclr_temperature"]
    warmup_epochs = warmup_epochs or CFG["simclr_warmup_epochs"]

    checkpoint_every = CFG["checkpoint_every"]
    eval_every = CFG["eval_every"]
    patience = CFG["pretrain_patience"]

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # ---- Data ----
    X, Y_clean = prepare_dataset()
    x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(Y_clean, X)

    ssl_dataset = ECGSSLData(x_train)
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Linear probe dataloaders ----
    np.random.seed(seed)
    probe_n = min(3000, len(y_train))
    probe_idx = np.random.choice(len(y_train), probe_n, replace=False)
    probe_train_ds = ECGDataset(x_train[probe_idx], y_train.iloc[probe_idx])
    probe_valid_ds = ECGDataset(x_valid, y_valid)

    probe_train_loader = DataLoader(probe_train_ds, batch_size=256, shuffle=False, num_workers=0)
    probe_valid_loader = DataLoader(probe_valid_ds, batch_size=256, shuffle=False, num_workers=0)

    # ---- Model / Optimizer / Scheduler ----
    model = ECGSimCLR(ECGEncoder()).to(device)
    criterion = NTXentLoss(temperature=temperature)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    # ---- Checkpoint resume ----
    start_epoch = 0
    history = []
    probe_history = []
    best_auroc = -1.0
    patience_counter = 0

    resume_path = checkpoint_dir / "latest.pth"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt["history"]
        probe_history = ckpt.get("probe_history", [])
        best_auroc = ckpt.get("best_auroc", -1.0)
        patience_counter = ckpt.get("patience_counter", 0)
        start_epoch = ckpt["epoch"]
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    else:
        print("No checkpoint found -- starting fresh")

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, ssl_loader, optimizer, criterion, device)
        scheduler.step()
        history.append(train_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {train_loss:.4f} | LR: {current_lr:.2e}"
        )

        # ---- resumable checkpoint ----
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "probe_history": probe_history,
                "best_auroc": best_auroc,
                "patience_counter": patience_counter,
            }, checkpoint_dir / "latest.pth")
            print(f"  -> checkpoint saved at epoch {epoch + 1}")

        # ---- linear-probe validation + early stopping ----
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            auroc = linear_probe_auroc(
                model.encoder, probe_train_loader, probe_valid_loader, device
            )
            probe_history.append((epoch + 1, auroc))
            print(f"  -> linear probe mean AUROC: {auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                patience_counter = 0
                torch.save(model.encoder.state_dict(), checkpoint_dir / "encoder_best.pth")
                print(f"  -> new best encoder saved (AUROC={auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"No AUROC improvement for {patience} evals "
                        f"-- stopping early at epoch {epoch + 1}"
                    )
                    break

    # Debug-only snapshot of final weights
    torch.save(model.encoder.state_dict(), "ecg_encoder_simclr_final.pth")

    return model


if __name__ == '__main__':
    run_simclr_training()
