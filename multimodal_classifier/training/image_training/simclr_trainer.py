import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from configs.dataset.imagedata_config import CFG, LABEL_COLS
from multimodal_classifier.losses.nt_xent_loss import NTXentLoss

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def print_log(msg):
    if HAS_XLA:
        xm.master_print(msg)
    else:
        print(msg)


def save_checkpoint(obj, path):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    if HAS_XLA:
        xm.save(obj, path)
    else:
        torch.save(obj, path)


def step_optimizer(optimizer):
    if HAS_XLA:
        xm.optimizer_step(optimizer, barrier=True)
    else:
        optimizer.step()


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=50):
    model.train()
    running_loss = torch.zeros((), device=device)
    num_batches = 0

    progress_bar = tqdm(loader, desc="SimCLR Train")
    for view1, view2 in progress_bar:
        view1, view2 = view1.to(device), view2.to(device)
        images = torch.cat([view1, view2], dim=0)

        optimizer.zero_grad(set_to_none=True)

        autocast_kwargs = {"device_type": "xla", "dtype": torch.bfloat16} if HAS_XLA else {"device_type": str(device).split(":")[0], "enabled": False}
        with torch.autocast(**autocast_kwargs):
            _, z = model(images)
            z1, z2 = z.chunk(2, dim=0)
            loss = criterion(z1, z2)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_optimizer(optimizer)

        running_loss += loss.detach()
        num_batches += 1

        if num_batches % log_every == 0:
            progress_bar.set_postfix(loss=f"{(running_loss / num_batches).item():.4f}")

    return (running_loss / max(num_batches, 1)).item()


@torch.no_grad()
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, targets_list = [], []
    for images, targets in loader:
        images = images.to(device)
        f = encoder(images)
        feats.append(f.detach().cpu())
        targets_list.append(targets.detach().cpu())
    return torch.cat(feats).numpy(), torch.cat(targets_list).numpy()


def linear_probe_auroc(encoder, probe_train_loader, probe_valid_loader, device, label_cols=LABEL_COLS, verbose=True):
    X_train, y_train = extract_features(encoder, probe_train_loader, device)
    X_valid, y_valid = extract_features(encoder, probe_valid_loader, device)

    per_label = {}
    for i, col in enumerate(label_cols):
        y_tr, y_va = y_train[:, i], y_valid[:, i]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X_train, y_tr)
        preds = clf.predict_proba(X_valid)[:, 1]
        per_label[col] = roc_auc_score(y_va, preds)

    if verbose:
        for col, score in per_label.items():
            print_log(f"      {col}: {score:.4f}")
    encoder.train()
    return float(np.mean(list(per_label.values()))) if per_label else float("nan")


def run_simclr_pretraining(model, train_loader, probe_train_loader=None, probe_valid_loader=None, device=None, epochs=None, checkpoint_dir="checkpoints"):
    if device is None:
        device = xm.xla_device() if HAS_XLA else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epochs is None:
        epochs = CFG["epochs"]

    model = model.to(device)
    criterion = NTXentLoss(temperature=CFG["temperature"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    warmup_epochs = min(CFG["warmup_epochs"], epochs - 1) if epochs > 1 else 0
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    resume_path = checkpoint_dir / "latest.pth"

    start_epoch = 0
    history, probe_history = [], []
    best_auroc = -1.0
    patience_counter = 0
    patience_limit = 3

    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt.get("history", [])
        start_epoch = ckpt["epoch"]
        print_log(f"Resumed SimCLR from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        history.append(loss)
        print_log(f"Epoch [{epoch + 1}/{epochs}] SimCLR Loss: {loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
            }, checkpoint_dir / "latest.pth")

        if probe_train_loader and probe_valid_loader and ((epoch + 1) % 10 == 0 or (epoch + 1) == epochs):
            auroc = linear_probe_auroc(model.encoder, probe_train_loader, probe_valid_loader, device)
            probe_history.append((epoch + 1, auroc))
            print_log(f"  -> linear probe mean AUROC: {auroc:.4f}")
            if auroc > best_auroc:
                best_auroc = auroc
                patience_counter = 0
                save_checkpoint(model.encoder.state_dict(), checkpoint_dir / "encoder_best.pth")
                print_log(f"  -> new best encoder saved (AUROC={auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print_log(f"No improvement for {patience_limit} evals -- stopping early.")
                    break

    save_checkpoint(model.encoder.state_dict(), checkpoint_dir / "image_encoder_simclr.pth")
    return model, history
