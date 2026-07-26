import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from configs.dataset.imagedata_config import CFG, SHARED_LABELS
from multimodal_classifier.losses.supaCon_loss import multilabel_supcon_loss
from .simclr_trainer import linear_probe_auroc, print_log, save_checkpoint, step_optimizer

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def train_supcon_epoch(model, loader, optimizer, temperature, device, log_every=50):
    model.train()
    running_loss = torch.zeros((), device=device)
    num_batches = 0

    progress_bar = tqdm(loader, desc="SupCon Train")
    for (view1, view2), labels in progress_bar:
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
        images = torch.cat([view1, view2], dim=0)

        optimizer.zero_grad(set_to_none=True)

        autocast_kwargs = {"device_type": "xla", "dtype": torch.bfloat16} if HAS_XLA else {"device_type": str(device).split(":")[0], "enabled": False}
        with torch.autocast(**autocast_kwargs):
            _, z = model(images)
            loss = multilabel_supcon_loss(z.float(), labels, temperature=temperature)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        step_optimizer(optimizer)

        running_loss += loss.detach()
        num_batches += 1

        if num_batches % log_every == 0:
            progress_bar.set_postfix(loss=f"{(running_loss / num_batches).item():.4f}")

    return (running_loss / max(num_batches, 1)).item()


def run_supcon_pretraining(model, train_loader, probe_train_loader=None, probe_valid_loader=None, device=None, epochs=None, checkpoint_dir="checkpoints_supcon"):
    if device is None:
        device = xm.xla_device() if HAS_XLA else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if epochs is None:
        epochs = CFG["epochs"]

    model = model.to(device)
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
    patience_counter, patience_limit = 0, 3

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
        probe_history = ckpt.get("probe_history", [])
        best_auroc = ckpt.get("best_auroc", -1.0)
        start_epoch = ckpt["epoch"]
        print_log(f"Resumed SupCon from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        loss = train_supcon_epoch(model, train_loader, optimizer, CFG["temperature"], device)
        scheduler.step()
        history.append(loss)
        print_log(f"Epoch [{epoch + 1}/{epochs}] SupCon Loss: {loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if probe_train_loader and probe_valid_loader and ((epoch + 1) % 5 == 0 or (epoch + 1) == epochs):
            auroc = linear_probe_auroc(model.encoder, probe_train_loader, probe_valid_loader, device, label_cols=SHARED_LABELS)
            probe_history.append((epoch + 1, auroc))
            print_log(f"      Probe AUROC (SHARED_LABELS): {auroc:.4f}")
            if auroc > best_auroc:
                best_auroc = auroc
                patience_counter = 0
                save_checkpoint(model.encoder.state_dict(), checkpoint_dir / "encoder_best.pth")
                print_log(f"      New best encoder saved (AUROC={best_auroc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print_log(f"Early stopping at epoch {epoch + 1}")
                    break

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "probe_history": probe_history,
                "best_auroc": best_auroc,
            }, checkpoint_dir / "latest.pth")

        save_checkpoint(model.encoder.state_dict(), checkpoint_dir / "image_encoder_multisupcon_final.pth")
    return model, history
