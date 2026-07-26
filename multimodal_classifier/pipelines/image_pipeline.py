import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from configs.dataset.imagedata_config import CFG, DEFAULT_DATA_DIR, LABEL_COLS, SHARED_LABELS
from multimodal_classifier.data.image_data import create_image_dataloaders, clean_labels
from multimodal_classifier.models.image_models import ImageSimCLR, ImageSupCon, CheXpertClassifier, ResNet50Encoder
from multimodal_classifier.training.image_training import run_simclr_pretraining, run_supcon_pretraining, run_classifier_training


def run_full_image_pipeline(train_df_path, valid_df_path, data_dir=DEFAULT_DATA_DIR, device=None, epochs=None):
    """Executes the complete two-phase image pretraining and classification pipeline."""
    train_df = pd.read_csv(train_df_path)
    valid_df = pd.read_csv(valid_df_path)

    print("Creating image dataloaders...")
    simclr_loader, supcon_loader, clf_train_loader, clf_valid_loader = create_image_dataloaders(
        train_df, valid_df, data_dir=data_dir, device=device
    )

    # Phase 1: SimCLR Pretraining
    print("Starting SimCLR unsupervised contrastive pretraining...")
    simclr_model = ImageSimCLR(pretrained=CFG["pretrained_encoder"])
    run_simclr_pretraining(simclr_model, simclr_loader, device=device, epochs=epochs, checkpoint_dir="checkpoints_simclr")

    # Phase 2: SupCon Pretraining on shared labels
    print("Starting SupCon pretraining on multi-modal shared labels...")
    supcon_model = ImageSupCon(pretrained=CFG["pretrained_encoder"])
    run_supcon_pretraining(supcon_model, supcon_loader, device=device, epochs=epochs, checkpoint_dir="checkpoints_supcon")

    # Phase 3: Supervised Classification downstream probe
    print("Training downstream CheXpert multi-label classifier...")
    encoder = ResNet50Encoder(pretrained=False)
    best_enc_path = Path("checkpoints_supcon/encoder_best.pth")
    if best_enc_path.exists():
        encoder.load_state_dict(torch.load(best_enc_path, map_location="cpu"))
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    clf_model = CheXpertClassifier(encoder, n_classes=len(LABEL_COLS))

    # Calculate class imbalance weights for BCE
    clean_labels_df = clean_labels(train_df, LABEL_COLS)
    pos_counts = clean_labels_df.sum(axis=0).values
    neg_counts = len(clean_labels_df) - pos_counts
    pos_weight = torch.tensor(neg_counts / np.clip(pos_counts, 1, None), dtype=torch.float32)

    run_classifier_training(clf_model, clf_train_loader, clf_valid_loader, device=device, epochs=20, pos_weight=pos_weight)
    print("Image pipeline execution completed successfully!")
