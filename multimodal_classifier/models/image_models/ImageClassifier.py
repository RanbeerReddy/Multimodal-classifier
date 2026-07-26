import torch
import torch.nn as nn
from configs.dataset.imagedata_config import LABEL_COLS


class CheXpertClassifier(nn.Module):
    """Multi-label classification head trained over a frozen ResNet50 encoder backbone."""

    def __init__(self, encoder, n_classes=len(LABEL_COLS), dropout=0.3):
        super().__init__()
        self.encoder = encoder  # kept frozen during classification head training

        self.head = nn.Sequential(
            nn.Linear(encoder.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        return self.head(feats)
