from torch import nn
import torch

class ECGClassifier(nn.Module):
    def __init__(self, encoder, n_meta=6, n_classes=5, dropout=0.3):
        super().__init__()

        self.encoder = encoder  # pretrained

        self.meta_mlp = nn.Sequential(
            nn.Linear(n_meta, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        fusion_dim = encoder.out_dim + 32

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, signal, meta):
        with torch.no_grad():  # freeze encoder
            x = self.encoder(signal)

        m = self.meta_mlp(meta)
        x = torch.cat([x, m], dim=1)

        return self.head(x)