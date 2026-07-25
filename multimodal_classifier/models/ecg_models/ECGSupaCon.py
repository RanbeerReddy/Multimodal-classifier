import torch.nn as nn
import torch.nn.functional as F
from .supaCon_projectionHead import SupConProjectionHead


class ECGSupConModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projector = SupConProjectionHead(encoder.feature_dim, 128)

    def forward(self, x):
        h = self.encoder(x)        # (B, 256)
        z = self.projector(h)      # (B, 128) -- already L2-normalized by the projector
        return h, z