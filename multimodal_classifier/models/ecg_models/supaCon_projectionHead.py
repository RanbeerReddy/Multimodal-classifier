import torch.nn as nn
import torch.nn.functional as F


class SupConProjectionHead(nn.Module):
    """SupCon projection head (renamed from SupaCon_ProjectionHead to match
    the image notebook's naming convention)."""

    def __init__(self, in_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, p=2, dim=1)  # L2-normalize here, once
        return z


# Backward-compatible alias
SupaCon_ProjectionHead = SupConProjectionHead