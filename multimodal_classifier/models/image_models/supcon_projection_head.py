import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.dataset.imagedata_config import CFG
from .image_encoder import ResNet50Encoder


class SupConProjectionHead(nn.Module):
    """Projection head with L2 normalization for Supervised Contrastive learning."""

    def __init__(self, feature_dim=None, hidden_dim=None, projection_dim=None):
        super().__init__()
        feat_dim = feature_dim if feature_dim is not None else CFG["feature_dim"]
        hid_dim = hidden_dim if hidden_dim is not None else CFG["hidden_dim"]
        proj_dim = projection_dim if projection_dim is not None else CFG["projection_dim"]

        self.net = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, proj_dim)
        )

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, p=2, dim=1)
        return z


class ImageSupCon(nn.Module):
    """SupCon wrapper combining ResNet50 encoder with normalized projection head."""

    def __init__(self, pretrained=False, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else ResNet50Encoder(pretrained=pretrained)
        self.projector = SupConProjectionHead(feature_dim=self.encoder.feature_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, z
