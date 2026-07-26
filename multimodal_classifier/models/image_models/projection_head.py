import torch.nn as nn
from configs.dataset.imagedata_config import CFG
from .image_encoder import ResNet50Encoder


class ProjectionHead(nn.Module):
    """2-layer MLP projection head for SimCLR contrastive learning."""

    def __init__(self, feature_dim=None, hidden_dim=None, projection_dim=None):
        super().__init__()
        feat_dim = feature_dim if feature_dim is not None else CFG["feature_dim"]
        hid_dim = hidden_dim if hidden_dim is not None else CFG["hidden_dim"]
        proj_dim = projection_dim if projection_dim is not None else CFG["projection_dim"]

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, proj_dim)
        )

    def forward(self, x):
        return self.projector(x)


class ImageSimCLR(nn.Module):
    """SimCLR pretraining wrapper combining ResNet50 encoder with projection head."""

    def __init__(self, pretrained=False, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else ResNet50Encoder(pretrained=pretrained)
        self.projector = ProjectionHead(feature_dim=self.encoder.feature_dim)

    def forward(self, x):
        features = self.encoder(x)
        embeddings = self.projector(features)
        return features, embeddings


# Alias for notebook compatibility
SimCLR = ImageSimCLR
