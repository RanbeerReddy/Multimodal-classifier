import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from configs.dataset.imagedata_config import CFG


class ResNet50Encoder(nn.Module):
    """ResNet50 feature extractor with classification head removed."""

    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        backbone.fc = nn.Identity()  # remove classification head

        self.backbone = backbone
        self.feature_dim = 2048

    def forward(self, x):
        return self.backbone(x)
