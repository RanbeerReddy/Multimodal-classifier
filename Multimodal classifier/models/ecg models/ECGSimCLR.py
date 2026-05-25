from torch import nn
import torch.nn.functional as F
from projection_head import ProjectionHead

class ECGSimCLR(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(encoder.out_dim, 128)

    def forward(self, x):
        h = self.encoder(x)      # (B, 256)
        z = self.projector(h)    # (B, 128)

        # Normalize (IMPORTANT for contrastive learning)
        z = F.normalize(z, dim=1)

        return h, z