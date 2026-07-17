

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────
# Depthwise Separable Conv
# ─────────────────────────────────────────────
class DepthwiseSepConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding='same'):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


# ─────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────
class ResBlock1d(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv1 = DepthwiseSepConv1d(channels, channels, kernel_size)
        self.conv2 = DepthwiseSepConv1d(channels, channels, kernel_size)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        return F.relu(self.bn(x + self.conv2(self.conv1(x))))


# ─────────────────────────────────────────────
# SSL Encoder (NO META, NO CLASSIFIER)
# ─────────────────────────────────────────────
class ECGEncoder(nn.Module):
    def __init__(self, n_leads=12, base_channels=64, emb_dim=256):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.block1 = nn.Sequential(
            ResBlock1d(base_channels, kernel_size=7),
            nn.MaxPool1d(2)
        )

        self.trans1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, 1, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            ResBlock1d(base_channels * 2, kernel_size=5),
            nn.MaxPool1d(2)
        )

        self.trans2 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 4, 1, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            ResBlock1d(base_channels * 4, kernel_size=3),
            nn.MaxPool1d(2)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.out_dim = base_channels * 4  # 256

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 12, 1000)
        x = self.stem(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)

        x = self.gap(x).squeeze(-1)  # (B, 256)
        return x