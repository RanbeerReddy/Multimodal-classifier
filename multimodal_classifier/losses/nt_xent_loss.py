import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for SimCLR.

    Refactored from a plain function to a module with cached masks/labels
    for efficiency on TPU (avoids re-creating tensors every forward pass).
    """

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self._cached_batch_size = None
        self._mask = None
        self._labels = None

    def _build_cache(self, batch_size, device):
        self._mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        # same positive-pair construction as before: view-1 index i's
        # positive is view-2 index i (offset by B in the concatenated
        # batch), and vice versa.
        self._labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size),
        ]).to(device)
        self._cached_batch_size = batch_size

    def forward(self, z1, z2):
        # explicit fp32 cast -- don't rely on autocast's op-allowlist
        # for normalize + matmul + cross_entropy
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)

        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        similarity = torch.matmul(z, z.T) / self.temperature

        if self._cached_batch_size != batch_size:
            self._build_cache(batch_size, similarity.device)

        similarity = similarity.masked_fill(self._mask, torch.finfo(similarity.dtype).min)
        return F.cross_entropy(similarity, self._labels)


def nt_xent_loss(z1, z2, temperature=0.2):
    """Backward-compatible functional wrapper around NTXentLoss."""
    return NTXentLoss(temperature=temperature)(z1, z2)