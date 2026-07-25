import torch
import torch.nn.functional as F


def label_similarity(y):
    """y: (B, C) multi-hot -> returns: (B, B) Jaccard-style label similarity"""
    y = y.float()
    intersection = torch.matmul(y, y.T)
    y_sum = y.sum(dim=1, keepdim=True)
    union = y_sum + y_sum.T - intersection + 1e-8
    return intersection / union


def multilabel_supcon_loss(z, y, temperature=0.2):
    """z: (2B, D) L2-normalized embeddings (both views). y: (B, C) multi-hot
    labels for one view (duplicated to 2B)."""
    B = y.size(0)
    y = torch.cat([y, y], dim=0)  # (2B, C)

    # label similarity
    sim_labels = label_similarity(y)                # (2B, 2B)
    # embedding similarity
    sim = torch.matmul(z, z.T) / temperature         # (2B, 2B)

    # mask self
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

    # remove self from label sim
    sim_labels = sim_labels.masked_fill(mask, 0)

    # avoid divide by zero
    row_sum = sim_labels.sum(dim=1, keepdim=True)
    sim_labels = sim_labels / (row_sum + 1e-6)

    # ignore rows with no positives
    valid = (row_sum > 1e-6).float()

    log_prob = F.log_softmax(sim, dim=1)
    loss = -(sim_labels * log_prob).sum(dim=1)

    # apply mask
    return (loss * valid.squeeze()).sum() / (valid.sum() + 1e-6)