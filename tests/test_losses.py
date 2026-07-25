import torch
import pytest
from multimodal_classifier.losses.nt_xent_loss import nt_xent_loss, NTXentLoss
from multimodal_classifier.losses.supaCon_loss import label_similarity, multilabel_supcon_loss


def test_nt_xent_loss_function():
    """Test backward-compatible functional API."""
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    loss = nt_xent_loss(z1, z2)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)


def test_nt_xent_loss_class():
    """Test the new NTXentLoss class with cached masks."""
    criterion = NTXentLoss(temperature=0.2)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    loss = criterion(z1, z2)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_nt_xent_loss_class_cache():
    """Verify mask caching works across calls with same batch size."""
    criterion = NTXentLoss(temperature=0.2)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    loss1 = criterion(z1, z2)
    assert criterion._cached_batch_size == 4

    # Same batch size should reuse cache
    z3 = torch.randn(4, 128)
    z4 = torch.randn(4, 128)
    loss2 = criterion(z3, z4)
    assert criterion._cached_batch_size == 4

    # Different batch size should rebuild cache
    z5 = torch.randn(8, 128)
    z6 = torch.randn(8, 128)
    loss3 = criterion(z5, z6)
    assert criterion._cached_batch_size == 8


def test_multilabel_supcon_loss():
    # z must be (2B, D) -- two concatenated views; y is (B, C) -- one copy of labels
    B = 4
    z = torch.randn(2 * B, 128)  # (2B, D) -- concatenated views
    y = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=torch.float32)  # (B, C)
    loss = multilabel_supcon_loss(z, y)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)


def test_label_similarity():
    y = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.float32)
    sim = label_similarity(y)
    assert sim.shape == (3, 3)
    assert sim[0, 1] == 1.0
    assert sim[0, 2] == 0.0
