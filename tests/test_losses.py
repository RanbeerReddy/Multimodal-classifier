import torch
import pytest
from nt_xent_loss import nt_xent_loss
from supaCon_loss import label_similarity, multilabel_supcon_loss

def test_nt_xent_loss():
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    loss = nt_xent_loss(z1, z2)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)

def test_multilabel_supcon_loss():
    z = torch.randn(4, 128)
    y = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=torch.float32)
    loss = multilabel_supcon_loss(z, y)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)

def test_label_similarity():
    y = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.float32)
    sim = label_similarity(y)
    assert sim.shape == (3, 3)
    assert sim[0, 1] == 1.0
    assert sim[0, 2] == 0.0
