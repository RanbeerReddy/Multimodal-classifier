import torch
import pytest
from multimodal_classifier.models.ecg_models.ecg_encoder import ECGEncoder
from multimodal_classifier.models.ecg_models.ECGSimCLR import ECGSimCLR
from multimodal_classifier.models.ecg_models.ECGClassifier import ECGClassifier
from multimodal_classifier.models.ecg_models.ECGSupaCon import ECGSupConModel
from multimodal_classifier.models.ecg_models.supaCon_projectionHead import SupConProjectionHead


def test_ecg_encoder():
    model = ECGEncoder(n_leads=12, base_channels=64, emb_dim=256)
    dummy_input = torch.randn(2, 12, 1000)
    output = model(dummy_input)
    assert output.shape == (2, 256)


def test_ecg_encoder_feature_dim():
    """Verify feature_dim attribute exists and is correct."""
    model = ECGEncoder(n_leads=12, base_channels=64)
    assert hasattr(model, 'feature_dim')
    assert model.feature_dim == 256


def test_ecg_encoder_groupnorm():
    """Verify GroupNorm is used instead of BatchNorm."""
    import torch.nn as nn
    model = ECGEncoder()
    has_groupnorm = any(isinstance(m, nn.GroupNorm) for m in model.modules())
    has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    assert has_groupnorm, "ECGEncoder should use GroupNorm"
    assert not has_batchnorm, "ECGEncoder should not use BatchNorm"


def test_ecg_simclr():
    encoder = ECGEncoder()
    model = ECGSimCLR(encoder)
    dummy_input = torch.randn(2, 12, 1000)
    h, z = model(dummy_input)
    assert h.shape == (2, 256)
    assert z.shape == (2, 128)


def test_ecg_classifier():
    encoder = ECGEncoder()
    model = ECGClassifier(encoder, n_meta=6, n_classes=5)
    dummy_signal = torch.randn(2, 12, 1000)
    dummy_meta = torch.randn(2, 6)
    output = model(dummy_signal, dummy_meta)
    assert output.shape == (2, 5)


def test_supcon_projection_head():
    head = SupConProjectionHead(in_dim=256, proj_dim=128)
    x = torch.randn(4, 256)
    z = head(x)
    assert z.shape == (4, 128)
    # Verify L2 normalization
    norms = torch.norm(z, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_ecg_supcon_model():
    encoder = ECGEncoder()
    model = ECGSupConModel(encoder)
    dummy_input = torch.randn(2, 12, 1000)
    h, z = model(dummy_input)
    assert h.shape == (2, 256)
    assert z.shape == (2, 128)
    # Verify L2 normalization
    norms = torch.norm(z, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)
