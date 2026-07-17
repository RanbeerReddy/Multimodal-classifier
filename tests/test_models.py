import torch
import pytest
from multimodal_classifier.models.ecg_models.ecg_encoder import ECGEncoder
from multimodal_classifier.models.ecg_models.ECGSimCLR import ECGSimCLR
from multimodal_classifier.models.ecg_models.ECGClassifier import ECGClassifier

def test_ecg_encoder():
    model = ECGEncoder(n_leads=12, base_channels=64, emb_dim=256)
    dummy_input = torch.randn(2, 12, 1000)
    output = model(dummy_input)
    assert output.shape == (2, 256)

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
