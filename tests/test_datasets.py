import numpy as np
import pytest
import torch
from multimodal_classifier.data.ecg_data.data_augmentation import ECGSSLData
from multimodal_classifier.data.ecg_data.supacon_data_augmentation import ECGSSLWithLabels


def test_ecg_ssl_data():
    # Need shape (N, T, 12) for transpose inside ECGSSLData
    dummy_data = np.random.randn(10, 1000, 12).astype(np.float32)
    dataset = ECGSSLData(dummy_data)
    assert len(dataset) == 10

    x1, x2 = dataset[0]
    assert x1.shape == (12, 1000)
    assert x2.shape == (12, 1000)


def test_ecg_ssl_data_augmentation():
    """Verify augmentation produces different views."""
    dummy_data = np.random.randn(5, 1000, 12).astype(np.float32)
    dataset = ECGSSLData(dummy_data)
    x1, x2 = dataset[0]
    # Views should generally differ due to random augmentation
    assert not torch.equal(x1, x2)


def test_ecg_ssl_with_labels():
    dummy_X = np.random.randn(10, 1000, 12).astype(np.float32)
    dummy_Y = np.random.randint(0, 2, (10, 5)).astype(np.float32)
    dataset = ECGSSLWithLabels(dummy_X, dummy_Y)
    assert len(dataset) == 10

    x1, x2, y = dataset[0]
    assert x1.shape == (12, 1000)
    assert x2.shape == (12, 1000)
    assert y.shape == (5,)
