import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Multimodal classifier', 'data', 'Ecg data')))

import pytest
import torch
from data_augmentation import ECGSSLData

def test_ecg_ssl_data():
    dummy_data = torch.randn(10, 12, 1000)
    dataset = ECGSSLData(dummy_data)
    assert len(dataset) == 10
    
    x1, x2 = dataset[0]
    assert x1.shape == (12, 1000)
    assert x2.shape == (12, 1000)
