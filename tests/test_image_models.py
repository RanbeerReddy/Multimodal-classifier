import torch
import pytest
from multimodal_classifier.models.image_models import ResNet50Encoder, ImageSimCLR, ImageSupCon, CheXpertClassifier
from configs.dataset.imagedata_config import CFG, LABEL_COLS


def test_resnet50_encoder():
    model = ResNet50Encoder(pretrained=False)
    assert hasattr(model, 'feature_dim')
    assert model.feature_dim == 2048
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2048)


def test_image_simclr():
    model = ImageSimCLR(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    features, embeddings = model(dummy_input)
    assert features.shape == (2, 2048)
    assert embeddings.shape == (2, CFG["projection_dim"])


def test_image_supcon():
    model = ImageSupCon(pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    features, embeddings = model(dummy_input)
    assert features.shape == (2, 2048)
    assert embeddings.shape == (2, CFG["projection_dim"])
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_chexpert_classifier():
    encoder = ResNet50Encoder(pretrained=False)
    model = CheXpertClassifier(encoder, n_classes=len(LABEL_COLS))
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, len(LABEL_COLS))
