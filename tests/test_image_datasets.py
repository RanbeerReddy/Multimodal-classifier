import pytest
import pandas as pd
import numpy as np
import torch
from PIL import Image
from multimodal_classifier.data.image_data import (
    clean_labels, SimCLRTransform, simclr_transform, eval_transform, CheXpertDataset
)
from configs.dataset.imagedata_config import LABEL_COLS, SHARED_LABELS


def test_clean_labels():
    data = {
        'Path': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
        LABEL_COLS[0]: [1.0, -1.0, np.nan],
        LABEL_COLS[1]: [0.0, 1.0, -1.0],
    }
    for col in LABEL_COLS[2:]:
        data[col] = [0.0, 0.0, 0.0]

    df = pd.DataFrame(data)
    cleaned = clean_labels(df, LABEL_COLS)
    assert cleaned.shape == (3, len(LABEL_COLS))
    assert list(cleaned[LABEL_COLS[0]].values) == [1.0, 0.0, 0.0]


def test_simclr_transform(tmp_path):
    img = Image.new('RGB', (300, 300), color='blue').convert("L")
    transform = SimCLRTransform(simclr_transform)
    view1, view2 = transform(img)
    assert view1.shape == (3, 224, 224)
    assert view2.shape == (3, 224, 224)
    assert isinstance(view1, torch.Tensor)


def test_chexpert_dataset_mock(tmp_path):
    img_path = tmp_path / "train" / "patient1"
    img_path.mkdir(parents=True)
    img_file = img_path / "study1_view1_frontal.jpg"
    Image.new('L', (256, 256), color=128).save(img_file)

    data = {
        'Path': [str(img_file.relative_to(tmp_path))],
        'Frontal/Lateral': ['Frontal'],
    }
    for col in LABEL_COLS:
        data[col] = [1.0]

    df = pd.DataFrame(data)
    ds = CheXpertDataset(df, data_dir=tmp_path, transform=eval_transform, label_cols=LABEL_COLS, return_labels=True)
    assert len(ds) == 1
    img_tensor, label_tensor = ds[0]
    assert img_tensor.shape == (3, 224, 224)
    assert label_tensor.shape == (len(LABEL_COLS),)
    assert label_tensor[0] == 1.0
