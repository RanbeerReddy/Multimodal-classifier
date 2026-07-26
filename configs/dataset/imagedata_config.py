import os
from pathlib import Path

# ==========================================
# Project Config for Image Modality
# ==========================================

CFG = {
    "img_size": 224,
    "batch_size": 128,
    "warmup_epochs": 5,
    "num_workers": 8,
    "prefetch_factor": 4,
    "projection_dim": 256,
    "hidden_dim": 1024,
    "feature_dim": 2048,
    "temperature": 0.07,
    "epochs": 100,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "pretrained_encoder": False,
}

# ==========================================
# Dataset Paths & Labels
# ==========================================

DEFAULT_DATA_DIR = Path(os.environ.get("CHEXPERT_DIR", "/kaggle/input/datasets/ashery/chexpert"))

LABEL_COLS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

# Shared labels across ECG and Image multimodal setup
SHARED_LABELS = [
    "No Finding",
    "Cardiomegaly",
    "Enlarged Cardiomediastinum",
    "Edema",
    "Pleural Effusion",
]

CONCEPT_MAP = {
    "No Finding": 0,
    "Cardiomegaly": 1,
    "Edema": 2,
    "Pleural Effusion": 2,
    "Support Devices": 3
}
