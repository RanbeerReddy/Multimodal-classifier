import torch
from multimodal_classifier.data.ecg_data.extractload_data import prepare_dataset
from multimodal_classifier.data.ecg_data.split_data import split_data
from multimodal_classifier.data.ecg_data.data_loader import create_ecg_dataloaders
from multimodal_classifier.training.ecg_training.simclr_encoder import run_simclr_training
from multimodal_classifier.training.ecg_training.simclr_classifier import run_classifier_training
from multimodal_classifier.training.ecg_training.MultiSupaCon_encoder import run_supcon_training


def run_full_pipeline():
    """Run the full ECG SSL + classifier pipeline:
    1. SimCLR pretraining
    2. SimCLR classifier training
    3. MultiSupCon pretraining
    4. MultiSupCon classifier training
    """
    # Phase 1: SimCLR pretraining
    print("=" * 60)
    print("Phase 1: SimCLR Pretraining")
    print("=" * 60)
    simclr_model = run_simclr_training()
    torch.save(simclr_model.encoder.state_dict(), "ecg_encoder_ssl.pth")

    # Phase 2: SimCLR classifier
    print("=" * 60)
    print("Phase 2: SimCLR Classifier Training")
    print("=" * 60)
    run_classifier_training(encoder_path="checkpoints_simclr/encoder_best.pth")

    # Phase 3: MultiSupCon pretraining
    print("=" * 60)
    print("Phase 3: MultiSupCon Pretraining")
    print("=" * 60)
    supcon_model = run_supcon_training()

    # Phase 4: MultiSupCon classifier
    print("=" * 60)
    print("Phase 4: MultiSupCon Classifier Training")
    print("=" * 60)
    run_classifier_training(
        encoder_path="checkpoints_supcon/encoder_best.pth",
        checkpoint_dir="checkpoints_classifier_supcon",
    )


if __name__ == '__main__':
    run_full_pipeline()