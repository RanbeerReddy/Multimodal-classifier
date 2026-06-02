from data.Ecg data.extractload_data import prepare_dataset
from data.Ecg data.split_data import split_data
from data.Ecg data.data_loader import create_ecg_dataloader
from training.ecg training.simclr_encoder import run_simclr_training





X, Y_clean = prepare_dataset()
x_train, x_valid, x_test, y_train, y_valid , y_test = split_data(Y_clean,X)
train_loader, valid_loader, test_loader = create_ecg_dataloader(Y_clean, X)
model = run_simclr_training()
# Saving the simclr encoder
torch.save(model.encoder.state_dict(), "ecg_encoder_ssl.pth")