

# ✅ Fix optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ✅ Reduce augmentations
# gentler noise, smaller shifts, shorter masks

# ✅ Ensure proper pairing in NT-Xent

# ✅ Optional: print grad norms
for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.norm())

# ✅ Check that the temperature and normalization are applied correctly


encoder = ECGEncoder()
encoder.load_state_dict(torch.load("ecg_encoder_ssl.pth"))
encoder = encoder.to(DEVICE)

for p in encoder.parameters():
    p.requires_grad = False

    model = ECGClassifier(encoder).to(DEVICE)



criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5

for epoch in tqdm(range(10)):
    model.train()
    total_loss = 0

    for signal, meta, labels in train_loader:
        signal = signal.to(DEVICE)
        meta = meta.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(signal, meta)  # (B, 5)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}")


    model.eval()
    val_loss = 0

    with torch.no_grad():
        for signal, meta, labels in valid_loader:
            signal = signal.to(DEVICE)
            meta = meta.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(signal, meta)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    val_loss /= len(valid_loader)
    print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")



import numpy as np
from sklearn.metrics import roc_auc_score

model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for signal, meta, labels in valid_loader:
        signal = signal.to(DEVICE)
        meta = meta.to(DEVICE)

        outputs = model(signal, meta)

        probs = torch.sigmoid(outputs).cpu().numpy()

        y_pred.append(probs)
        y_true.append(labels.numpy())

# Stack
y_pred = np.vstack(y_pred)
y_true = np.vstack(y_true)

# Macro AUC
auc = roc_auc_score(y_true, y_pred, average='macro')

print("Validation AUC:", auc)