

y_train_supacon = y_train[['NORM', 'MI', 'STTC', 'CD', 'HYP']]

x_train_supacon = x_train

indices = np.lexsort(y_train_supacon.to_numpy().T)

x_train_supacon = x_train_supacon[indices]

y_train_supacon = y_train_supacon.iloc[indices].reset_index(drop=True)


ssl_dataset = ECGSSLWithLabels(x_train_supacon, y_train_supacon.to_numpy())

ssl_loader = DataLoader(
    ssl_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)




encoder = ECGEncoder().to(DEVICE)
model = ECGSupConModel(encoder).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

EPOCHS = 10

for epoch in tqdm(range(EPOCHS)):
    model.train()
    total_loss = 0

    for x1, x2, labels in ssl_loader:
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        labels = labels.to(DEVICE)

        _, z1 = model(x1)
        _, z2 = model(x2)

        z = torch.cat([z1, z2], dim=0)

        loss = multilabel_supcon_loss(z, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
    print("z max:", z.max().item(), "z min:", z.min().item())
    print(f"Epoch {epoch}: {total_loss / len(ssl_loader):.4f}")