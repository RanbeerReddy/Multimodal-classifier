


ssl_dataset = ECGSSLData(x_train)
ssl_loader = DataLoader(ssl_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True)

model = ECGSimCLR(ECGEncoder()).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


from tqdm import tqdm

for epoch in tqdm(range(10)):
    model.train()
    total_loss = 0

    for x1, x2 in ssl_loader:
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

        _, z1 = model(x1)
        _, z2 = model(x2)

        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()