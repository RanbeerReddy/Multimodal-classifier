import torch

class ECGSSLWithLabels(Dataset):
    def __init__(self, X, Y):
        # X: (N, T, 12) → convert to (N, 12, T)
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def augment(self, x):
        x = x.clone()

        # noise
        x = x + 0.01 * torch.randn_like(x)

        # time shift
        shift = torch.randint(-150, 150, (1,))
        x = torch.roll(x, shifts=int(shift), dims=1)

        # scaling
        scale = torch.rand(1) * 0.4 + 0.8
        x = x * scale

        # one strong transform
        if torch.rand(1) < 0.5:
            length = torch.randint(150, 400, (1,))
            start = torch.randint(0, x.shape[1] - length, (1,))
            x[:, start:start+length] = 0
        else:
            lead = torch.randint(0, 12, (1,))
            x[lead] = 0

        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        x1 = self.augment(x)
        x2 = self.augment(x)

        return x1, x2, y