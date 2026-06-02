import torch


class ECGSSLData(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def augment(self, x):
        x = x.clone()
    
        # 1. Slight noise (always)
        x = x + 0.01 * torch.randn_like(x)
    
        # 2. Time shift (strong but consistent)
        shift = torch.randint(-150, 150, (1,))
        x = torch.roll(x, shifts=int(shift), dims=1)
    
        # 3. Scaling
        scale = torch.rand(1) * 0.4 + 0.8
        x = x * scale
    
        # 4. ONE strong transformation (not many)
        if torch.rand(1) < 0.5:
            # masking
            length = torch.randint(150, 400, (1,))
            start = torch.randint(0, 1000 - length, (1,))
            x[:, start:start+length] = 0
        else:
            # lead dropout
            lead = torch.randint(0, 12, (1,))
            x[lead] = 0
    
        return x

    def __getitem__(self, idx):
        x = self.X[idx]

        x1 = self.augment(x.clone())
        x2 = self.augment(x.clone())

        return x1, x2