class ECGSupConModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projector = SupaCon_ProjectionHead(encoder.out_dim, 128)

    def forward(self, x):
        h = self.encoder(x)        # (B, 256)
        z = self.projector(h)      # (B, 128)
        z = F.normalize(z, dim=1)  # IMPORTANT
        return h, z