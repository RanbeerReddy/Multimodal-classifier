class SupaCon_ProjectionHead(nn.Module):
    def __init__(self, in_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )

    def forward(self, x):
        projections = self.net(x)
        # This is the "Silver Bullet" fix:
        projections = F.normalize(projections, p=2, dim=1) 
        return projections