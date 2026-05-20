

def nt_xent_loss(z1, z2, temperature=0.2):
    B = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # remove self-similarity
    mask = torch.eye(2*B, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e9)

    # correct labels: positive pairs
    labels = torch.arange(B).to(z.device)
    labels = torch.cat([labels + B, labels])  # match pairs

    return F.cross_entropy(sim, labels)