


def label_similarity(y):
    """
    y: (B, C) multi-hot
    returns: (B, B)
    """
    y = y.float()
    intersection = torch.matmul(y, y.T)

    y_sum = y.sum(dim=1, keepdim=True)
    union = y_sum + y_sum.T - intersection + 1e-8

    sim = intersection / union
    return sim



def multilabel_supcon_loss(z, y, temperature=0.2):
    B = y.size(0)

    y = torch.cat([y, y], dim=0)  # (2B, C)

    # label similarity
    sim_labels = label_similarity(y)  # (2B, 2B)
    # embedding similarity
    sim = torch.matmul(z, z.T) / temperature

    # mask self
    mask = torch.eye(2*B, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e4)

    # remove self from label sim
    sim_labels = sim_labels.masked_fill(mask, 0)

    # 🔥 FIX 1: clamp minimum
    row_sum = sim_labels.sum(dim=1, keepdim=True)

    # 🔥 FIX 2: avoid divide by zero
    sim_labels = sim_labels / (row_sum + 1e-6)

    # 🔥 FIX 3: ignore rows with no positives
    valid = (row_sum > 1e-6).float()

    log_prob = F.log_softmax(sim, dim=1)

    loss = -(sim_labels * log_prob).sum(dim=1)

    # apply mask
    loss = (loss * valid.squeeze()).sum() / (valid.sum() + 1e-6)

    return loss