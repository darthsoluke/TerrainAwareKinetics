"""TD(lambda) cost-to-go targets for the closed-loop critic (torch and float64 reference)."""

import numpy as np


def lambda_returns(costs, values, gamma, lam):
    """Targets [B, H] from costs [B, H] and values V(s_0..s_H) [B, H+1] (torch tensors)."""
    import torch
    columns = [None] * costs.shape[1]
    following = values[:, -1]
    for t in range(costs.shape[1] - 1, -1, -1):
        following = costs[:, t] + gamma * ((1 - lam) * values[:, t + 1] + lam * following)
        columns[t] = following
    return torch.stack(columns, 1)


def lambda_returns_numpy(costs, values, gamma, lam):
    costs, values = np.asarray(costs, dtype=np.float64), np.asarray(values, dtype=np.float64)
    targets = np.empty_like(costs)
    following = values[:, -1]
    for t in range(costs.shape[1] - 1, -1, -1):
        following = costs[:, t] + gamma * ((1 - lam) * values[:, t + 1] + lam * following)
        targets[:, t] = following
    return targets
