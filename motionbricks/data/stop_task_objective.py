"""Loss-side stopping scales and actual publication-boundary velocities."""

import torch


STOP_TASK_SCALES = dict(root_velocity_mps=.15, support_gap_m=.02,
    planted_foot_velocity_mps=.1, penetration_m=.02)


def published_root_velocities(prediction, history):
    if prediction.ndim != 3 or history.ndim != 3 or history.shape[1] != 4:
        raise ValueError('Expected a native predicted window and four observed frames')
    if prediction.shape[1] < 12 or len(prediction) != len(history):
        raise ValueError('Need the actual eight published future frames')
    roots = torch.cat((history[:, -1:, [0, 2]], prediction[:, 4:12, [0, 2]]), 1)
    return torch.diff(roots, dim=1)*30


def scaled_zero_velocity_loss(velocity_mps):
    return (velocity_mps/STOP_TASK_SCALES['root_velocity_mps']).square().mean()
