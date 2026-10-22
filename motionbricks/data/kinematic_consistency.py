"""Consistency between redundant SOMA motion outputs, inspired by DartControl.

Units are metres and metres/second. This couples predictions to predictions;
source reconstruction remains a separate objective. It imposes no contact state.
"""

import math

import torch


def kinematic_consistency_losses(fk_positions, feature_positions, feature_velocities, fps):
    if (fk_positions.ndim != 4 or fk_positions.shape[-1] != 3
            or feature_positions.shape != fk_positions.shape
            or feature_velocities.shape != fk_positions.shape or fk_positions.shape[1] < 2):
        raise ValueError("Consistency inputs must share [batch, frames>=2, joints, 3]")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    fk_velocity = (fk_positions[:, 1:] - fk_positions[:, :-1]) * fps
    return {
        "l_kinematic_position_consistency_m": (fk_positions - feature_positions).norm(dim=-1).mean(),
        # The feature at t describes [t,t+1]. The duplicated final velocity is unused.
        "l_kinematic_velocity_consistency_mps": (fk_velocity - feature_velocities[:, :-1]).norm(dim=-1).mean(),
    }
