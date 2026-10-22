"""Loss-only supervision for decoder Root channels that affect future integration."""

import torch
from torch.nn import functional as F

from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


DECODED_HISTORY_ROOT_VERSION = 'decoded-factual-root-three-intervals-four-heights-v1'


def decoded_history_root_loss(inference, decoded_local, history, observation):
    """Supervise observed values, never overwrite the predicted motion.

    Four observed positions give three velocities. The fourth height is known,
    but its outgoing velocity is not. Futures and body/token targets are untouched.
    """
    if history.shape[1] != 4 or decoded_local.shape[1] < 4:
        raise ValueError('Need four factual history frames and a decoded prefix')
    with torch.no_grad():
        rep = inference.global_motion_rep
        centered = shift_global_motion(history.detach(), rep, -vertical_origin(inference, history, observation))
        centered[..., [0, 2]] -= history[:, :1, [0, 2]]
        root = extract_feature_from_motion_rep(centered, rep, 'root')
        target = inference.motion_rep.dual_rep.global_to_local(root, False, True,
            lengths=torch.full((len(root),), 4, dtype=torch.long, device=root.device))
        root_indices = list(inference.local_motion_rep.indices['root'])
        height_channel = root_indices.index(int(inference.local_motion_rep.indices['global_root_y'][0]))
        known = torch.ones_like(target, dtype=torch.bool)
        known[:, 3] = False
        known[:, 3, height_channel] = True
    predicted = extract_feature_from_motion_rep(decoded_local[:, :4], inference.local_motion_rep, 'root')
    return F.smooth_l1_loss(predicted[known], target[known])
