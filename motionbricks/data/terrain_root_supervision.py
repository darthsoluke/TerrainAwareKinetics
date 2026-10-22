"""Opt-in training targets for Root reconstruction of factual played history."""

import torch

from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


ROOT_HISTORY_SUPERVISION = 'factual-global4-local3-v1'


@torch.no_grad()
def factual_history_root_targets(inference, target, history, observation):
    """Replace only source-past Root labels, retaining all source future labels.

    This is a loss-side batch constructor, not an actor condition or a motion
    repair. The source-to-future boundary interval remains an explicit unresolved
    recovery target; this helper never synthesizes a transition or moves truth.
    Existing training runners must opt in and record ROOT_HISTORY_SUPERVISION.
    """
    rep = inference.global_motion_rep
    global_target = extract_feature_from_motion_rep(target['global_motions'], rep, 'root').detach().clone()
    local_target = extract_feature_from_motion_rep(target['local_motions'], inference.local_motion_rep, 'root').detach().clone()
    batch, frames = global_target.shape[:2]
    if frames < 8 or frames % 4 or local_target.shape[:2] != (batch, frames):
        raise ValueError('Root targets require matched native four-frame token windows')
    if history.shape != (batch, 4, len(rep.indices['all'])) or not torch.isfinite(history).all():
        raise ValueError('Exactly four finite factual global history frames required')
    observed = shift_global_motion(history.detach(), rep, -vertical_origin(inference, history, observation))
    observed[..., [0, 2]] -= history[:, :1, [0, 2]]
    observed_root = extract_feature_from_motion_rep(observed, rep, 'root')
    observed_local_root = inference.motion_rep.dual_rep.global_to_local(
        observed_root, False, True, lengths=torch.full((batch,), 4, device=history.device, dtype=torch.long))
    global_target[:, :4] = rep.normalize(observed_root)
    # Frame3's forward interval needs a future endpoint. Only0..2 are fully
    # observed, so using the four-frame converter's terminal derivative is wrong.
    local_target[:, :3] = observed_local_root[:, :3]
    return {'global_motions': global_target, 'local_motions': local_target,
            'groundtruth_num_tokens': torch.full((batch, 1), frames // 4, device=history.device, dtype=torch.long)}
