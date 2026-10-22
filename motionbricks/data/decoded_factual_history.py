"""Loss-only factual decoder prefix targets; no synthesized recovery future."""

import torch
from torch.nn import functional as F

from .native_mb_vertical_frame import shift_global_motion
from .terrain_decoder_pilot import vertical_origin


FACTUAL_PREFIX_VERSION = 'decoder-factual-root3-height4-body4-native-scale-v1'
SEPARATE_FACTUAL_ROOT_VERSION = 'decoder-factual-root13-independent-body-native-v2'


def decoded_factual_history_loss(inference, prediction, history, observation, *, root_coefficient=None):
    """Match known history channels at the native per-feature reconstruction scale.

    Four frames supply three Root velocity intervals and four heights/body poses.
    Contacts, redundant velocities and the unknown outgoing interval are excluded.
    Each known channel gets the same weight as an existing future reconstruction
    channel; the 28-frame denominator is not renormalized to amplify the prefix.
    """
    if history.shape[1] != 4 or prediction.shape[1] != 32:
        raise ValueError('This bounded objective requires four observed and32 decoded frames')
    rep = inference.global_motion_rep
    local_rep = inference.local_motion_rep
    if prediction.shape[-1] != len(local_rep.indices['all']):
        raise ValueError('Expected the full native normalized local representation')
    with torch.no_grad():
        observed = shift_global_motion(history.detach(), rep, -vertical_origin(inference, history, observation))
        observed = observed.clone()
        observed[..., [0, 2]] -= history[:, :1, [0, 2]]
        local = inference.motion_rep.dual_rep.global_to_local(observed, False, True,
            lengths=torch.full((len(history),), 4, device=history.device, dtype=torch.long))
        mask = torch.zeros_like(local, dtype=torch.bool)
        mask[:, :3, local_rep.indices['root']] = True
        mask[:, :, local_rep.indices['global_root_y']] = True
        mask[:, :, local_rep.indices['ric_data']] = True
        mask[:, :, local_rep.indices['global_rot_data']] = True
    values = F.smooth_l1_loss(prediction[:, :4], local, reduction='none')
    if root_coefficient is None:
        return (values * mask).sum() / (len(history) * 28 * prediction.shape[-1])
    if root_coefficient < 0:
        raise ValueError('Factual Root coefficient must be nonnegative')
    root_mask = torch.zeros_like(mask)
    root_mask[:, :3, local_rep.indices['root']] = True
    root_mask[:, 3, local_rep.indices['global_root_y']] = True
    # Factual intervals are independent of desired future pace. Remove their old
    # per-feature contribution so the two normalizations never double count.
    body = (values * (mask & ~root_mask)).sum() / (len(history) * 28 * prediction.shape[-1])
    return body + root_coefficient * values[root_mask].mean()
