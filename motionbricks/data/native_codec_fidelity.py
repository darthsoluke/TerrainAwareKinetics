"""Loss-side trajectory/FK fidelity for source reconstruction, never motion repair."""
import torch
from torch.nn import functional as F

from .terrain_root_supervision import factual_history_root_targets
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


VERSION = 'source-native-root-fk-fidelity-v1'


def source_fidelity_losses(inference, decoded_local, target, history, observation):
    rep, local_rep, dual = inference.global_motion_rep, inference.local_motion_rep, inference.motion_rep.dual_rep
    frames = decoded_local.shape[1]
    lengths = torch.full((len(decoded_local),), frames, dtype=torch.long, device=decoded_local.device)
    local_root = extract_feature_from_motion_rep(decoded_local, local_rep, 'root')
    root = dual.local_to_global(local_root, True, False, lengths=lengths)
    _, angle, _ = inference._extract_initial_root_info(history[..., :5])
    c, s = angle.cos().reshape(-1, 1), angle.sin().reshape(-1, 1)
    heading = torch.stack((root[..., 3] * c - root[..., 4] * s,
                           root[..., 4] * c + root[..., 3] * s), -1)
    root = rep.normalize(torch.cat((root[..., :3], heading), -1))
    expected = factual_history_root_targets(inference, target, history, observation)
    derived_local = dual.global_to_local(root, True, True, lengths=lengths)
    global_loss = F.smooth_l1_loss(root, expected['global_motions'])
    local_loss = F.smooth_l1_loss(derived_local[:, :-1], expected['local_motions'][:, :-1])
    native_root = (inference._root_model.args['global_root_loss_coeff'] * global_loss
                   + inference._root_model.args['local_root_loss_coeff'] * local_loss)
    # Both inverse paths are in the same centered world; global rotations define rendered joints.
    predicted_joints = local_rep.inverse(decoded_local, True, return_all=False,
        joint_positions_from='global_rot_data')['posed_joints']
    with torch.no_grad():
        source_joints = rep.inverse(target['global_motions'], True, return_all=False,
            joint_positions_from='global_rot_data')['posed_joints']
    scale = rep.stats.std[rep.indices['ric_data']].to(decoded_local)
    error = (predicted_joints[..., 1:, :] - source_joints[..., 1:, :]).flatten(-2)
    fk = F.smooth_l1_loss(error / torch.sqrt(scale.square() + 1e-5), torch.zeros_like(error))
    return dict(decoded_root_native=native_root, decoded_root_global=global_loss,
        decoded_root_local=local_loss, rendered_fk=fk, total=native_root + fk)
