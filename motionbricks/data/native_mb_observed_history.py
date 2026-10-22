"""Re-encode played skeletal state, not the decoder's redundant predictions.

Root XYZ and displayed SO(3) rotations are authoritative. This does not change
rendered motion, project to terrain, smooth poses or manufacture future history.
"""

import torch


def reencode_played_history(features, representation, info=None, tensor_checks=None, *, differentiable=False):
    with torch.set_grad_enabled(torch.is_grad_enabled() and differentiable):
        return _reencode_played_history(features, representation, info, tensor_checks)


def _reencode_played_history(features, representation, info=None, tensor_checks=None):
    # The window length is whatever the caller supplies; the ablation actor reads a
    # longer motion window than the original four-frame played history.
    length = features.shape[1] if features.ndim == 3 else -1
    if (features.ndim != 3 or length < 1
            or features.shape[-1] != len(representation.indices['all'])
            or representation.root_mode != 'global' or representation.removed_heading
            or not torch.isfinite(features).all()):
        raise ValueError('Expected finite unnormalized native global played frames')
    pose = representation.inverse(features, is_normalized=False, return_all=False,
                                  joint_positions_from='global_rot_data', return_quat=False)
    roots = pose['posed_joints'][..., representation.skeleton.root_idx, :]
    # Contact predictions are retained as unused forecast channels, not observed forces.
    canonical, origin = representation({
        'posed_joints': pose['posed_joints'], 'global_joint_rots': pose['global_joint_rots'],
        'local_joint_rots': pose['local_joint_rots'], 'translation': roots,
        'foot_contacts': features[..., representation.indices['foot_contacts']],
    }, to_normalize=False, return_init_heading_info=True,
       lengths=torch.full((features.shape[0],), length, device=features.device, dtype=torch.long))
    initial_quat = origin['init_heading_quat'].reshape(features.shape[0], 4)
    yaw = 2 * torch.atan2(initial_quat[:, 2], initial_quat[:, 0])
    observed = representation.change_first_heading(canonical, yaw, False, False)
    observed = observed.clone()
    observed[..., representation.indices['global_root_pos_2d']] += roots[:, :1, [0, 2]]
    reconstructed = representation.inverse(observed, False, joint_positions_from='global_rot_data',
                                         return_quat=False, return_all=False)
    position_error = (reconstructed['posed_joints'] - pose['posed_joints']).abs().max()
    rotation_error = (reconstructed['global_joint_rots'] - pose['global_joint_rots']).abs().max()
    if tensor_checks is not None:
        tensor_checks.update(position_error=position_error, rotation_error=rotation_error)
    if not torch.isfinite(observed).all() or position_error > 2e-5 or rotation_error > 2e-5:
        raise ValueError('Observed-history re-encoding changed the displayed FK state')
    if info is not None:
        original_positions = representation.inverse(features, False, joint_positions_from='ric_data',
                                                    return_quat=False, return_all=False)['posed_joints']
        difference = (original_positions - pose['posed_joints']).norm(dim=-1)
        info.update(history_source='played_fk_reencoded',
                    original_ric_fk_mean_error_m=float(difference.mean()),
                    original_ric_fk_max_error_m=float(difference.max()),
                    physical_state_reencoding_max_error_m=float(position_error))
    return observed
