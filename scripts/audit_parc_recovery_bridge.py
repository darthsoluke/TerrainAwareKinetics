"""Check whether factual SOMA history has an exact PARC kinematic representation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.parc_soma_retarget import MAPPING, Y_TO_Z, official_parc_character, reference_calibration
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def inverse_calibrated_tasks(soma_positions, soma_rotations, soma_names, character, config, scale):
    """Analytic task-frame inversion only; no IK, projection or fitted root offset."""
    parc_names = character.get_body_names()
    count = len(soma_positions)
    rotations = np.empty((count, len(parc_names), 3, 3))
    expected_positions = np.empty((count, len(parc_names), 3))
    lookup = {}
    for soma_name, parc_name in MAPPING.items():
        soma_id, parc_id = soma_names.index(soma_name), parc_names.index(parc_name)
        entry = config['ik_match_table1'][soma_name]
        offset = np.asarray(entry[3])
        rotation_offset = Rotation.from_quat(entry[4], scalar_first=True).as_matrix()
        observed_rotation = Y_TO_Z @ soma_rotations[:, soma_id] @ Y_TO_Z.T
        rotations[:, parc_id] = observed_rotation @ rotation_offset.T
        expected_positions[:, parc_id] = (soma_positions[:, soma_id] @ Y_TO_Z.T
                                           - (observed_rotation @ offset[..., None])[..., 0]) / scale
        lookup[parc_name] = soma_name
    root = expected_positions[:, 0]
    local = []
    for body in range(1, len(parc_names)):
        parent = int(character.get_parent_id(body))
        bind = Rotation.from_quat(character._local_rotation[body].cpu().numpy()).as_matrix()
        local.append(bind.T @ rotations[:, parent].swapaxes(-1, -2) @ rotations[:, body])
    local = np.stack(local, axis=1)
    root_quat = Rotation.from_matrix(rotations[:, 0]).as_quat().astype(np.float32)
    joint_quat = Rotation.from_matrix(local.reshape(-1, 3, 3)).as_quat().reshape(count, -1, 4).astype(np.float32)
    with torch.no_grad():
        fk_positions, fk_rotations = character.forward_kinematics(
            torch.from_numpy(root.astype(np.float32)), torch.from_numpy(root_quat), torch.from_numpy(joint_quat))
        # This observes the original character's joint-DOF contract. It is not
        # applied to the input or any future motion.
        legal_rotations = character.dof_to_rot(character.rot_to_dof(torch.from_numpy(joint_quat)))
    fk_positions = fk_positions.numpy()
    fk_rotation_matrices = Rotation.from_quat(fk_rotations.numpy().reshape(-1, 4)).as_matrix().reshape(rotations.shape)
    legal = Rotation.from_quat(legal_rotations.numpy().reshape(-1, 4)).as_matrix().reshape(local.shape)
    dof_residual = Rotation.from_matrix((local.swapaxes(-1, -2) @ legal).reshape(-1, 3, 3)).magnitude().reshape(count, -1)
    position_residual = np.linalg.norm((fk_positions - expected_positions) * scale, axis=-1)
    rotation_residual = Rotation.from_matrix((rotations.swapaxes(-1, -2) @ fk_rotation_matrices).reshape(-1, 3, 3)).magnitude()
    unmapped = [name for name in soma_names if name not in MAPPING]
    return {'root_pos_zup': root.astype(np.float32), 'root_rot_xyzw': root_quat, 'joint_rot_xyzw': joint_quat,
            'positions_expected_from_tasks': expected_positions, 'positions_actual_parc_fk': fk_positions,
            'metrics': {'task_position_residual_mean_soma_m': float(position_residual.mean()),
                        'task_position_residual_max_soma_m': float(position_residual.max()),
                        'task_position_residual_by_body_max_soma_m': dict(zip(parc_names, position_residual.max(0).tolist())),
                        'rotation_inverse_fk_max_deg': float(np.rad2deg(rotation_residual).max()),
                        'native_dof_incompatibility_max_deg': float(np.rad2deg(dof_residual).max()),
                        'native_dof_incompatibility_by_joint_max_deg': dict(zip(parc_names[1:], np.rad2deg(dof_residual).max(0).tolist())),
                        'unrepresented_soma_joints': unmapped,
                        'interpretation': 'Position error is simultaneous inverse-calibration task/FK inconsistency. '
                                          'It is not a measured solver roundtrip or a repaired SOMA pose.'}}


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rep = build_native_representation(args.stats)
    character = official_parc_character(args.parc)
    config, calibration = reference_calibration(character, rep.skeleton)
    scale = calibration['uniform_scale']
    checkpoint = torch.load(args.parc_checkpoint, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']
    sequence = {key: cfg.get(key) for key in ('sequence_duration', 'sequence_fps', 'num_prev_states',
                                            'target_type', 'features', 'relative_z_style')}
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)[:3]
    rows = []
    names = list(rep.skeleton.bone_order_names)
    contact_indices = rep.indices['foot_contacts'].tolist()
    for case in cases:
        with np.load(args.histories / ('feedback-' + case['label'] + '-histories.npz'), allow_pickle=False) as saved:
            histories = torch.from_numpy(saved['histories'].copy())
            offsets = saved['fixed_offsets'].copy()
        preparation = json.loads((args.stats.parent / case['row']['prepare_report_snapshot']).read_text())
        source_scale = preparation['calibration']['uniform_scale']
        if abs(source_scale - scale) > 1e-8:
            raise ValueError('Source morphology calibration differs from the bridge')
        for position in (0, 3, len(histories) - 1):
            offset = int(offsets[position])
            for mode, motion in (('source', case['world'][offset:offset + 4]), ('generated', histories[position])):
                pose = rep.inverse(motion[None], False, return_all=True, joint_positions_from='global_rot_data')
                decoded = inverse_calibrated_tasks(pose['posed_joints'][0].numpy(), pose['global_joint_rots'][0].numpy(),
                                                   names, character, config, scale)
                row = {'source_id': case['label'], 'history_mode': mode, 'position': position,
                       'source_frame_start': case['start_frame'] + offset, **decoded['metrics'],
                       'exact_bridge_approved': False, 'generated_continuation': False}
                arrays = {key: value for key, value in decoded.items() if key != 'metrics'}
                np.savez_compressed(args.output / f'{case["label"]}-{mode}-{position}.npz', **arrays,
                                    original_soma_world=motion.numpy())
                rows.append(row)
                print(json.dumps({key: row[key] for key in ('source_id', 'history_mode', 'position',
                                  'task_position_residual_mean_soma_m', 'task_position_residual_max_soma_m',
                                  'native_dof_incompatibility_max_deg')}), flush=True)
    result = {'rows': rows, 'parc_checkpoint_sha256': sha256(args.parc_checkpoint),
              'parc_configuration': sequence, 'parc_source_files': {name: sha256(args.parc / 'PARC' / name) for name in
                  ('motion_generator/gen_util.py', 'motion_generator/mdm.py', 'anim/kin_char_model.py')},
              'uniform_scale': scale, 'soma_history_frames': 4, 'soma_contact_feature_count': len(contact_indices),
              'parc_contact_body_count': len(character.get_body_names()), 'exact_bridge_approved': False,
              'actor_motion_modified': False, 'ik_or_projection_used': False,
              'scope': 'Read-only inverse task calibration and native FK/DOF compatibility. An arbitrary SOMA '
                       'history cannot be labelled as the same PARC body state merely by matching names. '
                       'SOMA predicted foot channels also do not supply all15 factual PARC body contacts.'}
    write_json(args.output / 'complete.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--histories', type=Path, required=True)
    parser.add_argument('--parc', type=Path, required=True)
    parser.add_argument('--parc-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
