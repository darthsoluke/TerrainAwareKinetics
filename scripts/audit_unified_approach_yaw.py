"""Same factual stair-approach history under rigid world rotations; no training."""

import argparse
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_quarter_turn import rotate_heightfield, rotate_motion, quarter_matrix
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.candidate) != 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979':
        raise ValueError('Audit the unchanged live candidate first')
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    saved = torch.load(args.candidate, map_location='cpu', weights_only=False)
    controller = UnifiedTerrainController(inference, bundle)
    controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
    with np.load(args.scene, allow_pickle=False) as values:
        scene = {key: torch.from_numpy(values[key].copy()) for key in
                 ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    with np.load(args.played, allow_pickle=False) as values:
        played = torch.from_numpy(values['global_features'].copy()).to(args.device)[None]
    rep = inference.global_motion_rep
    rows = []
    scenes = [rotate_heightfield(scene, k) for k in range(4)]
    for frame in (64, 72, 80, 88):
        history = reencode_played_history(played[:, frame - 4:frame], rep)
        reference_sdf = reference_points = reference_normals = reference_joints = None
        for turns, changed_scene in enumerate(scenes):
            changed = rotate_motion(rep, history, turns)
            matrix = quarter_matrix(turns, history)
            velocity = history.new_tensor([[0., 1.6]]) @ matrix.T
            angle = history.new_tensor(turns * torch.pi / 2)
            command = torch.cat((velocity, torch.stack((angle.cos(), angle.sin()))[None]), -1)
            output = controller(changed, command, changed_scene)
            volume = controller.volume_observation
            root = rep.unnormalize(output['root_output']['pred_global_root_values'])
            origin = vertical_origin(inference, changed, output['observation'])
            joint = rep.inverse(changed, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
            if turns == 0:
                reference_sdf = volume['distance'].clone()
                reference_points = volume['points'].clone()
                reference_normals = volume['normal'].clone()
                reference_joints = joint.clone()
            expected_points = reference_points.clone()
            expected_points[..., [0, 2]] = reference_points[..., [0, 2]] @ matrix.T
            expected_joints = reference_joints.clone()
            expected_joints[..., [0, 2]] = reference_joints[..., [0, 2]] @ matrix.T
            torch.testing.assert_close(volume['points'], expected_points, atol=6e-5, rtol=0)
            torch.testing.assert_close(joint, expected_joints, atol=6e-5, rtol=0)
            torch.testing.assert_close(volume['distance'], reference_sdf, atol=6e-5, rtol=0)
            row = dict(frame=frame, yaw_degrees=turns * 90,
                observed_y_m=float(changed[0, -1, 1]), origin_y_m=float(origin.item()),
                max_root_y_m=float((root[0, 4:28, 1] + origin[0]).max()),
                end_root_y_m=float(root[0, 27, 1] + origin[0]),
                max_decoded_y_m=float(output['world'][0, 4:28, 1].max()),
                point_covariance_error_m=float((volume['points'] - expected_points).abs().max()),
                joint_covariance_error_m=float((joint - expected_joints).abs().max()),
                sdf_invariance_error_m=float((volume['distance'] - reference_sdf).abs().max()))
            rows.append(row)
            write_json(args.output / 'progress.json', dict(rows=rows))
    write_json(args.output / 'complete.json', dict(rows=rows, optimizer_updates=0, live_changed=False,
        candidate_sha256=sha256(args.candidate), scene_sha256=sha256(args.scene),
        played_sha256=sha256(args.played), no_history_or_root_output_correction=True,
        scope='Same actual histories, all geometry/body/commands rotated together; source futures not supplied.',
        quality_approved=False))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'candidate', 'scene', 'played', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'failure.json', dict(error=repr(error)))
        raise
