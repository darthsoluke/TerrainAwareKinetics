"""Audit a raw PARC pilot against the released generator and quality gates.

This does not modify motions, vendor code, model weights, or training data.
Optional stride comparisons reuse the official generator without correction.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

import run_parc_local_terrain_pilot as pilot


def path_evidence(body_positions, body_names, goal, fps, radius=0.5):
    foot_ids = [body_names.index(name) for name in ('left_foot', 'right_foot')]
    feet = np.asarray(body_positions)[:, foot_ids].mean(axis=1)
    distances = np.linalg.norm(feet - np.asarray(goal), axis=-1)
    reached = np.flatnonzero(distances < radius)
    return {
        'criterion': 'official average-foot 3D distance, strictly below radius',
        'radius_m': radius,
        'ever_reached': bool(len(reached)),
        'first_reached_frame': int(reached[0]) if len(reached) else None,
        'first_reached_time_s': float(reached[0] / fps) if len(reached) else None,
        'closest_distance_m': float(distances.min()),
        'last_frame_distance_m': float(distances[-1]),
        'left_goal_region_after_reaching': bool(len(reached) and distances[-1] >= radius),
        'note': 'Reaching the waypoint is not proof of valid contacts or successful traversal.',
    }


def gate_evidence(contact, penetration, total, config):
    values = {'contact_loss': contact, 'pen_loss': penetration, 'total_loss': total}
    failed = [key for key, value in values.items()
              if not np.isfinite(value) or value >= config['max_' + key]]
    return {'scores': values, 'failed': failed, 'passes_released_filter': not failed,
            'thresholds': {key: config['max_' + key] for key in values},
            'scope': 'Released weighted sums, not millimeters or physics certification.'}


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-strides', action='store_true')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    repo = pilot.DEFAULT_REPO.resolve()
    sys.path.insert(0, str(repo))
    sys.modules.setdefault('parc', importlib.import_module('PARC'))
    os.environ['WANDB_MODE'] = 'disabled'
    torch.set_num_threads(8)

    from parc.util import file_io_helper, motion_util, terrain_util
    from parc.motion_generator import gen_util
    from parc.motion_synthesis.procgen import mdm_path

    config_path = repo / 'data/configs/parc_2_kin_gen_default.yaml'
    config = yaml.safe_load(config_path.read_text())
    checkpoint = torch.load(pilot.DEFAULT_CHECKPOINT, map_location='cpu', weights_only=False)
    model = pilot.load_official_model(repo, pilot.DEFAULT_CHECKPOINT, 'cpu')
    old_report = json.loads((run / 'pilot_report.json').read_text())
    report = {
        'source_run': str(run), 'torch_version': torch.__version__,
        'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checkpoint_sha256': pilot.sha256(pilot.DEFAULT_CHECKPOINT),
        'checkpoint_config': {k: checkpoint['cfg'].get(k) for k in
            ('d_model', 'num_layers', 'diffusion_timesteps', 'seq_len',
             'use_hf_collision_loss', 'EMA', 'motion_lib_file', 'output_dir')},
        'checkpoint_origin': 'Released smaller_model.zip; not verified as the paper final-stage model',
        'weights_used': 'model, not ema_model; same choice as official procgen loader default',
        'released_config': str(config_path),
        'released_config_sha256': pilot.sha256(config_path),
        'generation_differences': {
            'pilot_candidate_batch': 1, 'released_config_candidate_batch': config['mdm_path']['mdm_batch_size'],
            'paper_augmentation_candidate_batch': 64, 'paper_long_horizon_candidate_batch': 32,
            'pilot_stair_ddim_stride': 10, 'released_ddim_stride': config['mdm_gen']['ddim_stride'],
            'pilot_schedule': list(reversed(range(0, model._diffusion_timesteps, 10))),
            'released_stride_schedule_for_this_checkpoint': list(reversed(range(
                0, model._diffusion_timesteps, config['mdm_gen']['ddim_stride']))),
            'pilot_post_generation_stages': [],
            'released_stages': ['candidate quality filter', 'kinematic optimization',
                                'hesitation removal', 'physics tracking and recording'],
        },
        'existing_cases': [], 'stride_comparisons': [],
        'safety': {'training_started': False, 'motion_correction_applied': False,
                   'gpu_used': False, 'training_data_approved': False},
    }
    for case_name, diagnostic_name in [
        ('stairs_up', 'kinematic_world_terrain_diagnostics.npz'),
        ('stairs_down', 'kinematic_world_terrain_diagnostics.npz'),
        ('hand_contact_continuation', 'generated_kinematic_world_terrain_diagnostics.npz'),
    ]:
        case = run / case_name
        saved = file_io_helper.load_ms_file(str(case / 'generated_raw_kinematic.pkl'), device='cpu')
        frames = motion_util.MotionFrames.from_ms_motion_data(saved.motion_data, device='cpu')
        positions, rotations = model._kin_char_model.forward_kinematics(
            frames.root_pos, frames.root_rot, frames.joint_rot)
        names = model._kin_char_model.get_body_names()
        with np.load(case / diagnostic_name) as diag:
            fk_error = float(np.max(np.abs(positions.numpy() - diag['body_world_pos'])))
            quat_dot = np.abs((rotations.numpy() * diag['body_world_rot']).sum(axis=-1))
        assert fk_error < 1e-6 and np.max(np.abs(quat_dot - 1)) < 1e-5
        row = {'case': case_name, 'recomputed_fk_max_error_m': fk_error,
               'recomputed_quaternion_dot_max_error': float(np.max(np.abs(quat_dot - 1)))}
        if case_name.startswith('stairs'):
            old = next(x for x in old_report['cases'] if x['case'] == case_name)
            with np.load(case / 'terrain_and_route.npz') as route:
                row['path'] = path_evidence(positions.numpy(), names, route['path_nodes'][-1, :3], 30)
            row['old_failure_reason'] = old['failure_reason']
            row['released_filter'] = gate_evidence(old['official_heuristic_contact_loss'],
                old['official_heuristic_penetration_loss'], old['official_heuristic_total_loss'], config)
        report['existing_cases'].append(row)

    # Reproduce the expanded-view error that led the pilot to disable batching.
    reset_seed(old_report['seed'])
    terrain, nodes = pilot.build_stairs(terrain_util, 'up')
    settings = gen_util.MDMGenSettings()
    first = mdm_path.gen_mdm_motion_at_path_start(nodes, terrain, model._kin_char_model,
                                                model, settings, None, 2)
    try:
        gen_util.gen_mdm_motion(nodes[-1:].expand(2, -1), first.get_slice(slice(5, 7)),
                               terrain, model, model._kin_char_model, settings)
    except RuntimeError as error:
        report['batch_two_reproduction'] = {'status': 'failed', 'error': str(error)}
    else:
        report['batch_two_reproduction'] = {'status': 'ran_without_error'}

    if args.compare_strides:
        for direction in ('up', 'down'):
            for stride in (10, config['mdm_gen']['ddim_stride']):
                reset_seed(old_report['seed'])
                terrain, nodes = pilot.build_stairs(terrain_util, direction)
                path_settings = mdm_path.MDMPathSettings()
                path_settings.mdm_batch_size = path_settings.top_k = 1
                path_settings.max_motion_length = 5.0
                settings = gen_util.MDMGenSettings()
                settings.ddim_stride = stride
                candidates, _, info = mdm_path.generate_frames_until_end_of_path(
                    nodes, terrain, model._kin_char_model, model, None, path_settings,
                    settings, add_noise_to_loss=False, verbose=False, slice_terrain=False)
                frames = candidates[0].get_idx(0)
                positions, _ = model._kin_char_model.forward_kinematics(
                    frames.root_pos, frames.root_rot, frames.joint_rot)
                row = {'direction': direction, 'seed': old_report['seed'], 'stride': stride,
                       'candidate_batch': 1, 'frames': len(positions),
                       'path': path_evidence(positions.numpy(), names, nodes[-1, :3].numpy(), 30),
                       'released_filter': gate_evidence(float(info['contact_losses']),
                           float(info['pen_losses']), float(info['losses']), config)}
                report['stride_comparisons'].append(row)
                print(json.dumps(row), flush=True)
    report['interpretation'] = [
        'A single-seed stride comparison cannot establish population-level quality.',
        'Saved stair clips entered the official goal region; terminal distance is not a timeout proof.',
        'Upstream all(done) break precedes completion bookkeeping, so batch-one completion scores can be wrong.',
        'The released checkpoint and raw pilot pipeline are not a reproduction of the paper long-horizon demo.',
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
