"""Apply released PARC selection/optimization to immutable long-course candidates.

Inputs are a job manifest, not inferred from filenames. This stage does not
approve data for training. Physical recording and retarget acceptance follow.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from run_parc_official_correction import optimize_case, write_report
from run_parc_local_terrain_pilot import sha256


def selection(row, config):
    scores = (row['contact_loss'], row['penetration_loss'], row['total_loss'])
    if not all(np.isfinite(float(v)) for v in scores):
        raise ValueError('Candidate has nonfinite selection scores')
    return (row['completed'] is True
            and scores[0] < config['max_contact_loss']
            and scores[1] < config['max_pen_loss']
            and scores[2] < config['max_total_loss'])


def temporal_metrics(frames, char, fps):
    from parc.util import torch_util

    positions, _ = char.forward_kinematics(frames.root_pos, frames.root_rot, frames.joint_rot)
    angle = torch_util.quat_diff_angle(frames.joint_rot[1:], frames.joint_rot[:-1]).abs() * (180. / np.pi)
    jerk = torch.diff(positions, n=3, dim=0).norm(dim=-1) * fps ** 3
    return {'joint_step_max_deg': float(angle.max()),
            'joint_step_p99_deg': float(torch.quantile(angle.flatten(), .99)),
            'body_jerk_p99_mps3': float(torch.quantile(jerk.flatten(), .99)),
            'body_jerk_max_mps3': float(jerk.max()),
            'frame_count': len(frames.root_pos), 'fps': fps}


def validate_intent(row, frame_count=None):
    if 'intent' not in row:
        return None
    path = Path(row['intent'])
    if sha256(path) != row['intent_sha256']:
        raise ValueError('Source intent changed')
    with np.load(path, allow_pickle=False) as stored:
        commands = stored['command_world_xy']
        fps = float(stored['fps'])
    if (commands.ndim != 2 or commands.shape[1] != 2 or not np.isfinite(commands).all()
            or fps != 30 or (frame_count is not None and len(commands) != frame_count)):
        raise ValueError('Intent frame/axis contract mismatch')
    return commands


def run(repo, jobs_path, output, device, resume=False):
    repo, jobs_path, output = repo.resolve(), jobs_path.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=resume)
    sys.path.insert(0, str(repo))
    sys.modules.setdefault('parc', importlib.import_module('PARC'))
    if int(np.__version__.split('.')[0]) < 2:
        sys.modules.setdefault('numpy._core', np.core)
        sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
        sys.modules.setdefault('numpy._core.numeric', np.core.numeric)
    from parc.anim.kin_char_model import KinCharModel
    from parc.util import file_io_helper, geom_util, motion_util, terrain_util

    os.environ['WANDB_MODE'] = 'disabled'
    torch.set_num_threads(4)
    torch.manual_seed(20261009)
    np.random.seed(20261009)
    config_path = repo / 'data/configs/parc_2_kin_gen_default.yaml'
    config = yaml.safe_load(config_path.read_text())
    char = KinCharModel(torch.device(device))
    char.load_char_file(str(repo / 'data/assets/humanoid.xml'))
    sampling = dict(config['opt']['char_point_samples'])
    sampling['capsule_num_sphere_subdivisons'] = sampling.pop('capsule_num_sphere_subdivisions')
    points = geom_util.get_char_point_samples(char, **sampling)
    model = SimpleNamespace(_kin_char_model=char, _device=device, _sequence_fps=30,
                            motion_origin='official PARC iteration-3 EMA generator')
    jobs = json.loads(jobs_path.read_text())['jobs']
    names = [row['name'] for row in jobs]
    if not jobs or len(names) != len(set(names)) or any(not name.replace('_', '').isalnum() for name in names):
        raise ValueError('Empty jobs or noncanonical/duplicate names')
    by_name = {row['name']: row for row in jobs}
    for row in jobs:
        source = Path(row['motion'])
        if not source.is_absolute():
            source = jobs_path.parent / source
        if sha256(source) != row['sha256']:
            raise ValueError('Source hash mismatch: ' + str(source))
        validate_intent(row)
    report = {'status': 'running', 'config_sha256': sha256(config_path), 'released_config': config,
              'runner_sha256': sha256(Path(__file__)), 'jobs_sha256': sha256(jobs_path),
              'training_approved': False, 'cases': [], 'rejected': []}
    report_path = output / 'correction_report.json'
    if resume and report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous['jobs_sha256'] != report['jobs_sha256'] or previous['config_sha256'] != report['config_sha256']:
            raise ValueError('Cannot resume changed inputs or official optimization configuration')
        report['resume_history'] = previous.get('resume_history', []) + [
            {'previous_runner_sha256': previous['runner_sha256'], 'previous_status': previous['status'],
             'reason': 'Equivalent chunked diagnostic SDF workspace; optimizer unchanged'}]
        for case in previous['cases']:
            if case['status'] == 'kinematic_correction_complete_physics_pending':
                for entry in case['stages'].values():
                    if sha256(Path(entry['motion_file'])) != entry['sha256']:
                        raise ValueError('Completed stage changed before resume')
                row = by_name[case['case']]
                if case.get('source_sha256') != row['sha256']:
                    raise ValueError('Completed case does not bind original source')
                commands = validate_intent(row)
                if commands is not None:
                    if 'intent' not in case or sha256(Path(case['intent'])) != case.get('intent_sha256'):
                        raise ValueError('Completed case has no verified aligned intent')
                    with np.load(case['intent'], allow_pickle=False) as stored:
                        kept = stored['source_frame_indices']
                        if not np.array_equal(stored['command_world_xy'], commands[kept]):
                            raise ValueError('Completed intent mapping changed')
                report['cases'].append(case)
            else:
                directory = output / case['case']
                if directory.exists():
                    directory.rename(output / ('failed_' + case['case'] + '_' + str(time.time_ns())))
    write_report(report_path, report)
    try:
        for row in jobs:
            if any(case['case'] == row['name'] for case in report['cases']):
                continue
            if not selection(row, config):
                report['rejected'].append(dict(row, reason='released_filter_or_incomplete'))
                write_report(report_path, report)
                continue
            source = Path(row['motion'])
            if not source.is_absolute():
                source = jobs_path.parent / source
            if sha256(source) != row['sha256']:
                raise ValueError('Source hash mismatch: ' + str(source))
            data = file_io_helper.load_ms_file(str(source), device=device)
            if data.motion_data.fps != 30:
                raise ValueError('Official optimization constants require 30 FPS')
            frames = motion_util.MotionFrames.from_ms_motion_data(data.motion_data, device=device)
            terrain = terrain_util.SubTerrain.from_ms_terrain_data(data.terrain_data, device=device)
            if len(frames.root_pos) < 5:
                raise ValueError('Too few frames for jerk measurement')
            commands = validate_intent(row, len(frames.root_pos))
            before = temporal_metrics(frames, char, 30)
            nodes = getattr(data, 'path_nodes', None)
            if nodes is not None:
                nodes = torch.as_tensor(nodes, device=device, dtype=torch.float32)
            optimize_case(row['name'], 0, frames, terrain, nodes, model, config, output,
                          points, row, report, report_path)
            case = report['cases'][-1]
            case['status'] = 'finalizing_provenance'
            write_report(report_path, report)
            final_path = Path(case['stages']['final']['motion_file'])
            final_data = file_io_helper.load_ms_file(str(final_path), device=device)
            final = motion_util.MotionFrames.from_ms_motion_data(final_data.motion_data, device=device)
            case['temporal_before'] = before
            case['temporal_after'] = temporal_metrics(final, char, 30)
            case['source_sha256'] = row['sha256']
            case['source_group'] = row.get('source_group', row['name'])
            if 'intent' in row:
                kept = np.load(final_path.parent / 'retained_frame_indices.npy', allow_pickle=False)
                intent_output = final_path.parent / 'intent.npz'
                np.savez_compressed(intent_output, command_world_xy=commands[kept],
                                    source_frame_indices=kept, fps=30)
                case['intent'] = str(intent_output)
                case['intent_sha256'] = sha256(intent_output)
            case['status'] = 'kinematic_correction_complete_physics_pending'
            write_report(report_path, report)
        report['status'] = ('correction_finished_physics_pending' if len(report['cases']) == len(jobs)
                            else 'partial_correction_finished_physics_pending')
    except Exception as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        write_report(report_path, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.repo, args.jobs, args.output, args.device, args.resume)
