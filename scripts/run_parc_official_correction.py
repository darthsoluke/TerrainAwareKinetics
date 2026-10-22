"""Run released PARC selection and offline kinematic correction, not physics.

Uses the public small checkpoint and released procedural-generation config.
All candidate scores, raw clips, and optimization outputs are retained. A
kinematically corrected clip is never implicitly approved for MB training.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

import run_parc_local_terrain_pilot as pilot
from review_parc_pilot_quality import gate_evidence, path_evidence


def settings_from_config(cls, values):
    settings = cls()
    for key, value in values.items():
        setattr(settings, key, value)
    return settings


def candidate_rows(info, config):
    arrays = {key: torch.as_tensor(value).reshape(-1).detach().cpu().tolist()
              for key, value in info.items()}
    rows = []
    for rank in range(len(arrays['losses'])):
        row = gate_evidence(arrays['contact_losses'][rank], arrays['pen_losses'][rank],
                            arrays['losses'][rank], config)
        row.update(rank=rank, candidate_id=arrays['candidate_ids'][rank],
                   completed=arrays['completed'][rank], frames=arrays['frame_counts'][rank])
        row['eligible'] = row['passes_released_filter'] and row['completed'] is not False
        rows.append(row)
    return rows


def accumulate_candidates(pool, candidates, terrains, rows, seed, limit):
    """Keep provenance and references together across bounded retry batches."""
    for row in rows:
        if row['eligible']:
            rank = row['rank']
            pool.append({'row': dict(row, generation_seed=seed),
                         'frames': candidates[rank], 'terrain': terrains[rank]})
    pool.sort(key=lambda item: item['row']['scores']['total_loss'])
    del pool[limit:]
    return pool


def completed_run_status(cases, requested_cases, variants):
    expected = {name if i == 0 else name + '_variant_' + str(i)
                for name in requested_cases for i in range(variants)}
    complete = {entry['case'] for entry in cases
                if entry['status'] == 'kinematic_correction_complete_physics_pending'
                and all(stage in entry['stages'] for stage in ('raw', 'optimized', 'final'))}
    return ('kinematic_stage_finished_physics_pending' if expected and expected <= complete
            else 'partial_kinematic_stage_physics_pending')


@torch.no_grad()
def score_candidate_batches(frames, terrain, char, points, config, batch_size=8):
    """Bound SDF workspace without changing candidates or per-candidate scores."""
    from parc.motion_synthesis.procgen import mdm_path

    if batch_size < 1:
        raise ValueError('score batch size must be positive')
    chunks = []
    for start in range(0, frames.root_pos.shape[0], batch_size):
        chunks.append(mdm_path.compute_motion_loss(frames.get_idx(slice(start, start + batch_size)),
            None, terrain, char, points, config['mdm_path']['w_contact'],
            config['mdm_path']['w_pen'], 1., verbose=False))
    return {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def stage_metrics(frames, terrain, model, points, config, nodes, reference_contacts):
    from parc.motion_synthesis.procgen import mdm_path
    from parc.util import terrain_util, torch_util
    from generate_parc_long_stairs import chunked_motion_score

    char = model._kin_char_model
    positions, rotations = char.forward_kinematics(frames.root_pos, frames.root_rot, frames.joint_rot)
    scores = chunked_motion_score(mdm_path.compute_motion_loss, frames.unsqueeze(0), nodes,
        terrain, char, points, config['mdm_path']['w_contact'], config['mdm_path']['w_pen'],
        1., verbose=False, frame_chunk=4)
    all_penetration = []
    for body_id, local_points in enumerate(points):
        world = torch_util.quat_rotate(rotations[:, body_id, None, :], local_points[None])
        world = world + positions[:, body_id, None, :]
        flat = world.reshape(-1, 3)
        for begin in range(0, len(flat), 256):
            sdf = terrain_util.points_hf_sdf(flat[None, begin:begin + 256], terrain.hf[None],
                terrain.min_point[None], terrain.dxdy, base_z=float(terrain.hf.min()) - 10., inverted=True)
            all_penetration.append((-sdf).clamp_min(0).flatten())
    penetration = torch.cat(all_penetration)
    jerk = torch.diff(positions, n=3, dim=0).norm(dim=-1) * model._sequence_fps ** 3
    contact_mask = (reference_contacts[:-1] > .9) & (reference_contacts[1:] > .9)
    speeds = torch.diff(positions, dim=0).norm(dim=-1) * model._sequence_fps
    selected_speed = speeds[contact_mask]
    metrics = {
        'frames': len(positions),
        'weighted_contact_score': float(scores['contact_loss']),
        'weighted_penetration_score': float(scores['pen_loss']),
        'sampled_surface_penetration_max_m': float(penetration.max()),
        'sampled_surface_penetration_mean_m': float(penetration.mean()),
        'body_jerk_p95_mps3': float(torch.quantile(jerk.flatten(), .95)) if jerk.numel() else None,
        'body_jerk_max_mps3': float(jerk.max()) if jerk.numel() else None,
        'reference_contact_body_speed_mean_mps': float(selected_speed.mean()) if selected_speed.numel() else None,
        'reference_contact_intervals': int(contact_mask.sum()),
        'contact_mask_source': 'fixed raw-candidate contact probabilities above .9 on both interval endpoints',
        'metric_limits': 'Sampled official geometry only. Body-origin speed is not certified foot slip or force.',
    }
    if nodes is not None:
        metrics['path'] = path_evidence(positions.cpu().numpy(), char.get_body_names(),
            nodes[-1, :3].cpu().numpy(), model._sequence_fps)
    return metrics


def save_stage(directory, name, frames, terrain, model, nodes, config, points,
               reference_contacts, constraints=None):
    from parc.util import file_io_helper

    motion_file = directory / (name + '.pkl')
    diag_file = directory / (name + '_fk.npz')
    file_io_helper.save_ms_file(str(motion_file), motion_frames=frames, fps=model._sequence_fps,
        loop_mode='CLAMP', terrain=terrain, hf_mask_inds=None, body_constraints=constraints,
        min_point_offset=None, path_nodes=nodes,
        extra_misc_data={'stage': name, 'physics_corrected': False,
                         'motion_origin': getattr(model, 'motion_origin', 'official PARC generator; small released checkpoint'),
                         'training_approved': False,
                         'optimization': None if name == 'raw' else config['opt']})
    pilot.save_fk_world_terrain_diagnostics(diag_file, frames, terrain, model, np, torch)
    return {'motion_file': str(motion_file), 'diagnostics_file': str(diag_file),
            'sha256': pilot.sha256(motion_file),
            'metrics': stage_metrics(frames, terrain, model, points, config, nodes, reference_contacts)}


def optimize_case(case_name, variant, frames, terrain, nodes, model, config, output,
                  points, row, report, report_path):
    from parc.motion_synthesis.motion_opt import motion_optimization as optimization
    from parc.util import motion_edit_lib

    directory = output / case_name / ('variant_' + str(variant))
    directory.mkdir(parents=True)
    entry = {'case': case_name if variant == 0 else case_name + '_variant_' + str(variant),
             'directory': str(directory), 'status': 'optimizing', 'stages': {},
             'selected_candidate': row, 'physics_corrected': False, 'training_approved': False}
    report['cases'].append(entry)
    started = time.time()
    entry['stages']['raw'] = save_stage(directory, 'raw', frames, terrain, model, nodes,
                                      config, points, frames.contacts)
    write_report(report_path, report)
    options = config['opt']
    char = model._kin_char_model
    terrain.set_device(model._device)
    frames.set_device(model._device)
    if options['auto_compute_body_constraints']:
        constraints = optimization.compute_approx_body_constraints(frames.root_pos,
            frames.root_rot, frames.joint_rot, frames.contacts, char, terrain)
    else:
        constraints = None
    arguments = {key: options[key] for key in ('num_iters', 'step_size', 'w_root_pos',
        'w_root_rot', 'w_joint_rot', 'w_smoothness', 'w_penetration', 'w_contact',
        'w_sliding', 'w_body_constraints', 'w_jerk', 'max_jerk')}
    optimized = optimization.motion_contact_optimization(src_motion_frames=frames,
        body_points=points, terrain=terrain, char_model=char, body_constraints=constraints,
        exp_name=entry['case'], use_wandb=False, log_file=str(directory / 'optimization.log'), **arguments)
    entry['stages']['optimized'] = save_stage(directory, 'optimized', optimized, terrain,
        model, nodes, config, points, frames.contacts, constraints)
    # Keep a separate pre-removal output for frame-aligned comparisons.
    retained = np.arange(len(optimized.root_pos))
    if config['remove_hesitation']:
        original_slice = optimized.get_slice

        def observe_slice(indices):
            nonlocal retained
            retained = np.arange(len(optimized.root_pos))[indices]
            return original_slice(indices)

        optimized.get_slice = observe_slice
        try:
            final = motion_edit_lib.remove_hesitation_frames(optimized, char)
        finally:
            optimized.get_slice = original_slice
    else:
        final = optimized
    np.save(directory / 'retained_frame_indices.npy', retained)
    entry['stages']['final'] = save_stage(directory, 'final', final, terrain, model,
        nodes, config, points, final.contacts)
    entry['frames_removed_by_official_hesitation_step'] = len(optimized.root_pos) - len(final.root_pos)
    entry['status'] = 'kinematic_correction_complete_physics_pending'
    entry['elapsed_seconds'] = time.time() - started
    write_report(report_path, report)
    print(json.dumps({'finished': entry['case'], 'seconds': entry['elapsed_seconds'],
                      'before': entry['stages']['raw']['metrics'],
                      'after': entry['stages']['optimized']['metrics']}), flush=True)


def generate_hand(model, config, source_run, points):
    from parc.util import file_io_helper, motion_util, terrain_util
    from parc.motion_generator import gen_util
    from parc.motion_synthesis.procgen import mdm_path

    source_dir = source_run / 'hand_contact_continuation'
    old = json.loads((source_dir / 'result.json').read_text())
    data = file_io_helper.load_ms_file(str(source_dir / 'source_climb_motion_and_terrain.pkl'), device=model._device)
    frames = motion_util.MotionFrames.from_ms_motion_data(data.motion_data, device=model._device)
    terrain = terrain_util.SubTerrain.from_ms_terrain_data(data.terrain_data, device=model._device)
    current = old['source_frame_index']
    prev = frames.get_slice(slice(current - model._num_prev_states + 1, current + 1)).unsqueeze(0)
    batch = config['mdm_path']['mdm_batch_size']
    prev = prev.expand_first_dim(batch)
    target = prev.root_pos[:, -1].clone()
    target[:, :2] += torch.tensor(old['scripted_intent_world_xy'], device=model._device) * 2.
    generated = gen_util.gen_mdm_motion(target, prev, terrain, model, model._kin_char_model,
        settings_from_config(gen_util.MDMGenSettings, config['mdm_gen']))
    losses = score_candidate_batches(generated, terrain, model._kin_char_model, points, config)
    order = torch.argsort(losses['total_loss'])
    info = {'losses': losses['total_loss'][order], 'contact_losses': losses['contact_loss'][order],
            'pen_losses': losses['pen_loss'][order], 'candidate_ids': order,
            'completed': torch.ones(batch, dtype=torch.bool, device=model._device),
            'frame_counts': torch.full((batch,), model._seq_len, device=model._device)}
    candidates = [generated.get_idx(int(idx)).unsqueeze(0) for idx in order]
    return candidates, [terrain] * batch, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=pilot.ROOT / 'out/parc_official_correction_20261008')
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda:0'))
    parser.add_argument('--cases', nargs='+', choices=('stairs_up', 'stairs_down', 'hand_contact_continuation'),
                        default=['stairs_up', 'stairs_down', 'hand_contact_continuation'])
    parser.add_argument('--max-attempts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error('--max-attempts must be positive')
    repo = pilot.DEFAULT_REPO.resolve()
    sys.path.insert(0, str(repo))
    sys.modules.setdefault('parc', importlib.import_module('PARC'))
    os.environ['WANDB_MODE'] = 'disabled'
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.35)
    from parc.motion_synthesis.procgen import mdm_path
    from parc.motion_generator import gen_util
    from parc.util import geom_util, terrain_util

    config_path = repo / 'data/configs/parc_2_kin_gen_default.yaml'
    config = yaml.safe_load(config_path.read_text())
    output = pilot.unique_run_directory(args.output_root.resolve())
    report_path = output / 'correction_report.json'
    patch = subprocess.check_output(['git', '-C', str(repo), 'diff', '--',
        'PARC/motion_generator/gen_util.py', 'PARC/motion_synthesis/procgen/mdm_path.py'], text=True)
    (output / 'vendor_runtime.patch').write_bytes(patch.encode('utf-8'))
    runner_source = Path(__file__).read_bytes()
    (output / 'executed_runner.py').write_bytes(runner_source)
    seed_all(args.seed)
    model = pilot.load_official_model(repo, pilot.DEFAULT_CHECKPOINT, args.device)
    sample_cfg = dict(config['opt']['char_point_samples'])
    sample_cfg['capsule_num_sphere_subdivisons'] = sample_cfg.pop('capsule_num_sphere_subdivisions')
    points = geom_util.get_char_point_samples(model._kin_char_model, **sample_cfg)
    report = {
        'status': 'running', 'device': args.device, 'source_run': str(args.source_run.resolve()),
        'runner_sha256': hashlib.sha256(runner_source).hexdigest(),
        'torch_version': torch.__version__,
        'checkpoint': str(pilot.DEFAULT_CHECKPOINT), 'checkpoint_sha256': pilot.sha256(pilot.DEFAULT_CHECKPOINT),
        'checkpoint_scope': 'Released small model, not established as paper final-stage generator',
        'config_source': str(config_path), 'config_sha256': pilot.sha256(config_path),
        'released_config': config, 'seed': args.seed,
        'hand_candidate_score_batch_size': 8,
        'upstream_commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'runtime_patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
        'physics_corrected': False, 'training_approved': False,
        'physics_stage': {'status': 'not_run', 'reason': 'Matching released tracker checkpoint and supported Isaac Gym runtime are not available locally.'},
        'protocol': 'Official batch/stride/filter/optimization values. Fixed pilot terrains and two historical conditioning frames. No UE changes.',
        'generation_batches': [], 'cases': [],
    }
    write_report(report_path, report)
    try:
        for case_name in args.cases:
            pool = []
            for attempt in range(args.max_attempts):
                seed = args.seed + attempt
                seed_all(seed)
                nodes = None
                started = time.time()
                if case_name.startswith('stairs'):
                    terrain, nodes = pilot.build_stairs(terrain_util, case_name.removeprefix('stairs_'))
                    candidates, terrains, info = mdm_path.generate_frames_until_end_of_path(
                        nodes, terrain, model._kin_char_model, model, None,
                        settings_from_config(mdm_path.MDMPathSettings, config['mdm_path']),
                        settings_from_config(gen_util.MDMGenSettings, config['mdm_gen']),
                        add_noise_to_loss=False, verbose=False, slice_terrain=False)
                else:
                    candidates, terrains, info = generate_hand(model, config, args.source_run.resolve(), points)
                if 'completed' not in info:
                    raise RuntimeError('The PARC completion-state compatibility patch is missing')
                rows = candidate_rows(info, config)
                if nodes is None:
                    for row in rows:
                        row['completed'] = None
                        row['eligible'] = row['passes_released_filter']
                        row['scope'] += ' Local half-second continuation only; climbing success is not assessed.'
                valid = [row for row in rows if row['eligible']]
                accumulate_candidates(pool, candidates, terrains, rows, seed,
                                      config['mdm_path']['top_k'])
                report['generation_batches'].append({'case': case_name, 'seed': seed,
                    'seconds': time.time() - started, 'candidates': rows, 'accepted_count': len(valid),
                    'retained_across_attempts': len(pool)})
                write_report(report_path, report)
                print(json.dumps({'case': case_name, 'attempt': attempt, 'valid_candidates': len(valid),
                                  'total_candidates': len(rows)}), flush=True)
                if len(pool) >= config['mdm_path']['top_k']:
                    break
            else:
                report['cases'].append({'case': case_name, 'directory': str(output / case_name),
                    'status': 'rejected_no_sufficient_candidates', 'stages': {}, 'training_approved': False})
                write_report(report_path, report)
                continue
            for variant, selected in enumerate(pool):
                optimize_case(case_name, variant, selected['frames'].get_idx(0), selected['terrain'], nodes,
                    model, config, output, points, selected['row'], report, report_path)
        report['status'] = completed_run_status(report['cases'], args.cases, config['mdm_path']['top_k'])
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if args.device.startswith('cuda'):
            report['peak_allocated_gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
        write_report(report_path, report)
        print(str(report_path), flush=True)


if __name__ == '__main__':
    main()
