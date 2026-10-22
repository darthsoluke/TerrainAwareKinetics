"""Read-only entry supervision and SDF-only counterfactuals; never publish them."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.unified_terrain import UnifiedTerrainController, SCHEMA
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.audit_unified_stair_anticipation import fresh_trace
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def source_summary(window, scene, entry_floor):
    gap, known = heightfield_gap(window[:, :3], scene)
    if not known.all():
        raise ValueError('Source window contains unknown ground')
    ground = window[:, 1] - gap
    delta = ground[4:28] - ground[3]
    level = bool(ground[:4].max() - ground[:4].min() < .03)
    changes = bool(delta.abs().max() > .08)
    entry = bool((ground[:4] - entry_floor).abs().max() < .03)
    return dict(history_ground_range_m=float(ground[:4].max() - ground[:4].min()),
        history_on_entry_floor=entry,
        history_root_vertical_delta_m=float(window[3, 1] - window[0, 1]),
        flat_entry_before_height_change=level and changes and entry,
        future_ground_change_max_m=float(delta.abs().max()),
        future_source_root_rise_m=float((window[4:28, 1] - window[3, 1]).max()),
        future_source_root_drop_m=float((window[3, 1] - window[4:28, 1]).max()),
        flat_history_before_height_change=level and changes)


@torch.no_grad()
def inspect(controller, history, command, scene, reference=None):
    output, root, trace = fresh_trace(controller, history, command, scene)
    original_world = output['world'].clone()
    observation = controller.volume_observation
    memory = controller.memory.clone()
    floor = trace['origin_y_m']
    plane_distance = observation['points'][..., 1] - floor
    valid = observation['valid']
    difference = (observation['distance'] - plane_distance).abs()
    intervention = dict(observation)
    features = observation['features'].clone()
    features[..., 6] = torch.asinh(plane_distance)
    features[..., 7:10] = 0
    features[..., 8] = 1
    intervention['features'] = torch.where(valid[..., None], features, 0)
    intervention['distance'] = torch.where(valid, plane_distance, 0)
    intervention['normal'] = torch.zeros_like(observation['normal'])
    intervention['normal'][..., 1] = valid.to(plane_distance)

    # Keep actual native terrain/origin/poses, query locations and validity fixed.
    # Only the new SDF encoder sees the plane counterfactual.
    hook = controller.adaptation['encoder'].register_forward_pre_hook(
        lambda _module, _args: (intervention,))
    try:
        _, flat_root, flat_trace = fresh_trace(controller, history, command, scene)
        flat_memory = controller.memory.clone()
    finally:
        hook.remove()
    flat_scene = dict(scene, heightfield_y_m=torch.full_like(scene['heightfield_y_m'], floor))
    _, all_flat_root, all_flat_trace = fresh_trace(controller, history, command, flat_scene)
    replay, replay_root, _ = fresh_trace(controller, history, command, scene)
    torch.testing.assert_close(replay['world'], original_world, atol=0, rtol=0)
    torch.testing.assert_close(replay_root, root, atol=0, rtol=0)
    if trace['origin_y_m'] != flat_trace['origin_y_m']:
        raise ValueError('SDF-only intervention changed the native vertical origin')
    prediction = root[0, 4:28, 1]
    result = dict(trace=trace, command_vxz_mps=command[0, :2].tolist(),
        history_root_y_m=float(history[0, 3, 1]),
        predicted_root_rise_m=float((prediction - history[0, 3, 1]).max()),
        predicted_root_drop_m=float((history[0, 3, 1] - prediction).max()),
        sdf_plane_delta_max_m=float(difference[valid].max()),
        sdf_plane_delta_known_fraction=float((difference[valid] > .02).float().mean()),
        memory_plane_delta_rms=float((memory - flat_memory).square().mean().sqrt()),
        memory_plane_delta_max=float((memory - flat_memory).abs().max()),
        root_y_sdf_effect_max_m=float((prediction - flat_root[0, 4:28, 1]).abs().max()),
        root_y_sdf_effect_mean_m=float((prediction - flat_root[0, 4:28, 1]).mean()),
        root_y_all_terrain_effect_max_m=float((prediction - all_flat_root[0, 4:28, 1]).abs().max()),
        all_flat_origin_delta_m=all_flat_trace['origin_y_m'] - floor,
        replay_error=0., not_published=True)
    if reference is not None:
        result['root_height_rms_m'] = float((prediction - reference[4:28, 1]).square().mean().sqrt())
    return result


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle).eval().requires_grad_(False)
    readout = controller.root._conv_output.model[-1]
    native_readout = {key: value.clone() for key, value in readout.state_dict().items()}
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    dataset = TerrainLearningDataset(args.dataset, inference.global_motion_rep, 'train')
    by_id = {row['source_id']: i for i, row in enumerate(dataset.records)}
    archived = json.loads(args.training_report.read_text())
    sources, coverage, rows = [], [], []
    for case in cases:
        if case['native_split'] != 'train':
            continue
        original = dataset.full_reference_item(by_id[case['row']['source_id']])
        old_start = case['start_frame']
        torch.testing.assert_close(original['world'][old_start:old_start + len(case['world'])],
                                   case['world'], atol=0, rtol=0)
        new_start = old_start - 24
        if new_start < 0:
            raise ValueError('Require the previously audited24real prefix frames')
        extended = original['world'][new_start:old_start + len(case['world'])]
        entry_gap, entry_known = heightfield_gap(extended[:4, :3], case['scene'])
        if not entry_known.all():
            raise ValueError('Entry floor must be observed')
        entry_floor = (extended[:4, 1] - entry_gap).mean()
        windows = []
        for start in range(len(extended) - 32):
            summary = source_summary(extended[start:start + 33], case['scene'], entry_floor)
            old_index = start - 24
            summary.update(label=case['label'], start_in_extended=start,
                original_frame=new_start + start, start_in_old_case=old_index,
                archived_training_exposures=archived['exposure'].get(f'{case["label"]}:{old_index}', 0))
            windows.append(summary)
        coverage.extend(windows)
        chosen = set(range(0, len(windows), 8))
        chosen.add(24)
        anticipation = [i for i, row in enumerate(windows) if row['flat_entry_before_height_change']]
        if anticipation:
            chosen.update((anticipation[0], anticipation[-1]))
        sources.append((case, extended, windows, sorted(chosen)))
    write_json(args.output / 'coverage.json', dict(rows=coverage,
        training_report_sha256=sha256(args.training_report), cases_sha256=sha256(args.cases)))
    with np.load(args.live_recording, allow_pickle=False) as archive:
        live = torch.from_numpy(archive['global_features'].copy()).to(args.device)
    with np.load(args.scene, allow_pickle=False) as archive:
        live_scene = {key: torch.from_numpy(archive[key].copy()) for key in
                      ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    for candidate in args.candidates:
        saved = torch.load(candidate, map_location='cpu', weights_only=False)
        if saved['contract']['bundle_sha256'] != sha256(args.bundle):
            raise ValueError('Checkpoint baseline mismatch')
        readout.load_state_dict(native_readout, strict=True)
        if saved['schema'] == 'motionbricks.unified_terrain.root_height_readout.v1':
            for key, value in native_readout.items():
                torch.testing.assert_close(saved['root_height_readout'][key].to(value)[[0, 2, 3, 4]],
                                           value[[0, 2, 3, 4]], atol=0, rtol=0)
            readout.load_state_dict(saved['root_height_readout'], strict=True)
        elif saved['schema'] != SCHEMA:
            raise ValueError('Unsupported checkpoint schema')
        controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
        before = state_digest(controller, list(controller.state_dict()))
        for case, extended, windows, chosen in sources:
            for start in chosen:
                window = extended[start:start + 33].to(args.device)
                command = source_terminal_goal(window[None, :, :5])
                row = inspect(controller, window[None, :4], command, case['scene'], window)
                row.update(checkpoint=candidate.parent.name, scope='real_source_history', **windows[start])
                rows.append(row)
            write_json(args.output / 'progress.json', dict(rows=rows))
        for frame in (64, 72, 80, 88):
            history = reencode_played_history(live[None, frame:frame + 4], inference.global_motion_rep)
            row = inspect(controller, history, history.new_tensor([[0., 1.6, 1., 0.]]), live_scene)
            if sha256(candidate) == archived['contract']['initial_sha256']:
                baseline = next(r for r in archived['baseline']['live_stairs'] if r['speed_request_mps'] == 1.6)
                plan = next(r for r in baseline['plans'] if r['frame'] == frame)
                if plan['neural_generation']:
                    error = abs(row['trace']['root_future_max_world_y_m'] - plan['root_future_max_y_m'])
                    if error > 3e-5:
                        raise ValueError(f'Reencoded history did not replay the archived live Root: {error}')
                    row['archived_neural_root_replay_error_m'] = error
            row.update(checkpoint=candidate.parent.name, scope='fixed_live_history', frame=frame)
            rows.append(row)
        if state_digest(controller, list(controller.state_dict())) != before:
            raise ValueError('Read-only audit modified model weights')
        write_json(args.output / 'progress.json', dict(rows=rows))
    write_json(args.output / 'complete.json', dict(rows=rows, coverage=coverage,
        candidate_hashes={p.parent.name: sha256(p) for p in args.candidates},
        optimizer_updates=0, weights_unchanged=True, runtime_changed=False,
        replay_error=0., source_modified=False, live_history='official FK reencoded, matching publication',
        scope='SDF-only plane intervention keeps native terrain/origin factual; all-terrain plane '
              'intervention separately reports its origin change. Entry floor is anchored to the '
              'earliest real prefix, not each tread. '
              'This tests use of observed SDF, not a flat-motion label or deployment candidate.'))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'cases', 'dataset', 'training-report', 'scene', 'live-recording', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--candidates', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'failure.json', dict(error=repr(error)))
        raise
