"""Zero-update fixed-validation and closed-loop review of a saved P1 endpoint."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.pfnn_3d_scene import BatchedSceneField, CompositeSceneField, path_observation
from motionbricks.data.pfnn_frame_actor import FrameActor, advance, load_environment_encoder
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset, sha256
from motionbricks.data.unified_scene_field import VolumetricEgoSampler
from scripts.train_pfnn_frame_actor import model_digest


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def step(actor, rep, sampler, field, history, command, state):
    coarse = sampler.observe(history[..., :5], command[:, :2], field, .8)
    root = actor.root_step(history, command, coarse, state)
    fine = path_observation(sampler, history[..., :5], root['path_world_xz'], coarse, field)
    pred = actor.pose_step(history, command, fine, root)
    return advance(rep, history, pred), pred['state']


def window_metrics(rep, generated, source):
    """Each sequence includes its factual four-frame seed; errors exclude it."""
    pj = rep.inverse(generated, False, return_all=False,
                     joint_positions_from='global_rot_data')['posed_joints']
    sj = rep.inverse(source, False, return_all=False,
                     joint_positions_from='global_rot_data')['posed_joints']
    pr, sr = generated[..., :3], source[..., :3]
    relative_p, relative_s = pj - pr[:, :, None], sj - sr[:, :, None]
    pv = torch.diff(relative_p[:, 3:], dim=1) * 30
    sv = torch.diff(relative_s[:, 3:], dim=1) * 30
    pred_rms = pv.square().mean((1, 2, 3)).sqrt()
    source_rms = sv.square().mean((1, 2, 3)).sqrt()
    metrics = dict(
        root_mean_m=(pr[:, 4:] - sr[:, 4:]).norm(dim=-1).mean(1),
        root_endpoint_m=(pr[:, -1] - sr[:, -1]).norm(dim=-1),
        body_relative_mpjpe_m=(relative_p[:, 4:] - relative_s[:, 4:]).norm(dim=-1).mean((1, 2)),
        body_relative_velocity_rmse_mps=(pv - sv).square().mean((1, 2, 3)).sqrt(),
        predicted_body_velocity_rms_mps=pred_rms,
        source_body_velocity_rms_mps=source_rms,
        heading_mean_rad=torch.atan2(
            torch.sin(torch.atan2(generated[:, 4:, 4], generated[:, 4:, 3])
                      - torch.atan2(source[:, 4:, 4], source[:, 4:, 3])),
            torch.cos(torch.atan2(generated[:, 4:, 4], generated[:, 4:, 3])
                      - torch.atan2(source[:, 4:, 4], source[:, 4:, 3]))).abs().mean(1),
    )
    values = {key: value.cpu().tolist() for key, value in metrics.items()}
    rows = [{key: values[key][i] for key in values} for i in range(len(generated))]
    for row in rows:
        speed = row['source_body_velocity_rms_mps']
        row['body_velocity_rms_ratio'] = row['predicted_body_velocity_rms_mps'] / speed if speed > 1e-4 else None
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        for key in ('all', 'category:' + row['category'], 'terrain:' + row['terrain']):
            groups[key].append(row)
    keys = [key for key, value in rows[0].items() if isinstance(value, (float, type(None)))]
    return {group: dict(count=len(items), **{key: {
        'mean': float(np.mean(values)), 'p95': float(np.quantile(values, .95))}
        for key in keys if (values := [item[key] for item in items if item[key] is not None])})
        for group, items in groups.items()}


@torch.no_grad()
def windows(args, saved, rep, actor):
    cpu_rep = build_native_representation(args.stats)
    dataset = PFNNFrameDataset(args.cache, args.pfnn, cpu_rep, args.split)
    selection_path = args.selection if args.selection is not None else args.checkpoint.parent / 'validation_selection.json'
    selection = json.loads(selection_path.read_text())
    by_id = {r['cache_row_index']: i for i, r in enumerate(dataset.records)}
    ids = [by_id[r['cache_row_index']] for r in selection]
    if any(dataset.records[i] != row for i, row in zip(ids, selection)):
        raise ValueError('Fixed validation identities changed')
    save(args.output / 'selection.json', selection)
    sampler = VolumetricEgoSampler()
    report = dict(selection_sha256=sha256(selection_path), split=args.split, windows=len(ids), steps=26,
                  modes={}, optimizer_updates=0, test_windows=0)
    modes = ('initial_teacher', 'trained_teacher') if args.teacher_only else ('initial_teacher', 'trained_teacher', 'trained_closed')
    for mode in modes:
        if mode == 'initial_teacher':
            torch.manual_seed(saved['plan']['seed'])
            model = FrameActor(rep).to(args.device).eval()
            parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
            load_environment_encoder(model, parent['model_state'], freeze=True)
            del parent
            contract = json.loads((args.checkpoint.parent / 'contract.json').read_text())
            if model_digest(model) != contract['initial_sha256']:
                raise ValueError('Initialization no longer matches the training run')
        else:
            model = actor
        rows, motions = [], []
        for begin in range(0, len(ids), args.batch):
            items = [dataset[i] for i in ids[begin:begin + args.batch]]
            world = torch.stack([item['world'] for item in items]).to(args.device)
            commands = torch.stack([item['command_sequence']['command'] for item in items]).to(args.device)
            field = BatchedSceneField([CompositeSceneField(item['scene']) for item in items], device=args.device)
            history = reencode_played_history(world[:, :4], rep)
            state = model.initial_state(len(items))
            frames = [history]
            for offset in range(commands.shape[1]):
                if mode != 'trained_closed':
                    history = reencode_played_history(world[:, offset:offset + 4], rep)
                history, state = step(model, rep, sampler, field, history, commands[:, offset], state)
                frames.append(history[:, -1:].clone())
            generated = torch.cat(frames, dim=1)
            if not torch.isfinite(generated).all():
                raise ValueError('Nonfinite generated validation trajectory')
            measured = window_metrics(rep, generated, world[:, :generated.shape[1]])
            for item, metrics in zip(items, measured):
                row = item['record']
                terrain = dataset.index.sources[row['source_id']]['terrain_construction']['kind']
                rows.append(dict(cache_row_index=row['cache_row_index'], category=row['category'],
                                 terrain=terrain, **metrics))
            motions.append(generated.cpu())
            save(args.output / 'state.json', dict(stage=mode, windows_done=len(rows), optimizer_updates=0))
        torch.save(dict(world=torch.cat(motions), selection=selection), args.output / (mode + '.pt'))
        report['modes'][mode] = dict(rows=rows, groups=aggregate(rows))
        save(args.output / 'windows.json', report)
        print(json.dumps(dict(mode=mode, all=report['modes'][mode]['groups']['all'])), flush=True)
    return report


@torch.inference_mode()
def controls(args, rep, actor):
    from scripts.benchmark_pfnn_frame_actor import FrameChain, load_scene
    from scripts.evaluate_reactive_prefix_control import controlled_trials, motion_metrics, control_screen, skin_failures
    from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
    from motionbricks.data.terrain_joint_objective import JointGeometryObjective

    scene, seed = load_scene(args.flat_scene)
    seed = reencode_played_history(seed.to(args.device), rep)
    ground = np.asarray(scene['heightfield_y_m'])
    if not np.isfinite(ground).all() or np.ptp(ground) != 0:
        raise ValueError('Fixed control fixture must be the existing flat plane')
    field = CompositeSceneField(scene).prepare(args.device)
    chain = FrameChain(rep, actor, field, execution='graph')
    chain.prepare_graphs(seed, seed.new_tensor([[0., 0., 1., 0.]]), actor.initial_state(1))
    geometry = JointGeometryObjective(rep, args.device)
    rows = []
    for trial in controlled_trials(seed.cpu()):
        # The old MB trial uses four-frame input sampling. This actor consumes
        # the exact requested input on every frame, without modifying the task.
        history, state = seed.clone(), actor.initial_state(1)
        frames, timings = [history.clone()], []
        for command in trial['requested_controls'].to(args.device):
            history, state, _, _, _, timing = chain.step(history, command[None], state)
            if not torch.isfinite(history).all():
                raise ValueError('Nonfinite controlled trajectory')
            frames.append(history.clone()[:, -1:])
            timings.append(timing['total'])
        world = torch.cat(frames, dim=1)
        torch.save(dict(world=world.cpu(), controls=trial['requested_controls'], trial=trial),
                   args.output / (trial['name'] + '.pt'))
        joints = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
        motion = motion_metrics(world, joints, trial, float(ground.flat[0]), skeleton=rep.skeleton)
        skin = exact_skin_metrics(geometry, world, scene)
        screen = control_screen(motion, trial)
        row = dict(trial=trial['name'], motion=motion, control_screen=screen, skin=skin,
                   skin_failures=skin_failures(skin), compute_p99_ms=float(np.quantile(timings, .99)),
                   timing_role='remote graph compute; not local browser fps')
        rows.append(row)
        save(args.output / 'controls.json', rows)
        print(json.dumps(dict(trial=trial['name'], control_passed=screen['passed'], skin=skin)), flush=True)
    return rows


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    binding = saved['binding']
    for relative, expected in binding['sources'].items():
        if sha256(Path(relative)) != expected:
            raise ValueError('Training source changed: ' + relative)
    if sha256(args.parent) != binding['parent_sha256']:
        raise ValueError('Original environment checkpoint changed')
    for relative, expected in binding['stats_sha256'].items():
        if sha256(args.stats / relative) != expected:
            raise ValueError('Statistics changed: ' + relative)
    rep = build_native_representation(args.stats).to(args.device)
    actor = FrameActor(rep).to(args.device).eval().requires_grad_(False)
    actor.load_state_dict(saved['model_state'], strict=True)
    before = model_digest(actor)
    report = dict(checkpoint_sha256=sha256(args.checkpoint), checkpoint_updates=saved['optimizer_updates'],
                  checkpoint_exposures=saved['window_exposures'], evaluation_script_sha256=sha256(__file__),
                  optimizer_updates=0, training_exposures=0, quality_approved=False, delivery_approved=False)
    started = time.monotonic()
    save(args.output / 'contract.json', report)
    if args.part == 'windows':
        windows(args, saved, rep, actor)
    else:
        controls(args, rep, actor)
    if before != model_digest(actor) or report['checkpoint_sha256'] != sha256(args.checkpoint):
        raise ValueError('Evaluation changed model weights')
    report.update(completed=True, weights_unchanged=True, elapsed_seconds=time.monotonic() - started)
    save(args.output / 'complete.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'parent', 'stats', 'cache', 'pfnn', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--part', choices=('windows', 'controls'), required=True)
    parser.add_argument('--flat-scene', type=Path)
    parser.add_argument('--split', choices=('train', 'validation'), default='validation')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--teacher-only', action='store_true')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parsed = parser.parse_args()
    if parsed.part == 'controls' and parsed.flat_scene is None:
        parser.error('--flat-scene is required for controls')
    main(parsed)
