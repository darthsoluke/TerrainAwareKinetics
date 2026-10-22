"""Bounded PARC-only imitation, then complete generated-history stair training."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from motionbricks.data.joint_sdf_controller import SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.parc_stair_passages import validate_pure_cases
from motionbricks.data.source_sole_position_objective import source_sole_position_loss
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import control, gait_metrics, load_model
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.pilot_joint_sdf_reciprocal_source import configure_trainable, frozen_state
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def task_loss(model, result, history, source, target, geometry, coefficients):
    native = model.native_losses(result, source)['native_total']
    physical = geometry.prepared_losses(result['world'], history, target, offset=0, diagnostics=False)
    _, points = geometry.geometry(result['world'])
    end = result['world'].shape[1] - 4
    xyz, active = source_sole_position_loss(points[:, 4:end, :geometry.sole_count],
        target['points'][:, 4:end, :geometry.sole_count], target['heights'])
    if int(active.sum()) != physical['support_vertices']:
        raise ValueError('Source support mask changed')
    prefix = decoded_factual_history_loss(model.inference, result['local_prediction'], history,
        result['observation'], root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff']))
    source_ae = native_codec_reconstruction(model, result)['loss']
    loss = native + source_ae + prefix + coefficients['fk_m'] * xyz
    loss += sum(coefficients[k] * physical[k] for k in coefficients)
    return loss, dict(native=float(native.detach()), source_ae=float(source_ae.detach()),
        prefix=float(prefix.detach()), sole_xyz_m=float(xyz.detach()),
        physical={k: float(physical[k].detach()) for k in coefficients})


def episode_loss(model, case, geometry, prepared, field, coefficients, position):
    source = case['world'][None].cuda()
    initial = history = source[:, :4]
    state, parts, prefixes = SoftCommandState(history_gradients=True), [], []
    first = native = None
    rep = model.inference.global_motion_rep
    for block in range(case['training_segments']):
        intent = state.from_controls(history, control(case, block).cuda(), 8)
        result = model(history, intent, [case['scene']], straight_through=True,
                       history_gradients=True, source=source[:, :33] if block == 0 else None)
        world = result['world']
        prefixes.append(decoded_factual_history_loss(model.inference, model.last_local[:, :32], history,
            result['observation'], root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff'])))
        if block == 0:
            first = world
            native = model.native_losses(result, source[:, :33])['native_total']
        parts.append(world[:, 4:12])
        history = reencode_played_history(world[:, 8:12], rep, differentiable=True)
    gradient = torch.autograd.grad(world[:, 4:12].square().mean(), first, retain_graph=True)[0]
    if not torch.isfinite(gradient).all() or gradient.norm() == 0:
        raise ValueError('Missing continuous generated-history gradient')
    episode = torch.cat((initial, *parts, world[:, 12:16]), 1)
    exact = ExactPenetrationBatch([case['scene']], source.device, [field])
    target = dict(joints=prepared[0][None], points=prepared[1][None], heights=exact.heights, volumes=exact)
    physical = geometry.prepared_losses(episode, initial, target, offset=0, diagnostics=False)
    _, points = geometry.geometry(episode)
    end = episode.shape[1] - 4
    xyz, active = source_sole_position_loss(points[:, 4:end, :geometry.sole_count],
        target['points'][:, 4:end, :geometry.sole_count], exact.heights)
    if int(active.sum()) != physical['support_vertices']:
        raise ValueError('Whole-passage source mask changed')
    offset = position % case['training_segments'] * 8
    window = source[:, offset:offset + 33]
    with torch.no_grad():
        intent = SoftCommandState().from_controls(window[:, :4], control(case, position).cuda(), 8)
        auxiliary_result = model(window[:, :4], intent, [case['scene']], source=window)
    auxiliary = native_codec_reconstruction(model, auxiliary_result)['loss']
    loss = native + auxiliary + torch.stack(prefixes).mean() + coefficients['fk_m'] * xyz
    loss += sum(coefficients[k] * physical[k] for k in coefficients)
    return loss, dict(label=case['label'], loss=float(loss.detach()), primitives=len(parts),
        sole_xyz_m=float(xyz.detach()), history_gradient=float(gradient.norm()),
        physical={k: float(physical[k].detach()) for k in coefficients})


@torch.no_grad()
def review(model, cases, geometry, directory):
    directory.mkdir(parents=True, exist_ok=False)
    rows, terrain_response = [], []
    rep = model.inference.global_motion_rep
    for case in cases:
        source = case['world'][None].cuda()
        initial = source[:, :4]
        goal_index = case['evaluation_goal_source_frame'] - case['start_frame']
        goal = source[0, goal_index, :3]
        delta = goal[[0, 2]] - initial[0, 3, [0, 2]]
        axis = delta / delta.norm()
        save_motion(directory / (case['label'] + '-source.npz'), source, case['scene'], rep,
                    dict(role='unmodified_PARC_source', postprocessing=None))
        for scale in (1., .7):
            history = initial.clone()
            state, parts, failure, reached = SoftCommandState(), [], None, False
            request = control(case, 0).cuda().clone()
            request[:, :2] *= scale
            for block in range(60):
                try:
                    result = model(history, state.from_controls(history, request, 8), [case['scene']])
                except ValueError as error:
                    failure = str(error)
                    break
                parts.append(result['world'][:, 4:12])
                history = reencode_played_history(result['world'][:, 8:12], rep)
                reached = float((history[0, 3, [0, 2]] - goal[[0, 2]]) @ axis) >= 0
                if reached:
                    break
            row = dict(label=case['label'], entry=case['entry'], pilot_split=case['pilot_split'],
                speed_scale=scale, frames=len(parts) * 8, requested_frame_limit=480,
                failure=failure, goal_reached=reached, passed=False, postprocessing=None)
            if parts:
                world = torch.cat(parts, 1)
                row.update(exact_skin_metrics(geometry, world, case['scene']))
                row.update(gait_metrics(rep, world, axis))
                displacement = world[0, -1, [0, 2]] - initial[0, 3, [0, 2]]
                row.update(progress_m=float(displacement @ axis), required_progress_m=float(delta.norm()),
                    cross_track_m=float((displacement @ torch.stack((-axis[1], axis[0]))).abs()),
                    root_height_error_m=abs(float(world[0, -1, 1] - goal[1])))
                _, points = geometry.geometry(world)
                gap, known = heightfield_gap(points[0, :, :geometry.sole_count], case['scene'])
                row['sole_vertical_p99_m'] = float(torch.quantile((-gap[known]).clamp_min(0), .99)) if known.any() else None
                row['passed'] = bool(failure is None and reached and row['cross_track_m'] <= .35
                    and row['root_height_error_m'] <= .2 and known.all() and row['unknown_body_fraction'] == 0
                    and row['uncertain_inside_samples'] == 0 and row['sole_vertical_p99_m'] <= .04
                    and row['body_prism_p99_m'] <= .02 and row['body_prism_max_m'] <= .02)
                save_motion(directory / (case['label'] + f'-speed{scale}-{len(parts) * 8}f.npz'), world,
                            case['scene'], rep, row)
            rows.append(row)
            write_json(directory / 'metrics.json', dict(rows=rows, quality_approved=False))
            print(json.dumps(row), flush=True)
        intent = SoftCommandState().from_controls(initial, control(case, 0).cuda(), 8)
        actual = model(initial, intent, [case['scene']])['world']
        gap, _ = heightfield_gap(initial[0, 3:4, :3], case['scene'])
        floor = float(initial[0, 3, 1] - gap[0])
        flat = dict(case['scene'], heightfield_y_m=torch.full_like(torch.as_tensor(case['scene']['heightfield_y_m']), floor))
        counterfactual = model(initial, intent, [flat])['world']
        aj = rep.inverse(actual, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
        fj = rep.inverse(counterfactual, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
        terrain_response.append(dict(label=case['label'], body_relative_response_m=float(
            ((aj - actual[:, :, None, :3]) - (fj - counterfactual[:, :, None, :3])).norm(dim=-1)[:, 4:12].mean()),
            root_response_m=float((actual[:, 4:12, :3] - counterfactual[:, 4:12, :3]).norm(dim=-1).mean()),
            interpretation='same factual approach history/command; flat counterfactual only; difference is not a success gate'))
    write_json(directory / 'terrain_response.json', dict(rows=terrain_response, optimizer_updates=0))
    return rows


def main(args):
    if not 8 <= args.window_updates <= 100 or not 1 <= args.episode_updates <= 8:
        raise ValueError('Pilot caps:100 source-window updates and8 whole-passage updates')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    expected = 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf'
    if sha256(args.checkpoint) != expected:
        raise ValueError('Expected approved orange initialization')
    preparation = json.loads((args.cases.parent / 'complete.json').read_text())
    if sha256(args.cases) != preparation['cases_sha256']:
        raise ValueError('Prepared source identity changed')
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    validate_pure_cases(cases)
    training = [c for c in cases if c['pilot_split'] == 'train']
    if args.regression_cases:
        for prior in torch.load(args.regression_cases, map_location='cpu', weights_only=False):
            yaw = torch.tensor(prior['command_heading'])
            request = torch.cat((torch.tensor(prior['command_velocity']), yaw.cos()[None], yaw.sin()[None]))[None]
            cases.append(dict(prior, exact_constant_control=request, pilot_split='approved_regression',
                              label='approved-' + prior['entry']))
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if sha256(Path(saved['contract']['source'])) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle changed')
    model, _ = load_model(saved)
    configure_trainable(model, 'coordinated')
    terrain_names = ('sdf_encoder', 'sdf_joint_embedding', 'sdf_root', 'sdf_pose', 'sdf_decoder')
    if args.train_terrain:
        for name in terrain_names:
            model.get_submodule(name).requires_grad_(True)

    def immutable_state():
        return {name: digest for name, digest in frozen_state(model, 'coordinated').items()
                if not args.train_terrain or name not in terrain_names}

    geometry = JointGeometryObjective(model.inference.global_motion_rep, 'cuda:0')
    fields = [HeightfieldPrismSDF(c['scene']) for c in training]
    with torch.no_grad():
        prepared = [tuple(v[0] for v in geometry.geometry(c['world'][None].cuda())) for c in training]
    frozen = immutable_state()
    coefficients = saved['coefficients']
    contract = dict(schema='motionbricks.pure_parc_stair_pilot.v1', parent_sha256=expected,
        cases_sha256=sha256(args.cases), window_updates=args.window_updates,
        episode_updates=args.episode_updates, window_batch=4, pure_parc_fraction=1., bones_fraction=0.,
        stop_task=False, teacher_forced_phase='complete real33-frame windows, independent cursors',
        feedback_phase='whole original passage, actual generated history, temporal gradients, no resets',
        trainable='native Root, native Pose, codec decoder; frozen encoder/vocabulary/terrain adapters/control',
        terrain_adapters_trainable=args.train_terrain,
        frozen_hashes=frozen, optimizer='fresh Adam5e-5, clip0.5, no LR scaling',
        loss='existing native, sourceAE, factual prefix, FK/geometry/source-sole XYZ; coefficients unchanged',
        future_truth_in_actor=False, postprocessing=None, quality_approved=False,
        holdout_scope=preparation['holdout_scope'],
        scope='PARC-only NEW fitting, inherited pretrained weights retain their original mixed-data provenance')
    if args.train_terrain:
        contract['trainable'] = 'native Root/Pose/codec.decoder AND SDF encoder/joint embeddings/Root/Pose/decoder adapters; frozen codec encoder/vocabulary/command/old terrain modules'
    write_json(args.output / 'contract.json', contract)
    write_json(args.output / 'state.json', dict(status='baseline', quality_approved=False))
    baseline = review(model, cases, geometry, args.output / 'baseline')
    if args.reference_run:
        errors = {}
        for path in (args.output / 'baseline').glob('*.npz'):
            with np.load(path, allow_pickle=False) as actual, np.load(args.reference_run / 'baseline' / path.name, allow_pickle=False) as expected_record:
                np.testing.assert_allclose(actual['world'], expected_record['world'], atol=3e-5, rtol=1e-5)
                errors[path.name] = float(np.abs(actual['world'] - expected_record['world']).max())
        write_json(args.output / 'baseline_parity.json', dict(reference=str(args.reference_run), errors=errors))
    torch.manual_seed(2333)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    positions, logs, exposure = [0] * 4, [], Counter()
    start = time.perf_counter()

    def checkpoint(stage, step):
        if immutable_state() != frozen:
            raise ValueError('Frozen module changed')
        torch.save(dict(schema=SDF_VERSION, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
            step=step, contract=dict(saved['contract'], pure_parc_pilot=contract), coefficients=coefficients,
            exposure=dict(exposure), phase=stage, positions=positions,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), args.output / (stage + '-weights.pt'))

    def update(loss, row):
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite PARC loss')
        loss.backward()
        if any(p.grad is not None for name in frozen for p in model.get_submodule(name).parameters()):
            raise ValueError('Frozen encoder/vocabulary/conditioning received a gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid PARC gradient')
        optimizer.step()
        row.update(gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - start,
                   exposure=dict(exposure), peak_memory_bytes=torch.cuda.max_memory_allocated())
        if row['update'] == 1:
            row['gradient_groups'] = {name: sum(float(p.grad.square().sum()) for p in model.get_submodule(name).parameters()
                if p.grad is not None) ** .5 for name in ('root', 'pose', 'codec.decoder')}
            if min(row['gradient_groups'].values()) <= 0:
                raise ValueError('A declared predictor has no training gradient')
            if args.train_terrain:
                row['terrain_gradient_groups'] = {name: sum(float(p.grad.square().sum()) for p in model.get_submodule(name).parameters()
                    if p.grad is not None) ** .5 for name in terrain_names}
                if max(row['terrain_gradient_groups'].values()) <= 0:
                    raise ValueError('Declared terrain adaptation has no gradient')
        logs.append(row)
        write_json(args.output / 'progress.json', row)
        print(json.dumps(row), flush=True)

    write_json(args.output / 'state.json', dict(status='window_training', quality_approved=False))
    for step in range(args.window_updates):
        windows = torch.stack([c['world'][p * 8:p * 8 + 33] for c, p in zip(training, positions)]).cuda()
        history = windows[:, :4]
        intents = torch.cat([SoftCommandState().from_controls(history[i:i + 1], control(c, p).cuda(), 8)
                             for i, (c, p) in enumerate(zip(training, positions))])
        scenes = [c['scene'] for c in training]
        if step == 0:
            with torch.no_grad():
                plain = model(history, intents, scenes)['world'].clone()
                changed = windows.clone()
                changed[:, 4:] += 3.
                altered = model(history, intents, scenes, source=changed)['world']
            with torch.inference_mode():
                runtime = model(history, intents, scenes)['world']
            torch.testing.assert_close(plain, altered, atol=0, rtol=0)
            torch.testing.assert_close(plain, runtime, atol=3e-5, rtol=1e-5)
            parity = dict(future_target_error=float((plain - altered).abs().max()), runtime_error=float((plain - runtime).abs().max()))
        result = model(history, intents, scenes, source=windows, straight_through=True)
        if step == 0:
            torch.testing.assert_close(plain, result['world'], atol=3e-5, rtol=1e-5)
            parity['training_forward_error'] = float((plain - result['world']).abs().max())
        exact = ExactPenetrationBatch(scenes, history.device, fields)
        target = dict(joints=torch.stack([v[0][p * 8:p * 8 + 33] for v, p in zip(prepared, positions)]),
                      points=torch.stack([v[1][p * 8:p * 8 + 33] for v, p in zip(prepared, positions)]),
                      heights=exact.heights, volumes=exact)
        loss, detail = task_loss(model, result, history, windows, target, geometry, coefficients)
        optimizer.zero_grad(set_to_none=True)
        for c, p in zip(training, positions):
            exposure['window:' + c['label'] + ':' + str(p)] += 1
        update(loss, dict(phase='window', update=step + 1, loss=float(loss.detach()), **detail))
        positions = [(p + 1) % c['training_segments'] for c, p in zip(training, positions)]
        if step == 7 or step + 1 == args.window_updates:
            checkpoint('window', step + 1)
            write_json(args.output / 'preflight.json', dict(parity, frozen_hashes_match=True, checked_window_updates=step + 1))
        del loss, result
    window_review = review(model, cases, geometry, args.output / 'window_review')
    write_json(args.output / 'state.json', dict(status='episode_training', quality_approved=False))
    for step in range(args.episode_updates):
        optimizer.zero_grad(set_to_none=True)
        details = []
        for i, c in enumerate(training):
            loss, detail = episode_loss(model, c, geometry, prepared[i], fields[i], coefficients, step)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite generated-history passage objective')
            (loss / 4).backward()
            details.append(detail)
            exposure['episode:' + c['label']] += c['training_segments']
            del loss
        if any(p.grad is not None for name in frozen for p in model.get_submodule(name).parameters()):
            raise ValueError('Frozen module received an episode gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid temporal gradient')
        optimizer.step()
        row = dict(phase='episode', update=step + 1, cases=details, gradient_norm=float(norm),
                   elapsed_seconds=time.perf_counter() - start, exposure=dict(exposure),
                   peak_memory_bytes=torch.cuda.max_memory_allocated())
        logs.append(row)
        write_json(args.output / 'progress.json', row)
        print(json.dumps(row), flush=True)
        checkpoint('episode', args.window_updates + step + 1)
    model.requires_grad_(False)
    write_json(args.output / 'state.json', dict(status='final_review', quality_approved=False))
    final = review(model, cases, geometry, args.output / 'final')
    write_json(args.output / 'training.json', dict(logs=logs, contract=contract))
    write_json(args.output / 'complete.json', dict(baseline=baseline, window_review=window_review, final=final,
        actual_optimizer_updates=args.window_updates + args.episode_updates,
        source_window_primitives=4 * args.window_updates,
        generated_primitives=args.episode_updates * sum(c['training_segments'] for c in training),
        source_autoencoder_examples=4 * (args.window_updates + args.episode_updates),
        exposure=dict(exposure), frozen_hashes_match=immutable_state() == frozen,
        parent_file_unchanged=sha256(args.checkpoint) == expected, quality_approved=False))
    write_json(args.output / 'state.json', dict(status='complete', quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--window-updates', type=int, default=80)
    parser.add_argument('--episode-updates', type=int, default=6)
    parser.add_argument('--regression-cases', type=Path)
    parser.add_argument('--train-terrain', action='store_true')
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'state.json', dict(status='failed', error=str(error), quality_approved=False))
        raise
