"""Offline gait attribution with fixed actor histories and factorial codec probes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import training_targets, vertical_origin
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.evaluate_joint_sdf import save_motion
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_decoder_height_response import capture_decoder_call
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def restore(infer, local, history, observation):
    world = infer.motion_rep.dual_rep.local_to_global(
        local, True, False, lengths=torch.full((len(history),), 32, device=history.device, dtype=torch.long))
    offsets, heading, _ = infer._extract_initial_root_info(history[..., :5])
    world = infer._reapply_initial_root_info(dict(pred_global_poses=world,
        reference_start_root_global_offsets=offsets, reference_start_root_global_heading=heading))
    return shift_global_motion(world, infer.global_motion_rep, vertical_origin(infer, history, observation))[:, :32]


def gait(rep, world, direction):
    joints = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints'][0]
    names = rep.skeleton.bone_order_names
    feet = joints[:, [names.index('LeftFoot'), names.index('RightFoot')]] - world[0, :, None, :3]
    foreaft = feet[..., [0, 2]] @ direction
    separation = foreaft[:, 0] - foreaft[:, 1]
    def part(first):
        values = separation[first:]
        return dict(foreaft_rms_m=float(values.square().mean().sqrt()),
                    close_feet_fraction=float((values.abs() < .1).float().mean()))
    return dict(all=part(0), last32=part(-32), foreaft_m=foreaft.tolist())


def select_bones(dataset, count=2):
    candidates = []
    for index, row in enumerate(dataset.records):
        if row.get('origin') != 'bones_seed' or row.get('control_category') != 'run' or row['frames_30hz'] < 113:
            continue
        item = dataset.full_reference_item(index)
        world = item['world']
        best = None
        for start in range(0, len(world) - 112, 8):
            window = world[start:start + 113]
            velocity = torch.diff(window[:, [0, 2]], dim=0) * 30
            mean = velocity.mean(0)
            deviation = (velocity - mean).norm(dim=-1).mean()
            score = float((mean.norm() - 3.).abs() + deviation)
            if best is None or score < best[0]:
                best = (score, start, mean, deviation)
        if best is not None:
            candidates.append((best[0], index, best, item))
    result, used = [], set()
    for score, index, (_, start, velocity, deviation), item in sorted(candidates, key=lambda value: value[0]):
        identity = item['row'].get('filename') or item['row']['source_id']
        if identity in used:
            continue
        used.add(identity)
        if torch.count_nonzero(item['scene']['heightfield_y_m']):
            raise ValueError('BONES scene is not its declared zero plane')
        world = item['world'][start:start + 113]
        center = world[3, [0, 2]]
        scene = dict(heightfield_y_m=torch.zeros(201, 201), dx_m=.5,
                     min_point_world_xyz_m=torch.tensor([float(center[0]) - 50, 0, float(center[1]) - 50]))
        result.append(dict(world=world, scene=scene, row=item['row'], source_start=start,
            command_velocity=velocity, command_heading=torch.atan2(world[:, 4].mean(), world[:, 3].mean()),
            selection=dict(score=score, mean_speed_mps=float(velocity.norm()), deviation_mps=float(deviation),
                dataset_index=index, candidate_count=len(candidates), selection_uses_actor=False)))
        if len(result) == count:
            return result
    raise ValueError(f'Need {count} independent real run sources, found {len(result)}')


def flat_gradient(model, source, scene, velocity, heading):
    history = source[:, :4]
    task = SoftCommandState().intent(history, velocity, heading, 8)
    model.freeze_base()
    with torch.enable_grad():
        result = model(history, task, [scene], source=source[:, :33], straight_through=True)
        memory_gap = float((model._sdf_memory - model._sdf_plane).abs().max())
        loss = model.native_losses(result, source[:, :33])['native_total']
        loss.backward()
        gradients = {}
        for name in ('sdf_encoder', 'sdf_joint_embedding', 'sdf_root', 'sdf_pose', 'sdf_decoder'):
            grads = [parameter.grad for parameter in getattr(model, name).parameters() if parameter.grad is not None]
            gradients[name] = dict(max_abs=max(float(g.abs().max()) for g in grads),
                                   l2=sum(float(g.double().square().sum()) for g in grads) ** .5,
                                   tensors=len(grads))
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    with torch.no_grad():
        model.sdf_enabled = True
        enabled = model(history, task, [scene])['world'].clone()
        model.sdf_enabled = False
        disabled = model(history, task, [scene])['world'].clone()
        model.sdf_enabled = True
    return dict(memory_gap_max=memory_gap, enabled_disabled_max_difference=float((enabled - disabled).abs().max()),
                native_loss=float(loss.detach()), gradient=gradients, optimizer_updates=0)


@torch.no_grad()
def review_case(model, case, output):
    output.mkdir()
    infer = model.inference
    rep = infer.global_motion_rep
    source = case['world'][None].cuda()
    velocity = case['command_velocity'].to(source)
    heading = case['command_heading'].to(source)
    direction = velocity / velocity.norm()
    scene = case['scene']
    report = dict(row=case['row'], source_start=case.get('source_start'), selection=case.get('selection'),
                  velocity_mps=velocity.tolist(), oracle_is_offline_only=True, modes={},
                  counterfactual_flat_scene=case.get('counterfactual_flat_scene', False))
    if case['row'].get('origin') == 'bones_seed':
        report['flat_cancellation'] = flat_gradient(model, source, scene, velocity, heading)
    for mode in ('generated_history', 'source_history'):
        history = source[:, :4].clone()
        state = SoftCommandState()
        blocks = []
        motions = {key: [] for key in ('actor', 'source_tokens', 'source_root', 'source_tokens_and_root')}
        failure = None
        for block in range(10):
            offset = block * 8
            reference = source[:, offset:offset + 33]
            if mode == 'source_history':
                history = reference[:, :4].clone()
            task = state.intent(history, velocity, heading, 8)
            try:
                with capture_decoder_call(model.codec) as calls:
                    result = model(history, task, [scene])
            except ValueError as error:
                failure = str(error)
                break
            if len(calls) != 2 or calls[-1]['kwargs']['has_target_cond'][:, 4:].any():
                raise ValueError('Unexpected actor decoder contract')
            actual = result['world'].clone()
            call = calls[-1]
            replay_local = model.codec.forward_decoder(*call['args'], **call['kwargs'])['recon_state']
            replay = restore(infer, replay_local, history, result['observation'])
            torch.testing.assert_close(replay, actual, atol=1e-5, rtol=1e-5)
            target = training_targets(infer, reference, history, result['observation'])
            source_tokens = model.codec.encode_into_idx(target['local_motions'], fetch_overall_indices=False)
            tokens = call['args'][0].clone()
            tokens[:, :8] = source_tokens
            root = extract_feature_from_motion_rep(target['local_motions'], model.codec.motion_rep,
                                                    model.codec.decoder_external_cond_feature_mode)
            external = call['kwargs']['external_cond'].clone()
            external[:, :32] = root
            block_row = dict(block=block, replay_max_difference=float((replay - actual).abs().max()),
                token_accuracy=float((call['args'][0][:, :8] == source_tokens).float().mean()),
                predicted_tokens=call['args'][0][:, :8].tolist(), source_tokens=source_tokens.tolist(),
                root_condition_error_rms=float((call['kwargs']['external_cond'][:, :32] - root).square().mean().sqrt()),
                variants={})
            decoded = dict(actor=actual)
            for label, use_tokens, use_root in (('source_tokens', True, False), ('source_root', False, True),
                                               ('source_tokens_and_root', True, True)):
                kwargs = dict(call['kwargs'])
                if use_root:
                    kwargs['external_cond'] = external
                args = (tokens if use_tokens else call['args'][0], *call['args'][1:])
                local = model.codec.forward_decoder(*args, **kwargs)['recon_state']
                decoded[label] = restore(infer, local, history, result['observation'])
            ref_joints = rep.inverse(reference[:, 4:12], False, return_all=False,
                                     joint_positions_from='global_rot_data')['posed_joints']
            for label, world in decoded.items():
                published = world[:, 4:12]
                joints = rep.inverse(published, False, return_all=False,
                                     joint_positions_from='global_rot_data')['posed_joints']
                relative_error = ((joints - published[:, :, None, :3]) -
                                  (ref_joints - reference[:, 4:12, None, :3])).norm(dim=-1)
                block_row['variants'][label] = dict(relative_mpjpe_m=float(relative_error.mean()),
                    root_rms_m=float((published[..., :3] - reference[:, 4:12, :3]).square().sum(-1).mean().sqrt()))
                motions[label].append(published.clone())
            blocks.append(block_row)
            # Oracle results never change the factual actor history or command.
            history = reencode_played_history(actual[:, 8:12], rep)
        mode_report = dict(blocks=blocks, failure=failure, requested_frames=80, frames=len(blocks) * 8, variants={})
        for label, parts in motions.items():
            if not parts:
                continue
            world = torch.cat(parts, 1)
            if mode == 'generated_history' and label == 'actor' and case.get('recording'):
                with np.load(case['recording'], allow_pickle=False) as archive:
                    expected = torch.from_numpy(archive['world'].copy()).to(world)[None]
                torch.testing.assert_close(world, expected, atol=3e-5, rtol=1e-5)
                mode_report['recording_replay_max_difference'] = float((world - expected).abs().max())
            mode_report['variants'][label] = dict(gait=gait(rep, world, direction),
                relative_mpjpe_m=float(np.mean([b['variants'][label]['relative_mpjpe_m'] for b in blocks])),
                root_rms_mean_m=float(np.mean([b['variants'][label]['root_rms_m'] for b in blocks])))
            save_motion(output / f'{mode}-{label}.npz', world, scene, rep,
                dict(role='continuous_actor' if mode == 'generated_history' and label == 'actor' else 'offline_window_diagnostic',
                     future_truth_used=label != 'actor', quality_approved=False, blocks_independently_diagnosed=label != 'actor'))
        report['modes'][mode] = mode_report
        write_json(output / 'progress.json', report)
    report['source'] = gait(rep, source[:, 4:84], direction)
    save_motion(output / 'source.npz', source[:, 4:84], scene, rep, dict(role='unchanged_source'))
    write_json(output / 'complete.json', report)
    return report


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    path = Path(saved['contract']['source'])
    if sha256(path) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle identity mismatch')
    infer, bundle = load_terrain_bundle(path, 'cuda:0')
    model = JointSDFController(infer, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    data_path = Path(saved['contract']['dataset'])
    data = TerrainLearningDataset(data_path, build_native_representation(data_path / 'stats'), 'train')
    if args.reported_cases:
        case = next(c for c in torch.load(args.reported_cases, map_location='cpu', weights_only=False)
                    if c['row']['source_id'] == 'generated_8030ae7a2a7ad640')
        index = next(i for i, row in enumerate(data.records) if row['source_id'] == case['row']['source_id'])
        full = data.full_reference_item(index)
        start = case['start_frame']
        world = full['world'][start:start + 113]
        if len(world) != 113:
            raise ValueError('Missing real tail for the final oracle windows')
        torch.testing.assert_close(world[:len(case['world'])], case['world'], atol=0, rtol=0)
        source = world[None].cuda()
        edge = case['edge_frame']
        velocity = (source[0, edge - 8, [0, 2]] - source[0, 3, [0, 2]]) / ((edge - 11) / 30)
        heading = torch.atan2(source[:, 3:edge - 8, 4].mean(1), source[:, 3:edge - 8, 3].mean(1))
        case = dict(case, world=world, source_start=start, command_velocity=velocity, command_heading=heading,
                    recording=args.recording)
        if args.flat_scene:
            gap, valid = heightfield_gap(source[0, 3:4, :3], case['scene'])
            if not valid.all():
                raise ValueError('Counterfactual plane needs known support')
            height = float(source[0, 3, 1] - gap[0])
            center = source[0, 3, [0, 2]]
            scene = dict(heightfield_y_m=torch.full((201, 201), height), dx_m=.5,
                         min_point_world_xyz_m=torch.tensor([float(center[0]) - 50, 0, float(center[1]) - 50]))
            case = dict(case, scene=scene, counterfactual_flat_scene=True)
        cases = [case]
    else:
        cases = select_bones(data)
    write_json(args.output / 'selection.json', [dict(row=c['row'], start=c['source_start'], selection=c.get('selection')) for c in cases])
    results = []
    for index, case in enumerate(cases):
        results.append(review_case(model, case, args.output / f'run-{index}'))
        print(json.dumps(dict(case=index, status='complete')), flush=True)
    write_json(args.output / 'complete.json', dict(checkpoint_sha256=sha256(args.checkpoint),
        optimizer_updates=0, quality_approved=False, results=results,
        scope=('Reported initial state and command with additional unchanged real source tail for oracle labels. '
               if args.reported_cases else 'Two real BONES run sources chosen by source-only speed and straightness near3m/s. ') +
              'Factorial token/Root truth substitution is offline only; no oracle is fed back or deployed. '
              'Source-history concatenations are independent prediction windows, not a continuous actor.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reported-cases', type=Path)
    parser.add_argument('--recording', type=Path)
    parser.add_argument('--flat-scene', action='store_true')
    main(parser.parse_args())
