"""Matched Root-adapter-only comparison of legacy proxy and native terminal labels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_online_conditions import velocity_goal
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.unified_scene_field import field_from_heightfield
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE, SCHEMA
from motionbricks.data.unified_training_goals import source_terminal_goal, GOAL_SOURCE
from scripts.audit_unified_stair_anticipation import first_riser
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.pilot_unified_terrain import prepare, evaluate
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_unified_flat import runs as flat_runs
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def freeze_for_root_comparison(controller, *, height_readout=False):
    controller.requires_grad_(False)
    for name in ('root_stage1', 'root_stage2'):
        controller.adaptation[name].requires_grad_(True)
    extra = set()
    if height_readout:
        readout = controller.root._conv_output.model[-1]
        if not isinstance(readout, torch.nn.Conv1d) or readout.out_channels != 5 or readout.bias is None:
            raise ValueError('Expected the native five-component final Root convolution')
        readout.requires_grad_(True)
        for parameter in (readout.weight, readout.bias):
            mask = torch.zeros_like(parameter)
            mask[1] = 1
            parameter.register_hook(lambda gradient, mask=mask: gradient * mask)
        extra = {name for name, p in controller.named_parameters()
                 if p is readout.weight or p is readout.bias}
    names = [name for name, p in controller.named_parameters() if p.requires_grad]
    if not names or any(not name.startswith(('adaptation.root_stage1.', 'adaptation.root_stage2.'))
                        and name not in extra for name in names):
        raise ValueError('Only Root environment adapters may be trained')
    return names


@torch.no_grad()
def check_goals(controller, cases, device, window_stride=24):
    rep = controller.inference.global_motion_rep
    rows = []
    for case in cases:
        for start in source_windows(len(case['world']), window_stride):
            source = case['world'][None, start:start + 33].to(device)
            goal = source_terminal_goal(rep.extract_root(source))
            local = controller.inference.motion_rep.dual_rep.global_to_local(source, False, False,
                lengths=torch.tensor([33], device=device))
            native = velocity_goal(controller.inference.local_motion_rep.extract_root(local)[:, 31],
                                   rep.extract_root(source)[:, 31])
            torch.testing.assert_close(goal, native, atol=1e-5, rtol=1e-5)
            old = case['exact_constant_control'].to(device)
            row = dict(label=case['label'], start=start, terminal_goal=goal[0].cpu().tolist(),
                legacy_proxy=old[0].cpu().tolist(), vector_error_mps=float((goal[:, :2] - old[:, :2]).norm()),
                native_converter_max_error=float((goal - native).abs().max()))
            rows.append(row)
    return rows


def source_windows(length, stride):
    if stride not in (1, 24) or length < 33:
        raise ValueError('Use complete real33frame windows and an explicit1or24frame stride')
    return list(range(0, length - 32, stride))


def training_objective(parts, objective):
    if objective == 'root':
        return parts['native_root']
    if objective == 'joint':
        return parts['native_total']
    raise ValueError('Choose an explicit native Root or joint objective')


@torch.no_grad()
def extend_real_approaches(train, args, controller, geometry):
    if args.approach_dataset is None and args.approach_audit is None:
        return train, []
    if args.approach_dataset is None or args.approach_audit is None:
        raise ValueError('Real approach extension requires both dataset and audited receipt')
    if args.arm != 'terminal' or args.objective != 'root' or args.window_stride != 1 or args.height_readout:
        raise ValueError('Isolate source coverage with the matched frozen-native Root-stage arm')
    audit = json.loads(args.approach_audit.read_text())
    if (audit['source_cases_sha256'] != sha256(args.cases)
            or audit['dataset_manifest_sha256'] != sha256(args.approach_dataset / 'native_manifest.json')):
        raise ValueError('Approach audit belongs to different source data')
    dataset = TerrainLearningDataset(args.approach_dataset, controller.inference.global_motion_rep, 'train')
    by_id = {row['source_id']: i for i, row in enumerate(dataset.records)}
    extended_train, checks = [], []
    for case in train:
        if case['name'] != 'stairs':
            extended_train.append(case)
            continue
        row = next(r for r in audit['rows'] if r['label'] == case['label'])
        if not row['eligible_for_flat_approach_extension'] or row['added_real_prefix_frames'] != 24:
            raise ValueError('Source did not pass the real approach audit')
        reference = dataset.full_reference_item(by_id[case['row']['source_id']])
        start = case['start_frame']
        torch.testing.assert_close(reference['world'][start:start + len(case['world'])],
                                   case['world'], atol=0, rtol=0)
        if row['new_start'] != start - 24 or row['old_start'] != start:
            raise ValueError('Audited source interval differs')
        world = reference['world'][start - 24:start + len(case['world'])].clone()
        gap, known = heightfield_gap(world[:28, :3], case['scene'])
        floor = world[:28, 1] - gap
        metrics = exact_skin_metrics(geometry, world[None].to(args.device), case['scene'])
        if (not known.all() or float(floor.max() - floor.min()) > .03
                or metrics['body_prism_max_m'] > .02 or metrics['unknown_body_fraction']
                or metrics['uncertain_inside_samples']):
            raise ValueError('Extended source no longer passes the unchanged geometry screen')
        updated = dict(case, world=world, start_frame=start - 24)
        extended_train.append(updated)
        checks.append(dict(label=case['label'], old_start=start, new_start=start - 24,
            old_frames=len(case['world']), new_frames=len(world), prefix_frames=24,
            original_overlap_error=0., metrics=metrics, synthetic_frames=0,
            training_index_to_old_case_offset=-24, review_case_unchanged=True))
    if len(checks) != 2:
        raise ValueError('Extend only the two training stairs; leave all review cases intact')
    return extended_train, checks


def load_world(args, rep, device):
    if sha256(args.scene) != 'e19876a349c54ba4a2c663b358f9e33bc0bab2e115f302e91ff3f63b7f819ffa':
        raise ValueError('Live scene identity changed')
    with np.load(args.scene, allow_pickle=False) as archive:
        scene = {key: torch.from_numpy(archive[key].copy()) for key in
                 ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
        seed = torch.from_numpy(archive['initial_history'].copy()).to(device)
    receipt = json.loads(args.scene_receipt.read_text())
    if receipt['scene_sha256'] != sha256(args.scene):
        raise ValueError('Scene/spawn identity mismatch')
    spawn = next(s for s in receipt['spawns'] if s['id'] == 'zone-03')
    placed = rep.change_first_heading(seed.clone(), float(spawn['heading']), False, False)
    placed[..., [0, 2]] += placed.new_tensor(spawn['spawn_xz']) - placed[:, -1:, [0, 2]].clone()
    return scene, seed, placed, spawn


@torch.no_grad()
def world_review(controller, geometry, scene, seed, spawn, output, *, max_blocks=60, speeds=(.8, 1.6)):
    if not 1 <= max_blocks <= 225 or any(not np.isfinite(s) or s <= 0 for s in speeds):
        raise ValueError('Use finite positive requests and an explicit review budget up to60seconds')
    output.mkdir(parents=True, exist_ok=False)
    field = field_from_heightfield(scene)
    riser = first_riser(scene, *spawn['spawn_xz'])
    summaries = []
    for speed in speeds:
        actor = UnifiedTerrainBackend(controller, seed, scene)
        actor.configure_runtime(dict(PROFILE))
        pieces, rows, failure = [seed.clone()], [], None
        for block in range(max_blocks):
            observed = actor.history.clone()
            try:
                result = actor.predict_plan([0., speed], 0.)
            except (ValueError, RuntimeError) as error:
                failure = repr(error)
                break
            played = result['planned_motion'].clone()
            _, surface = geometry.geometry(played)
            distance, _, known = field.query(surface)
            gap, ground_known = heightfield_gap(surface, scene)
            elevated = surface[..., 1] - gap > riser['floor_y_m'] + .08
            depths = torch.where(known, (-distance).clamp_min(0), 0)
            stair_sole = torch.where(elevated & ground_known, depths, 0)[:, :, :geometry.sole_count]
            hit = (stair_sole.amax(-1)[0] > .02).nonzero().flatten()
            root_plan = np.asarray(result['debug']['root_stage_world_xyz_m'])
            rows.append(dict(frame=block * 8, observed_root_z_m=float(observed[0, -1, 2]),
                distance_to_first_riser_m=riser['z_m'] - float(observed[0, -1, 2]),
                neural_generation=bool(actor.last_generation['neural_generation']),
                root_future_max_y_m=float(root_plan[:, 1].max()), root_future_end_z_m=float(root_plan[-1, 2]),
                stair_sole_peak_m=float(stair_sole.max()),
                first_stair_sole_hit_2cm_frame=block * 8 + int(hit[0]) if len(hit) else None))
            pieces.append(played)
            actor.history = result['published_history'].clone()
            if float(played[0, -1, 2]) >= float(spawn['center_xz'][1]):
                break
        world = torch.cat(pieces, 1)
        metrics = exact_skin_metrics(geometry, world, scene)
        crossed = bool(world[0, -1, 2] >= float(spawn['center_xz'][1]))
        hits = [r['first_stair_sole_hit_2cm_frame'] for r in rows if r['first_stair_sole_hit_2cm_frame'] is not None]
        lateral = float((world[0, :, 0] - spawn['spawn_xz'][0]).abs().max())
        row = dict(speed_request_mps=speed, actual_frames=world.shape[1], max_requested_frames=4 + 8 * max_blocks,
            completion_status='goal_reached' if crossed else 'incomplete_within_explicit_time_budget',
            output_speed_bounds=None,
            reached_after_stair_goal=crossed, max_lateral_m=lateral, failure=failure, metrics=metrics,
            first_stair_sole_hit_2cm_frame=min(hits) if hits else None, plans=rows,
            traversal_screen_pass=bool(crossed and failure is None and lateral <= .25
                and metrics['body_prism_max_m'] <= .02 and metrics['unknown_body_fraction'] == 0
                and metrics['uncertain_inside_samples'] == 0), quality_approved=False)
        np.savez_compressed(output / f'forward-{speed:g}.npz', global_features=world[0].cpu().numpy(), fps=np.int32(30))
        summaries.append(row)
        write_json(output / 'progress.json', dict(rows=summaries))
    write_json(output / 'complete.json', dict(rows=summaries, no_padding=True, quality_approved=False))
    return summaries


def review(controller, cases, geometry, args, output):
    rep, device = controller.inference.global_motion_rep, args.device
    source = evaluate(controller, cases, geometry, output / 'sources', device)
    scene, seed, placed, spawn = load_world(args, rep, device)
    live = world_review(controller, geometry, scene, placed, spawn, output / 'live_stairs')
    flat_seed = rep.change_first_heading(seed, 0., False, False)
    flat_seed[..., [0, 2]] -= flat_seed[:, -1:, [0, 2]].clone()
    flat_scene = dict(heightfield_y_m=torch.zeros(201, 201),
                      min_point_world_xyz_m=torch.tensor([-50., 0., -50.]), dx_m=torch.tensor(.5))
    flats = flat_runs(controller.inference, controller, flat_seed, flat_scene, output / 'flat')
    result = dict(sources=source, live_stairs=live, flat=flats)
    write_json(output / 'complete.json', result)
    return result


def main(args):
    if not 1 <= args.steps <= 200:
        raise ValueError('Priority-one comparison is bounded to200updates per arm')
    if args.height_readout and (args.arm != 'terminal' or args.objective != 'root'):
        raise ValueError('Height-readout capacity testing requires matched labels and the native Root objective')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(14293)
    np.random.seed(14293)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.initial) != 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979':
        raise ValueError('Both arms must start from the unchanged28941candidate')
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)
    if sha256(args.bundle) != initial['contract']['bundle_sha256'] or sha256(args.cases) != initial['contract']['cases_sha256']:
        raise ValueError('Baseline or source cases changed')
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
    trainable = freeze_for_root_comparison(controller, height_readout=args.height_readout)
    readout = controller.root._conv_output.model[-1]
    fixed_rows = torch.tensor([0, 2, 3, 4], device=args.device)
    readout_before = (readout.weight.detach()[fixed_rows].clone(), readout.bias.detach()[fixed_rows].clone())
    protected_keys = [key for key in controller.state_dict() if key not in trainable]
    protected = state_digest(controller, protected_keys)
    geometry = JointGeometryObjective(inference.global_motion_rep, args.device)
    train, cases, audit = prepare(args.cases, args.flat_cases, geometry, args.device, args.output)
    train, approach_checks = extend_real_approaches(train, args, controller, geometry)
    write_json(args.output / 'approach_checks.json', dict(rows=approach_checks))
    goal_audit = check_goals(controller, train, args.device, args.window_stride)
    write_json(args.output / 'goal_audit.json', dict(rows=goal_audit))
    schema = 'motionbricks.unified_terrain.root_height_readout.v1' if args.height_readout else SCHEMA
    contract = dict(schema=schema, arm=args.arm, initial_sha256=sha256(args.initial),
        bundle_sha256=sha256(args.bundle), cases_sha256=sha256(args.cases), flat_cases_sha256=sha256(args.flat_cases),
        profile=PROFILE, updates=args.steps, effective_batch=4, primitive_exposures=args.steps * 4,
        actual_parc_fraction=.5, actual_bones_fraction=.5, seed=14293,
        command_labels=GOAL_SOURCE if args.arm == 'terminal' else 'legacy fixed approach proxies; explicit negative control',
        trainable_parameters=trainable, shared_encoder_pose_decoder_and_native_frozen=not args.height_readout,
        shared_encoder_pose_decoder_and_other_native_weights_frozen=True,
        root_height_readout_only=args.height_readout,
        root_height_readout_effective_parameters=int(readout.weight[1].numel() + 1) if args.height_readout else 0,
        losses='native components retain their original coefficients; explicit selection for Root-only trainable scope',
        objective=args.objective,
        history='source prefix only', window_stride=args.window_stride,
        real_approach_extension=approach_checks,
        approach_manifest_sha256=sha256(args.approach_dataset / 'native_manifest.json') if approach_checks else None,
        approach_audit_sha256=sha256(args.approach_audit) if approach_checks else None,
        windows='real source windows only; same cyclic rule, explicit coverage stride',
        runtime_changed=False, quality_approved=False)
    write_json(args.output / 'contract.json', contract)
    write_json(args.output / 'state.json', dict(status='baseline_review'))
    baseline = review(controller, cases, geometry, args, args.output / 'baseline')
    # Review cannot affect the matched masking/optimizer RNG schedule.
    torch.manual_seed(14293)
    np.random.seed(14293)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    exposure, losses, schedule = Counter(), [], hashlib.sha256()
    checks = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        current = []
        for case in train:
            windows = source_windows(len(case['world']), args.window_stride)
            start = windows[step % len(windows)]
            source = case['world'][None, start:start + 33].to(args.device)
            label = source_terminal_goal(inference.global_motion_rep.extract_root(source))
            command = label if args.arm == 'terminal' else case['exact_constant_control'].to(args.device)
            if step == 0:
                with torch.no_grad():
                    expected = controller(source[:, :4], command, case['scene'])['world'].clone()
                    altered = source.clone()
                    altered[:, 4:, 1] += 3
                    altered[:, 4:, 5:] += .2
                    same_command = (source_terminal_goal(inference.global_motion_rep.extract_root(altered))
                                    if args.arm == 'terminal' else command)
                    again = controller(altered[:, :4], same_command, case['scene'])['world']
                    torch.testing.assert_close(again, expected, atol=0, rtol=0)
            result = controller(source[:, :4], command, case['scene'], training=True, straight_through=True)
            if step == 0:
                forward_error = float((result['world'].detach() - expected).abs().max())
                torch.testing.assert_close(result['world'].detach(), expected, atol=3e-5, rtol=1e-5)
                checks.append(dict(label=case['label'], future_height_pose_actor_error=0., training_runtime_forward_error=forward_error))
            parts = controller.native_losses(result, source, source[:, :4], legacy_proxy_control_arm=args.arm == 'legacy')
            selected_loss = training_objective(parts, args.objective)
            loss = selected_loss / len(train)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward()
            current.append(float(selected_loss.detach()))
            identity = f'{case["label"]}:{start}'
            exposure[identity] += 1
            schedule.update((identity + '\n').encode())
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or not any(p.grad is not None and p.grad.abs().max() > 0 for p in parameters):
            raise ValueError('Nonfinite or absent Root adapter gradient')
        if any(p.grad is not None for p in controller.parameters() if not p.requires_grad):
            raise ValueError('Protected module acquired gradients')
        optimizer.step()
        torch.testing.assert_close(readout.weight.detach()[fixed_rows], readout_before[0], atol=0, rtol=0)
        torch.testing.assert_close(readout.bias.detach()[fixed_rows], readout_before[1], atol=0, rtol=0)
        losses.append(float(np.mean(current)))
        if (step + 1) % 10 == 0:
            write_json(args.output / 'state.json', dict(status='root_goal_fit', step=step + 1,
                target=args.steps, exposures=(step + 1) * 4, mean_last10_loss=float(np.mean(losses[-10:]))))
            print(f'{args.arm} step={step+1} loss={np.mean(losses[-10:]):.6f}', flush=True)
        if step == 7:
            if state_digest(controller, protected_keys) != protected:
                raise ValueError('Protected weights changed in preflight')
            write_json(args.output / 'preflight.json', dict(checks=checks, frozen_hash_unchanged=True,
                updates=8, root_gradient_norm=float(norm), deployed=False))
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected weights changed')
    checkpoint = dict(schema=schema, contract=contract, adaptation=controller.adaptation.state_dict(),
        steps=args.steps, optimizer=optimizer.state_dict(), exposure=dict(exposure))
    if args.height_readout:
        checkpoint['root_height_readout'] = readout.state_dict()
    torch.save(checkpoint, args.output / 'candidate.pt')
    write_json(args.output / 'state.json', dict(status='final_review', step=args.steps))
    final = review(controller, cases, geometry, args, args.output / 'final')
    write_json(args.output / 'complete.json', dict(contract=contract, baseline=baseline, final=final,
        goal_audit=goal_audit, checks=checks, losses=losses, exposure=dict(exposure),
        source_window_schedule_sha256=schedule.hexdigest(), frozen_hash_unchanged=True,
        nonheight_root_output_rows_unchanged=True,
        candidate_sha256=sha256(args.output / 'candidate.pt'), quality_approved=False, deployed=False))
    write_json(args.output / 'state.json', dict(status='complete', quality_approved=False, deployed=False))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'initial', 'cases', 'flat-cases', 'scene', 'scene-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arm', choices=('legacy', 'terminal'), required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--window-stride', type=int, choices=(1, 24), default=24)
    parser.add_argument('--objective', choices=('joint', 'root'), default='joint')
    parser.add_argument('--height-readout', action='store_true')
    parser.add_argument('--approach-dataset', type=Path)
    parser.add_argument('--approach-audit', type=Path)
    parser.add_argument('--device', default='cuda:0')
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        if arguments.output.exists():
            write_json(arguments.output / 'state.json', dict(status='failed', error=repr(error)))
        raise
