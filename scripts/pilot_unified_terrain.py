"""Bounded source-history capacity fit; no synthetic recovery supervision."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE, SCHEMA
from motionbricks.data.unified_training_goals import source_terminal_goal, GOAL_SOURCE
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@torch.no_grad()
def prepare(cases_path, flat_path, geometry, device, output):
    stairs = torch.load(cases_path, map_location='cpu', weights_only=False)
    flats = [c for c in torch.load(flat_path, map_location='cpu', weights_only=False) if c['name'] in ('idle', 'walk')]
    if len(flats) != 2 or any(c['row']['origin'] != 'bones_seed' for c in flats):
        raise ValueError('Require the two real BONES reference clips')
    audit = []
    for case in flats:
        hf = torch.as_tensor(case['scene']['heightfield_y_m'])
        if not torch.isfinite(hf).all() or not hf.eq(hf[0, 0]).all():
            raise ValueError('BONES requires its existing complete fixed plane')
        world = case['world'][None].to(device)
        metrics = exact_skin_metrics(geometry, world, case['scene'])
        _, points = geometry.geometry(world)
        nearest_sole = (points[0, :, :geometry.sole_count, 1]-float(hf[0, 0])).abs().amin(-1)
        row = dict(label=case['label'], origin=case['row']['origin'], frames=world.shape[1],
                   source_id=case['row'].get('source_id'), fixed_floor_m=float(hf[0, 0]),
                   closest_sole_gap_p95_m=float(torch.quantile(nearest_sole, .95)), metrics=metrics)
        audit.append(row)
        write_json(output/'bones_source_audit.json', dict(rows=audit, source_modified=False))
        if (metrics['body_prism_max_m'] > .02 or metrics['unknown_body_fraction']
                or metrics['uncertain_inside_samples'] or row['closest_sole_gap_p95_m'] > .05):
            raise ValueError('BONES source failed the fixed-plane eligibility screen')
        velocity = ((world[0, -1, [0, 2]]-world[0, 3, [0, 2]])/((world.shape[1]-4)/30)
                    if case['name'] == 'walk' else world.new_zeros(2))
        yaw = world[0, 3, 3:5]
        case['exact_constant_control'] = torch.cat((velocity, yaw))[None].cpu()
        case['native_segments'] = (len(case['world'])-9)//24
        case['native_split'] = 'train'
    for case in stairs:
        case['name'] = 'stairs'
    train = [c for c in stairs if c['native_split'] == 'train']+flats
    if len(train) != 4:
        raise ValueError('Expected two stairs and two flat sources')
    return train, stairs+flats, audit


@torch.no_grad()
def evaluate(controller, cases, geometry, output, device):
    output.mkdir(parents=True, exist_ok=False)
    rep = controller.inference.global_motion_rep
    rows = []
    for case in cases:
        world = case['world'][None].to(device)
        command = case['exact_constant_control'].to(device)
        teacher_errors = []
        for start in range(0, world.shape[1]-32, 24):
            window_goal = source_terminal_goal(rep.extract_root(world[:, start:start+33]))
            prediction = controller(world[:, start:start+4], window_goal, case['scene'])['world']
            pj = rep.inverse(prediction[:, 4:28], False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
            sj = rep.inverse(world[:, start+4:start+28], False, return_all=False, joint_positions_from='global_rot_data')['posed_joints']
            teacher_errors.append(float((pj-sj).norm(dim=-1).mean()))
        backend = UnifiedTerrainBackend(controller, world[:, :4], case['scene'])
        backend.configure_runtime(dict(PROFILE))
        pieces = [world[:, :4].clone()]
        failure = None
        for _ in range(case['native_segments']*3):
            try:
                value = backend.predict_plan(command[0, :2], torch.atan2(command[0, 3], command[0, 2]))
                pieces.append(value['planned_motion'].clone())
                backend.history = value['published_history'].clone()
            except (ValueError, RuntimeError) as error:
                failure = repr(error)
                break
        played = torch.cat(pieces, 1)
        metrics = exact_skin_metrics(geometry, played, case['scene'])
        intended_end = 3+case['native_segments']*24
        source_delta = world[0, intended_end, [0, 2]]-world[0, 3, [0, 2]]
        length = source_delta.norm()
        axis = source_delta/length.clamp_min(1e-6)
        displacement = played[0, -1, [0, 2]]-played[0, 3, [0, 2]]
        progress = float(displacement@axis/length.clamp_min(1e-6))
        lateral = float((displacement-axis*(displacement@axis)).norm())
        row = dict(label=case['label'], native_split=case['native_split'], source_id=case['row'].get('source_id'),
                   actual_frames=played.shape[1], requested_frames=intended_end+1, failure=failure,
                   teacher_history_world_mpjpe_m=float(np.mean(teacher_errors)),
                   teacher_history_goal=GOAL_SOURCE,
                   rollout_goal='fixed approach proxy, not a source reconstruction condition',
                   source_relative_progress=progress, final_lateral_m=lateral,
                   final_source_root_y_error_m=float((played[0, -1, 1]-world[0, intended_end, 1]).abs()),
                   metrics=metrics, quality_approved=False)
        np.savez_compressed(output/(case['label']+'.npz'), global_features=played[0].cpu().numpy(), fps=np.int32(30))
        rows.append(row)
        write_json(output/'progress.json', dict(rows=rows))
    write_json(output/'complete.json', dict(rows=rows, no_padding=True, quality_approved=False))
    return rows


def main(args):
    if not 1 <= args.steps <= 200:
        raise ValueError('This capacity pilot is bounded to 200 updates')
    gate = json.loads(args.gate.read_text())
    if gate.get('status') != 'complete' or gate.get('quick') or not gate.get('native_hash_unchanged'):
        raise ValueError('Full real-model gate required before this fit')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(8214)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.bundle) != gate['baseline_sha256']:
        raise ValueError('Approved bundle hash mismatch')
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    keys = list(inference.state_dict())
    frozen = state_digest(inference, keys)
    geometry = JointGeometryObjective(inference.global_motion_rep, args.device)
    write_json(args.output/'state.json', dict(status='source_audit'))
    train, review, audit = prepare(args.cases, args.flat_cases, geometry, args.device, args.output)
    controller = UnifiedTerrainController(inference, bundle)
    contract = dict(schema=SCHEMA, purpose='source-history capacity pilot, not a recovery curriculum',
                    bundle_sha256=sha256(args.bundle), cases_sha256=sha256(args.cases), flat_cases_sha256=sha256(args.flat_cases),
                    updates=args.steps, effective_batch=4, primitive_exposures=args.steps*4,
                    actual_parc_fraction=.5, actual_bones_fraction=.5, profile=PROFILE,
                    losses='unchanged native reconstruction, Root and masked-token objectives',
                    command_labels=GOAL_SOURCE + '; source-derived labels, not recorded keys; fixed rollout goals remain review commands',
                    augmentation='none in this first wiring/capacity comparison',
                    history='unaltered source prefix only; no invented recovery targets',
                    frozen_modules='original Root/Pose/codec and original terrain modules',
                    review_only_labels=[c['label'] for c in review if c['native_split'] != 'train'])
    write_json(args.output/'contract.json', contract)
    write_json(args.output/'state.json', dict(status='baseline_review'))
    baseline = evaluate(controller, review, geometry, args.output/'baseline', args.device)
    optimizer = torch.optim.Adam(controller.adaptation.parameters(), lr=5e-5)
    exposure = Counter()
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        per_source = []
        for case in train:
            windows = list(range(0, len(case['world'])-32, 24))
            start = windows[step % len(windows)]
            source = case['world'][None, start:start+33].to(args.device)
            command = source_terminal_goal(inference.global_motion_rep.extract_root(source))
            result = controller(source[:, :4], command, case['scene'],
                                training=True, straight_through=True)
            parts = controller.native_losses(result, source, source[:, :4])
            loss = parts['native_total']/len(train)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite capacity loss')
            loss.backward()
            per_source.append(float(parts['native_total'].detach()))
            exposure[case['label']+':'+str(start)] += 1
        norm = torch.nn.utils.clip_grad_norm_(controller.adaptation.parameters(), .5)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite capacity gradient')
        optimizer.step()
        losses.append(float(np.mean(per_source)))
        if (step+1) % 10 == 0:
            write_json(args.output/'state.json', dict(status='source_history_fit', step=step+1, target=args.steps,
                primitive_exposures=(step+1)*4, mean_last10_native_loss=float(np.mean(losses[-10:]))))
    if state_digest(inference, keys) != frozen or any(p.grad is not None for p in inference.parameters()):
        raise ValueError('Protected native weights or gradients changed')
    torch.save(dict(schema=SCHEMA, contract=contract, adaptation=controller.adaptation.state_dict(),
                    optimizer=optimizer.state_dict(), steps=args.steps, exposure=dict(exposure)), args.output/'candidate.pt')
    write_json(args.output/'state.json', dict(status='final_review', step=args.steps))
    final = evaluate(controller, review, geometry, args.output/'final', args.device)
    write_json(args.output/'complete.json', dict(contract=contract, baseline=baseline, final=final,
        native_hash_unchanged=True, losses=losses, exposure=dict(exposure), source_audit=audit,
        candidate_sha256=sha256(args.output/'candidate.pt'), quality_approved=False,
        deployed=False, note='Source-history fit is not proof of closed-loop traversal or WASD quality.'))
    write_json(args.output/'state.json', dict(status='complete', quality_approved=False, deployed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'gate', 'cases', 'flat-cases', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/'state.json', dict(status='failed', error=repr(error)))
        raise
