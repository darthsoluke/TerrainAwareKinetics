"""Bounded STOP-only command learning; approved moving branch remains immutable."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.stop_command_adapter import StopCommandAdapter, STOP_ADAPTER_VERSION
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.evaluate_joint_sdf import save_motion
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare(model, args):
    rep = build_native_representation(args.dataset/'stats')
    data = TerrainLearningDataset(args.dataset, rep, 'train')
    audit = json.loads(args.audit.read_text())
    cases, used, screens = [], set(), []
    geometry = JointGeometryObjective(model.inference.global_motion_rep, 'cuda:0')
    for example in audit['examples']:
        if (example['split'] != 'train' or example['domain'] != 'bones'
                or not example['restart'] or example['source_id'] in used
                or example['rest_end']-example['rest_start'] < 40):
            continue
        begin = example['rest_start']-24
        if begin < 0 or begin+33 > example['frames']:
            continue
        item = data.full_reference_item(example['source_index'])
        if item['row']['source_id'] != example['source_id']:
            raise ValueError('Source identity differs from the stop audit')
        source = item['world'][None, begin:begin+33].cuda()
        input_speed = float((source[0, 3, [0, 2]]-source[0, 2, [0, 2]]).norm()*30)
        terminal_speed = float(torch.diff(source[0, -6:, [0, 2]], dim=0).norm(dim=-1).max()*30)
        if input_speed < .3 or terminal_speed > .12:
            continue
        pair = []
        for phase, start in [('brake', begin), ('rest', example['rest_start'])]:
            world = item['world'][None, start:start+33].cuda()
            if world.shape[1] != 33:
                raise ValueError('No padding is permitted')
            yaw = torch.atan2(world[:, 3, 4], world[:, 3, 3])
            intent = SoftCommandState().intent(world[:, :4], world.new_zeros(1, 2), yaw, 8)
            with torch.no_grad():
                metrics = exact_skin_metrics(geometry, world, item['scene'])
            passed = (metrics['body_prism_max_m'] is not None
                and metrics['body_prism_max_m'] <= .02 and metrics['unknown_body_fraction'] == 0
                and metrics['uncertain_inside_samples'] == 0)
            screens.append(dict(source_id=example['source_id'], phase=phase,
                source_start=start, passed=passed, **metrics))
            write(args.output/'source_screen.json', dict(rows=screens))
            pair.append(dict(world=world, scene=item['scene'], intent=intent,
                metadata=dict(source_id=example['source_id'], phase=phase,
                    source_start=start, source_end=start+33,
                    source_screen=metrics,
                    provenance='Inferred STOP from unmodified real future braking/rest, not recorded keys')))
        if not all(row['passed'] for row in screens[-2:]):
            continue
        used.add(example['source_id'])
        cases.extend(pair)
        if len(used) == 4:
            break
    if len(used) != 4:
        raise ValueError('Need four distinct real moving-stop-restart sources')
    return cases


@torch.no_grad()
def review(model, args, folder, reference=None, stop_goal_policy='retained', transition=False):
    if stop_goal_policy not in ('retained', 'current'):
        raise ValueError('Unknown causal stop-goal policy')
    folder.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.artifact/'export.json').read_text())
    sources = torch.load(args.artifact/'offline-cases.pt', map_location='cpu', weights_only=False)
    rep = model.inference.global_motion_rep
    geometry = JointGeometryObjective(rep, 'cuda:0')
    rows = []
    for case, entry in zip(sources, receipt['entries']):
        with np.load(args.artifact/(entry['entry']+'-scene.npz'), allow_pickle=False) as data:
            initial = torch.from_numpy(data['initial_history'].copy()).cuda()
            scene = {k:torch.from_numpy(data[k].copy()).cuda()
                     for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
        velocity = entry['command_velocity_mps']
        yaw = entry['command_heading']
        direction = initial.new_tensor(velocity)
        direction /= direction.norm()
        goal = case['world'][case['evaluation_goal_source_frame']-case['start_frame']].cuda()
        _, known_start = heightfield_gap(initial[0, -1:, :3], scene)
        initial_gap, _ = heightfield_gap(initial[0, -1:, :3], scene)
        goal_gap, known_goal = heightfield_gap(goal[None, :3], scene)
        if not known_start.all() or not known_goal.all():
            raise ValueError('Source endpoints must be on known terrain')
        initial_floor = initial[0, -1, 1]-initial_gap[0]
        goal_floor = goal[1]-goal_gap[0]
        for profile in ('cruise', 'release_resume'):
            history = initial.clone()
            driver = SoftCommandState()
            if transition:
                from motionbricks.data.stop_transition_adapter import CommandTransitionState, set_transition_active
                phase = CommandTransitionState()
            parts, commands, stops = [], [], []
            failure = None
            goal_reached = False
            for block in range(24):
                stopped = profile == 'release_resume' and 3 <= block < 13
                v = [0., 0.] if stopped else velocity
                stop_goal = history[:, -1, [0, 2]] if stopped and stop_goal_policy == 'current' else None
                intent = driver.intent(history, v, yaw, 8, goal_xz=stop_goal)
                if transition:
                    observed_speed = float(((history[:,-1,[0,2]]-history[:,0,[0,2]])*10).norm())
                    set_transition_active(model, phase.update(v, observed_speed))
                try:
                    result = model(history, intent, [scene])
                except (ValueError, RuntimeError) as error:
                    failure = str(error)
                    break
                part = result['world'][:, 4:12].clone()
                history = reencode_played_history(result['world'][:, 8:12], rep)
                parts.append(part)
                commands.append(dict(block=block, requested_velocity_mps=v, stopped=stopped))
                if stopped:
                    stops.append(part)
                goal_reached = float((history[0, -1, [0, 2]]-goal[[0, 2]]) @ direction) >= 0
                if goal_reached and (profile == 'cruise' or block >= 16):
                    break
            row = dict(entry=entry['entry'], profile=profile, frames=len(parts)*8,
                requested_max_frames=192, failure=failure, goal_reached=goal_reached,
                stop_goal_policy=stop_goal_policy,
                transition_conditioning=transition,
                quality_approved=False, postprocessing=None, commands=commands)
            if parts:
                world = torch.cat(parts, 1)
                row.update(exact_skin_metrics(geometry, world, scene))
                if profile == 'cruise' and reference is not None:
                    with np.load(reference/(entry['entry']+'-'+profile+'.npz'), allow_pickle=False) as old:
                        previous = torch.from_numpy(old['world']).to(world)
                    torch.testing.assert_close(world[0], previous, atol=0, rtol=0)
                    row['approved_moving_output_error'] = float((world[0]-previous).abs().max())
                if len(stops) == 10:
                    stopped_world = torch.cat(stops, 1)
                    tail = stopped_world[0, -15:, [0, 2]]
                    speed = torch.diff(tail, dim=0).norm(dim=-1)*30
                    drift = (tail[-1]-tail[0]).norm()
                    root = stopped_world[0, -1, :3]
                    gap, known = heightfield_gap(root[None], scene)
                    floor = root[1]-gap[0]
                    low, high = torch.minimum(initial_floor, goal_floor), torch.maximum(initial_floor, goal_floor)
                    inside = bool(known.all() and low+.05 < floor < high-.05)
                    resume_progress = float((history[0, -1, [0, 2]]-root[[0, 2]]) @ direction)
                    row.update(stop_tail_speed_p90_mps=float(torch.quantile(speed, .9)),
                        stop_tail_drift_m=float(drift), stopped_inside_flight=inside,
                        resume_progress_m=resume_progress,
                        stop_resume_screen=bool(torch.quantile(speed, .9) <= .15
                            and drift <= .08 and inside and resume_progress >= .25),
                        stop_scope='Requested2.667s; natural braking allowed. Native Root-floor location screen, not proof of foot support or final visual approval.')
                save_motion(folder/(entry['entry']+'-'+profile+'.npz'), world, scene, rep, row)
            rows.append(row)
            write(folder/'progress.json', dict(rows=rows))
            print(json.dumps({k:v for k,v in row.items() if k != 'commands'}), flush=True)
    write(folder/'complete.json', dict(rows=rows, quality_approved=False))
    return rows


def main(args):
    if not 1 <= args.updates <= 50:
        raise ValueError('Hard50-update limit for this targeted pilot')
    args.output.mkdir(parents=True, exist_ok=False)
    digest = sha256(args.checkpoint)
    if digest != 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf':
        raise ValueError('Approved checkpoint differs')
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model, _ = load_model(saved)
    model.requires_grad_(False)
    model.control = StopCommandAdapter(model.control)
    model.eval()
    fixed = [k for k in model.state_dict() if not k.startswith('control.stop.')]
    frozen_digest = state_digest(model, fixed)
    cases = prepare(model, args)
    write(args.output/'contract.json', dict(parent_sha256=digest, adapter_version=STOP_ADAPTER_VERSION,
        actual_updates=args.updates, batch=4, actor_examples=args.updates*4,
        trainable='Only new STOP command-embedding residual; all approved parameters frozen',
        native_losses='Existing native motion/Root/token objectives, coefficients unchanged',
        source_cases=[c['metadata'] for c in cases], weak_intent=True,
        source_body='Real BONES SOMA; no PARC-body inverse bridge or fake stair stop label',
        future_truth_in_actor=False, postprocessing=None, quality_approved=False))
    baseline = review(model, args, args.output/'baseline')
    optimizer = torch.optim.Adam(model.control.stop.parameters(), lr=5e-5, weight_decay=0.)
    logs = []
    for step in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        batch_rows = []
        for offset in range(4):
            case = cases[(step*4+offset) % len(cases)]
            source = case['world']
            history = source[:, :4]
            if step == 0 and offset == 0:
                with torch.no_grad():
                    plain = model(history, case['intent'], [case['scene']])['world'].clone()
                    altered = source.clone()
                    altered[:, 4:] += 3.
                    changed = model(history, case['intent'], [case['scene']], source=altered)['world']
                torch.testing.assert_close(plain, changed, atol=0, rtol=0)
            result = model(history, case['intent'], [case['scene']], source=source, straight_through=True)
            if step == 0 and offset == 0:
                torch.testing.assert_close(plain, result['world'], atol=3e-5, rtol=1e-5)
                write(args.output/'preflight.json', dict(future_target_actor_error=float((plain-changed).abs().max()),
                    training_forward_error=float((plain-result['world']).abs().max())))
            losses = model.native_losses(result, source)
            loss = losses['native_total']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            (loss/4).backward()
            batch_rows.append(dict(**case['metadata'], loss=float(loss.detach())))
        gradient = torch.nn.utils.clip_grad_norm_(model.control.stop.parameters(), .5)
        if not torch.isfinite(gradient) or gradient <= 0:
            raise ValueError('No finite nonzero STOP learning gradient')
        optimizer.step()
        if state_digest(model, fixed) != frozen_digest:
            raise ValueError('Approved model parameters changed')
        row = dict(step=step+1, gradient_norm=float(gradient), samples=batch_rows)
        logs.append(row)
        write(args.output/'progress.json', row)
        if step % 10 == 0:
            print(json.dumps(dict(step=step+1, loss=sum(r['loss'] for r in batch_rows)/4)), flush=True)
    torch.save(dict(schema='motionbricks.stop_command_adapter.pilot.v1', adapter_version=STOP_ADAPTER_VERSION,
        parent_sha256=digest, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
        actual_updates=args.updates, actor_examples=args.updates*4,
        contract=saved['contract'], quality_approved=False), args.output/'weights.pt')
    final = review(model, args, args.output/'final', reference=args.output/'baseline')
    write(args.output/'complete.json', dict(baseline=baseline, final=final, logs=logs,
        actual_updates=args.updates, actor_examples=args.updates*4, approved_parameters_unchanged=True,
        parent_sha256=digest, candidate_sha256=sha256(args.output/'weights.pt'), quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'audit', 'artifact', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=50)
    main(parser.parse_args())
