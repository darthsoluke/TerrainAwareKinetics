"""Bounded matched Root/Pose fitting with explicit freeze and checkpoint contracts."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController, PROFILE
from motionbricks.data.unified_pose_training import (POSE_SCHEMA, configure_pose_scope, pose_example,
                                                     pose_teacher, load_pose_checkpoint, decoded_native_losses)
from scripts.check_unified_pose_preconditions import scores
from scripts.pilot_unified_goal_alignment import review
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def build_schedule(cases, steps, mode='per-source-phase-v2'):
    if mode not in ('global-step-phase-v1', 'per-source-phase-v2'):
        raise ValueError('Unknown source-window scheduling contract')
    stairs = {entry: [i for i, c in enumerate(cases) if c.get('entry') == entry] for entry in ('up', 'down')}
    flat = [i for i, c in enumerate(cases) if c['row']['origin'] == 'bones_seed']
    if min(len(stairs['up']), len(stairs['down'])) < 1 or len(flat) < 6:
        raise ValueError('Require real up/down and six BONES sources')
    rng = np.random.default_rng(15114)
    schedule = []
    visits = Counter()
    for step in range(steps):
        ids = [stairs[e][step % len(stairs[e])] for e in ('up', 'down')]
        ids += [flat[(step * 2 + j) % len(flat)] for j in range(2)]
        row = []
        for index in ids:
            count = len(cases[index]['world']) - 32
            # Global-step rotation aliases with source selection when its period is3or6.
            groups = np.array_split(np.arange(count), 3)
            phase = step % 3 if mode == 'global-step-phase-v1' else visits[index] % 3
            group = groups[phase]
            start = int(group[int(rng.integers(len(group)))])
            draw = dict(case=index, label=cases[index]['label'], start=start, mask_seed=int(rng.integers(1, 2**30)))
            if mode == 'per-source-phase-v2':
                draw['phase'] = phase
            row.append(draw)
            visits[index] += 1
        schedule.append(row)
    return schedule


def full_mask_schedule(schedule, fraction):
    flags = [[False] * len(batch) for batch in schedule]
    if fraction == 0:
        return flags
    if fraction != .25:
        raise ValueError('Only the specified quarter-mask factor is supported')
    rng = np.random.default_rng(15115)
    for domain in ('stairs', 'flat'):
        groups = {}
        for step, batch in enumerate(schedule):
            for slot, draw in enumerate(batch):
                if ('stairs' if slot < 2 else 'flat') == domain:
                    groups.setdefault((draw['case'], draw.get('phase', step % 3)), []).append((step, slot))
        counts = {key: int(len(value) * fraction) for key, value in groups.items()}
        remaining = round(sum(map(len, groups.values())) * fraction) - sum(counts.values())
        order = sorted(groups, key=lambda k: (-(len(groups[k]) * fraction - counts[k]), k))
        for key in order[:remaining]:
            counts[key] += 1
        for key, values in groups.items():
            for index in rng.permutation(len(values))[:counts[key]]:
                step, slot = values[index]
                flags[step][slot] = True
    return flags


@torch.no_grad()
def token_review(controller, cases, device):
    rows = []
    for case in cases:
        for start in sorted({0, (len(case['world']) - 33) // 2, len(case['world']) - 33}):
            source = case['world'][None, start:start + 33].to(device)
            example = pose_example(controller, source, case['scene'])
            loss, logits, _, _ = pose_teacher(controller, example, 15114 + start)
            _, all_logits, _, _ = pose_teacher(controller, example, 15114 + start, all_mask=True)
            rows.append(dict(label=case['label'], start=start, teacher_focused_ce=float(loss),
                teacher=scores(logits, example['tokens']), all_mask=scores(all_logits, example['tokens']),
                final=scores(example['result']['pose_logits'][:, :8], example['tokens'])))
    return rows


def main(args):
    joint = args.arm == 'root_pose'
    if joint and (args.teacher_history != 'full' or args.full_mask_fraction or args.joint_preflight is None):
        raise ValueError('Joint scope requires the explicit gradient gate and unchanged full-history/native CE')
    if not 8 <= args.steps <= 300:
        raise ValueError('Pose capacity pair is bounded to8..300actual updates')
    if args.full_mask_fraction not in (0., .25):
        raise ValueError('Only the explicit native or quarter-full-mask comparison is allowed')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(15114); np.random.seed(15114)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    gate = json.loads(args.preflight.read_text())
    if joint:
        joint_gate = json.loads(args.joint_preflight.read_text())
        if not joint_gate['weights_unchanged'] or joint_gate['optimizer_updates'] or not joint_gate['all_checks_passed']:
            raise ValueError('Joint-gradient preflight failed')
    prepared = json.loads(args.train_cases.with_name('complete.json').read_text())
    if not gate['weights_unchanged'] or gate['optimizer_updates'] or gate['future_actor_error']:
        raise ValueError('Causal Pose preconditions failed')
    if sha256(args.train_cases) != prepared['cases_sha256']:
        raise ValueError('Prepared data identity mismatch')
    if sha256(args.initial) != 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979':
        raise ValueError('Use the same preserved live initialization for both arms')
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle, pose_condition_stage=args.pose_stage)
    if args.teacher_history == 'native-sparse-start-v1' and (
            inference._pose_model.args.get('max_num_start_keyframes') != 4
            or inference._pose_model.args.get('no_start_keyframe_prob') != 0.):
        raise ValueError('The sparse teacher requires the checked native MB four-start-keyframe configuration')
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)
    if initial['contract']['bundle_sha256'] != sha256(args.bundle):
        raise ValueError('Baseline mismatch')
    controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
    if args.pose_stage == 'output' and not args.zero_pose_residual:
        raise ValueError('Placement comparison requires a declared matched zero-residual initialization')
    if args.zero_pose_residual:
        with torch.no_grad():
            controller.adaptation['pose'].output.weight.zero_()
    names = configure_pose_scope(controller, args.arm)
    protected_keys = [name for name in controller.state_dict() if name not in names]
    protected = state_digest(controller, protected_keys)
    train = torch.load(args.train_cases, map_location='cpu', weights_only=False)
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    old_flats = torch.load(args.flat_cases, map_location='cpu', weights_only=False)
    from scripts.pilot_unified_terrain import prepare
    geometry = JointGeometryObjective(inference.global_motion_rep, args.device)
    _, review_cases, _ = prepare(args.cases, args.flat_cases, geometry, args.device, args.output)
    schedule = build_schedule(train, args.steps, args.window_schedule)
    mask_flags = full_mask_schedule(schedule, args.full_mask_fraction)
    schedule_hash = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    write_json(args.output / 'schedule.json', schedule)
    write_json(args.output / 'full_mask_schedule.json', mask_flags)
    contract = dict(schema=POSE_SCHEMA, arm=args.arm, initial_sha256=sha256(args.initial),
        pose_condition_stage=args.pose_stage, zero_pose_residual=args.zero_pose_residual,
        bundle_sha256=sha256(args.bundle), data_sha256=sha256(args.train_cases), preflight_sha256=sha256(args.preflight),
        review_cases_sha256=sha256(args.cases), schedule_sha256=schedule_hash, profile=PROFILE,
        updates=args.steps, effective_batch=4, primitive_exposures=args.steps * 4,
        actual_parc_fraction=.5, actual_bones_fraction=.5, seed=15114,
        trainable_parameters=names, trainable_count=sum(p.numel() for p in controller.parameters() if p.requires_grad),
        actual_source_counts=prepared['counts'], requested_source_counts=prepared['requested'],
        window_schedule=args.window_schedule,
        source_count_target_met=prepared['target_counts_met'],
        objective=('native CE + native decoded-motion + native Root; existing coefficients unchanged'
                   if joint else 'native focused-token CE only; original cosine80percent-mask corruption'),
        joint_preflight_sha256=sha256(args.joint_preflight) if joint else None,
        full_mask_fraction=args.full_mask_fraction,
        full_mask_exposures=sum(sum(row) for row in mask_flags),
        full_mask_assignment='stratified by terrain/flat, source identity and early/middle/late windows',
        root_condition=('jointly trained Root-adapter causal prediction, no source future Root'
                        if joint else 'same frozen causal model prediction, no source future Root'),
        history='source only; runtime feedback uses official FK reencoding',
        teacher_history=args.teacher_history,
        runtime_history='unchanged four actual frames for Root/Pose/codec',
        module_mode='eval, including native dropout; deterministic conditioning for matched adaptation',
        no_geometry_training_loss=True, quality_approved=False)
    write_json(args.output / 'contract.json', contract)
    write_json(args.output / 'state.json', dict(status='baseline_review'))
    baseline_tokens = token_review(controller, train + review_cases, args.device)
    baseline = review(controller, review_cases, geometry, args, args.output / 'baseline')
    if args.reference_control:
        reference = json.loads(args.reference_control.read_text())
        keys = ('initial_sha256', 'bundle_sha256', 'data_sha256', 'profile', 'updates',
                'full_mask_fraction', 'pose_condition_stage', 'zero_pose_residual')
        if not joint:
            keys += ('trainable_parameters',)
        for key in keys:
            if contract[key] != reference['contract'][key]:
                raise ValueError('Schedule control contract differs: ' + key)
        if baseline != reference['baseline'] or baseline_tokens != reference['baseline_tokens']:
            raise ValueError('Schedule control initial results differ')
        write_json(args.output / 'baseline_parity.json', dict(passed=True, reference=str(args.reference_control)))
    parameters = [p for p in controller.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    losses, checks, anchors, exposures = [], [], [], Counter()
    component_losses = []
    corruption_hash = hashlib.sha256()
    history_mask_hash, history_counts = hashlib.sha256(), Counter()
    for step, batch in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for slot, draw in enumerate(batch):
            case = train[draw['case']]
            source = case['world'][None, draw['start']:draw['start'] + 33].to(args.device)
            example = pose_example(controller, source, case['scene'], actor_gradient=joint)
            if step == 0:
                anchors.append((source.clone(), case['scene'], example['result']['root_output']['pred_global_root_values'].clone()))
                with torch.no_grad():
                    mutated = source.clone(); mutated[:, 4:, 1] += 3; mutated[:, 4:, 5:] += .2
                    replay = controller(mutated[:, :4], example['command'], case['scene'])['world']
                    normal = controller(source[:, :4], example['command'], case['scene'])['world']
                    torch.testing.assert_close(replay, normal, atol=0, rtol=0)
                    torch.testing.assert_close(normal, example['result']['world'], atol=3e-5 if joint else 0, rtol=0)
                checks.append(dict(label=case['label'], future_target_actor_difference=0.,
                    straight_through_forward_difference=float((normal - example['result']['world']).abs().max())))
            all_mask = mask_flags[step][slot]
            loss, logits, target, corrupted = pose_teacher(controller, example, draw['mask_seed'],
                all_mask=all_mask, history_policy=args.teacher_history)
            if joint:
                parts = decoded_native_losses(controller, example, source)
                component_losses.append(dict(step=step + 1, slot=slot, ce=float(loss.detach()),
                    **{k: float(v.detach()) for k, v in parts.items()}))
                loss = loss + parts['native_motion'] + parts['native_root']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite Pose CE')
            (loss / 4).backward()
            corruption_hash.update(corrupted.detach().cpu().numpy().tobytes())
            corruption_hash.update(target['focused_token_mask'].cpu().numpy().tobytes())
            history_mask_hash.update(target['history_condition_mask'].cpu().numpy().tobytes())
            history_counts[int(target['history_condition_mask'].sum())] += 1
            step_losses.append(float(loss.detach()))
            exposures[f'{case["label"]}:{draw["start"]}'] += 1
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Absent/nonfinite selected Pose gradient')
        if any(p.grad is not None for p in controller.parameters() if not p.requires_grad):
            raise ValueError('Protected module acquired gradient')
        optimizer.step()
        losses.append(float(np.mean(step_losses)))
        if step == 7:
            if state_digest(controller, protected_keys) != protected:
                raise ValueError('Protected weights changed')
            root_checks = []
            for source, scene, expected in anchors:
                actual = pose_example(controller, source, scene)['result']['root_output']['pred_global_root_values']
                if not joint:
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                root_checks.append(float((actual - expected).abs().max()))
            write_json(args.output / 'preflight.json', dict(checks=checks, root_prediction_difference=root_checks,
                protected_hash_unchanged=True, updates=8, gradient_norm=float(norm)))
        if (step + 1) % 10 == 0:
            write_json(args.output / 'state.json', dict(status='pose_fit', step=step + 1, target=args.steps,
                exposures=(step + 1) * 4, mean_last10_loss=float(np.mean(losses[-10:])),
                objective='native_joint' if joint else 'native_ce'))
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected weights changed')
    delta = {name.removeprefix('pose.'): value.detach().cpu().clone() for name, value in controller.named_parameters()
             if name in names and name.startswith('pose.')}
    checkpoint = dict(schema=POSE_SCHEMA, contract=contract, adaptation=controller.adaptation.state_dict(),
        pose_delta=delta, steps=args.steps, optimizer=optimizer.state_dict(), exposures=dict(exposures))
    torch.save(checkpoint, args.output / 'candidate.pt')
    # Reload every trained native tensor before final review, catching incomplete exports.
    test_source, test_scene, _ = anchors[0]
    expected = pose_example(controller, test_source, test_scene)['result']['world'].clone()
    loaded = torch.load(args.output / 'candidate.pt', map_location='cpu', weights_only=False)
    load_pose_checkpoint(controller, loaded, sha256(args.bundle))
    torch.testing.assert_close(pose_example(controller, test_source, test_scene)['result']['world'], expected, atol=0, rtol=0)
    write_json(args.output / 'state.json', dict(status='final_review', step=args.steps))
    final_tokens = token_review(controller, train + review_cases, args.device)
    final = review(controller, review_cases, geometry, args, args.output / 'final')
    write_json(args.output / 'complete.json', dict(contract=contract, baseline=baseline, final=final,
        baseline_tokens=baseline_tokens, final_tokens=final_tokens, losses=losses, checks=checks,
        component_losses=component_losses,
        schedule_sha256=schedule_hash, corruption_sha256=corruption_hash.hexdigest(),
        history_condition_mask_sha256=history_mask_hash.hexdigest(), history_keyframe_counts=dict(history_counts),
        protected_hash_unchanged=True, native_pose_export_reload_difference=0.,
        candidate_sha256=sha256(args.output / 'candidate.pt'), quality_approved=False, deployed=False))
    write_json(args.output / 'state.json', dict(status='complete', quality_approved=False, deployed=False))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'train-cases', 'preflight', 'cases', 'flat-cases', 'scene', 'scene-receipt', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--arm', choices=('adapter', 'capacity', 'root_pose'), required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--full-mask-fraction', type=float, default=0.)
    p.add_argument('--pose-stage', choices=('input', 'output'), default='input')
    p.add_argument('--zero-pose-residual', action='store_true')
    p.add_argument('--window-schedule', choices=('global-step-phase-v1', 'per-source-phase-v2'), default='per-source-phase-v2')
    p.add_argument('--reference-control', type=Path)
    p.add_argument('--teacher-history', choices=('full', 'native-sparse-start-v1'), default='full')
    p.add_argument('--joint-preflight', type=Path)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'state.json', dict(status='failed', error=repr(error)))
        raise
