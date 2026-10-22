"""Frozen endpoint review with source-defined support and a valid command clock."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_3d_scene import BatchedSceneField, CompositeSceneField, path_observation
from motionbricks.data.pfnn_frame_actor import advance, LOSS_WEIGHTS
from motionbricks.data.pfnn_frame_actor_v2 import FrameActorV2
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset, sha256
from motionbricks.data.unified_scene_field import VolumetricEgoSampler


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return dict(count=0, p50=None, p90=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite metric')
    return dict(count=int(values.size), p50=float(np.median(values)), p90=float(np.quantile(values, .9)))


def motion_measures(predicted_joints, source_joints, previous_source, contacts, feet):
    """All scoring masks come from source, including both interval endpoints."""
    if predicted_joints.shape != source_joints.shape or len(contacts) != len(source_joints) + 1:
        raise ValueError('Prediction, source and contact clocks differ')
    error = np.linalg.norm(predicted_joints - source_joints, axis=-1)
    p = np.concatenate((previous_source[None], predicted_joints))
    s = np.concatenate((previous_source[None], source_joints))
    pv, sv = np.diff(p[:, feet], axis=0) * 30, np.diff(s[:, feet], axis=0) * 30
    stable = contacts[1:] & contacts[:-1]
    return dict(stance_error=error[:, feet][contacts[1:]], swing_error=error[:, feet][~contacts[1:]],
                body_error=error.ravel(), source_stance_speed=np.linalg.norm(sv, axis=-1)[stable],
                source_mask_speed=np.linalg.norm(pv, axis=-1)[stable],
                stance_velocity_error=np.linalg.norm(pv - sv, axis=-1)[stable])


def loss_component_probe():
    """Output-coordinate local probe, not a parameter-space intervention."""
    rows = []
    for axes in (1, 3):
        for delta in (.005, .01, .02, .05):
            error = torch.zeros(1, 1, 30, 3, dtype=torch.float64)
            error[:, :, :4, :axes] = delta
            error.requires_grad_()
            body = F.smooth_l1_loss(error / .05, torch.zeros_like(error)) * LOSS_WEIGHTS['body_fk']
            boundary = F.smooth_l1_loss(error * 30 / .5, torch.zeros_like(error)) * LOSS_WEIGHTS['boundary_joint_velocity']
            total = body + boundary
            gradient = torch.autograd.grad(total, error)[0]
            rows.append(dict(axes=axes, coordinate_error_m=delta,
                             foot_error_norm_m=delta * np.sqrt(axes), weighted_body=float(body.detach()),
                             weighted_boundary=float(boundary.detach()), weighted_sum=float(total.detach()),
                             summed_coordinate_gradient=float(gradient.abs().sum())))
    return rows


@torch.no_grad()
def rollout(actor, rep, item, field, sampler, *, teacher):
    source = item['world'][None].to(next(actor.parameters()).device)
    commands = item['command_sequence']['command'][None].to(source)
    frames = actor.frames
    # commands[i] corresponds to current source frame i+3, emitted frame i+4.
    end = min(source.shape[1], commands.shape[1] + 4)
    history = source[:, :frames].clone()
    state = actor.initial_state(1)
    emitted = []
    for t in range(frames, end):
        command = commands[:, t - 4]
        coarse = sampler.observe(history[:, -4:, :5], command[:, :2], field, .8)
        root = actor.root_step(history, command, coarse, state)
        fine = path_observation(sampler, history[:, -4:, :5], root['path_world_xz'], coarse, field)
        pred = actor.pose_step(history, command, fine, root)
        frame = advance(rep, history, pred)[:, -1:].clone()
        emitted.append(frame)
        history = source[:, t + 1 - frames:t + 1].clone() if teacher else torch.cat((history[:, 1:], frame), dim=1)
        state = pred['state']
    generated = torch.cat(emitted, dim=1)
    pj = rep.inverse(generated, False, return_all=False, joint_positions_from='global_rot_data')['posed_joints'][0]
    sj = rep.inverse(source[:, frames - 1:end], False, return_all=False,
                     joint_positions_from='global_rot_data')['posed_joints'][0]
    contacts = source[0, frames - 1:end, rep.indices['foot_contacts']] > .5
    return generated.cpu(), pj.cpu().numpy(), sj.cpu().numpy(), contacts.cpu().numpy()


def main(args):
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    cpu_rep = build_native_representation(args.stats)
    ds = PFNNFrameDataset(args.cache, args.pfnn, cpu_rep, 'validation')
    counts, selected = Counter(), []
    for i, row in enumerate(ds.records):
        if counts[row['category']] < 8:
            counts[row['category']] += 1
            selected.append(i)
    if len(selected) != 48 or any(count != 8 for count in counts.values()):
        raise ValueError('Expected six strata of eight rows')
    identities = [dict(ds.records[i], dataset_index=i) for i in selected]
    items = [ds[i] for i in selected]
    rep = build_native_representation(args.stats).to(args.device)
    feet = rep.skeleton.foot_joint_idx
    fields = [BatchedSceneField([CompositeSceneField(item['scene'])], device=args.device) for item in items]
    sampler = VolumetricEgoSampler()
    report = dict(optimizer_updates=0, new_training_exposures=0, selection=identities,
                  categories=dict(counts), source_mask='both source interval endpoints',
                  frames_per_rollout=14, held_command_frames=0,
                  families=dict(Counter(r['source_group'] for r in identities)),
                  original_windows=len(set((r['source_id'].split(':terrain:')[0], r['start']) for r in identities)),
                  loss_component_probe=loss_component_probe(), arms={})
    baseline = None
    for arm in 'ABC':
        path = args.root / ('arm_' + arm) / 'weights.pt'
        before = sha256(path)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        plan = saved['plan']
        actor = FrameActorV2(rep, backbone=plan['backbone'], geometry=plan['geometry'], frames=plan['frames']).to(args.device)
        actor.load_state_dict(saved['model_state'], strict=True)
        actor.eval().requires_grad_(False)
        result = dict(checkpoint_sha256=before, parameters=sum(p.numel() for p in actor.parameters()), modes={})
        for mode in ('teacher', 'closed'):
            all_values, rows, trajectories, baseline_values = {}, [], [], {}
            for i, (item, field) in enumerate(zip(items, fields)):
                world, pj, sj, contacts = rollout(actor, rep, item, field, sampler, teacher=mode == 'teacher')
                values = motion_measures(pj, sj[1:], sj[0], contacts, feet)
                for key, value in values.items():
                    all_values.setdefault(key, []).append(value)
                if mode == 'teacher' and arm == 'A':
                    original = motion_measures(sj[:-1], sj[1:], sj[0], contacts, feet)
                    for key in ('stance_error', 'swing_error', 'body_error'):
                        baseline_values.setdefault(key, []).append(original[key])
                predicted_contact = world[0, :, rep.indices['foot_contacts']].numpy() > .5
                mask = contacts[1:]
                rows.append(dict(cache_row_index=identities[i]['cache_row_index'],
                                 metrics={k: quantiles(v) for k, v in values.items()},
                                 mean_root_speed_mps=float(np.linalg.norm(np.diff(np.concatenate((sj[:1, 0], pj[:, 0])), axis=0), axis=-1).mean() * 30),
                                 mean_source_root_speed_mps=float(np.linalg.norm(np.diff(sj[:, 0], axis=0), axis=-1).mean() * 30),
                                 contact_counts=dict(true_positive=int((predicted_contact & mask).sum()),
                                     false_positive=int((predicted_contact & ~mask).sum()),
                                     false_negative=int((~predicted_contact & mask).sum()))))
                trajectories.append(world)
            summary = {k: quantiles(np.concatenate(v)) for k, v in all_values.items()}
            summary['per_window_mean_root_speed_mps'] = quantiles([r['mean_root_speed_mps'] for r in rows])
            summary['per_window_mean_source_root_speed_mps'] = quantiles([r['mean_source_root_speed_mps'] for r in rows])
            if baseline_values:
                baseline = {k: quantiles(np.concatenate(v)) for k, v in baseline_values.items()}
            result['modes'][mode] = dict(summary=summary, rows=rows,
                metric_role='source-stratified position error; teacher velocity measures prediction consistency, closed velocity measures actual emitted intervals')
            torch.save(dict(selection=identities, world=torch.cat(trajectories), frames_per_rollout=14),
                       args.output / (arm + '_' + mode + '.pt'))
            print(json.dumps(dict(arm=arm, mode=mode, summary=summary)), flush=True)
        if sha256(path) != before:
            raise ValueError('Checkpoint changed during review')
        report['arms'][arm] = result
        report['persistence'] = baseline
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        del actor
    (args.output / 'complete.json').write_text(json.dumps(dict(completed=True, optimizer_updates=0,
        script_sha256=sha256(__file__), checkpoint_hashes_unchanged=True), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'stats', 'cache', 'pfnn', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args())
