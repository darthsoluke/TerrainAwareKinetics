"""Closed-loop rollout of a trained ablation arm, scored with the same audit.

The first W cached frames are factual; every later frame is predicted and fed
back. Only the predicted segment is scored, so the reference cannot leak into
the slip statistic. GPU work is a single forward pass per frame.
"""
import argparse, json, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_3d_scene import BatchedSceneField, CompositeSceneField, path_observation
from motionbricks.data.pfnn_frame_actor import advance
from motionbricks.data.pfnn_frame_actor_v2 import FrameActorV2
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from motionbricks.data.unified_scene_field import VolumetricEgoSampler
from scripts.audit_pfnn_physical_consistency import audit, foot_indices
from scripts.feasibility_residual_pfnn import analyse as wrench_residual


def load_actor(path, rep, plan, device):
    actor = FrameActorV2(rep, backbone=plan['backbone'], geometry=plan['geometry'],
                         frames=plan['frames']).to(device)
    state = torch.load(path, map_location='cpu', weights_only=False)
    actor.load_state_dict(state['model_state'], strict=True)
    actor.eval()
    return actor


@torch.no_grad()
def rollout(actor, rep, sampler, item, frames, device, steps, teacher_forced=False):
    world = item['world'].to(device)[None]
    commands = item['command_sequence']['command'].to(device)[None]
    field = BatchedSceneField([CompositeSceneField(item['scene'])], device=device)
    total = world.shape[1]
    start = min(frames, total - 1)
    history = world[:, :start].clone()
    state = actor.initial_state(1)
    predicted = []
    for t in range(start, min(start + steps, total)):
        # The cached command sequence can be shorter than the world window; the last
        # available command is held rather than indexing past the end.
        command = commands[:, min(max(0, t - 4), commands.shape[1] - 1)]
        recent = history[:, -4:, :5]
        coarse = sampler.observe(recent, command[:, :2], field, .8)
        root = actor.root_step(history, command, coarse, state)
        fine = path_observation(sampler, recent, root['path_world_xz'], coarse, field)
        pred = actor.pose_step(history, command, fine, root)
        emitted = advance(rep, history, pred)
        predicted.append(emitted[:, -1])
        if teacher_forced:
            # Every step starts from the factual window, isolating one-step prediction
            # error from the accumulation that closed-loop rollouts suffer.
            history = world[:, max(0, t + 1 - frames):t + 1].clone()
        else:
            history = torch.cat((history, emitted[:, -1:]), dim=1)[:, -frames:]
        state = pred['state']
    if not predicted:
        return None, None
    predicted = torch.stack(predicted, dim=1)[0]
    truth = world[0, start:start + predicted.shape[1]]
    return predicted.detach().cpu().numpy(), truth.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=pathlib.Path, required=True)
    parser.add_argument('--pfnn', type=pathlib.Path, required=True)
    parser.add_argument('--stats', type=pathlib.Path, required=True)
    parser.add_argument('--proxy', type=pathlib.Path, required=True)
    parser.add_argument('--weights', type=pathlib.Path)
    parser.add_argument('--contract', type=pathlib.Path,
                        help='contract.json for --random-init smoke runs')
    parser.add_argument('--random-init', action='store_true')
    parser.add_argument('--teacher-forced', action='store_true')
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--steps', type=int, default=17)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.random_init:
        if args.contract is None:
            raise ValueError('--random-init needs --contract')
        plan = json.loads(args.contract.read_text())['plan']
        checkpoint = None
    else:
        checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False)
        plan = checkpoint['plan']
    rep = build_native_representation(args.stats).to(device)
    actor = (FrameActorV2(rep, backbone=plan['backbone'], geometry=plan['geometry'],
                          frames=plan['frames']).to(device).eval()
             if args.random_init else load_actor(args.weights, rep, plan, device))
    proxy = np.load(args.proxy, allow_pickle=True)
    neutral = rep.skeleton.neutral_joints.detach().cpu().numpy()
    foot_joints = foot_indices(proxy)
    dataset = PFNNFrameDataset(args.cache, args.pfnn, build_native_representation(args.stats), args.split)
    sampler = VolumetricEgoSampler()
    rows, dropped, one_step = [], 0, []
    for i in range(min(args.limit, len(dataset))):
        sequence, truth = rollout(actor, rep, sampler, dataset[i], plan['frames'], device,
                                  args.steps, args.teacher_forced)
        if sequence is None or len(sequence) < 6:
            dropped += 1
            continue
        if args.teacher_forced:
            # One-step accuracy: the predicted frame against the factual frame. Slip
            # across independently predicted frames would measure prediction noise.
            decoded = rep.inverse(torch.from_numpy(sequence)[None].to(device), False,
                                  return_all=False, joint_positions_from='global_rot_data')
            reference = rep.inverse(torch.from_numpy(truth)[None].to(device), False,
                                    return_all=False, joint_positions_from='global_rot_data')
            feet_error = (decoded['posed_joints'][0][:, foot_joints]
                          - reference['posed_joints'][0][:, foot_joints]).norm(dim=-1)
            body_error = (decoded['posed_joints'][0] - reference['posed_joints'][0]).norm(dim=-1)
            one_step.append(dict(feet=feet_error.detach().cpu().numpy().ravel(),
                                 body=body_error.detach().cpu().numpy().ravel()))
            continue
        entry = audit(sequence, rep, proxy, neutral, foot_joints)
        # Dynamics amplitude. The wrench residual falls for any degenerate motion, so
        # it is only comparable between rollouts of similar amplitude.
        decoded = rep.inverse(torch.from_numpy(sequence)[None].to(device), False,
                              return_all=False, joint_positions_from='global_rot_data')['posed_joints'][0]
        step = (decoded[1:] - decoded[:-1]).norm(dim=-1) * 30.0
        entry['motion_amplitude'] = float(step.mean())
        entry['root_amplitude'] = float((decoded[1:, 0] - decoded[:-1, 0]).norm(dim=-1).mean() * 30.0)
        entry['wrench_linear'], entry['wrench_angular'] = wrench_residual(
            sequence, rep, proxy, neutral, foot_joints, args.mu)
        rows.append(entry)
    pooled = np.concatenate([row.pop('pooled_speed') for row in rows]) if rows else np.zeros(0)
    if one_step:
        feet = np.concatenate([o['feet'] for o in one_step])
        body = np.concatenate([o['body'] for o in one_step])
        summary = dict(arm=plan.get('arm'), frames=plan['frames'], updates=plan.get('updates'),
                       teacher_forced=True, rollouts=len(one_step),
                       feet_error_p50_m=float(np.median(feet)),
                       feet_error_p90_m=float(np.percentile(feet, 90)),
                       body_error_p50_m=float(np.median(body)),
                       body_error_p90_m=float(np.percentile(body, 90)))
        print(json.dumps(summary, indent=2))
        if args.output:
            args.output.write_text(json.dumps(summary, indent=2))
        return
    speeds = [r['stance_speed'] for r in rows if r['stance_speed']['median'] is not None]
    summary = dict(arm=plan.get('arm'), frames=plan['frames'], updates=plan.get('updates'),
                   random_init=bool(args.random_init),
                   teacher_forced=bool(args.teacher_forced),
                   rollouts=len(rows), dropped=dropped,
                   stance_frames=sum(r['stance_frames'] for r in rows))
    wrench_lin = [r['wrench_linear'] for r in rows if r['wrench_linear'].size]
    wrench_ang = [r['wrench_angular'] for r in rows if r['wrench_angular'].size]
    amplitudes = [r['motion_amplitude'] for r in rows if 'motion_amplitude' in r]
    if amplitudes:
        summary.update(motion_amplitude_p50=float(np.median(amplitudes)),
                       root_amplitude_p50=float(np.median([r['root_amplitude'] for r in rows])))
    if wrench_lin:
        lin = np.concatenate(wrench_lin); ang = np.concatenate(wrench_ang)
        summary.update(wrench_mu=args.mu,
                       wrench_linear_p50=float(np.median(lin)),
                       wrench_linear_p90=float(np.percentile(lin, 90)),
                       wrench_angular_p50=float(np.median(ang)),
                       wrench_angular_p90=float(np.percentile(ang, 90)),
                       wrench_feasible_fraction=float(((lin < 1e-3) & (ang < 1e-3)).mean()))
    if speeds:
        for key in ('p10', 'median', 'p90', 'p95', 'mean', 'over_5cmps', 'over_10cmps'):
            summary['rollout_stance_' + key] = float(np.median([s[key] for s in speeds]))
    print(json.dumps(summary, indent=2))
    if args.output:
        # Per-row wrench arrays are numpy; keep them out of the JSON payload.
        for row in rows:
            row.pop('wrench_linear', None)
            row.pop('wrench_angular', None)
        args.output.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
        np.savez(args.output.with_suffix('.npz'), rollout=pooled)


if __name__ == '__main__':
    main()
