"""Read-only gait/advance and decoder-prefix audits of saved stair failures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def phase_events(separation, threshold=.1):
    """Alternating fore-aft excursions with hysteresis, not force contacts."""
    state, events = 0, []
    for frame, value in enumerate(separation):
        side = 1 if value >= threshold else -1 if value <= -threshold else 0
        if side and side != state:
            events.append(frame)
            state = side
    return events


def measure(roots, feet, direction, fps=30., block_size=8):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    forward = roots[:, [0, 2]] @ direction
    forward -= forward[0]
    delta = np.diff(forward)
    relative = (feet - roots[:, None])[:, :, [0, 2]] @ direction
    separation = relative[:, 0] - relative[:, 1]
    events = phase_events(separation)
    cycles = []
    # Nonoverlapping full left-right cycles; incomplete ends remain in the series.
    for index in range(0, len(events) - 2, 2):
        start, end = events[index], events[index + 2]
        local_delta = delta[start:end]
        cycles.append(dict(start_frame=start, end_frame=end, seconds=(end-start)/fps,
            net_forward_m=float(forward[end]-forward[start]),
            root_height_change_m=float(roots[end, 1]-roots[start, 1]),
            gross_forward_m=float(local_delta.clip(0).sum()),
            backward_m=float((-local_delta).clip(0).sum()),
            foot_separation_range_m=float(np.ptp(separation[start:end+1]))))
    frames = np.arange(1, len(roots))
    seam = frames % block_size == 0
    negative = delta < -1e-5
    large_backward_seams = frames[seam & (delta < -.05)].tolist()
    return dict(frames=len(roots), fps=fps, net_forward_m=float(forward[-1]),
        planar_path_m=float(np.linalg.norm(np.diff(roots[:, [0, 2]], axis=0), axis=1).sum()),
        gross_forward_m=float(delta.clip(0).sum()), backward_m=float((-delta).clip(0).sum()),
        backward_frames=frames[negative].tolist(),
        backward_seam_frames=frames[negative & seam].tolist(),
        backward_seams_over5cm=large_backward_seams,
        recurrent_backward_replanning=len(large_backward_seams) >= 3,
        max_backward_step_m=float((-delta).clip(0).max(initial=0)),
        seam_step_mean_m=float(delta[seam].mean()) if seam.any() else None,
        inside_step_mean_m=float(delta[~seam].mean()) if (~seam).any() else None,
        phase_events=events, cycles=cycles,
        series=dict(progress_m=forward.tolist(), root_height_m=roots[:, 1].tolist(),
                    foot_separation_m=separation.tolist(), forward_step_m=delta.tolist()),
        scope='Original frames, displayed FK joints; phase excursions are not contact labels. '
              'Three backward published seam jumps over5cm are an explicit regression screen, '
              'not a minimum forward-speed requirement or a complete naturalness test.')


def recorded(args):
    import torch
    from motionbricks.data.native_mb_data import build_native_representation
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    torch.set_num_threads(4)
    rep = build_native_representation()
    ids = [rep.skeleton.bone_order_names.index(s + 'Foot') for s in ('Left', 'Right')]
    rows = {}
    for item in args.recordings:
        arm, filename = item.split('=', 1)
        path = Path(filename)
        with np.load(path, allow_pickle=False) as archive:
            world = archive['world'].copy()
            fps = float(archive['fps'])
        with torch.no_grad():
            joints = rep.inverse(torch.from_numpy(world)[None], False, return_all=False,
                                 joint_positions_from='global_rot_data')['posed_joints'][0].numpy()
        rows[arm] = measure(world[:, :3], joints[:, ids], args.direction, fps)
        rows[arm].update(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'complete.json').write_text(json.dumps(dict(rows=rows, optimizer_updates=0,
        quality_approved=False), indent=2), encoding='utf-8')
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    colors = dict(source='#278250', before='#2476b3', slower='#d56527')
    for arm, row in rows.items():
        s, fps = row['series'], row['fps']
        clock = np.arange(row['frames']) / fps
        color = colors.get(arm)
        axes[0].plot(clock, s['progress_m'], label=arm, color=color)
        axes[1].plot(clock[1:], np.array(s['forward_step_m']) * 100, label=arm, color=color)
        axes[2].plot(clock, s['foot_separation_m'], label=arm, color=color, alpha=.8)
    axes[0].set_ylabel('Net forward travel (m)')
    axes[1].set_ylabel('Per-frame forward step (cm)')
    axes[2].set_ylabel('Left-right foot separation (m)')
    axes[2].set_xlabel('Original recording time (s), no retiming')
    for ax in axes:
        ax.axhline(0, color='#888888', linewidth=.5)
        ax.grid(alpha=.2)
        ax.legend(loc='upper left')
    fig.suptitle('Slower candidate: repeated backward block seams, not just slower climbing')
    fig.savefig(args.output / 'diagnosis.png', dpi=160)
    fig.savefig(args.output / 'diagnosis.pdf')
    print(json.dumps({arm: {key: val for key, val in row.items() if key != 'series'}
                      for arm, row in rows.items()}))


def replay(args):
    import torch
    from motionbricks.data.native_mb_observed_history import reencode_played_history
    from motionbricks.data.terrain_control import SoftCommandState
    from scripts.pilot_joint_sdf_contact_preserve import load_model, prepare_cases, control

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for item in args.runs:
        arm, filename, recording = item.split('=', 2)
        checkpoint = Path(filename)
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        model, _ = load_model(saved)
        rep = model.inference.global_motion_rep
        case = next(c for c in prepare_cases(args.cases, Path(saved['contract']['dataset']))
                    if c['row']['source_id'] == 'generated_8030ae7a2a7ad640')
        with np.load(recording, allow_pickle=False) as archive:
            expected = torch.from_numpy(archive['world'].copy()).cuda()[None]
        if expected.shape[1] % 8:
            raise ValueError('Replay must retain complete original released blocks')
        history = case['world'][None, :4].cuda()
        axis = control(case, 0)[0, :2]
        axis = axis / axis.norm()
        state, rows, parts = SoftCommandState(), [], []
        with torch.no_grad():
            for block in range(expected.shape[1] // 8):
                result = model(history, state.from_controls(history, control(case, block), 8), [case['scene']])
                generated = result['world']
                local = model.inference.local_motion_rep.unnormalize(model.last_local[:, :32])
                velocity = local[..., model.inference.local_motion_rep.indices['local_root_vel']]
                observed = history[:, 3, [0, 2]] - history[:, 0, [0, 2]]
                past_error = velocity[:, :3].sum(1) / 30 - observed
                direct_error = generated[:, 3, [0, 2]] - history[:, 3, [0, 2]]
                first_step = generated[:, 4, [0, 2]] - history[:, 3, [0, 2]]
                outgoing = velocity[:, 3] / 30
                torch.testing.assert_close(past_error, direct_error, atol=1e-5, rtol=0)
                torch.testing.assert_close(first_step, past_error+outgoing, atol=1e-5, rtol=0)
                rows.append(dict(frame=block*8, past_error_forward_m=float(past_error[0] @ axis),
                    outgoing_step_forward_m=float(outgoing[0] @ axis),
                    published_step_forward_m=float(first_step[0] @ axis),
                    factual_past_forward_m=float(observed[0] @ axis),
                    predicted_past_forward_m=float((velocity[0, :3].sum(0)/30) @ axis)))
                parts.append(generated[:, 4:12])
                history = reencode_played_history(generated[:, 8:12], rep)
        actual = torch.cat(parts, 1)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=1e-5)
        reports[arm] = dict(frames=actual.shape[1], blocks=rows,
            replay_max_difference=float((actual-expected).abs().max()),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        del model, saved
        torch.cuda.empty_cache()
    report = dict(runs=reports, optimizer_updates=0, output_modified=False,
                  scope='Exact same-state decoder replay. Decomposition only; no Root replacement.')
    (args.output / 'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    p = commands.add_parser('recorded')
    p.add_argument('--recordings', nargs='+', required=True)
    p.add_argument('--direction', type=float, nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('replay')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (recorded if args.mode == 'recorded' else replay)(args)
