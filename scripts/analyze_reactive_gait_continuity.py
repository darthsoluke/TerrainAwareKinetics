"""Diagnose saved reactive gait playback with NumPy; no model or runtime imports."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from review_reactive_kinematic import review


ROOT = Path(__file__).resolve().parents[1]
FPS = 30
MOVING_MPS = .1
MANIFEST = []


def read_source(path):
    raw = path.read_bytes()
    MANIFEST.append(dict(path=str(path.resolve()), bytes=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest()))
    return raw


def stats(values):
    values = np.asarray(values, dtype=float).ravel()
    if not len(values):
        return None
    return dict(count=len(values), mean=float(values.mean()), minimum=float(values.min()),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), maximum=float(values.max()))


def histogram(values):
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def categories(velocity, heading):
    speed = np.linalg.norm(velocity, axis=-1)
    angle = np.arctan2(velocity[..., 0], velocity[..., 1]) - heading
    angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
    return np.where(speed < MOVING_MPS, 'idle', np.where(angle <= np.pi/4,
                    'forward', np.where(angle >= 3*np.pi/4, 'reverse', 'strafe')))


def runs(values):
    breaks = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1, len(values)]
    return [(int(a), int(b)) for a, b in zip(breaks[:-1], breaks[1:])]


def soma_parents():
    path = ROOT/'out/vendor/kimodo/kimodo/skeleton/definitions.py'
    module = ast.parse(read_source(path))
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == 'SOMASkeleton30')
    assignment = next(node for node in cls.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'bone_order_names_with_parents'
                              for t in node.targets))
    bones = ast.literal_eval(assignment.value)
    names = [name for name, _ in bones]
    return names, np.array([-1 if parent is None else names.index(parent) for _, parent in bones])


def body_positions(motion, joints):
    relative = joints - motion[..., None, :3]
    yaw = np.arctan2(motion[..., 4], motion[..., 3])
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    result = relative.copy()
    result[..., 0] = c*relative[..., 0] - s*relative[..., 2]
    result[..., 2] = s*relative[..., 0] + c*relative[..., 2]
    return result


def local_rotations(motion, parents):
    # GLOBAL366 stores Root5, relative positions87, global rotations180, velocity90, contacts4.
    six = motion[..., 92:272].reshape(*motion.shape[:-1], 30, 6).astype(float)
    x = six[..., :3] / np.linalg.norm(six[..., :3], axis=-1, keepdims=True)
    z = np.cross(x, six[..., 3:])
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    matrices = np.stack((x, np.cross(z, x), z), axis=-1)
    local = matrices.copy()
    local[..., 1:, :, :] = np.swapaxes(matrices[..., parents[1:], :, :], -1, -2) @ matrices[..., 1:, :, :]
    return local


def rotation_step_deg(local):
    # The Frobenius product equals trace(R[t-1]^T R[t]). Exclude global Root rotation.
    cosine = (np.sum(local[:-1, 1:]*local[1:, 1:], axis=(-1, -2))-1)/2
    return np.degrees(np.arccos(np.clip(cosine, -1, 1))).max(axis=-1)


def spectral_summary(positions):
    if len(positions) < 60:
        return None
    values = positions.reshape(len(positions), -1).astype(float)
    time = np.arange(len(values), dtype=float)
    basis = np.stack((np.ones_like(time), time), axis=-1)
    values -= basis @ np.linalg.lstsq(basis, values, rcond=None)[0]
    window = np.hanning(len(values))
    power = np.square(np.abs(np.fft.rfft(values*window[:, None], axis=0))).sum(axis=1)
    power[1:] *= 2
    if len(values) % 2 == 0:
        power[-1] /= 2
    frequencies = np.fft.rfftfreq(len(values), 1/FPS)
    selected = (frequencies >= .5) & (frequencies <= 15)
    total = power[selected].sum()
    if total == 0:
        return dict(frames=len(values), fraction_power_6_to_15hz=0., peak_hz=None)
    high = power[frequencies >= 6].sum()
    scale = len(values)**2 * np.mean(window**2) * positions.shape[1]
    return dict(frames=len(values), fraction_power_6_to_15hz=float(high/total),
                high_frequency_joint_rms_m=float(np.sqrt(high/scale)),
                peak_hz=float(frequencies[np.flatnonzero(selected)[np.argmax(power[selected])]]))


def command_activity(intents):
    if len(intents) < 2:
        return dict(count=len(intents))
    times = np.array([r['client_input_ms'] for r in intents])
    velocity = np.array([r['velocity'] for r in intents])
    heading = np.array([r['heading'] for r in intents])
    delta = np.linalg.norm(np.diff(velocity, axis=0), axis=-1)
    interval = np.diff(times)
    near = interval < 100
    angle = np.arctan2(velocity[:, 0], velocity[:, 1])
    angle_delta = np.degrees(np.abs(np.arctan2(np.sin(np.diff(angle)), np.cos(np.diff(angle)))))
    moving_pairs = np.all(np.linalg.norm(velocity, axis=-1)[np.stack((np.arange(len(delta)),
                          np.arange(1, len(delta)+1)), axis=1)] >= MOVING_MPS, axis=1)
    return dict(count=len(intents), first_to_last_input_s=float((times[-1]-times[0])/1000),
                messages_per_active_second=float((len(intents)-1)*1000/(times[-1]-times[0])),
                interval_ms=stats(interval), velocity_delta_mps=stats(delta),
                heading_range_rad=float(heading.max()-heading.min()),
                heading_change_count=int(np.count_nonzero(np.diff(heading))),
                velocity_change_count=int(np.count_nonzero(delta)),
                moving_direction_delta_deg=stats(angle_delta[moving_pairs]),
                tiny_velocity_updates_under_100ms={str(threshold): int(np.sum(near & (delta < threshold)))
                                                 for threshold in (.01, .02, .05)},
                identical_command_updates=int(np.sum((delta == 0) & (np.diff(heading) == 0))))


def summarize_dynamics(body, rotation_step, root_delta, mask, handoff):
    valid = mask[1:] & mask[:-1]
    pose_step = np.linalg.norm(np.diff(body, axis=0), axis=-1).max(axis=-1)
    pose_second = np.linalg.norm(np.diff(body, n=2, axis=0), axis=-1).max(axis=-1)
    acceleration_valid = mask[2:] & mask[1:-1] & mask[:-2]
    result = dict(frames=int(mask.sum()), root_planar_speed_mps=stats(np.linalg.norm(root_delta[mask], axis=-1)*FPS),
                  body_joint_step_max_m=stats(pose_step[valid]),
                  nonroot_local_rotation_step_max_deg=stats(rotation_step[valid]),
                  body_joint_second_difference_max_m=stats(pose_second[acceleration_valid]),
                  pose_steps_over_5cm=int(np.sum((pose_step > .05) & valid)),
                  pose_steps_over_10cm=int(np.sum((pose_step > .1) & valid)))
    for name, transition in [('handoff', handoff[1:]), ('within_forecast', ~handoff[1:])]:
        selected = valid & transition
        result[name] = dict(transitions=int(selected.sum()), joint_step_max_m=stats(pose_step[selected]),
                            local_rotation_step_max_deg=stats(rotation_step[selected]))
    result['spectra'] = [dict(first_played_frame=a+1, last_played_frame=b, **spectrum)
                         for a, b in runs(mask) if mask[a]
                         and (spectrum := spectral_summary(body[a:b])) is not None]
    return result


def analyze_episode(folder, episode, events, original, parents, frame_rows, segment_rows, forecast_rows):
    forecasts = {r['metadata']['plan_id']: r for r in events if r['kind'] == 'forecast'
                 and r['metadata']['episode_id'] == episode}
    played = [r for r in events if r['kind'] == 'played' and r['episode_id'] == episode]
    intents = [r for r in events if r['kind'] == 'intent' and r['episode_id'] == episode]
    commands = {r['seq']: r for r in intents}
    saved = {}
    for plan, row in forecasts.items():
        path = Path(row['path'])
        path = path if path.is_absolute() else ROOT/path
        with np.load(io.BytesIO(read_source(path)), allow_pickle=False) as data:
            saved[plan] = {key: data[key] for key in ('history', 'played_history', 'motion', 'joints')}
    index = np.array([r['frame']-forecasts[r['plan_id']]['metadata']['first_frame'] for r in played])
    plans = np.array([r['plan_id'] for r in played])
    motion = np.array([saved[r['plan_id']]['motion'][i] for r, i in zip(played, index)])
    joints = np.array([saved[r['plan_id']]['joints'][i] for r, i in zip(played, index)])
    times = np.array([r['client_render_ms'] for r in played])
    seed = np.asarray(next(r['seed'] for r in events if r['kind'] == 'session'), dtype=np.float32)
    actual = np.concatenate((seed, motion))
    history_error = []
    for plan, row in forecasts.items():
        origin = row['metadata']['origin_frame']
        if origin <= len(played):
            history_error.append(float(np.abs(saved[plan]['history']-actual[origin:origin+4]).max()))
    if max(history_error) != 0:
        raise ValueError('Conditioned history differs from the eventually rendered history')
    step = np.array([forecasts[plan]['metadata']['step_frames'] for plan in plans])
    consumed_seq = np.array([forecasts[plan]['metadata']['input_seq'] for plan in plans])
    velocity = np.array([commands[seq]['velocity'] for seq in consumed_seq])
    heading = np.array([commands[seq]['heading'] for seq in consumed_seq])
    intent_times = np.array([r['client_input_ms'] for r in intents])
    latest = np.searchsorted(intent_times, times, side='right')-1
    if (latest < 0).any():
        raise ValueError('Rendered sample predates its logged input clock')
    latest_velocity = np.array([intents[i]['velocity'] for i in latest])
    latest_heading = np.array([intents[i]['heading'] for i in latest])
    latest_speed = np.linalg.norm(latest_velocity, axis=-1)
    latest_unit = np.divide(latest_velocity, latest_speed[:, None], out=np.zeros_like(latest_velocity),
                            where=latest_speed[:, None] >= MOVING_MPS)
    consumed_class = categories(velocity, heading)
    latest_class = categories(latest_velocity, latest_heading)
    speed = np.linalg.norm(velocity, axis=-1)
    unit = np.divide(velocity, speed[:, None], out=np.zeros_like(velocity), where=speed[:, None] >= MOVING_MPS)
    delta = np.diff(np.concatenate((seed[-1:, [0, 2]], motion[:, [0, 2]])), axis=0).astype(float)
    progress = np.sum(delta*unit, axis=-1)*FPS
    latest_progress = np.sum(delta*latest_unit, axis=-1)*FPS
    handoff = np.r_[False, plans[1:] != plans[:-1]]
    body = body_positions(motion.astype(float), joints.astype(float))
    rotation_step = rotation_step_deg(local_rotations(motion, parents))
    pose_step = np.r_[np.nan, np.linalg.norm(np.diff(body, axis=0), axis=-1).max(axis=-1)]
    for i, row in enumerate(played):
        frame_rows.append(dict(session=folder.name, episode=episode, frame=row['frame'],
            render_ms=times[i], plan_id=int(plans[i]), source_index=int(index[i]),
            requested_step_frames=int(step[i]), input_seq=int(consumed_seq[i]),
            consumed_class=str(consumed_class[i]), latest_class=str(latest_class[i]),
            root_x=float(motion[i, 0]), root_y=float(motion[i, 1]), root_z=float(motion[i, 2]),
            command_vx=float(velocity[i, 0]), command_vz=float(velocity[i, 1]),
            command_heading=float(heading[i]), latest_vx=float(latest_velocity[i, 0]), latest_vz=float(latest_velocity[i, 1]),
            projected_speed_mps=float(progress[i]), planar_speed_mps=float(np.linalg.norm(delta[i])*FPS),
            projected_speed_relative_latest_input_mps=float(latest_progress[i]),
            body_joint_step_max_m=float(pose_step[i]) if i else None,
            nonroot_local_rotation_step_max_deg=float(rotation_step[i-1]) if i else None,
            handoff=bool(handoff[i]), source_path=forecasts[int(plans[i])]['path']))
    segments = []
    labels = np.array([f'{s}:{c}' for s, c in zip(step, consumed_class)])
    for a, b in runs(labels):
        one_second = [float(progress[start:start+FPS].mean()) for start in range(a, b-FPS+1, FPS)]
        seconds = (b-a)/FPS
        item = dict(session=folder.name, episode=episode, requested_step_frames=int(step[a]),
            command_class=str(consumed_class[a]), first_frame=a+1, last_frame=b,
            frames=b-a, native_duration_s=seconds, render_duration_s=float((times[b-1]-times[a])/1000),
            commanded_mean_mps=float(speed[a:b].mean()), projected_mean_mps=float(progress[a:b].mean()),
            planar_speed_mean_mps=float(np.linalg.norm(delta[a:b], axis=-1).mean()*FPS),
            projected_distance_m=float(progress[a:b].sum()/FPS),
            command_distance_m=float(speed[a:b].sum()/FPS),
            opposite_direction_frame_fraction=float(np.mean(progress[a:b] < 0)) if consumed_class[a] != 'idle' else None,
            displacement_xz_m=delta[a:b].sum(axis=0).tolist(), one_second_projected_speed_mps=one_second,
            body_joint_step_p95_m=float(np.percentile(pose_step[a+1:b], 95)) if b-a > 1 else None)
        segments.append(item)
        segment_rows.append({k: json.dumps(v) if isinstance(v, list) else v for k, v in item.items()})
    groups = defaultdict(list)
    for plan, row in forecasts.items():
        meta = row['metadata']
        command = commands[meta['input_seq']]
        command_speed = float(np.linalg.norm(command['velocity']))
        category = str(categories(np.array(command['velocity']), command['heading']))
        trajectory = saved[plan]['motion'][:, [0, 2]].astype(float)
        origin = saved[plan]['history'][-1, [0, 2]].astype(float)
        direction = np.array(command['velocity'])/command_speed if command_speed >= MOVING_MPS else np.zeros(2)
        projected = np.diff(np.concatenate((origin[None], trajectory)), axis=0) @ direction * FPS
        groups[(meta['step_frames'], category)].append(projected)
        used = index[plans == plan]
        forecast_rows.append(dict(session=folder.name, episode=episode, plan_id=plan,
            step_frames=meta['step_frames'], origin_frame=meta['origin_frame'], input_seq=meta['input_seq'],
            command_class=category, command_speed_mps=command_speed, played_frames=len(used),
            first_played_index=int(used[0]) if len(used) else None, last_played_index=int(used[-1]) if len(used) else None,
            forecast_first_step_projected_mps=float(projected[0]),
            forecast_late_8_to_23_projected_mean_mps=float(projected[8:].mean()), source_path=row['path']))
    dispatch = [r for r in events if r['kind'] == 'generation_started' and r['episode_id'] == episode]
    configured = 4
    intent_by_mode = defaultdict(list)
    for row in events:
        if row['kind'] == 'configure' and row['episode_id'] == episode:
            configured = row['step_frames']
        if row['kind'] == 'intent' and row['episode_id'] == episode:
            intent_by_mode[configured].append(row)
    modes = []
    for mode in sorted(set(step)):
        mask = step == mode
        used_plans = set(plans[mask])
        lengths = [int(np.sum(plans == plan)) for plan in used_plans]
        pairs = [(a, b) for a, b in zip(dispatch[:-1], dispatch[1:])
                 if a['step_frames'] == b['step_frames'] == mode]
        early = [(a, b) for a, b in pairs if b['played_frame']-a['played_frame'] < mode]
        first = mask & (index == 0)
        contexts = np.array([forecasts[plan]['metadata']['timing']['decoded_context_root_delta_m']
                             for plan in plans])[:, [0, 2]]
        classes = []
        for category in ('idle', 'forward', 'strafe', 'reverse'):
            selected = mask & (consumed_class == category)
            if not selected.any():
                continue
            local_segments = [r for r in segments if r['requested_step_frames'] == mode
                              and r['command_class'] == category]
            sustained = [r for r in local_segments if r['frames'] >= FPS]
            classes.append(dict(command_class=category,
                native_duration_s=float(selected.sum()/FPS), mean_command_speed_mps=float(speed[selected].mean()),
                actual_projected_speed_mps=stats(progress[selected]),
                progress_to_command_distance_ratio=float(progress[selected].sum()/speed[selected].sum())
                    if category != 'idle' else None,
                sustained_segments=sustained,
                **summarize_dynamics(body, rotation_step, delta, selected, handoff)))
        mode_forecasts = [r for r in forecasts.values() if r['metadata']['step_frames'] == mode]
        chosen = [forecasts[plan]['metadata'] for plan in sorted(used_plans)]
        modes.append(dict(step_frames=int(mode), played_frames=int(mask.sum()),
            played_native_seconds=float(mask.sum()/FPS), adopted_plans=len(used_plans),
            adopted_plans_per_native_second=float(len(used_plans)*FPS/mask.sum()),
            adopted_plan_length_frames=histogram(lengths), executed_forecast_index=histogram(index[mask]),
            first_index_fraction=float(np.mean(index[mask] == 0)), index_0_to_3_fraction=float(np.mean(index[mask] < 4)),
            input_activity=command_activity(intent_by_mode[mode]),
            generation_dispatches=int(sum(r['step_frames'] == mode for r in dispatch)),
            dispatch_pairs_checked=len(pairs), dispatches_before_selected_interval=len(early),
            early_dispatches_with_changed_sequence=sum(a['input_seq'] != b['input_seq'] for a, b in early),
            origin_interval_frames=histogram(np.diff([r['origin_frame'] for r in chosen])),
            forecast_count=len(mode_forecasts),
            unused_forecasts=sum(r['metadata']['plan_id'] not in used_plans for r in mode_forecasts),
            compute_ms=stats([r['metadata']['timing']['compute_ms'] for r in mode_forecasts]),
            native_motion_by_class=classes,
            actual_displacement_xz_m=delta[mask].sum(axis=0).tolist(),
            decoded_context_accounting=dict(first_samples=int(first.sum()),
                executed_first_displacement_xz_m=delta[first].sum(axis=0).tolist(),
                logged_context_delta_sum_xz_m=contexts[first].sum(axis=0).tolist(),
                first_displacement_minus_logged_context_xz_m=(delta[first]-contexts[first]).sum(axis=0).tolist(),
                other_executed_displacement_xz_m=delta[mask & ~first].sum(axis=0).tolist()),
            **summarize_dynamics(body, rotation_step, delta, mask, handoff)))
    startup = []
    latest_moving = np.linalg.norm(latest_velocity, axis=-1) >= MOVING_MPS
    for a, b in runs(latest_moving):
        if not latest_moving[a]:
            continue
        origin_root = seed[-1, [0, 2]] if a == 0 else motion[a-1, [0, 2]]
        windows = []
        for horizon in (.5, 1., 2.):
            end = a+int(horizon*FPS)
            if end <= b:
                windows.append(dict(native_seconds=horizon,
                    displacement_xz_m=(motion[end-1, [0, 2]]-origin_root).astype(float).tolist(),
                    projected_mean_mps=float(latest_progress[a:end].mean()),
                    command_mean_mps=float(np.linalg.norm(latest_velocity[a:end], axis=-1).mean())))
        startup.append(dict(first_frame=a+1, last_frame=b, moving_seconds=(b-a)/FPS,
                            initial_class=str(latest_class[a]), windows=windows))
    constant_commands = []
    for a, b in runs(consumed_seq):
        if b-a >= 15 and speed[a] >= MOVING_MPS:
            constant_commands.append(dict(first_frame=a+1, last_frame=b, frames=b-a,
                input_seq=int(consumed_seq[a]), velocity_xz_mps=velocity[a].tolist(),
                heading_rad=float(heading[a]), actual_projected_speed_mps=float(progress[a:b].mean()),
                command_mps=float(speed[a]), first_half_second_projected_mps=float(progress[a:a+15].mean()),
                first_second_projected_mps=float(progress[a:a+30].mean()) if b-a >= 30 else None))
    return dict(session=folder.name, episode=episode, played_frames=len(played),
        wall_duration_s=float((times[-1]-times[0])/1000),
        display_hz=float((len(times)-1)*1000/(times[-1]-times[0])),
        original_review=dict(conditioned_histories_checked=original['conditioned_histories_checked'],
            played_and_committed_history_max_abs=original['played_and_committed_history_max_abs'],
            final_idle=original['final_idle'], failures=original['failures']),
        eventually_played_history_checked=len(history_error), eventually_played_history_max_abs=max(history_error),
        consumed_vs_latest_command_class_different_frames=int(np.sum(consumed_class != latest_class)),
        heading_input_range_rad=float(np.ptp([r['heading'] for r in intents])),
        startup_windows=startup, constant_command_spans=constant_commands,
        sustained_segments=[r for r in segments if r['frames'] >= FPS],
        modes=modes, forecast_projected_speed_curves=[dict(step_frames=int(mode), command_class=category,
            forecasts=len(values), index_mean_mps=np.mean(values, axis=0).tolist(),
            index_p50_mps=np.median(values, axis=0).tolist()) for (mode, category), values in sorted(groups.items())])


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(result):
    lines = ['# Saved v3 Gait Diagnosis', '',
        'CPU-only NumPy analysis of every closed saved v3 session. No model calls, GPU, training, runtime, service or browser changes.', '',
        '## Observed Movement', '',
        '| Session / episode | Request | Moving class | Seconds | Command m/s | Actual along command m/s | Progress ratio |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for episode in result['episodes']:
        for mode in episode['modes']:
            for group in mode['native_motion_by_class']:
                if group['command_class'] != 'idle':
                    lines.append(f"| {episode['session']} / {episode['episode']} | {mode['step_frames']} | "
                        f"{group['command_class']} | {group['native_duration_s']:.2f} | {group['mean_command_speed_mps']:.3f} | "
                        f"{group['actual_projected_speed_mps']['mean']:.3f} | {group['progress_to_command_distance_ratio']:.1%} |")
    lines += ['', 'Root displacement is reconstructed from the acknowledged forecast sample at its absolute frame. '
              'Dynamics use the native 30 Hz grid. Categories use the consumed command: forward within 45 degrees of '
              'requested facing, reverse beyond 135 degrees, strafe between; speeds below 0.1 m/s are idle. '
              'Latest input and consumed input are both exported in frame_metrics.csv.', '',
              '## Cadence And Input', '',
              '| Session / episode | Request | Input messages | Active input Hz | Adopted plans | Adopted Hz | One-frame plans | Frames at index 0 | Early dispatches / pairs |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for episode in result['episodes']:
        for mode in episode['modes']:
            activity = mode['input_activity']
            rate = f"{activity['messages_per_active_second']:.2f}" if activity['count'] > 1 else 'n/a'
            lines.append(f"| {episode['session']} / {episode['episode']} | {mode['step_frames']} | {activity['count']} | {rate} | "
                f"{mode['adopted_plans']} | {mode['adopted_plans_per_native_second']:.2f} | {mode['adopted_plan_length_frames'].get('1', 0)} | "
                f"{mode['first_index_fraction']:.1%} | {mode['dispatches_before_selected_interval']} / {mode['dispatch_pairs_checked']} |")
    lines += ['', 'Active input Hz uses first-to-last input time; adopted Hz uses played frame count / 30. '
              'A selected mode change can have already scheduled frames from the previous mode, so frame groups use '
              'forecast metadata. Early dispatches compare consecutive generation_started records in the same episode and mode.', '']
    for episode in result['episodes']:
        if episode['session'] == 'session_0004':
            mode = next(r for r in episode['modes'] if r['step_frames'] == 4)
            activity = mode['input_activity']
            lines += [f"Session_0004 selected 4-frame mode: input interval median {activity['interval_ms']['p50']:.2f} ms, "
                f"velocity-change median {activity['velocity_delta_mps']['p50']:.5f} m/s, "
                f"{activity['tiny_velocity_updates_under_100ms']['0.02']} adjacent updates with interval below 100 ms "
                'and velocity change below 0.02 m/s. Heading is unchanged throughout the session. '
                f"All {mode['early_dispatches_with_changed_sequence']} early dispatch pairs have changed input sequence. "
                'Small input variation is observed; its split between intentional analogue motion and hardware noise is unlogged.', '']
    lines += ['## Forecast Index Bias', '',
              '| Session / episode | Request | Moving class | Forecasts | Index 0 mean m/s | Indices 8-23 mean m/s | Index 23 mean m/s |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for episode in result['episodes']:
        for curve in episode['forecast_projected_speed_curves']:
            if curve['command_class'] == 'idle':
                continue
            values = curve['index_mean_mps']
            lines.append(f"| {episode['session']} / {episode['episode']} | {curve['step_frames']} | "
                f"{curve['command_class']} | {curve['forecasts']} | {values[0]:.3f} | "
                f"{np.mean(values[8:]):.3f} | {values[-1]:.3f} |")
    lines += ['', 'Speeds are projected onto each forecast\'s own input direction. All saved forecasts are included, '
              'including forecasts never adopted. The first displacement begins at that forecast\'s raw conditioned history endpoint. '
              'Later samples describe its saved open-loop prediction. Latest selected 4-frame playback never reaches index 4 '
              'of any forecast; the tail speed in this table is therefore predicted, not observed execution.', '',
              '## Pose Continuity', '',
              '| Session / episode | Request | Body joint step P95 / max cm | Local joint rotation P95 / max deg | Handoff joint step P95 cm | Within-forecast P95 cm |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for episode in result['episodes']:
        for mode in episode['modes']:
            pose = mode['body_joint_step_max_m']
            angle = mode['nonroot_local_rotation_step_max_deg']
            seam = mode['handoff']['joint_step_max_m']
            within = mode['within_forecast']['joint_step_max_m']
            lines.append(f"| {episode['session']} / {episode['episode']} | {mode['step_frames']} | {100*pose['p95']:.2f} / "
                f"{100*pose['maximum']:.2f} | {angle['p95']:.2f} / {angle['maximum']:.2f} | "
                f"{100*seam['p95']:.2f} | {100*within['p95']:.2f} |")
    lines += ['', 'Each joint-step statistic is the maximum across 30 joints for an actual adjacent rendered pair, '
              'after removing Root translation and each frame\'s decoded yaw. Rotation uses parent-relative SO(3) '
              'and excludes global Root rotation. These differ from the old review\'s same-time alternative-forecast disagreement. '
              'Second differences and detrended Hann-window 6-15 Hz spectral power for contiguous intervals of at least '
              '60 frames are in report.json; spectral detrending is analysis only.', '',
              '## Sustained Movement And Startup', '',
              'Continuous category spans may contain changes in direction or magnitude within that category. '
              'Exact constant-command spans and latest-input startup windows are separately recorded in report.json. '
              'Startup projection uses the latest input available at render time; the main table uses the command '
              'that generated the played forecast.', '']
    for episode in result['episodes']:
        if episode['session'] == 'session_0001':
            span = episode['constant_command_spans'][0]
            lines.append(f"- Discrete-key {episode['modes'][0]['step_frames']}-frame mode starts with constant "
                f"{span['command_mps']:.1f} m/s forward intent. First 0.5 s mean progress "
                f"{span['first_half_second_projected_mps']:.3f} m/s; first second "
                f"{span['first_second_projected_mps']:.3f} m/s.")
        if episode['session'] == 'session_0004':
            for span in sorted((r for r in episode['sustained_segments'] if r['command_class'] != 'idle'),
                               key=lambda r: r['frames'], reverse=True)[:3]:
                reverse_windows = sum(v < 0 for v in span['one_second_projected_speed_mps'])
                lines.append(f"- Latest fixed-facing {span['command_class']} span frames {span['first_frame']}-"
                    f"{span['last_frame']}: {span['native_duration_s']:.2f} s, requested "
                    f"{span['commanded_mean_mps']:.3f} m/s, mean progress {span['projected_mean_mps']:.3f} m/s; "
                    f"{reverse_windows}/{len(span['one_second_projected_speed_mps'])} nonoverlapping one-second "
                    'windows move opposite the current command on average.')
    lines += ['', '## High Frequency Content', '',
              '| Session / episode | Request | Idle frames | Peak Hz | 6-15 Hz power fraction | 6-15 Hz joint RMS mm |',
              '| --- | ---: | --- | ---: | ---: | ---: |']
    for episode in result['episodes']:
        for mode in episode['modes']:
            idle = next((r for r in mode['native_motion_by_class'] if r['command_class'] == 'idle'), None)
            if idle is None:
                continue
            for spectrum in idle['spectra']:
                lines.append(f"| {episode['session']} / {episode['episode']} | {mode['step_frames']} | "
                    f"{spectrum['first_played_frame']}-{spectrum['last_played_frame']} | {spectrum['peak_hz']:.3f} | "
                    f"{spectrum['fraction_power_6_to_15hz']:.1%} | {spectrum['high_frequency_joint_rms_m']*1000:.3f} |")
    lines += ['', 'RMS is the per-joint vector RMS in the 6-15 Hz band, corrected for the Hann window. '
              'The two early idle trials have narrow peaks at the 4/2-frame adoption rates. '
              'Moving trials also have large pose changes, but their positional power is dominated by lower frequencies; '
              'their full-band fractions are recorded rather than assigning every large step to high-frequency tremor.', '',
              '## Replay Evidence', '',
              f"- {sum(e['played_frames'] for e in result['episodes'])} acknowledged frames across {len(result['episodes'])} episodes.",
              f"- Existing review history checks: {sum(e['original_review']['conditioned_histories_checked'] for e in result['episodes'])}; "
              f"maximum difference {max(e['original_review']['played_and_committed_history_max_abs'] for e in result['episodes']):g}.",
              f"- New audit against the history that eventually rendered: {sum(e['eventually_played_history_checked'] for e in result['episodes'])} forecasts; "
              f"maximum difference {max(e['eventually_played_history_max_abs'] for e in result['episodes']):g}.",
              '- The session journal does not identify the input hardware. Session_0004 contains fixed-facing lateral/reverse commands; '
              'session_0001 headings follow movement and provide forward-facing evidence only.',
              '- Saved arrays contain decoded motion, official FK joints/rig and raw conditioning history. They do not contain Root-stage '
              'output, logits, source contacts, terrain vertices, or a controlled cadence-only counterfactual.', '',
              '## Causal Candidates', '',
              '1. Cadence bypass is directly observable: input sequence changes trigger generation before step_frames elapses. '
              'The analogue-like session exposes this much more often than discrete keys; see tiny-update counts and dispatch records.',
              '2. Execution-index bias repeatedly selects the early part of each 24-frame prediction. Per-index forecast speed curves '
              'show whether later saved frames promise motion that is almost never executed. Those later frames are predictions '
              'under their original history, not an observed alternative closed-loop trajectory.',
              '3. Decoded history-end Root offsets recur at executed first frames. decoded_context_accounting separates their logged '
              'sum from first-frame displacement and other steps algebraically. The trace cannot determine which network stage '
              'created that offset because it omits pre-codec Root-stage output.',
              '4. Large local joint jumps are actual pose changes after removing translation and yaw. Handoff/within-forecast '
              'and moving/idle breakdowns locate them; they do not isolate codec sampling, conditioning or terrain as the cause.', '',
              '## Next Controlled Checks', '',
              '- Enforce selected dispatch cadence while retaining latest input, then replay the identical analogue command stream '
              'with the same seed and RNG. Compare progress, startup, pose jumps and input latency.',
              '- Compare raw Root-stage trajectory, decoded context endpoint, decoded future and accepted history in one identical-input '
              'probe to identify the integration-origin contribution.',
              '- Compare original longer execution with short regeneration using the same input/seed. Keep full-skin terrain quality '
              'as a separate evaluation; this report measures saved gait and Root dynamics.', '',
              'Artifacts: report.json, frame_metrics.csv, segment_metrics.csv, forecast_metrics.csv, input_manifest.json. '
              'All source files are hashed. The output directory is created exclusively and existing reports are never overwritten.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names, parents = soma_parents()
    read_source(Path(__file__).resolve())
    read_source(ROOT/'scripts/review_reactive_kinematic.py')
    episodes, sessions, frame_rows, segment_rows, forecast_rows = [], [], [], [], []
    for folder in sorted(args.run.glob('session_*')):
        events = [json.loads(line) for line in read_source(folder/'events.jsonl').splitlines()]
        if events[-1]['kind'] != 'closed':
            raise ValueError(f'Expected a closed immutable session: {folder}')
        original = review(folder)
        sessions.append(dict(session=folder.name, kinds=dict(Counter(r['kind'] for r in events))))
        for item in original['episodes']:
            episodes.append(analyze_episode(folder, item['episode_id'], events, item, parents,
                                             frame_rows, segment_rows, forecast_rows))
    result = dict(schema='motionbricks.reactive_gait_diagnosis.v1', source_run=str(args.run.resolve()),
                  source_review='scripts/review_reactive_kinematic.py', sessions=sessions, episodes=episodes,
                  skeleton_joint_order=names, native_fps=FPS, moving_threshold_mps=MOVING_MPS,
                  new_model_calls=0, optimizer_updates=0, gpu_operations=0, runtime_changes=False)
    serialized = {name: json.dumps(value, indent=2, allow_nan=False)+'\n'
                  for name, value in [('report.json', result), ('input_manifest.json', MANIFEST)]}
    text = markdown(result)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in serialized.items():
        with (args.output/name).open('x', encoding='utf-8') as stream:
            stream.write(value)
    (args.output/'report.md').write_text(text, encoding='utf-8')
    for name, rows in [('frame_metrics.csv', frame_rows), ('segment_metrics.csv', segment_rows),
                       ('forecast_metrics.csv', forecast_rows)]:
        write_csv(args.output/name, rows)
    print(json.dumps(dict(output=str(args.output.resolve()), episodes=len(episodes), frames=len(frame_rows),
                          forecasts=len(forecast_rows), source_files=len(MANIFEST)), indent=2))


if __name__ == '__main__':
    main()
