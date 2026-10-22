"""Compare saved native MB, baked G1 and physical tracking; no new simulation."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import mujoco
import numpy as np
from scipy.signal import welch
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.reference import G1_JOINT_NAMES, OFFICIAL_BODY_NAMES, file_sha256, load_soma_reference
from motionbricks.mb_sonic.scene import soma_to_mujoco
from scripts.audit_mb_tracker_binary import decode, read_index
from scripts.isolate_mb_continuous_reference import load_fixed_reference, load_record, save_json


def spectrum(values, dt):
    values = np.asarray(values, float).reshape(len(values), -1)
    f, p = welch(values, fs=1/dt, nperseg=min(len(values), round(4/dt)), axis=0, detrend='linear')
    df = f[1]-f[0]
    energy = float(p[(f >= 8) & (f <= 14)].sum()*df)
    return dict(energy_8_14=energy, total_energy=float(p.sum()*df), resolution_hz=float(df),
        velocity_rms=float(np.sqrt(np.mean((np.diff(values, axis=0)/dt)**2))),
        acceleration_rms=float(np.sqrt(np.mean((np.diff(values, n=2, axis=0)/dt**2)**2))))


def root_metrics(times, roots):
    velocity = np.diff(roots, axis=0)/np.diff(times)[:, None]
    dt = float(np.median(np.diff(times)))
    return dict(start_xyz_m=roots[0].tolist(), end_xyz_m=roots[-1].tolist(),
        progress_negative_y_m=float(roots[0, 1]-roots[-1, 1]),
        mean_forward_mps=float((roots[0, 1]-roots[-1, 1])/(times[-1]-times[0])),
        lateral_range_m=float(np.ptp(roots[:, 0])), height_range_m=float(np.ptp(roots[:, 2])),
        max_root_step_m=float(np.linalg.norm(np.diff(roots, axis=0), axis=1).max()),
        velocity_mean_xyz_mps=velocity.mean(0).tolist(), spectrum=spectrum(roots, dt))


def dataset(case, tracking):
    source = load_soma_reference(case/'generation/proposal.npz')
    reference, receipt = load_fixed_reference(case)
    metadata, ticks, artifacts, _, _ = read_index(tracking/'tracking.jsonl')
    initial, warmup = None, []
    for line in (tracking/'tracking.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['kind'] == 'explicit_initial_reset':
            initial = row['actual']
        if row['kind'] == 'actual_history_warmup':
            warmup.append(row['actual'])
    controls = {key: [] for key in ('time', 'joints', 'target', 'root', 'reference_root')}
    physical = {key: [] for key in ('time_s', 'qpos', 'body_positions_world', 'actuator_forces')}
    load_masks = []
    model = mujoco.MjModel.from_xml_path(metadata['robot']['path'])
    foot_ids = [model.body(side+'_ankle_roll_link').id for side in ('left', 'right')]
    for tick in ticks.values():
        artifact = artifacts[tick['numeric_record_id']]
        if file_sha256(artifact['path']) != artifact['sha256']:
            raise ValueError('Physical record changed')
        record = load_record(artifact['path'])
        batch, command = record['physics'], record['tick']['policy_command']
        for key in physical:
            physical[key].append(batch[key])
        controls['time'].append(batch['time_s'][-1])
        controls['joints'].append(batch['joint_positions'][-1])
        controls['target'].append(command['joint_target'])
        controls['root'].append(batch['qpos'][-1, :3])
        index = int(round(batch['time_s'][-1]/.02))
        controls['reference_root'].append(reference.qpos[index, :3])
        c = batch['contacts']
        start, end = c['offsets'][-2:]
        mask = []
        for body in foot_ids:
            selected = ((c['body_ids'][start:end] == body).any(axis=1)
                        & (c['body_ids'][start:end] == 0).any(axis=1))
            mask.append(float(c['wrench_contact'][start:end, 0][selected].sum()) > 10.)
        load_masks.append(mask)
    controls = {key: np.asarray(value) for key, value in controls.items()}
    physical = {key: np.concatenate(value) for key, value in physical.items()}
    prefix = [initial, *warmup]
    for key in ('time_s', 'qpos', 'body_positions_world'):
        physical['render_'+key] = np.concatenate((np.asarray([row[key] for row in prefix]), physical[key]))
    return source, reference, metadata, controls, physical, np.asarray(load_masks), model, receipt


def analyze(case, tracking, output):
    output.mkdir(parents=True, exist_ok=False)
    s, g, metadata, c, p, support, model, receipt = dataset(case, tracking)
    valid = s.timestamps_s >= 0
    native_root = soma_to_mujoco(s.root_positions_m[valid])
    native_pose = soma_to_mujoco(s.world_positions_m[valid])-native_root[:, None]
    reference_relative = g.body_pos_w-g.qpos[:, None, :3]
    body_ids = [model.body(name).id for name in g.body_names]
    actual_bodies = p['body_positions_world'][:, body_ids]
    actual_relative = actual_bodies-p['qpos'][:, None, :3]
    control_ids = np.searchsorted(p['time_s'], c['time']-1e-8)
    actual_body_control = actual_bodies[control_ids]
    foot_columns = [g.body_names.index(side+'_ankle_roll_link') for side in ('left', 'right')]
    actual_foot_velocity = np.diff(actual_body_control[:, foot_columns, :2], axis=0)/.02
    reference_foot_velocity = g.body_lin_vel_w[np.rint(c['time'][1:]/.02).astype(int)][:, foot_columns, :2]
    stance_speed = np.linalg.norm(actual_foot_velocity, axis=-1)[support[1:]]
    demanded_speed = np.linalg.norm(reference_foot_velocity, axis=-1)[support[1:]]
    seam = (4+24*np.arange(1, 21)).astype(int)
    report = dict(schema='continuous_reference_layer_review.v1', optimizer_updates=0, new_physics_steps=0,
        source_root=root_metrics(s.timestamps_s[valid], native_root),
        g1_reference_root=root_metrics(g.timestamps_s, g.qpos[:, :3]),
        actual_root=root_metrics(p['time_s'], p['qpos'][:, :3]),
        final_reference_root_error_xyz_m=(c['root'][-1]-c['reference_root'][-1]).tolist(),
        source_pose_root_relative=spectrum(native_pose, 1/30),
        g1_reference_pose_root_relative=spectrum(reference_relative, .02),
        actual_pose_root_relative=spectrum(actual_relative, .001),
        g1_reference_joints=spectrum(g.joint_pos, .02),
        actor_targets=spectrum(c['target'], .02), actual_joints=spectrum(c['joints'], .02),
        actuator_force=spectrum(p['actuator_forces'], .001),
        source_ar_boundary_root_steps_m=np.linalg.norm(np.diff(s.root_positions_m, axis=0)[seam-1], axis=-1).tolist(),
        measured_support=dict(threshold_N=10., sampled_foot_frames=int(support.sum()),
            actual_foot_body_speed_mps_mean=float(stance_speed.mean()) if len(stance_speed) else None,
            reference_foot_body_speed_mps_mean=float(demanded_speed.mean()) if len(demanded_speed) else None,
            marker='ankle_roll_link_body_origin;not_contact_point_slip'),
        source_sha256=file_sha256(case/'generation/proposal.npz'), reference_sha256=receipt['reference_sha256'],
        recording_sha256=file_sha256(tracking/'tracking.jsonl'),
        model_tensors_unchanged=json.loads((case/'generation/generation.json').read_text())['generation']['model_tensors_unchanged'],
        online_generation=False, online_reference_switches=0, post_filter=False)
    save_json(output/'review.json', report)
    colors = ['#3559a6', '#b87614', '#17806b']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for roots, times, label, color in ((native_root, s.timestamps_s[valid], 'MB source', colors[0]),
            (g.qpos[:, :3], g.timestamps_s, 'G1 reference', colors[1]),
            (p['qpos'][:, :3], p['time_s'], 'G1 actual', colors[2])):
        axes[0].plot(roots[:, 0], roots[:, 1], label=label, color=color)
        axes[1].plot(times, roots[:, 2], label=label, color=color)
        axes[2].plot(times[1:], np.diff(roots[:, 1])/-np.diff(times), label=label, color=color, linewidth=.8)
    axes[0].set(xlabel='World X (m)', ylabel='World Y (m)', title='Unmodified Root paths')
    axes[0].axis('equal')
    axes[1].set(xlabel='Time (s)', ylabel='Root height (m)', title='Root height; different body proportions')
    axes[2].set(xlabel='Time (s)', ylabel='Forward velocity (m/s)', title='Forward speed')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(output/'root_paths.png', dpi=150)
    plt.close(fig)
    hf = np.diff(c['target'], n=2, axis=0)**2
    joint = int(np.argmax(hf.mean(0)))
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, constrained_layout=True)
    axes[0].plot(g.timestamps_s, g.joint_pos[:, joint], label='G1 reference', color=colors[1])
    axes[0].plot(c['time'], c['joints'][:, joint], label='Actual', color=colors[2])
    axes[0].plot(c['time']-.02, c['target'][:, joint], label='Actor PD setpoint', color='#7f3c8d', alpha=.7, linewidth=.8)
    axes[0].set(ylabel='Angle (rad)', title=G1_JOINT_NAMES[joint])
    axes[1].plot(c['time'], np.linalg.norm(c['root']-c['reference_root'], axis=1), color=colors[2])
    axes[1].set(xlabel='Time (s)', ylabel='Root error (m)', title='Actual minus fixed reference')
    for axis in axes:
        axis.grid(alpha=.2)
    axes[0].legend()
    fig.savefig(output/'tracking_signals.png', dpi=150)
    plt.close(fig)
    print(json.dumps({key: report[key] for key in ('source_root', 'actual_root', 'final_reference_root_error_xyz_m',
        'g1_reference_joints', 'actor_targets', 'actual_joints', 'measured_support')}, indent=2))


def render(case, tracking, output):
    import imageio_ffmpeg
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30

    s, g, _, c, p, _, model, _ = dataset(case, tracking)
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    source_positions = soma_to_mujoco(s.world_positions_m)
    body_ids = [model.body(name).id for name in g.body_names]
    actual_positions = p['render_body_positions_world'][:, body_ids]
    parents = [SOMASkeleton30().joint_parents.tolist(), [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12]]
    for name, stages in (('source_and_g1_reference', 2), ('three_stage_tracking', 3)):
        path = output/(name+'.mp4')
        if path.exists():
            raise ValueError('Rendered evidence already exists')
        fig = plt.figure(figsize=(12, 4.8), layout='constrained')
        axes, lines = [], []
        labels = ['MB native SOMA30', 'Frozen G1 reference', 'ProtoMotions actual G1']
        colors = ['#3559a6', '#b87614', '#17806b']
        for stage in range(stages):
            axis = fig.add_subplot(1, stages, stage+1, projection='3d')
            axis.set_title(labels[stage])
            axis.set_zlim(-.1, 1.9)
            axis.set_box_aspect((1.7, 1.7, 2.))
            axis.view_init(elev=15, azim=-55)
            axis.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
            edge = parents[0] if stage == 0 else parents[1]
            stage_lines = [(axis.plot([], [], [], color=colors[stage], linewidth=2)[0], j, parent)
                           for j, parent in enumerate(edge) if parent >= 0]
            axes.append(axis)
            lines.append(stage_lines)
        clock = fig.suptitle('')
        end = min(16., float(p['time_s'][-1])) if stages == 3 else 16.
        writer = FFMpegWriter(fps=30, codec='libx264', bitrate=2000, extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(fig, str(path), dpi=100):
            for frame in range(int(np.floor(end*30))):
                time = frame/30
                source_index = int(np.argmin(abs(s.timestamps_s-time)))
                ref_index = int(np.argmin(abs(g.timestamps_s-time)))
                actual_index = int(np.argmin(abs(p['render_time_s']-time)))
                poses = [source_positions[source_index], g.body_pos_w[ref_index], actual_positions[actual_index]]
                for stage, axis in enumerate(axes):
                    pose = poses[stage]
                    root = pose[0]
                    axis.set_xlim(root[0]-.85, root[0]+.85)
                    axis.set_ylim(root[1]-.85, root[1]+.85)
                    for line, joint, parent in lines[stage]:
                        pair = pose[[joint, parent]]
                        line.set_data_3d(pair[:, 0], pair[:, 1], pair[:, 2])
                clock.set_text(f'{time:5.2f}s | 1x replay | cameras follow each Root; world paths in separate plot')
                writer.grab_frame()
                if frame == min(90, int(end*30)-1):
                    fig.savefig(output/(name+'.png'), dpi=100)
        plt.close(fig)
        print(str(path), flush=True)


def render_models(case, tracking, output, render_name):
    import subprocess
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib.font_manager import findfont
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    from motionbricks.mb_sonic.runtime import load_physics_scene

    if Path(render_name).name != render_name:
        raise ValueError('Render name must be a plain basename')
    path = output/(render_name+'.mp4')
    if path.exists():
        raise ValueError('Full-model replay already exists')
    s, g, metadata, _, p, _, _, _ = dataset(case, tracking)
    _, model, _ = load_physics_scene(case/'scene/scene_manifest.json', metadata['robot']['path'],
        flat_ground_primitive='exact_finite_box_v1')
    data = mujoco.MjData(model)
    model.vis.headlight.ambient[:] = .65
    model.vis.headlight.diffuse[:] = .65
    model.vis.headlight.specular[:] = .1
    model.geom_rgba[model.geom_bodyid == 0] = [.85, .89, .87, 1.]
    renderer = mujoco.Renderer(model, height=420, width=400)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.7, 135., -30.
    camera.lookat[2] = .8
    human = mujoco.MjvOption()
    human.geomgroup[:] = 0
    human.geomgroup[0] = 1
    robot = mujoco.MjvOption()
    robot.geomgroup[:] = 0
    robot.geomgroup[[0, 1]] = 1
    parents = SOMASkeleton30().joint_parents.tolist()
    native_points = soma_to_mujoco(s.world_positions_m)
    font = ImageFont.truetype(findfont('DejaVu Sans'), 17)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1200x480', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '20', '-pix_fmt', 'yuv420p', '-n', str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        for frame in range(480):
            time = frame/30
            si = int(np.argmin(abs(s.timestamps_s-time)))
            gi = int(np.argmin(abs(g.timestamps_s-time)))
            ai = int(np.argmin(abs(p['render_time_s']-time)))
            image = Image.new('RGB', (1200, 480), '#f7f8fa')
            draw = ImageDraw.Draw(image)
            for stage, label in enumerate(('MB native SOMA30', 'Frozen G1 reference', 'ProtoMotions actual')):
                root = native_points[si, 0] if stage == 0 else (g.qpos[gi, :3] if stage == 1 else p['render_qpos'][ai, :3])
                camera.lookat[:2] = root[:2]
                if stage == 1:
                    data.qpos[:] = g.qpos[gi]
                elif stage == 2:
                    data.qpos[:] = p['render_qpos'][ai]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=human if stage == 0 else robot)
                if stage == 0:
                    scene = renderer.scene
                    for joint, parent in enumerate(parents):
                        if parent < 0:
                            continue
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.ones(3)*.012,
                            np.zeros(3), np.eye(3).ravel(), np.array([.14, .36, .65, 1.]))
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .012,
                            native_points[si, parent], native_points[si, joint])
                        scene.ngeom += 1
                pixels = renderer.render()
                if frame == 90 and pixels.std() < 2:
                    raise ValueError('Full-model replay has a blank render')
                image.paste(Image.fromarray(pixels), (400*stage, 32))
                draw.text((400*stage+10, 7), label, fill='#20282d', font=font)
            draw.text((10, 456), f'{time:5.2f}s | 1x recorded replay | cameras follow each Root; see separate world-path plot',
                fill='#20282d', font=font)
            process.stdin.write(np.asarray(image).tobytes())
            if frame in (0, 90, 240, 360):
                image.save(output/f'{render_name}_frame_{frame:03d}.png')
    finally:
        process.stdin.close()
        code = process.wait(timeout=60)
        renderer.close()
    if code != 0:
        raise RuntimeError(f'Replay encoding failed: {code}')
    print(str(path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--tracking', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('analyze', 'render', 'render_models'), required=True)
    parser.add_argument('--render-name', default='full_model_three_stage')
    args = parser.parse_args()
    if args.stage == 'analyze':
        analyze(args.case, args.tracking, args.output)
    elif args.stage == 'render':
        render(args.case, args.tracking, args.output)
    else:
        render_models(args.case, args.tracking, args.output, args.render_name)


if __name__ == '__main__':
    main()
