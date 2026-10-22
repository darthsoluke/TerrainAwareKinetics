"""Raylib viewer for MotionBricks locomotion.

This script is a small rendering platform around the existing MotionBricks
Python rollout path. It precomputes a short G1 rollout, retargets it to the
human preview skeleton used by the other demos, and plays it in a raylib 3D
window.
"""

import argparse
import math
import time

import numpy as np
import torch as t

try:
    import pyray as rl
except ImportError as exc:  # pragma: no cover - depends on local env
    raise SystemExit(
        "raylib is not installed in this Python environment. Install with:\n"
        "  python -m pip install raylib"
    ) from exc

from motionbricks.motion_backbone.demo.utils import navigation_demo
from render_human_skeleton_demo import HUMAN_BONES, HUMAN_JOINTS, _retarget_frame


SKIN_BONES = [
    ("pelvis", "spine", 0.10, rl.Color(106, 123, 140, 255)),
    ("spine", "chest", 0.13, rl.Color(112, 132, 150, 255)),
    ("chest", "neck", 0.07, rl.Color(112, 132, 150, 255)),
    ("pelvis", "left_hip", 0.07, rl.Color(68, 78, 89, 255)),
    ("pelvis", "right_hip", 0.07, rl.Color(68, 78, 89, 255)),
    ("left_hip", "left_knee", 0.06, rl.Color(205, 159, 111, 255)),
    ("left_knee", "left_ankle", 0.052, rl.Color(205, 159, 111, 255)),
    ("left_ankle", "left_foot", 0.045, rl.Color(42, 42, 46, 255)),
    ("right_hip", "right_knee", 0.06, rl.Color(205, 159, 111, 255)),
    ("right_knee", "right_ankle", 0.052, rl.Color(205, 159, 111, 255)),
    ("right_ankle", "right_foot", 0.045, rl.Color(42, 42, 46, 255)),
    ("chest", "left_shoulder", 0.06, rl.Color(112, 132, 150, 255)),
    ("left_shoulder", "left_elbow", 0.045, rl.Color(205, 159, 111, 255)),
    ("left_elbow", "left_wrist", 0.04, rl.Color(205, 159, 111, 255)),
    ("chest", "right_shoulder", 0.06, rl.Color(112, 132, 150, 255)),
    ("right_shoulder", "right_elbow", 0.045, rl.Color(205, 159, 111, 255)),
    ("right_elbow", "right_wrist", 0.04, rl.Color(205, 159, 111, 255)),
]


def _approach_angle(current: float, target: float, max_delta: float) -> float:
    delta = (target - current + math.pi) % (math.tau) - math.pi
    delta = max(-max_delta, min(max_delta, delta))
    return current + delta


def _v3(
    value: np.ndarray,
    root_offset: np.ndarray,
    scale: float,
    actor_pos: np.ndarray,
    actor_yaw: float,
) -> rl.Vector3:
    p = (value - root_offset) * scale
    s = math.sin(actor_yaw)
    c = math.cos(actor_yaw)
    x = p[0] * c - p[2] * s + actor_pos[0]
    z = p[0] * s + p[2] * c + actor_pos[2]
    return rl.Vector3(float(x), float(p[1] + actor_pos[1]), float(z))


def _lerp_frame(a: dict[str, np.ndarray], b: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    return {name: a[name] * (1.0 - alpha) + b[name] * alpha for name in HUMAN_JOINTS}


def _precompute_motion(args: argparse.Namespace) -> tuple[list[dict[str, np.ndarray]], float]:
    demo_agent = navigation_demo(args)
    demo_agent.full_agent.reset()
    names = demo_agent.full_agent._motion_rep.skeleton.bone_order_names
    frames: list[dict[str, np.ndarray]] = []

    for step in range(args.precompute_steps):
        force_idle = step + 30 > args.precompute_steps
        qpos = demo_agent.full_agent.get_next_frame()
        context_motion_features = demo_agent.full_agent.get_context_motion_features()
        context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
        demo_agent.mj_data.qpos[:] = qpos

        control_signals = demo_agent.controller.generate_control_signals(
            None,
            demo_agent.mj_model,
            demo_agent.mj_data,
            visualize=False,
            control_info={"force_idle": force_idle, "allowed_mode": args.allowed_mode},
        )
        if args.use_qpos:
            control_signals["context_mujoco_qpos"] = context_mujoco_qpos
        else:
            control_signals["context_motion_features"] = context_motion_features

        with t.no_grad():
            demo_agent.full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * args.generate_dt,
            )

        if step % args.frame_stride != 0:
            continue

        qpos_tensor = t.as_tensor(qpos, dtype=t.float32, device=demo_agent.full_agent._device).view(1, 1, -1)
        g1_positions, _ = demo_agent.full_agent._converter.convert_mujoco_qpos_to_motion_transforms(qpos_tensor)
        frames.append(_retarget_frame(g1_positions[0, 0].detach().cpu().numpy(), names))

    fps = float(demo_agent.inferencer.motion_rep.fps / args.frame_stride)
    return frames, fps


def _draw_character(
    frame: dict[str, np.ndarray],
    root_offset: np.ndarray,
    scale: float,
    actor_pos: np.ndarray,
    actor_yaw: float,
    draw_skin: bool,
) -> None:
    if draw_skin:
        for start, end, radius, color in SKIN_BONES:
            rl.draw_cylinder_ex(
                _v3(frame[start], root_offset, scale, actor_pos, actor_yaw),
                _v3(frame[end], root_offset, scale, actor_pos, actor_yaw),
                radius,
                radius,
                12,
                color,
            )
        rl.draw_sphere(_v3(frame["head"], root_offset, scale, actor_pos, actor_yaw), 0.11, rl.Color(205, 159, 111, 255))
        rl.draw_sphere(_v3(frame["left_wrist"], root_offset, scale, actor_pos, actor_yaw), 0.045, rl.Color(205, 159, 111, 255))
        rl.draw_sphere(_v3(frame["right_wrist"], root_offset, scale, actor_pos, actor_yaw), 0.045, rl.Color(205, 159, 111, 255))

    for start, end in HUMAN_BONES:
        rl.draw_line_3d(
            _v3(frame[start], root_offset, scale, actor_pos, actor_yaw),
            _v3(frame[end], root_offset, scale, actor_pos, actor_yaw),
            rl.Color(235, 242, 250, 255),
        )

    for joint in HUMAN_JOINTS:
        radius = 0.04 if joint not in {"head", "pelvis"} else 0.065
        rl.draw_sphere(_v3(frame[joint], root_offset, scale, actor_pos, actor_yaw), radius, rl.Color(82, 168, 255, 255))


def run_viewer(args: argparse.Namespace) -> None:
    start = time.time()
    frames, fps = _precompute_motion(args)
    if not frames:
        raise RuntimeError("MotionBricks produced no frames; increase --precompute_steps or lower --frame_stride.")

    print(f"Precomputed {len(frames)} MotionBricks frames at {fps:.1f} fps in {time.time() - start:.1f}s")
    if args.headless_check:
        return

    rl.set_config_flags(rl.FLAG_MSAA_4X_HINT)
    rl.init_window(args.width, args.height, "MotionBricks Raylib Viewer")
    rl.set_target_fps(args.render_fps)

    camera = rl.Camera3D(
        rl.Vector3(0.0, 2.0, 4.8),
        rl.Vector3(0.0, 1.0, 0.0),
        rl.Vector3(0.0, 1.0, 0.0),
        48.0,
        rl.CAMERA_PERSPECTIVE,
    )
    frame_cursor = 0.0
    paused = False
    draw_skin = True
    actor_pos = np.zeros(3, dtype=np.float32)
    actor_yaw = 0.0
    camera_yaw = 0.0
    camera_pitch = args.camera_pitch
    camera_distance = args.camera_distance
    camera_target = rl.Vector3(0.0, 1.0, 0.0)
    opened_at = time.time()

    while not rl.window_should_close():
        if args.auto_close_seconds > 0 and time.time() - opened_at >= args.auto_close_seconds:
            break

        dt = rl.get_frame_time()
        if rl.is_key_pressed(rl.KEY_SPACE):
            paused = not paused
        if rl.is_key_pressed(rl.KEY_M):
            draw_skin = not draw_skin
        if rl.is_key_pressed(rl.KEY_R):
            frame_cursor = 0.0
            actor_pos[:] = 0.0
            actor_yaw = 0.0
            camera_yaw = 0.0
            camera_pitch = args.camera_pitch

        if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_RIGHT):
            rl.disable_cursor()
        if rl.is_mouse_button_released(rl.MOUSE_BUTTON_RIGHT):
            rl.enable_cursor()
        if rl.is_mouse_button_down(rl.MOUSE_BUTTON_RIGHT):
            mouse_delta = rl.get_mouse_delta()
            camera_yaw -= mouse_delta.x * args.mouse_sensitivity
            camera_pitch -= mouse_delta.y * args.mouse_sensitivity
            camera_pitch = max(args.min_camera_pitch, min(args.max_camera_pitch, camera_pitch))

        camera_forward = np.array([-math.sin(camera_yaw), 0.0, -math.cos(camera_yaw)], dtype=np.float32)
        camera_right = np.array([math.cos(camera_yaw), 0.0, -math.sin(camera_yaw)], dtype=np.float32)
        move = np.zeros(3, dtype=np.float32)
        if rl.is_key_down(rl.KEY_W):
            move += camera_forward
        if rl.is_key_down(rl.KEY_S):
            move -= camera_forward
        if rl.is_key_down(rl.KEY_D):
            move += camera_right
        if rl.is_key_down(rl.KEY_A):
            move -= camera_right

        move_len = float(np.linalg.norm(move))
        moving = move_len > 1e-5
        speed = args.run_speed if rl.is_key_down(rl.KEY_LEFT_SHIFT) or rl.is_key_down(rl.KEY_RIGHT_SHIFT) else args.walk_speed
        locomotion_rate = 0.0
        if moving:
            move /= move_len
            actor_pos += move * speed * dt
            desired_yaw = math.atan2(float(move[0]), float(move[2]))
            actor_yaw = _approach_angle(actor_yaw, desired_yaw, args.turn_speed * dt)
            locomotion_rate = speed / max(args.walk_speed, 1e-5)

        if not paused and moving:
            frame_cursor = (frame_cursor + dt * fps * args.playback_speed * locomotion_rate) % len(frames)
        if not moving and args.freeze_idle:
            frame_cursor = 0.0
        idx0 = int(frame_cursor) % len(frames)
        idx1 = (idx0 + 1) % len(frames)
        alpha = frame_cursor - int(frame_cursor)
        frame = _lerp_frame(frames[idx0], frames[idx1], alpha)

        root_offset = frames[0]["pelvis"].copy()
        target_height = 1.05
        desired_target = rl.Vector3(float(actor_pos[0]), target_height, float(actor_pos[2]))
        camera_target = rl.vector3_lerp(camera_target, desired_target, args.camera_lag)
        pitch_cos = math.cos(camera_pitch)
        camera_offset = rl.Vector3(
            math.sin(camera_yaw) * pitch_cos * camera_distance,
            math.sin(camera_pitch) * camera_distance + args.camera_height,
            math.cos(camera_yaw) * pitch_cos * camera_distance,
        )
        camera.position = rl.Vector3(
            camera_target.x + camera_offset.x,
            camera_target.y + camera_offset.y,
            camera_target.z + camera_offset.z,
        )
        camera.target = camera_target

        rl.begin_drawing()
        rl.clear_background(rl.Color(17, 20, 26, 255))
        rl.begin_mode_3d(camera)
        rl.draw_grid(24, 0.25)
        _draw_character(frame, root_offset, args.world_scale, actor_pos, actor_yaw, draw_skin)
        rl.end_mode_3d()

        rl.draw_rectangle(12, 12, 470, 104, rl.Color(18, 22, 29, 210))
        rl.draw_text("MotionBricks on raylib", 24, 22, 20, rl.Color(147, 197, 253, 255))
        rl.draw_text("WASD move | Shift run | RMB mouse camera | M mesh | R reset", 24, 50, 14, rl.RAYWHITE)
        rl.draw_text("Space pause animation", 24, 70, 14, rl.LIGHTGRAY)
        rl.draw_text(f"frames {len(frames)}  fps {fps:.1f}  speed {speed if moving else 0.0:.2f} m/s", 24, 90, 14, rl.LIGHTGRAY)
        rl.end_drawing()

    if rl.is_cursor_hidden():
        rl.enable_cursor()
    rl.close_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a MotionBricks rollout in raylib")

    parser.add_argument("--humanoid_xml", type=str, default="assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--result_dir", type=str, default="./out")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)

    parser.add_argument("--controller", type=str, default="random", choices=["wasd", "random"])
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)

    parser.add_argument("--precompute_steps", type=int, default=260)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--render_fps", type=int, default=60)
    parser.add_argument("--playback_speed", type=float, default=1.0)
    parser.add_argument("--world_scale", type=float, default=1.0)
    parser.add_argument("--walk_speed", type=float, default=1.25)
    parser.add_argument("--run_speed", type=float, default=2.6)
    parser.add_argument("--turn_speed", type=float, default=9.0)
    parser.add_argument("--camera_distance", type=float, default=4.6)
    parser.add_argument("--camera_height", type=float, default=0.28)
    parser.add_argument("--camera_pitch", type=float, default=0.24)
    parser.add_argument("--min_camera_pitch", type=float, default=-0.15)
    parser.add_argument("--max_camera_pitch", type=float, default=0.9)
    parser.add_argument("--camera_lag", type=float, default=0.18)
    parser.add_argument("--mouse_sensitivity", type=float, default=0.004)
    parser.add_argument("--freeze_idle", type=int, default=1)
    parser.add_argument("--headless_check", action="store_true")
    parser.add_argument("--auto_close_seconds", type=float, default=0.0)

    args = parser.parse_args()
    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]
    return args


if __name__ == "__main__":
    run_viewer(parse_args())
