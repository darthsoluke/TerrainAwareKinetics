"""Retarget a MotionBricks G1 rollout to a simple human skeleton and render it.

This is a visualization-oriented retargeter: it keeps MotionBricks' generated
joint directions, but projects them onto fixed human-like limb lengths. It does
not retrain MotionBricks on human motion data, so the result is a quick preview
of "MotionBricks motion on a human rig" rather than a production human model.

Use --render_style skeleton for colored bones or --render_style skin for a
simple procedural capsule skin.
"""

import argparse
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch as t

from motionbricks.motion_backbone.demo.utils import navigation_demo


HUMAN_JOINTS = [
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

HUMAN_BONES = [
    ("pelvis", "spine"),
    ("spine", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("chest", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("chest", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

SKIN_BONES = [
    ("pelvis", "spine", 0.18, "#6f7f8f"),
    ("spine", "chest", 0.20, "#7f8fa3"),
    ("chest", "neck", 0.11, "#7f8fa3"),
    ("pelvis", "left_hip", 0.12, "#8b5a3c"),
    ("left_hip", "left_knee", 0.105, "#d8b089"),
    ("left_knee", "left_ankle", 0.085, "#d8b089"),
    ("left_ankle", "left_foot", 0.075, "#2d2d2d"),
    ("pelvis", "right_hip", 0.12, "#8b5a3c"),
    ("right_hip", "right_knee", 0.105, "#d8b089"),
    ("right_knee", "right_ankle", 0.085, "#d8b089"),
    ("right_ankle", "right_foot", 0.075, "#2d2d2d"),
    ("chest", "left_shoulder", 0.105, "#7f8fa3"),
    ("left_shoulder", "left_elbow", 0.085, "#d8b089"),
    ("left_elbow", "left_wrist", 0.070, "#d8b089"),
    ("chest", "right_shoulder", 0.105, "#7f8fa3"),
    ("right_shoulder", "right_elbow", 0.085, "#d8b089"),
    ("right_elbow", "right_wrist", 0.070, "#d8b089"),
]

LENGTHS = {
    "pelvis_to_spine": 0.24,
    "spine_to_chest": 0.30,
    "chest_to_neck": 0.14,
    "neck_to_head": 0.20,
    "hip_half_width": 0.17,
    "shoulder_half_width": 0.24,
    "upper_leg": 0.48,
    "lower_leg": 0.47,
    "foot": 0.18,
    "upper_arm": 0.31,
    "forearm": 0.28,
}


def _unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return fallback.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = _unit(direction, np.array([0.0, 1.0, 0.0]))
    helper = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(direction, helper))) > 0.92:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = _unit(np.cross(direction, helper), np.array([1.0, 0.0, 0.0]))
    v = _unit(np.cross(direction, u), np.array([0.0, 0.0, 1.0]))
    return u, v


def _plot_capsule(
    ax,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: str,
    segments: int = 14,
    rings: int = 5,
) -> None:
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return
    direction = axis / length
    u, v = _orthonormal_basis(direction)

    theta = np.linspace(0, 2 * np.pi, segments)
    z = np.linspace(0, length, rings)
    theta_grid, z_grid = np.meshgrid(theta, z)
    center = start[None, None, :] + z_grid[..., None] * direction[None, None, :]
    circle = radius * (
        np.cos(theta_grid)[..., None] * u[None, None, :]
        + np.sin(theta_grid)[..., None] * v[None, None, :]
    )
    pts = center + circle
    ax.plot_surface(pts[..., 0], pts[..., 2], pts[..., 1], color=color, linewidth=0, shade=True, alpha=0.96)


def _plot_sphere(ax, center: np.ndarray, radius: float, color: str, segments: int = 16) -> None:
    phi = np.linspace(0, np.pi, segments)
    theta = np.linspace(0, 2 * np.pi, segments)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    x = center[0] + radius * np.sin(phi_grid) * np.cos(theta_grid)
    y = center[2] + radius * np.sin(phi_grid) * np.sin(theta_grid)
    z = center[1] + radius * np.cos(phi_grid)
    ax.plot_surface(x, y, z, color=color, linewidth=0, shade=True, alpha=0.98)


def _idx(names: list[str], name: str) -> int:
    return names.index(name)


def _retarget_frame(g1_pos: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Map G1 motion-space joints to a fixed-proportion human skeleton."""
    p = lambda name: g1_pos[_idx(names, name)]

    pelvis_src = p("pelvis_skel")
    left_hip_src = p("left_hip_pitch_skel")
    right_hip_src = p("right_hip_pitch_skel")
    left_shoulder_src = p("left_shoulder_pitch_skel")
    right_shoulder_src = p("right_shoulder_pitch_skel")
    chest_src = p("waist_pitch_skel")

    up = _unit(chest_src - pelvis_src, np.array([0.0, 1.0, 0.0]))
    right = _unit(right_hip_src - left_hip_src, np.array([1.0, 0.0, 0.0]))
    forward = _unit(np.cross(right, up), np.array([0.0, 0.0, 1.0]))
    right = _unit(np.cross(up, forward), right)

    human = {}
    human["pelvis"] = np.array([pelvis_src[0], pelvis_src[1], pelvis_src[2]], dtype=np.float32)
    human["spine"] = human["pelvis"] + up * LENGTHS["pelvis_to_spine"]
    human["chest"] = human["spine"] + up * LENGTHS["spine_to_chest"]
    human["neck"] = human["chest"] + up * LENGTHS["chest_to_neck"]
    human["head"] = human["neck"] + up * LENGTHS["neck_to_head"]

    human["left_hip"] = human["pelvis"] - right * LENGTHS["hip_half_width"]
    human["right_hip"] = human["pelvis"] + right * LENGTHS["hip_half_width"]
    human["left_shoulder"] = human["chest"] - right * LENGTHS["shoulder_half_width"]
    human["right_shoulder"] = human["chest"] + right * LENGTHS["shoulder_half_width"]

    limb_specs = [
        ("left_hip", "left_knee", "left_knee_skel", LENGTHS["upper_leg"], -up),
        ("left_knee", "left_ankle", "left_ankle_roll_skel", LENGTHS["lower_leg"], -up),
        ("left_ankle", "left_foot", "left_toe_base", LENGTHS["foot"], forward),
        ("right_hip", "right_knee", "right_knee_skel", LENGTHS["upper_leg"], -up),
        ("right_knee", "right_ankle", "right_ankle_roll_skel", LENGTHS["lower_leg"], -up),
        ("right_ankle", "right_foot", "right_toe_base", LENGTHS["foot"], forward),
        ("left_shoulder", "left_elbow", "left_elbow_skel", LENGTHS["upper_arm"], -right),
        ("left_elbow", "left_wrist", "left_hand_roll_skel", LENGTHS["forearm"], -right),
        ("right_shoulder", "right_elbow", "right_elbow_skel", LENGTHS["upper_arm"], right),
        ("right_elbow", "right_wrist", "right_hand_roll_skel", LENGTHS["forearm"], right),
    ]

    source_for_parent = {
        "left_hip": "left_hip_pitch_skel",
        "left_knee": "left_knee_skel",
        "left_ankle": "left_ankle_roll_skel",
        "right_hip": "right_hip_pitch_skel",
        "right_knee": "right_knee_skel",
        "right_ankle": "right_ankle_roll_skel",
        "left_shoulder": "left_shoulder_pitch_skel",
        "left_elbow": "left_elbow_skel",
        "right_shoulder": "right_shoulder_pitch_skel",
        "right_elbow": "right_elbow_skel",
    }
    for parent, child, child_src, length, fallback in limb_specs:
        direction = _unit(p(child_src) - p(source_for_parent[parent]), fallback)
        human[child] = human[parent] + direction * length

    return human


def _render_human_frame(
    human: dict[str, np.ndarray],
    path: Path,
    frame_idx: int,
    render_style: str = "skeleton",
) -> None:
    fig = plt.figure(figsize=(7.2, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f7f7f7")
    fig.patch.set_facecolor("#f7f7f7")

    if render_style == "skin":
        for start, end, radius, color in SKIN_BONES:
            _plot_capsule(ax, human[start], human[end], radius, color)
        _plot_sphere(ax, human["head"], 0.15, "#d8b089")
        _plot_sphere(ax, human["left_wrist"], 0.075, "#d8b089")
        _plot_sphere(ax, human["right_wrist"], 0.075, "#d8b089")
        _plot_sphere(ax, human["left_ankle"], 0.07, "#2d2d2d")
        _plot_sphere(ax, human["right_ankle"], 0.07, "#2d2d2d")
    else:
        for start, end in HUMAN_BONES:
            a, b = human[start], human[end]
            color = "#1f77b4"
            lw = 4.0
            if "left" in start or "left" in end:
                color = "#2ca02c"
            elif "right" in start or "right" in end:
                color = "#d62728"
            ax.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], color=color, linewidth=lw)

        points = np.stack([human[name] for name in HUMAN_JOINTS], axis=0)
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], c="#111111", s=16, depthshade=False)

    center = human["pelvis"]
    radius = 1.25
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(max(0.0, center[1] - 0.15), center[1] + 1.9)
    ax.view_init(elev=12, azim=-70)
    ax.set_axis_off()
    label = "procedural skin" if render_style == "skin" else "human skeleton"
    ax.set_title(f"MotionBricks retargeted {label} | frame {frame_idx:04d}", pad=10)
    plt.tight_layout(pad=0)
    fig.savefig(path)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    demo_agent = navigation_demo(args)
    demo_agent.full_agent.reset()

    names = demo_agent.full_agent._motion_rep.skeleton.bone_order_names
    rendered = []

    for step in range(args.max_steps):
        force_idle = step + 30 > args.max_steps
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
        human = _retarget_frame(g1_positions[0, 0].detach().cpu().numpy(), names)

        frame_path = frames_dir / f"frame_{step:04d}.png"
        _render_human_frame(human, frame_path, step, args.render_style)
        rendered.append(Image.open(frame_path).copy())

    if rendered:
        gif_name = "motionbricks_human_skin.gif" if args.render_style == "skin" else "motionbricks_human_skeleton.gif"
        gif_path = out_dir / gif_name
        rendered[0].save(
            gif_path,
            save_all=True,
            append_images=rendered[1:],
            duration=int(1000 / demo_agent.inferencer.motion_rep.fps * args.frame_stride),
            loop=0,
        )
        print(f"Wrote {len(rendered)} human skeleton frames to {frames_dir}")
        print(f"Wrote GIF to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retarget MotionBricks G1 motion to a simple human skeleton")
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

    parser.add_argument("--max_steps", type=int, default=180)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    parser.add_argument("--output_dir", type=str, default="out/human_skeleton_render")
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--render_style", type=str, default="skeleton", choices=["skeleton", "skin"])
    args = parser.parse_args()

    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]

    start = time.time()
    main(args)
    print(f"Done in {time.time() - start:.1f}s")
