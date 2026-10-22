"""Render a short MotionBricks G1 rollout without opening the MuJoCo viewer.

This script is intentionally close to scripts/interactive_demo_g1.py, but it
uses MuJoCo's offscreen Renderer and writes a PNG sequence plus an animated GIF.
It is useful for quickly verifying that the pretrained checkpoints can generate
motion in a non-interactive environment.
"""

import argparse
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
import torch as t

from motionbricks.motion_backbone.demo.utils import navigation_demo


def _camera_track_robot(cam: mujoco.MjvCamera, qpos: np.ndarray) -> None:
    cam.lookat[:] = qpos[:3]
    cam.lookat[2] += 0.6
    cam.distance = 3.2
    cam.azimuth = 145
    cam.elevation = -18


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    demo_agent = navigation_demo(args)
    demo_agent.full_agent.reset()

    renderer = mujoco.Renderer(demo_agent.mj_model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    rendered_frames = []

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

        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
        _camera_track_robot(camera, qpos)
        renderer.update_scene(demo_agent.mj_data, camera=camera)
        frame = renderer.render()

        if step % args.frame_stride == 0:
            image = Image.fromarray(frame)
            image.save(frames_dir / f"frame_{step:04d}.png")
            rendered_frames.append(image)

    renderer.close()

    if rendered_frames:
        gif_path = out_dir / "motionbricks_headless.gif"
        rendered_frames[0].save(
            gif_path,
            save_all=True,
            append_images=rendered_frames[1:],
            duration=int(1000 / demo_agent.inferencer.motion_rep.fps * args.frame_stride),
            loop=0,
        )
        print(f"Wrote {len(rendered_frames)} frames to {frames_dir}")
        print(f"Wrote GIF to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless MotionBricks rendering demo")
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

    parser.add_argument("--output_dir", type=str, default="out/headless_render")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frame_stride", type=int, default=2)
    args = parser.parse_args()

    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]

    start = time.time()
    main(args)
    print(f"Done in {time.time() - start:.1f}s")
