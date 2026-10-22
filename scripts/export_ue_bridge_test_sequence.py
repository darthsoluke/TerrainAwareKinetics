"""Export a minimal MotionBricks G1 data contract sample for UE debug playback.

The exporter intentionally uses the official navigation_demo/full_agent path and
only serializes qpos plus global 34-bone transforms. It does not modify model
logic, controller logic, or any raylib viewer code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch as t

from motionbricks.geometry.quaternions import matrix_to_quaternion
from motionbricks.motion_backbone.demo.utils import navigation_demo


SCHEMA_VERSION = "motionbricks_ue_bridge.v1"
QPOS_LEN = 36
ROOT_QUAT_SLICE = slice(3, 7)
ROOT_DOF_OFFSET = 7
QUAT_NORM_TOL = 1e-3


def _as_float_list(array: np.ndarray | list[float]) -> list[float]:
    return [float(x) for x in np.asarray(array, dtype=np.float64).reshape(-1)]


def _as_nested_float_list(array: np.ndarray) -> list[Any]:
    return np.asarray(array, dtype=np.float64).tolist()


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema_hash(schema_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(schema_without_hash).encode("utf-8")).hexdigest()


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("Quaternion norm is zero.")
    return q / norm


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _ue_candidate_position_cm(position_m: np.ndarray) -> np.ndarray:
    p = np.asarray(position_m, dtype=np.float64)
    out = p.copy()
    out[..., 1] *= -1.0
    return out * 100.0


def _extract_dof_names(mj_model: mujoco.MjModel) -> list[str]:
    joints: list[tuple[int, str, int]] = []
    for joint_id in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(mj_model.jnt_type[joint_id])
        qpos_addr = int(mj_model.jnt_qposadr[joint_id])
        joints.append((qpos_addr, name or f"joint_{joint_id}", joint_type))
    joints.sort(key=lambda item: item[0])
    return [name for _addr, name, joint_type in joints if joint_type != int(mujoco.mjtJoint.mjJNT_FREE)]


def _build_schema(demo_agent: Any, args: argparse.Namespace) -> dict[str, Any]:
    skeleton = demo_agent.full_agent._motion_rep.skeleton
    bone_names = list(skeleton.bone_order_names)
    parents = [int(x) for x in skeleton.joint_parents.detach().cpu().numpy().tolist()]
    dof_names = _extract_dof_names(demo_agent.mj_model)
    joint_order = [
        "root_x_m",
        "root_y_m",
        "root_z_m",
        "root_quat_w",
        "root_quat_x",
        "root_quat_y",
        "root_quat_z",
        *dof_names,
    ]

    schema = {
        "schema_version": SCHEMA_VERSION,
        "source": "MotionBricks official navigation_demo(args) / full_agent rollout",
        "source_hz": float(demo_agent.inferencer.motion_rep.fps),
        "frame_stride": int(args.frame_stride),
        "unit": {
            "root_xyz": "meters",
            "bone_positions": "meters",
            "angles": "radians",
            "ue_candidate_positions": "centimeters",
        },
        "coordinate_system": {
            "source": "MuJoCo X forward / Y left / Z up",
            "ue_candidate_position_cm": "UE_cm=(x, -y, z)*100",
            "rotation_note": "Quaternions are wxyz in MuJoCo basis; UE must apply a basis change, not just reorder components.",
        },
        "root_quat_order": "wxyz",
        "qpos_length": QPOS_LEN,
        "joint_order": joint_order,
        "dof_names": dof_names,
        "bone_count": len(bone_names),
        "bone_names": bone_names,
        "parents": parents,
        "bone_position_space": "mujoco_world_meters",
        "bone_rotation_space": "mujoco_world_wxyz",
        "has_bone_global_rotations": True,
    }
    schema["schema_hash"] = _schema_hash(schema)
    return schema


def _validate_schema(schema: dict[str, Any]) -> None:
    if schema["qpos_length"] != QPOS_LEN:
        raise ValueError(f"Expected qpos_length {QPOS_LEN}, got {schema['qpos_length']}.")
    if len(schema["joint_order"]) != QPOS_LEN:
        raise ValueError(f"Expected joint_order length {QPOS_LEN}, got {len(schema['joint_order'])}.")
    if len(schema["dof_names"]) != QPOS_LEN - ROOT_DOF_OFFSET:
        raise ValueError(f"Expected 29 dof_names, got {len(schema['dof_names'])}.")
    if len(schema["bone_names"]) != len(schema["parents"]):
        raise ValueError("bone_names and parents length mismatch.")
    expected_hash = _schema_hash({k: v for k, v in schema.items() if k != "schema_hash"})
    if schema["schema_hash"] != expected_hash:
        raise ValueError("schema_hash does not match schema contents.")


def _build_runtime_contract(schema: dict[str, Any]) -> dict[str, Any]:
    header = {
        "schema_version": schema["schema_version"],
        "source_hz": float(schema["source_hz"]),
        "sample_rate_hz": float(schema["source_hz"]),
        "root_quat_order": schema["root_quat_order"],
        "qpos_length": int(schema["qpos_length"]),
        "joint_order": list(schema["joint_order"]),
        "dof_names": list(schema["dof_names"]),
        "schema_hash": schema["schema_hash"],
    }
    skeleton = {
        "bone_count": int(schema["bone_count"]),
        "bone_names": list(schema["bone_names"]),
        "parents": [int(x) for x in schema["parents"]],
    }
    return {
        **header,
        "schema_header": dict(header),
        "bone_count": skeleton["bone_count"],
        "bone_names": list(skeleton["bone_names"]),
        "parents": list(skeleton["parents"]),
        "skeleton": skeleton,
    }


def _validate_qpos(qpos: np.ndarray) -> None:
    if qpos.shape != (QPOS_LEN,):
        raise ValueError(f"Expected qpos shape ({QPOS_LEN},), got {qpos.shape}.")
    quat = qpos[ROOT_QUAT_SLICE]
    norm = float(np.linalg.norm(quat))
    if abs(norm - 1.0) > QUAT_NORM_TOL:
        raise ValueError(f"Root quaternion must be normalized wxyz; got norm {norm:.6f}.")


def _qpos_to_mujoco_bone_transforms(demo_agent: Any, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _validate_qpos(qpos)
    device = demo_agent.full_agent._device
    qpos_tensor = t.as_tensor(qpos, dtype=t.float32, device=device).view(1, 1, -1)
    positions_motion, rotations_motion = demo_agent.full_agent._converter.convert_mujoco_qpos_to_motion_transforms(qpos_tensor)

    converter = demo_agent.full_agent._converter
    motion_to_mujoco = converter.motion_to_mujoco_matrix.to(device=device, dtype=positions_motion.dtype)
    mujoco_to_motion = converter.mujoco_to_motion_matrix.to(device=device, dtype=positions_motion.dtype)

    positions_mujoco = t.matmul(
        motion_to_mujoco[None, None, None, :, :],
        positions_motion[..., None],
    )[..., 0]
    rotations_mujoco = t.matmul(
        t.matmul(motion_to_mujoco[None, None, None, :, :], rotations_motion),
        mujoco_to_motion[None, None, None, :, :],
    )
    rotations_wxyz = matrix_to_quaternion(rotations_mujoco)

    positions_np = positions_mujoco[0, 0].detach().cpu().numpy().astype(np.float64)
    rotations_np = rotations_wxyz[0, 0].detach().cpu().numpy().astype(np.float64)
    rotations_np = np.asarray([_normalize_quat_wxyz(q) for q in rotations_np], dtype=np.float64)
    return positions_np, rotations_np


def _validate_transforms(schema: dict[str, Any], positions: np.ndarray, rotations: np.ndarray) -> None:
    bone_count = int(schema["bone_count"])
    if positions.shape != (bone_count, 3):
        raise ValueError(f"Expected bone positions shape ({bone_count}, 3), got {positions.shape}.")
    if rotations.shape != (bone_count, 4):
        raise ValueError(f"Expected bone rotations shape ({bone_count}, 4), got {rotations.shape}.")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("Bone transforms contain non-finite values.")
    quat_norms = np.linalg.norm(rotations, axis=1)
    if not np.allclose(quat_norms, 1.0, atol=QUAT_NORM_TOL):
        raise ValueError("One or more bone rotations are not normalized.")


def _frame_from_qpos(
    demo_agent: Any,
    schema: dict[str, Any],
    qpos: np.ndarray,
    source_step: int,
    time_s: float,
) -> dict[str, Any]:
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    _validate_qpos(qpos)
    positions, rotations = _qpos_to_mujoco_bone_transforms(demo_agent, qpos)
    _validate_transforms(schema, positions, rotations)
    return {
        "source_step": int(source_step),
        "time": float(time_s),
        "qpos": _as_float_list(qpos),
        "root_xyz_m": _as_float_list(qpos[:3]),
        "root_quat_wxyz": _as_float_list(_normalize_quat_wxyz(qpos[ROOT_QUAT_SLICE])),
        "bone_global_positions_m": _as_nested_float_list(positions),
        "bone_global_rotations_wxyz": _as_nested_float_list(rotations),
    }


def _rollout_frames(demo_agent: Any, schema: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    full_agent = demo_agent.full_agent
    full_agent.reset()

    if args.random_seed is not None:
        np.random.seed(int(args.random_seed))
        t.manual_seed(int(args.random_seed))

    for step in range(int(args.max_steps)):
        force_idle = step + 30 > int(args.max_steps)
        qpos = np.asarray(full_agent.get_next_frame(), dtype=np.float64).copy()
        context_motion_features = full_agent.get_context_motion_features()
        context_mujoco_qpos = full_agent.get_context_mujoco_qpos()
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
        if args.random_seed is not None:
            control_signals["random_seed"] = t.tensor([int(args.random_seed)], dtype=t.int64)

        with t.no_grad():
            full_agent.generate_new_frames(
                control_signals,
                demo_agent.controller.get_controller_dt() * float(args.generate_dt),
            )

        mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)

        if step % int(args.frame_stride) == 0:
            frames.append(
                _frame_from_qpos(
                    demo_agent,
                    schema,
                    qpos,
                    source_step=step,
                    time_s=step / float(schema["source_hz"]),
                )
            )

    return frames


def _axis_test_from_qpos(
    demo_agent: Any,
    schema: dict[str, Any],
    name: str,
    description: str,
    qpos: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = _frame_from_qpos(demo_agent, schema, qpos, source_step=-1, time_s=0.0)
    root_ue_cm = _ue_candidate_position_cm(np.asarray(frame["root_xyz_m"], dtype=np.float64))
    bone_ue_cm = _ue_candidate_position_cm(np.asarray(frame["bone_global_positions_m"], dtype=np.float64))
    record = {
        "name": name,
        "description": description,
        "qpos": frame["qpos"],
        "root_xyz_m": frame["root_xyz_m"],
        "root_quat_wxyz": frame["root_quat_wxyz"],
        "bone_global_positions_m": frame["bone_global_positions_m"],
        "bone_global_rotations_wxyz": frame["bone_global_rotations_wxyz"],
        "ue_candidate_root_xyz_cm": _as_float_list(root_ue_cm),
        "ue_candidate_bone_positions_cm": _as_nested_float_list(bone_ue_cm),
    }
    if extra:
        record["extra"] = extra
    return record


def _build_axis_tests(
    demo_agent: Any,
    schema: dict[str, Any],
    sequence_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    neutral = np.asarray(demo_agent.mj_model.qpos0, dtype=np.float64).copy()
    if neutral.shape != (QPOS_LEN,):
        neutral = np.zeros(QPOS_LEN, dtype=np.float64)
        neutral[ROOT_QUAT_SLICE] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    neutral[ROOT_QUAT_SLICE] = _normalize_quat_wxyz(neutral[ROOT_QUAT_SLICE])

    plus_x = neutral.copy()
    plus_x[0] += 1.0

    plus_y = neutral.copy()
    plus_y[1] += 1.0

    yaw_90 = neutral.copy()
    yaw_quat = np.array([math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)], dtype=np.float64)
    yaw_90[ROOT_QUAT_SLICE] = _normalize_quat_wxyz(_quat_mul_wxyz(yaw_quat, yaw_90[ROOT_QUAT_SLICE]))

    single_joint = neutral.copy()
    single_joint[ROOT_DOF_OFFSET] += 0.5
    single_joint_name = schema["dof_names"][0]

    tests = [
        _axis_test_from_qpos(
            demo_agent,
            schema,
            "neutral",
            "MuJoCo model qpos0 neutral pose; validates qpos length, wxyz root quaternion, and 34-bone arrays.",
            neutral,
        ),
        _axis_test_from_qpos(
            demo_agent,
            schema,
            "mujoco_x_plus_1m",
            "Neutral root translated +1m along MuJoCo +X; UE candidate delta should be +100cm X.",
            plus_x,
            {"expected_ue_delta_cm_from_neutral": [100.0, 0.0, 0.0]},
        ),
        _axis_test_from_qpos(
            demo_agent,
            schema,
            "mujoco_y_plus_1m",
            "Neutral root translated +1m along MuJoCo +Y; UE candidate delta should be -100cm Y.",
            plus_y,
            {"expected_ue_delta_cm_from_neutral": [0.0, -100.0, 0.0]},
        ),
        _axis_test_from_qpos(
            demo_agent,
            schema,
            "root_yaw_plus_90deg",
            "Neutral root rotated +90deg around MuJoCo +Z; UE must basis-change rotation, not only reorder quaternion components.",
            yaw_90,
            {"axis": "mujoco +Z", "angle_degrees": 90.0},
        ),
        _axis_test_from_qpos(
            demo_agent,
            schema,
            "single_joint",
            f"Neutral pose with {single_joint_name} increased by +0.5 rad.",
            single_joint,
            {"dof_index": ROOT_DOF_OFFSET, "dof_name": single_joint_name, "delta_radians": 0.5},
        ),
    ]

    if sequence_frames:
        first = sequence_frames[0]
        tests.append(
            {
                "name": "same_frame_compare",
                "description": "Main sequence frame 0 copied here so UE can recompute FK from qpos and compare against the exported same-frame 34-bone transforms.",
                "sequence_file": "motionbricks_g1_sequence.json",
                "sequence_frame_index": 0,
                "qpos": first["qpos"],
                "root_xyz_m": first["root_xyz_m"],
                "root_quat_wxyz": first["root_quat_wxyz"],
                "bone_global_positions_m": first["bone_global_positions_m"],
                "bone_global_rotations_wxyz": first["bone_global_rotations_wxyz"],
                "ue_candidate_root_xyz_cm": _as_float_list(
                    _ue_candidate_position_cm(np.asarray(first["root_xyz_m"], dtype=np.float64))
                ),
                "ue_candidate_bone_positions_cm": _as_nested_float_list(
                    _ue_candidate_position_cm(np.asarray(first["bone_global_positions_m"], dtype=np.float64))
                ),
            }
        )

    return {
        **_build_runtime_contract(schema),
        "schema": schema,
        "tests": tests,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    if int(args.max_steps) <= 0:
        raise ValueError("--max_steps must be positive.")
    if int(args.frame_stride) <= 0:
        raise ValueError("--frame_stride must be positive.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_agent = navigation_demo(args)
    schema = _build_schema(demo_agent, args)
    _validate_schema(schema)

    frames = _rollout_frames(demo_agent, schema, args)
    sequence = {
        **_build_runtime_contract(schema),
        "schema": schema,
        "frames": frames,
    }
    axis_tests = _build_axis_tests(demo_agent, schema, frames)

    sequence_path = out_dir / "motionbricks_g1_sequence.json"
    axis_tests_path = out_dir / "axis_tests.json"
    _write_json(sequence_path, sequence)
    _write_json(axis_tests_path, axis_tests)

    qpos_len = len(frames[0]["qpos"]) if frames else 0
    print(f"Wrote {sequence_path}")
    print(f"Wrote {axis_tests_path}")
    print(f"frames={len(frames)} bone_count={schema['bone_count']} qpos_length={qpos_len}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export MotionBricks G1 UE bridge test sequence JSON.")
    parser.add_argument("--humanoid_scene_xml", type=str, default="assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--skeleton_xml", type=str, default="assets/skeletons/g1/g1.xml")
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

    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="out/ue_bridge_test")
    return parser


if __name__ == "__main__":
    parsed_args = _build_arg_parser().parse_args()
    parsed_args.return_model_configs = True
    parsed_args.return_dataloader = True
    parsed_args.recording_dir = None
    parsed_args.EXP = parsed_args.planner
    parsed_args.speed_scale = [float(x) for x in parsed_args.speed_scale.split(",")]
    main(parsed_args)
