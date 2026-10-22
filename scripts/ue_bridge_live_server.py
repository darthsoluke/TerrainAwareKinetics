"""MotionBricks G1 live server for UE debug playback.

This module keeps the official navigation_demo/full_agent rollout path intact:
UE (or a smoke client) sends small UDP JSON control commands, and the server
returns UDP JSON frame packets containing qpos plus 34-bone global transforms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import select
import socket
import sys
import time
from typing import Any

import mujoco
import numpy as np
import torch as t


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from motionbricks.geometry.quaternions import matrix_to_quaternion
from motionbricks.motion_backbone.demo.utils import navigation_demo


SCHEMA_VERSION = "motionbricks_ue_live.v1"
QPOS_LEN = 36
ROOT_QUAT_SLICE = slice(3, 7)
ROOT_DOF_OFFSET = 7
QUAT_NORM_TOL = 1e-3
DEFAULT_IDLE_COMMAND = {
    "type": "control",
    "mode": "idle",
    "movement_direction_mj": [0.0, 0.0, 0.0],
    "facing_direction_mj": [1.0, 0.0, 0.0],
}


def _as_float_list(array: np.ndarray | list[float]) -> list[float]:
    return [float(x) for x in np.asarray(array, dtype=np.float64).reshape(-1)]


def _as_nested_float_list(array: np.ndarray) -> list[Any]:
    return np.asarray(array, dtype=np.float64).tolist()


def _stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _schema_hash(schema_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(schema_without_hash).encode("utf-8")).hexdigest()


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("Quaternion norm is zero.")
    return q / norm


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
        "source": "MotionBricks official navigation_demo(args) / full_agent live rollout",
        "protocol": "udp_json",
        "source_hz": float(demo_agent.inferencer.motion_rep.fps),
        "tick_hz": float(args.tick_hz),
        "unit": {
            "root_xyz": "meters",
            "bone_positions": "meters",
            "angles": "radians",
        },
        "coordinate_system": {
            "source": "MuJoCo X forward / Y left / Z up",
            "ue_candidate_position_cm": "UE_cm=(x, -y, z)*100",
            "rotation_note": "Quaternions are wxyz in MuJoCo basis; UE must basis-change rotations.",
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


def _validate_qpos(qpos: np.ndarray) -> None:
    if qpos.shape != (QPOS_LEN,):
        raise ValueError(f"Expected qpos shape ({QPOS_LEN},), got {qpos.shape}.")
    norm = float(np.linalg.norm(qpos[ROOT_QUAT_SLICE]))
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
    step: int,
    time_s: float,
    command_info: dict[str, Any],
) -> dict[str, Any]:
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    _validate_qpos(qpos)
    positions, rotations = _qpos_to_mujoco_bone_transforms(demo_agent, qpos)
    _validate_transforms(schema, positions, rotations)
    return {
        **_build_runtime_contract(schema),
        "type": "frame",
        "protocol": "udp_json",
        "source_step": int(step),
        "time": float(time_s),
        "qpos": _as_float_list(qpos),
        "root_xyz_m": _as_float_list(qpos[:3]),
        "root_quat_wxyz": _as_float_list(_normalize_quat_wxyz(qpos[ROOT_QUAT_SLICE])),
        "bone_global_positions_m": _as_nested_float_list(positions),
        "bone_global_rotations_wxyz": _as_nested_float_list(rotations),
        "command": command_info,
    }


def _warn_once(seen: set[str], key: str, message: str) -> None:
    if key not in seen:
        seen.add(key)
        print(f"warning: {message}", flush=True)


def _mode_fallback(mode_names: list[str]) -> str:
    for candidate in ("idle", "walk"):
        if candidate in mode_names:
            return candidate
    return mode_names[0]


def _coerce_vec3(command: dict[str, Any], primary_key: str, fallback_key: str, default: list[float]) -> np.ndarray:
    value = command.get(primary_key, command.get(fallback_key, default))
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 3 or not np.isfinite(arr).all():
        arr = np.asarray(default, dtype=np.float32)
    return arr.astype(np.float32)


def _command_to_control_signals(
    demo_agent: Any,
    command: dict[str, Any],
    warnings_seen: set[str],
) -> tuple[dict[str, t.Tensor], dict[str, Any]]:
    mode_names = list(demo_agent.full_agent._clip_holder.CLIPS.keys())
    requested_mode = command.get("mode", "idle")

    if isinstance(requested_mode, int):
        if 0 <= requested_mode < len(mode_names):
            mode_id = int(requested_mode)
            mode_name = mode_names[mode_id]
        else:
            mode_name = _mode_fallback(mode_names)
            mode_id = mode_names.index(mode_name)
            _warn_once(warnings_seen, f"mode_id:{requested_mode}", f"unknown mode id {requested_mode}; using {mode_name}.")
    else:
        requested_mode = str(requested_mode)
        if requested_mode in mode_names:
            mode_name = requested_mode
        else:
            mode_name = _mode_fallback(mode_names)
            _warn_once(
                warnings_seen,
                f"mode:{requested_mode}",
                f"unknown mode '{requested_mode}'; using {mode_name}. Available modes: {', '.join(mode_names)}",
            )
        mode_id = mode_names.index(mode_name)

    movement_direction = _coerce_vec3(command, "movement_direction_mj", "movement_direction", [0.0, 0.0, 0.0])
    facing_direction = _coerce_vec3(command, "facing_direction_mj", "facing_direction", [1.0, 0.0, 0.0])
    if float(np.linalg.norm(facing_direction)) < 1e-6:
        facing_direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    control_signals = {
        "movement_direction": t.from_numpy(movement_direction.astype(np.float32)).view(1, 3),
        "facing_direction": t.from_numpy(facing_direction.astype(np.float32)).view(1, 3),
        "mode": t.tensor([[mode_id]], dtype=t.int64),
    }

    if "allowed_pred_num_tokens" in command and command["allowed_pred_num_tokens"] is not None:
        allowed = t.as_tensor(command["allowed_pred_num_tokens"], dtype=t.int64).view(1, -1)
    else:
        allowed = demo_agent.controller.get_default_allowed_pred_num_tokens(mode_id).clone().detach()
    control_signals["allowed_pred_num_tokens"] = allowed

    command_info = {
        "mode": mode_name,
        "mode_id": int(mode_id),
        "movement_direction_mj": _as_float_list(movement_direction),
        "facing_direction_mj": _as_float_list(facing_direction),
        "allowed_pred_num_tokens": [int(x) for x in allowed.reshape(-1).detach().cpu().numpy().tolist()],
    }
    return control_signals, command_info


def _recv_latest_udp_command(sock: socket.socket, latest: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, int] | None]:
    sender: tuple[str, int] | None = None
    while True:
        readable, _, _ = select.select([sock], [], [], 0.0)
        if not readable:
            break
        data, addr = sock.recvfrom(65535)
        sender = addr
        try:
            decoded = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as exc:
            print(f"warning: dropped invalid UDP JSON from {addr}: {exc}", flush=True)
            continue
        if isinstance(decoded, dict):
            latest = decoded
        else:
            print(f"warning: dropped non-object UDP JSON from {addr}", flush=True)
    return latest, sender


class MotionBricksLiveServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.random_seed is not None:
            np.random.seed(int(args.random_seed))
            t.manual_seed(int(args.random_seed))
        self.demo_agent = navigation_demo(args)
        self.demo_agent.full_agent.reset()
        self.schema = _build_schema(self.demo_agent, args)
        _validate_schema(self.schema)
        self.controller_dt = self.demo_agent.controller.get_controller_dt() * float(args.generate_dt)
        self.warnings_seen: set[str] = set()
        self.latest_command: dict[str, Any] = dict(DEFAULT_IDLE_COMMAND)
        self.last_sender: tuple[str, int] | None = None

    def step(self, command: dict[str, Any], step_idx: int) -> dict[str, Any]:
        qpos = np.asarray(self.demo_agent.full_agent.get_next_frame(), dtype=np.float64).copy()
        context_motion_features = self.demo_agent.full_agent.get_context_motion_features()
        context_mujoco_qpos = self.demo_agent.full_agent.get_context_mujoco_qpos()
        self.demo_agent.mj_data.qpos[:] = qpos

        control_signals, command_info = _command_to_control_signals(self.demo_agent, command, self.warnings_seen)
        if int(self.args.use_qpos):
            control_signals["context_mujoco_qpos"] = context_mujoco_qpos
            command_info["context"] = "context_mujoco_qpos"
        else:
            control_signals["context_motion_features"] = context_motion_features
            command_info["context"] = "context_motion_features"

        if self.args.random_seed is not None:
            control_signals["random_seed"] = t.tensor([int(self.args.random_seed)], dtype=t.int64)

        with t.no_grad():
            self.demo_agent.full_agent.generate_new_frames(control_signals, self.controller_dt)

        mujoco.mj_forward(self.demo_agent.mj_model, self.demo_agent.mj_data)
        return _frame_from_qpos(
            self.demo_agent,
            self.schema,
            qpos,
            step=step_idx,
            time_s=step_idx / max(float(self.args.tick_hz), 1e-6),
            command_info=command_info,
        )

    def run_udp(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        sock.bind((self.args.host, int(self.args.port)))
        configured_dest = (self.args.send_host, int(self.args.send_port)) if self.args.send_host and int(self.args.send_port) > 0 else None
        print(
            f"MotionBricks UE live UDP server listening on {self.args.host}:{self.args.port}; "
            f"sending to {configured_dest or 'last command sender'}",
            flush=True,
        )

        period = 1.0 / max(float(self.args.tick_hz), 1e-6)
        step_idx = 0
        try:
            while int(self.args.max_steps) <= 0 or step_idx < int(self.args.max_steps):
                loop_start = time.perf_counter()
                self.latest_command, sender = _recv_latest_udp_command(sock, self.latest_command)
                if sender is not None:
                    self.last_sender = sender

                packet = self.step(self.latest_command, step_idx)
                payload = json.dumps(packet, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
                destination = configured_dest or self.last_sender
                if destination is not None:
                    sock.sendto(payload, destination)
                if self.args.print_packets:
                    print(payload.decode("utf-8"), flush=True)

                step_idx += 1
                sleep_s = period - (time.perf_counter() - loop_start)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        finally:
            sock.close()

    def run_smoke_test(self) -> dict[str, Any]:
        frames: list[dict[str, Any]] = []
        commands = [
            dict(DEFAULT_IDLE_COMMAND),
            {
                "type": "control",
                "mode": "walk",
                "movement_direction_mj": [1.0, 0.0, 0.0],
                "facing_direction_mj": [1.0, 0.0, 0.0],
            },
        ]
        max_steps = int(self.args.max_steps) if int(self.args.max_steps) > 0 else 5
        for step_idx in range(max_steps):
            packet = self.step(commands[min(step_idx, 1)], step_idx)
            frames.append(packet)
            if self.args.print_packets:
                print(json.dumps(packet, ensure_ascii=True, separators=(",", ":")), flush=True)

        last = frames[-1]
        qpos = np.asarray(last["qpos"], dtype=np.float64)
        root_quat = np.asarray(last["root_quat_wxyz"], dtype=np.float64)
        bone_positions = np.asarray(last["bone_global_positions_m"], dtype=np.float64)
        result = {
            "ok": bool(
                qpos.shape == (QPOS_LEN,)
                and bone_positions.shape == (34, 3)
                and math.isclose(float(np.linalg.norm(root_quat)), 1.0, abs_tol=QUAT_NORM_TOL)
            ),
            "frames": len(frames),
            "qpos_len": int(qpos.size),
            "bone_count": int(bone_positions.shape[0]),
            "root_quat_norm": float(np.linalg.norm(root_quat)),
            "schema_hash": self.schema["schema_hash"],
            "last_mode": last["command"]["mode"],
        }
        if not result["ok"]:
            raise RuntimeError(f"Smoke validation failed: {result}")

        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "smoke_last_frame.json"
        out_path.write_text(json.dumps(last, ensure_ascii=True, indent=2), encoding="utf-8")
        result["wrote"] = str(out_path)
        print(
            "smoke ok: "
            f"frames={result['frames']} qpos_len={result['qpos_len']} "
            f"bone_count={result['bone_count']} root_quat_norm={result['root_quat_norm']:.6f} "
            f"wrote={out_path}",
            flush=True,
        )
        return result


def _project_default(*parts: str) -> str:
    return str(PROJECT_DIR.joinpath(*parts))


def _parse_speed_scale(value: str) -> list[float]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("--speed_scale must look like '0.8,1.2'")
    return [float(pieces[0]), float(pieces[1])]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MotionBricks G1 UDP JSON live server for UE.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--send_host", type=str, default="127.0.0.1")
    parser.add_argument("--send_port", type=int, default=9011)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until interrupted.")
    parser.add_argument("--tick_hz", type=float, default=30.0)
    parser.add_argument("--print_packets", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")

    parser.add_argument("--humanoid_scene_xml", type=str, default=_project_default("assets", "skeletons", "g1", "scene_29dof.xml"))
    parser.add_argument("--skeleton_xml", type=str, default=_project_default("assets", "skeletons", "g1", "g1.xml"))
    parser.add_argument("--result_dir", type=str, default=_project_default("out"))
    parser.add_argument("--data_root", type=str, default=_project_default("datasets"))
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
    parser.add_argument("--speed_scale", type=_parse_speed_scale, default=[0.8, 1.2])
    parser.add_argument("--generate_dt", type=float, default=2.0)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")
    parser.add_argument("--output_dir", type=str, default=_project_default("out", "ue_bridge_live"))
    return parser


def _finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    return args


def main(args: argparse.Namespace) -> None:
    server = MotionBricksLiveServer(args)
    if args.smoke_test:
        server.run_smoke_test()
    else:
        server.run_udp()


if __name__ == "__main__":
    main(_finalize_args(_build_arg_parser().parse_args()))
