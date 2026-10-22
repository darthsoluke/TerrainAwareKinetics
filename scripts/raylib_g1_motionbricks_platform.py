"""Raylib platform for consuming official MotionBricks G1 rollout qpos.

This viewer treats raylib as a lightweight output surface for the official
MotionBricks navigation stack. Keyboard input is adapted into the same
control_signals fields used by the Python controller, while motion always comes
from full_agent.get_next_frame() / full_agent.generate_new_frames(...).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Callable

import numpy as np

try:
    import torch as t
except ImportError:  # pragma: no cover - lets --print_control_mapping run in lightweight shells
    t = None


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

ColorSpec = tuple[int, int, int, int]


@dataclass
class MotionFrame:
    qpos: np.ndarray
    positions: np.ndarray
    root: np.ndarray
    mode_name: str
    movement_direction: np.ndarray
    facing_direction: np.ndarray


def _require_raylib() -> ModuleType:
    try:
        import pyray as rl
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "raylib/pyray is not installed in this Python environment.\n"
            "Install it in the active conda env, for example:\n"
            "  conda activate motionbricks\n"
            "  python -m pip install raylib"
        ) from exc
    return rl


def _require_torch() -> ModuleType:
    if t is None:  # pragma: no cover - depends on active env
        raise SystemExit(
            "PyTorch is not installed in this Python environment.\n"
            "Run MotionBricks rollout commands in the motionbricks conda env."
        )
    return t


def _create_navigation_demo(args: argparse.Namespace):
    _require_torch()
    from motionbricks.motion_backbone.demo.utils import navigation_demo

    return navigation_demo(args)


def _project_default(*parts: str) -> str:
    return str(PROJECT_DIR.joinpath(*parts))


def _parse_speed_scale(value: str) -> list[float]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("--speed_scale must look like '0.8,1.2'")
    return [float(pieces[0]), float(pieces[1])]


def _mode_index(mode_names: list[str], mode_name: str) -> int:
    if mode_name not in mode_names:
        raise ValueError(f"Unknown MotionBricks mode '{mode_name}'. Available: {', '.join(mode_names)}")
    return mode_names.index(mode_name)


def _xy_to_control_vec(xy: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(xy))
    if norm < 1e-6:
        if fallback is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return fallback.astype(np.float32)
    xy = xy / norm
    return np.array([xy[0], xy[1], 0.0], dtype=np.float32)


def _camera_control_axes(camera_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([math.cos(camera_yaw), math.sin(camera_yaw)], dtype=np.float32)
    right = np.array([math.sin(camera_yaw), -math.cos(camera_yaw)], dtype=np.float32)
    return forward, right


def _wasd_movement_direction(camera_yaw: float, key_down: Callable[[str], bool]) -> np.ndarray:
    forward, right = _camera_control_axes(camera_yaw)
    move_xy = np.zeros(2, dtype=np.float32)
    if key_down("KEY_W"):
        move_xy += forward
    if key_down("KEY_S"):
        move_xy -= forward
    if key_down("KEY_D"):
        move_xy += right
    if key_down("KEY_A"):
        move_xy -= right
    return _xy_to_control_vec(move_xy)


def _control_to_world_xz(control_vec: np.ndarray) -> np.ndarray:
    return np.array([control_vec[1], 0.0, control_vec[0]], dtype=np.float32)


def _make_control_signals(
    mode_names: list[str],
    token_provider: Callable[[str | int], t.Tensor],
    mode_name: str,
    movement_direction: np.ndarray,
    facing_direction: np.ndarray,
) -> dict[str, t.Tensor]:
    torch = _require_torch()
    mode_id = _mode_index(mode_names, mode_name)
    return {
        "movement_direction": torch.from_numpy(movement_direction.astype(np.float32)).view(1, -1),
        "facing_direction": torch.from_numpy(facing_direction.astype(np.float32)).view(1, -1),
        "mode": torch.tensor([[mode_id]], dtype=torch.int64),
        "allowed_pred_num_tokens": token_provider(mode_id).clone().detach(),
    }


class RaylibControlAdapter:
    """Convert raylib key state into official MotionBricks control_signals."""

    NUMBER_KEYS = [
        "KEY_ONE",
        "KEY_TWO",
        "KEY_THREE",
        "KEY_FOUR",
        "KEY_FIVE",
        "KEY_SIX",
        "KEY_SEVEN",
        "KEY_EIGHT",
        "KEY_NINE",
        "KEY_ZERO",
    ]

    PREFERRED_MODES = [
        "idle",
        "slow_walk",
        "walk",
        "stealth_walk",
        "walk_stealth",
        "injured_walk",
        "walk_boxing",
        "walk_zombie",
        "walk_gun",
        "walk_scared",
    ]

    def __init__(
        self,
        rl: ModuleType,
        mode_names: list[str],
        default_keys: dict[str, str],
        token_provider: Callable[[str | int], t.Tensor],
        lookat_movement_direction: bool,
        idle_when_stopped: bool,
        default_mode: str,
    ) -> None:
        self.rl = rl
        self.mode_names = mode_names
        self.default_keys = default_keys
        self.token_provider = token_provider
        self.lookat_movement_direction = lookat_movement_direction
        self.idle_when_stopped = idle_when_stopped
        self.selected_mode = default_mode if default_mode in mode_names else self._fallback_walk_mode()
        self.number_modes = [mode for mode in self.PREFERRED_MODES if mode in mode_names]

    def _fallback_walk_mode(self) -> str:
        for candidate in ("walk", "slow_walk", "idle"):
            if candidate in self.mode_names:
                return candidate
        return self.mode_names[0]

    def _key_const(self, name: str) -> int | None:
        return getattr(self.rl, name, None)

    def _pressed(self, name: str) -> bool:
        key = self._key_const(name)
        return False if key is None else bool(self.rl.is_key_pressed(key))

    def _down(self, name: str) -> bool:
        key = self._key_const(name)
        return False if key is None else bool(self.rl.is_key_down(key))

    def _update_selected_mode(self) -> None:
        for idx, key_name in enumerate(self.NUMBER_KEYS):
            if idx < len(self.number_modes) and self._pressed(key_name):
                self.selected_mode = self.number_modes[idx]
                return

        for mode_name, key_char in self.default_keys.items():
            if not key_char or mode_name not in self.mode_names:
                continue
            if self._pressed(f"KEY_{key_char.upper()}"):
                self.selected_mode = mode_name
                return

    def read(self, camera_yaw: float) -> tuple[dict[str, t.Tensor], dict[str, object]]:
        self._update_selected_mode()

        forward, _ = _camera_control_axes(camera_yaw)
        movement_direction = _wasd_movement_direction(camera_yaw, self._down)
        moving = float(np.linalg.norm(movement_direction)) > 1e-6
        camera_facing = _xy_to_control_vec(forward, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        facing_direction = movement_direction if self.lookat_movement_direction and moving else camera_facing

        if moving:
            active_mode = self.selected_mode if self.selected_mode != "idle" else self._fallback_walk_mode()
        elif self.idle_when_stopped and "idle" in self.mode_names:
            active_mode = "idle"
        else:
            active_mode = self.selected_mode

        control_signals = _make_control_signals(
            self.mode_names,
            self.token_provider,
            active_mode,
            movement_direction,
            facing_direction,
        )
        info = {
            "active_mode": active_mode,
            "selected_mode": self.selected_mode,
            "moving": moving,
            "movement_direction": movement_direction,
            "facing_direction": facing_direction,
        }
        return control_signals, info


class MotionBricksPlatform:
    """Small wrapper around navigation_demo/full_navigation_agent."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.demo_agent = _create_navigation_demo(args)
        self.demo_agent.full_agent.reset()

        skeleton = self.demo_agent.full_agent._motion_rep.skeleton
        self.names = list(skeleton.bone_order_names)
        self.parents = skeleton.joint_parents.detach().cpu().numpy().astype(np.int64)
        self.mode_names = list(self.demo_agent.full_agent._clip_holder.CLIPS.keys())
        self.default_keys = dict(getattr(self.demo_agent.full_agent._clip_holder, "DEFAULT_KEYS", {}))
        self.motion_fps = float(self.demo_agent.inferencer.motion_rep.fps)
        self.controller_dt = self._resolve_controller_dt()
        self.last_frame: MotionFrame | None = None

    def _resolve_controller_dt(self) -> float:
        base_dt = self.demo_agent.controller.get_controller_dt()
        if self.args.controller_dt is not None:
            base_dt = float(self.args.controller_dt)
        return base_dt * float(self.args.generate_dt)

    def token_provider(self, mode: str | int) -> t.Tensor:
        return self.demo_agent.controller.get_default_allowed_pred_num_tokens(mode)

    def idle_control(self, mode_name: str = "idle") -> dict[str, t.Tensor]:
        if mode_name not in self.mode_names:
            mode_name = "idle" if "idle" in self.mode_names else self.mode_names[0]
        return _make_control_signals(
            self.mode_names,
            self.token_provider,
            mode_name,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )

    def step(self, control_signals: dict[str, t.Tensor], control_info: dict[str, object] | None = None) -> MotionFrame:
        qpos = self.demo_agent.full_agent.get_next_frame()
        context_motion_features = self.demo_agent.full_agent.get_context_motion_features()
        context_mujoco_qpos = self.demo_agent.full_agent.get_context_mujoco_qpos()
        self.demo_agent.mj_data.qpos[:] = qpos

        control_signals = dict(control_signals)
        if self.args.use_qpos:
            control_signals["context_mujoco_qpos"] = context_mujoco_qpos
        else:
            control_signals["context_motion_features"] = context_motion_features

        if self.args.random_seed is not None:
            control_signals["random_seed"] = t.tensor([int(self.args.random_seed)], dtype=t.int64)

        with t.no_grad():
            self.demo_agent.full_agent.generate_new_frames(control_signals, self.controller_dt)

        positions = self.qpos_to_positions(qpos)
        root_idx = int(getattr(self.demo_agent.full_agent._motion_rep.skeleton, "root_idx", 0))
        mode_id = int(control_signals["mode"].item())
        mode_name = self.mode_names[max(0, min(mode_id, len(self.mode_names) - 1))]
        frame = MotionFrame(
            qpos=qpos,
            positions=positions,
            root=positions[root_idx].copy(),
            mode_name=mode_name,
            movement_direction=control_signals["movement_direction"][0].detach().cpu().numpy(),
            facing_direction=control_signals["facing_direction"][0].detach().cpu().numpy(),
        )
        self.last_frame = frame
        return frame

    def qpos_to_positions(self, qpos: np.ndarray) -> np.ndarray:
        qpos_tensor = t.as_tensor(qpos, dtype=t.float32, device=self.demo_agent.full_agent._device).view(1, 1, -1)
        positions, _ = self.demo_agent.full_agent._converter.convert_mujoco_qpos_to_motion_transforms(qpos_tensor)
        return positions[0, 0].detach().cpu().numpy().astype(np.float32)


def _color(rl: ModuleType, spec: ColorSpec):
    return rl.Color(*spec)


def _clamp_float(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _shade_color(spec: ColorSpec, factor: float, alpha: int | None = None) -> ColorSpec:
    return (
        int(_clamp_float(spec[0] * factor, 0, 255)),
        int(_clamp_float(spec[1] * factor, 0, 255)),
        int(_clamp_float(spec[2] * factor, 0, 255)),
        spec[3] if alpha is None else int(_clamp_float(alpha, 0, 255)),
    )


def _mix_color(a: ColorSpec, b: ColorSpec, t_value: float, alpha: int | None = None) -> ColorSpec:
    t_value = _clamp_float(t_value, 0.0, 1.0)
    return (
        int(a[0] + (b[0] - a[0]) * t_value),
        int(a[1] + (b[1] - a[1]) * t_value),
        int(a[2] + (b[2] - a[2]) * t_value),
        a[3] if alpha is None else int(_clamp_float(alpha, 0, 255)),
    )


def _with_alpha(spec: ColorSpec, alpha: int) -> ColorSpec:
    return (spec[0], spec[1], spec[2], int(_clamp_float(alpha, 0, 255)))


def _is_pretty(args: argparse.Namespace) -> bool:
    return args.quality == "pretty"


def _joint_color(name: str) -> ColorSpec:
    if name.startswith("left_"):
        return (73, 190, 130, 255)
    if name.startswith("right_"):
        return (239, 104, 104, 255)
    if name.startswith("waist") or name.startswith("pelvis"):
        return (132, 185, 255, 255)
    return (234, 241, 248, 255)


def _bone_material_color(child_name: str, start: np.ndarray, end: np.ndarray) -> ColorSpec:
    base = _joint_color(child_name)
    if "toe" in child_name or "foot" in child_name or "hand" in child_name:
        base = _mix_color(base, (255, 255, 255, 255), 0.10)

    bone_vec = end - start
    length = float(np.linalg.norm(bone_vec))
    if length > 1e-6:
        bone_vec = bone_vec / length
    light_dir = np.array([-0.34, 0.84, 0.42], dtype=np.float32)
    light_dir = light_dir / max(float(np.linalg.norm(light_dir)), 1e-6)
    directional = abs(float(np.dot(bone_vec, light_dir)))
    height_t = _clamp_float(((start[1] + end[1]) * 0.5 + 0.15) / 1.8, 0.0, 1.0)
    return _shade_color(base, 0.72 + 0.17 * height_t + 0.17 * directional)


def _bone_radius_for_name(name: str, base_radius: float) -> float:
    if base_radius <= 0:
        return 0.0
    lower = name.lower()
    if "waist" in lower or "pelvis" in lower or "torso" in lower or "spine" in lower:
        return base_radius * 1.45
    if "hand" in lower or "toe" in lower:
        return base_radius * 0.72
    if "wrist" in lower or "ankle" in lower:
        return base_radius * 0.88
    return base_radius


def _joint_radius_for_name(name: str, parent: int, base_radius: float) -> float:
    if base_radius <= 0:
        return 0.0
    lower = name.lower()
    if parent < 0 or "waist" in lower or "pelvis" in lower:
        return base_radius * 1.25
    if "hand" in lower or "toe" in lower:
        return base_radius * 0.62
    if "ankle" in lower or "wrist" in lower:
        return base_radius * 0.74
    if "knee" in lower or "elbow" in lower:
        return base_radius * 0.86
    return base_radius


def _to_v3(rl: ModuleType, value: np.ndarray, origin: np.ndarray, scale: float):
    p = (value - origin) * scale
    return rl.Vector3(float(p[0]), float(p[1]), float(p[2]))


def _draw_background(rl: ModuleType, args: argparse.Namespace) -> None:
    if not _is_pretty(args):
        rl.clear_background(rl.Color(15, 18, 24, 255))
        return

    if hasattr(rl, "draw_rectangle_gradient_v"):
        rl.draw_rectangle_gradient_v(
            0,
            0,
            args.width,
            args.height,
            rl.Color(18, 25, 35, 255),
            rl.Color(4, 8, 13, 255),
        )
    else:
        rl.clear_background(rl.Color(10, 14, 20, 255))


def _draw_ground_environment(rl: ModuleType, args: argparse.Namespace) -> None:
    if not _is_pretty(args):
        rl.draw_grid(args.grid_slices, args.grid_spacing)
        return

    spacing = max(float(args.grid_spacing), 0.025)
    half_extent = max(float(args.ground_size) * 0.5, float(args.grid_slices) * spacing * 0.5)
    line_count = max(2, int(math.ceil(half_extent / spacing)))
    ground_y = -0.012

    if hasattr(rl, "draw_plane"):
        rl.draw_plane(
            rl.Vector3(0.0, ground_y, 0.0),
            rl.Vector2(half_extent * 2.2, half_extent * 2.2),
            rl.Color(19, 24, 29, 255),
        )
        rl.draw_plane(
            rl.Vector3(0.0, ground_y + 0.003, 0.0),
            rl.Vector2(half_extent * 1.18, half_extent * 1.18),
            rl.Color(24, 31, 38, 180),
        )

    minor = (58, 69, 80, 92)
    major = (91, 110, 128, 132)
    axis = (125, 211, 252, 145)
    y = 0.004
    for index in range(-line_count, line_count + 1):
        coord = index * spacing
        is_axis = index == 0
        is_major = index % 4 == 0
        color = axis if is_axis else major if is_major else minor
        rl.draw_line_3d(
            rl.Vector3(coord, y, -half_extent),
            rl.Vector3(coord, y, half_extent),
            _color(rl, color),
        )
        rl.draw_line_3d(
            rl.Vector3(-half_extent, y, coord),
            rl.Vector3(half_extent, y, coord),
            _color(rl, color),
        )

    horizon = (37, 48, 58, 95)
    edge = half_extent
    rl.draw_line_3d(rl.Vector3(-edge, y, -edge), rl.Vector3(edge, y, -edge), _color(rl, horizon))
    rl.draw_line_3d(rl.Vector3(-edge, y, edge), rl.Vector3(edge, y, edge), _color(rl, horizon))
    rl.draw_line_3d(rl.Vector3(-edge, y, -edge), rl.Vector3(-edge, y, edge), _color(rl, horizon))
    rl.draw_line_3d(rl.Vector3(edge, y, -edge), rl.Vector3(edge, y, edge), _color(rl, horizon))


def _draw_flat_disc(rl: ModuleType, center, radius: float, color: ColorSpec, sides: int = 40) -> None:
    if radius <= 0 or color[3] <= 0:
        return
    if hasattr(rl, "draw_cylinder"):
        rl.draw_cylinder(center, radius, radius, 0.003, sides, _color(rl, color))
    elif hasattr(rl, "draw_circle_3d"):
        rl.draw_circle_3d(center, radius, rl.Vector3(1.0, 0.0, 0.0), 90.0, _color(rl, color))


def _draw_soft_shadow_disc(rl: ModuleType, center, radius: float, alpha: int) -> None:
    if radius <= 0 or alpha <= 0:
        return
    for scale, weight in ((1.70, 0.20), (1.10, 0.30), (0.66, 0.42)):
        _draw_flat_disc(rl, center, radius * scale, (0, 0, 0, int(alpha * weight)))


def _draw_contact_shadows(
    rl: ModuleType,
    args: argparse.Namespace,
    frame: MotionFrame,
    names: list[str],
    origin: np.ndarray,
    scale: float,
) -> None:
    if args.shadow_strength <= 0 or not _is_pretty(args):
        return

    base_alpha = int(150 * _clamp_float(args.shadow_strength, 0.0, 1.0))
    falloff = max(float(args.shadow_falloff), 0.05)
    ground_y = origin[1] + 0.006 / max(scale, 1e-6)

    root_shadow_pos = frame.root.copy()
    root_shadow_pos[1] = ground_y
    root_height = max(0.0, float(frame.root[1] - origin[1]))
    root_t = _clamp_float(1.0 - root_height / (falloff * 2.3), 0.0, 1.0)
    root_center = _to_v3(rl, root_shadow_pos, origin, scale)
    _draw_soft_shadow_disc(rl, root_center, args.contact_shadow_radius * 2.4, int(base_alpha * root_t * 0.42))

    for idx, name in enumerate(names):
        p = frame.positions[idx]
        height = max(0.0, float(p[1] - origin[1]))
        contact = _clamp_float(1.0 - height / falloff, 0.0, 1.0)
        if contact <= 0.025:
            continue

        lower = name.lower()
        weight = 0.20
        radius = args.contact_shadow_radius * (0.52 + 0.58 * contact)
        if "foot" in lower or "toe" in lower or "ankle" in lower:
            weight = 0.92
            radius *= 1.34
        elif "knee" in lower or "hip" in lower or "waist" in lower or "pelvis" in lower:
            weight = 0.36
            radius *= 1.08
        elif "hand" in lower or "wrist" in lower:
            weight = 0.12
            radius *= 0.78

        shadow_pos = p.copy()
        shadow_pos[1] = ground_y
        center = _to_v3(rl, shadow_pos, origin, scale)
        _draw_soft_shadow_disc(rl, center, radius, int(base_alpha * contact * weight))


def _draw_bone_segment(rl: ModuleType, start, end, radius: float, color: ColorSpec) -> None:
    if radius <= 0 or not hasattr(rl, "draw_cylinder_ex"):
        rl.draw_line_3d(start, end, _color(rl, color))
        return
    rl.draw_cylinder_ex(start, end, radius, radius, 14, _color(rl, color))


def _draw_bone_rim(rl: ModuleType, start, end, color: ColorSpec) -> None:
    rim = _mix_color(color, (255, 255, 255, 255), 0.34, alpha=92)
    rl.draw_line_3d(start, end, _color(rl, rim))


def _draw_vector(
    rl: ModuleType,
    root: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    scale: float,
    color: ColorSpec,
    length: float = 0.85,
) -> None:
    world_dir = _control_to_world_xz(direction)
    norm = float(np.linalg.norm(world_dir))
    if norm < 1e-6:
        return
    world_dir = world_dir / norm
    start = root.copy()
    start[1] += 0.08
    end = start + world_dir * length
    rl.draw_line_3d(_to_v3(rl, start, origin, scale), _to_v3(rl, end, origin, scale), _color(rl, color))
    rl.draw_sphere(_to_v3(rl, end, origin, scale), 0.045, _color(rl, color))


def _draw_g1_skeleton(
    rl: ModuleType,
    args: argparse.Namespace,
    frame: MotionFrame,
    names: list[str],
    parents: np.ndarray,
    origin: np.ndarray,
    scale: float,
    draw_joints: bool,
) -> None:
    pretty = _is_pretty(args)
    for idx, parent in enumerate(parents):
        if parent < 0:
            continue
        child_name = names[idx]
        start_world = frame.positions[parent]
        end_world = frame.positions[idx]
        start = _to_v3(rl, start_world, origin, scale)
        end = _to_v3(rl, end_world, origin, scale)
        if pretty:
            color = _bone_material_color(child_name, start_world, end_world)
            radius = _bone_radius_for_name(child_name, args.bone_radius)
            _draw_bone_segment(rl, start, end, radius, color)
            _draw_bone_rim(rl, start, end, color)
        else:
            rl.draw_line_3d(start, end, _color(rl, _joint_color(child_name)))

    if draw_joints:
        for idx, name in enumerate(names):
            if pretty:
                radius = _joint_radius_for_name(name, int(parents[idx]), args.joint_radius)
                color = _mix_color(_joint_color(name), (255, 255, 255, 255), 0.08)
            else:
                radius = 0.035
                if parents[idx] < 0:
                    radius = 0.06
                elif "hand" in name or "toe" in name or "ankle" in name:
                    radius = 0.028
                color = _joint_color(name)
            rl.draw_sphere(_to_v3(rl, frame.positions[idx], origin, scale), radius, _color(rl, color))
            if pretty and hasattr(rl, "draw_sphere_wires") and radius > 0.0:
                rl.draw_sphere_wires(
                    _to_v3(rl, frame.positions[idx], origin, scale),
                    radius * 1.015,
                    8,
                    8,
                    _color(rl, _mix_color(color, (255, 255, 255, 255), 0.30, alpha=76)),
                )

    _draw_vector(rl, frame.root, origin, frame.movement_direction, scale, (250, 204, 21, 255), length=0.75)
    _draw_vector(rl, frame.root, origin, frame.facing_direction, scale, (56, 189, 248, 255), length=0.95)


def _update_camera_from_input(rl: ModuleType, args: argparse.Namespace, state: dict[str, float]) -> None:
    if rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_RIGHT):
        rl.disable_cursor()
    if rl.is_mouse_button_released(rl.MOUSE_BUTTON_RIGHT):
        rl.enable_cursor()
    if rl.is_mouse_button_down(rl.MOUSE_BUTTON_RIGHT):
        mouse_delta = rl.get_mouse_delta()
        state["yaw"] -= mouse_delta.x * args.mouse_sensitivity
        state["pitch"] -= mouse_delta.y * args.mouse_sensitivity

    if rl.is_key_down(rl.KEY_LEFT):
        state["yaw"] += args.keyboard_camera_speed * rl.get_frame_time()
    if rl.is_key_down(rl.KEY_RIGHT):
        state["yaw"] -= args.keyboard_camera_speed * rl.get_frame_time()
    if rl.is_key_down(rl.KEY_UP):
        state["pitch"] += args.keyboard_camera_speed * 0.5 * rl.get_frame_time()
    if rl.is_key_down(rl.KEY_DOWN):
        state["pitch"] -= args.keyboard_camera_speed * 0.5 * rl.get_frame_time()

    state["pitch"] = max(args.min_camera_pitch, min(args.max_camera_pitch, state["pitch"]))


def _set_camera_pose(
    rl: ModuleType,
    camera,
    target,
    yaw: float,
    pitch: float,
    distance: float,
    height: float,
) -> None:
    pitch_cos = math.cos(pitch)
    offset = rl.Vector3(
        math.sin(yaw) * pitch_cos * distance,
        math.sin(pitch) * distance + height,
        math.cos(yaw) * pitch_cos * distance,
    )
    camera.position = rl.Vector3(target.x + offset.x, target.y + offset.y, target.z + offset.z)
    camera.target = target


def _draw_hud(
    rl: ModuleType,
    args: argparse.Namespace,
    platform: MotionBricksPlatform,
    frame: MotionFrame,
    selected_mode: str,
    paused: bool,
    steps: int,
) -> None:
    panel_w = min(max(args.width - 24, 360), 710)
    rl.draw_rectangle(12, 12, panel_w, 118, rl.Color(18, 22, 29, 218))
    rl.draw_text("MotionBricks G1 official qpos consumer", 24, 22, 20, rl.Color(147, 197, 253, 255))
    shadow_note = "fake contact shadows on" if _is_pretty(args) and args.shadow_strength > 0 else "contact shadows off"
    rl.draw_text(f"{args.quality} render | {shadow_note} | qpos from full_agent", 24, 50, 14, rl.RAYWHITE)
    rl.draw_text("WASD move | RMB/Arrows camera | Space pause | J joints | R reset camera", 24, 72, 14, rl.LIGHTGRAY)
    status = (
        f"active {frame.mode_name}  selected {selected_mode}  "
        f"fps {platform.motion_fps:.0f}  controller_dt {platform.controller_dt:.3f}s  "
        f"step {steps if args.max_steps > 0 else steps}"
    )
    if paused:
        status += "  paused"
    rl.draw_text(status, 24, 96, 14, rl.Color(203, 213, 225, 255))


def print_control_mapping() -> None:
    yaw = 0.0
    print("Control mapping at yaw=0.000000 rad")
    for key_name in ("KEY_W", "KEY_A", "KEY_S", "KEY_D"):
        direction = _wasd_movement_direction(yaw, lambda candidate, key_name=key_name: candidate == key_name)
        print(
            f"{key_name[-1]}: movement_direction="
            f"({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})"
        )


def run_headless_check(args: argparse.Namespace) -> None:
    platform = MotionBricksPlatform(args)
    if args.headless_mode not in platform.mode_names:
        available = ", ".join(platform.mode_names)
        raise ValueError(f"--headless_mode '{args.headless_mode}' is not available. Available modes: {available}")

    steps = args.max_steps if args.max_steps > 0 else 5
    control = _make_control_signals(
        platform.mode_names,
        platform.token_provider,
        args.headless_mode,
        np.array([1.0, 0.0, 0.0], dtype=np.float32) if args.headless_mode != "idle" else np.zeros(3, dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    start = time.time()
    frame = None
    for _ in range(steps):
        frame = platform.step(control)
    elapsed = time.time() - start
    root = frame.root if frame is not None else np.zeros(3, dtype=np.float32)
    print(
        f"Headless MotionBricks qpos smoke OK: {steps} steps, "
        f"mode={args.headless_mode}, root=({root[0]:.3f}, {root[1]:.3f}, {root[2]:.3f}), "
        f"{elapsed:.2f}s"
    )


def run_viewer(args: argparse.Namespace) -> None:
    torch = _require_torch()
    rl = _require_raylib()
    platform = MotionBricksPlatform(args)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    adapter = RaylibControlAdapter(
        rl=rl,
        mode_names=platform.mode_names,
        default_keys=platform.default_keys,
        token_provider=platform.token_provider,
        lookat_movement_direction=bool(args.lookat_movement_direction),
        idle_when_stopped=bool(args.idle_when_stopped),
        default_mode=args.default_mode,
    )

    rl.set_config_flags(rl.FLAG_MSAA_4X_HINT)
    rl.init_window(args.width, args.height, "MotionBricks G1 Official qpos Platform")
    rl.set_target_fps(args.render_fps)

    camera = rl.Camera3D(
        rl.Vector3(0.0, 1.8, 4.6),
        rl.Vector3(0.0, 0.9, 0.0),
        rl.Vector3(0.0, 1.0, 0.0),
        args.camera_fovy,
        rl.CAMERA_PERSPECTIVE,
    )
    camera_state = {"yaw": args.camera_yaw, "pitch": args.camera_pitch}
    target = rl.Vector3(0.0, 0.9, 0.0)
    origin = np.zeros(3, dtype=np.float32)
    origin_set = False
    accumulator = 1.0 / platform.motion_fps
    paused = False
    draw_joints = True
    steps = 0
    opened_at = time.time()
    frame = platform.step(platform.idle_control())

    try:
        while not rl.window_should_close():
            if args.auto_close_seconds > 0 and time.time() - opened_at >= args.auto_close_seconds:
                break
            if args.max_steps > 0 and steps >= args.max_steps:
                break

            if rl.is_key_pressed(rl.KEY_SPACE):
                paused = not paused
            if rl.is_key_pressed(rl.KEY_J):
                draw_joints = not draw_joints
            if rl.is_key_pressed(rl.KEY_R):
                camera_state = {"yaw": args.camera_yaw, "pitch": args.camera_pitch}

            _update_camera_from_input(rl, args, camera_state)

            dt = rl.get_frame_time()
            if not paused:
                accumulator += dt
            frame_interval = 1.0 / max(platform.motion_fps, 1.0)
            max_substeps = max(1, args.max_substeps_per_render)
            substeps = 0
            control_info: dict[str, object] = {"selected_mode": adapter.selected_mode}
            while accumulator >= frame_interval and not paused and substeps < max_substeps:
                control, control_info = adapter.read(camera_state["yaw"])
                frame = platform.step(control, control_info)
                accumulator -= frame_interval
                steps += 1
                substeps += 1
                if args.max_steps > 0 and steps >= args.max_steps:
                    break

            if not origin_set:
                origin[:] = frame.root
                origin[1] = 0.0
                origin_set = True

            desired = _to_v3(rl, frame.root + np.array([0.0, args.camera_target_height, 0.0], dtype=np.float32), origin, args.world_scale)
            target = rl.vector3_lerp(target, desired, args.camera_lag)
            _set_camera_pose(
                rl,
                camera,
                target,
                camera_state["yaw"],
                camera_state["pitch"],
                args.camera_distance,
                args.camera_height,
            )

            rl.begin_drawing()
            _draw_background(rl, args)
            rl.begin_mode_3d(camera)
            _draw_ground_environment(rl, args)
            _draw_contact_shadows(rl, args, frame, platform.names, origin, args.world_scale)
            _draw_g1_skeleton(rl, args, frame, platform.names, platform.parents, origin, args.world_scale, draw_joints)
            rl.end_mode_3d()

            _draw_hud(
                rl,
                args,
                platform,
                frame,
                str(control_info.get("selected_mode", adapter.selected_mode)),
                paused,
                steps,
            )
            rl.end_drawing()
    finally:
        if rl.is_cursor_hidden():
            rl.enable_cursor()
        rl.close_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raylib platform that consumes official MotionBricks G1 qpos output."
    )

    parser.add_argument("--humanoid_scene_xml", type=str, default=_project_default("assets", "skeletons", "g1", "scene_29dof.xml"))
    parser.add_argument("--humanoid_xml", type=str, default=None, help="Alias for --humanoid_scene_xml")
    parser.add_argument("--skeleton_xml", type=str, default=_project_default("assets", "skeletons", "g1", "g1.xml"))
    parser.add_argument("--result_dir", type=str, default=_project_default("out"))
    parser.add_argument("--data_root", type=str, default=_project_default("datasets"))
    parser.add_argument("--explicit_dataset_folder", type=str, default=_project_default("datasets", "motionbricks-G1"))
    parser.add_argument("--clips_ckpt", type=str, default=_project_default("out", "G1-clip.ckpt"))
    parser.add_argument("--reprocess_clips", type=int, default=0)

    parser.add_argument("--controller", type=str, default="wasd", choices=["wasd", "random"])
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=_parse_speed_scale, default=[0.8, 1.2])
    parser.add_argument("--generate_dt", type=float, default=2.0)
    parser.add_argument("--controller_dt", type=float, default=None, help="Override base controller dt in seconds")
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--clips", type=str, default="G1")
    parser.add_argument("--random_seed", type=int, default=1234)

    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until the window closes")
    parser.add_argument("--headless_check", action="store_true", help="Run a short qpos rollout without opening raylib")
    parser.add_argument("--print_control_mapping", action="store_true", help="Print yaw=0 WASD movement directions without opening raylib")
    parser.add_argument("--headless_mode", type=str, default="idle")
    parser.add_argument("--default_mode", type=str, default="walk")
    parser.add_argument("--idle_when_stopped", type=int, default=1)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--render_fps", type=int, default=60)
    parser.add_argument("--world_scale", type=float, default=1.0)
    parser.add_argument("--grid_slices", type=int, default=28)
    parser.add_argument("--grid_spacing", type=float, default=0.25)
    parser.add_argument("--quality", type=str, default="pretty", choices=["debug", "pretty"])
    parser.add_argument("--pretty_render", type=int, default=None, help="Alias: 1 selects --quality pretty, 0 selects debug")
    parser.add_argument("--shadow_strength", type=float, default=0.55, help="Fake blob/contact shadow opacity, 0 disables it")
    parser.add_argument("--shadow_falloff", type=float, default=0.85, help="World-space height where contact shadows fade out")
    parser.add_argument("--bone_radius", type=float, default=0.026, help="Pretty-mode rendered bone cylinder radius")
    parser.add_argument("--joint_radius", type=float, default=0.048, help="Pretty-mode rendered joint sphere radius")
    parser.add_argument("--contact_shadow_radius", type=float, default=0.16, help="Base fake contact shadow radius")
    parser.add_argument("--ground_size", type=float, default=9.0, help="Pretty-mode ground plane size in rendered world units")
    parser.add_argument("--camera_distance", type=float, default=4.6)
    parser.add_argument("--camera_height", type=float, default=0.18)
    parser.add_argument("--camera_target_height", type=float, default=0.12)
    parser.add_argument("--camera_yaw", type=float, default=0.0)
    parser.add_argument("--camera_pitch", type=float, default=0.28)
    parser.add_argument("--camera_fovy", type=float, default=48.0)
    parser.add_argument("--min_camera_pitch", type=float, default=-0.15)
    parser.add_argument("--max_camera_pitch", type=float, default=0.95)
    parser.add_argument("--camera_lag", type=float, default=0.22)
    parser.add_argument("--mouse_sensitivity", type=float, default=0.004)
    parser.add_argument("--keyboard_camera_speed", type=float, default=2.2)
    parser.add_argument("--max_substeps_per_render", type=int, default=3)
    parser.add_argument("--auto_close_seconds", type=float, default=0.0)

    args = parser.parse_args()
    if args.humanoid_xml is not None:
        args.humanoid_scene_xml = args.humanoid_xml
    if args.pretty_render is not None:
        args.quality = "pretty" if int(args.pretty_render) else "debug"
    args.shadow_strength = _clamp_float(args.shadow_strength, 0.0, 1.0)
    args.shadow_falloff = max(float(args.shadow_falloff), 0.05)
    args.bone_radius = max(float(args.bone_radius), 0.0)
    args.joint_radius = max(float(args.joint_radius), 0.0)
    args.contact_shadow_radius = max(float(args.contact_shadow_radius), 0.0)
    args.ground_size = max(float(args.ground_size), 1.0)
    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    return args


def main() -> None:
    args = parse_args()
    if args.print_control_mapping:
        print_control_mapping()
    elif args.headless_check:
        run_headless_check(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
