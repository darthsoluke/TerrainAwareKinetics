"""Generate a small, auditable PARC kinematic pilot on local terrain."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "out" / "vendor" / "PARC"
DEFAULT_OUTPUT = ROOT / "out" / "parc_generation_pilot_20261008"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT / "model_40000.ckpt"
DEFAULT_STAIRS_SOURCE = (
    ROOT / "out" / "parc_release_audit_20261008" / "stairs"
    / "7789fd14f77f_beyond_stairs_001_before_opt_dm.pkl"
)
DEFAULT_CLIMB_SOURCE = (
    ROOT / "out" / "parc_release_audit_20261008" / "stairs"
    / "f900cf9e6fd0_mdm_castle_stairs_dm.pkl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def transform_local_intent(local_xy, heading_rad):
    """Normalize a scripted root-local command and rotate it by current heading."""
    import math

    x, y = float(local_xy[0]), float(local_xy[1])
    norm = math.hypot(x, y)
    if norm == 0.0:
        raise ValueError("scripted local intent must be nonzero")
    x, y = x / norm, y / norm
    c, s = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    return (x, y), (c * x - s * y, s * x + c * y)


def motion_metadata(processing):
    return {
        "motion_origin": "PARC official generator raw kinematic output",
        "physics_corrected": False,
        "processing": processing,
    }


def first_contact_with_history(contact_flags, previous_frame_count):
    first_valid = max(0, int(previous_frame_count) - 1)
    for frame_index in range(first_valid, len(contact_flags)):
        if contact_flags[frame_index]:
            return frame_index
    raise ValueError("no current contact frame has enough preceding history")


def build_stairs(terrain_util, direction: str, device: str = "cpu"):
    """Create a deterministic 2.5D staircase; this only defines terrain."""
    import torch

    nx, ny, spacing = 32, 20, 0.4
    terrain = terrain_util.SubTerrain(
        "pilot_stairs_" + direction,
        x_dim=nx,
        y_dim=ny,
        dx=spacing,
        dy=spacing,
        min_x=-6.4,
        min_y=-4.0,
        device=device,
    )
    for x in range(nx):
        if x < 4:
            level = 0
        elif x >= 20:
            level = 8
        else:
            level = min(8, (x - 4) // 2 + 1)
        terrain.hf[x, :] = level * 0.2

    # The source terrain is identical for ascent and descent; only the route
    # endpoints differ. A* and the generator receive the same official terrain.
    start_x, end_x = (2, 29) if direction == "up" else (29, 2)
    start = torch.tensor([start_x, ny // 2], dtype=torch.int64, device=device)
    end = torch.tensor([end_x, ny // 2], dtype=torch.int64, device=device)
    from parc.motion_synthesis.procgen import astar

    settings = astar.AStarSettings()
    settings.max_z_diff = 2.1
    settings.max_jump_xy_dist = 0.0
    settings.w_bumpy = 0.0
    settings.uniform_cost_min = 0.0
    settings.uniform_cost_max = 0.0
    settings.max_cost = 1000.0
    path = astar.run_a_star_on_start_end_nodes(terrain, start, end, settings)
    if path is False or path is None or path.shape[0] < 2:
        raise RuntimeError(f"official A* did not find a {direction} route")
    return terrain, path


def load_official_model(repo: Path, checkpoint_path: Path, device: str, weight_key="model"):
    """Load checkpoint weights into the official PARC MDM class."""
    import torch

    from parc.motion_generator import mdm

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if weight_key not in ("model", "ema_model") or weight_key not in checkpoint:
        raise ValueError(f"Requested checkpoint weights unavailable: {weight_key}")
    cfg = dict(checkpoint["cfg"])
    cfg["device"] = device
    cfg["char_file"] = str((repo / "data" / "assets" / "humanoid.xml").resolve())
    model = mdm.MDM(cfg)
    model._denoise_model.load_state_dict(checkpoint[weight_key], strict=True)
    for key, value in checkpoint.get("extra", {}).items():
        setattr(model, key, value.to(device) if isinstance(value, torch.Tensor) else value)
    model._denoise_model.eval()
    model._device = device
    return model


def save_terrain(path: Path, terrain, path_nodes, np):
    np.savez_compressed(
        path,
        hf=terrain.hf.detach().cpu().numpy(),
        hf_maxmin=terrain.hf_maxmin.detach().cpu().numpy(),
        min_point=terrain.min_point.detach().cpu().numpy(),
        dxdy=terrain.dxdy.detach().cpu().numpy(),
        path_nodes=path_nodes.detach().cpu().numpy(),
    )


def save_motion(path: Path, frames, terrain, path_nodes, model, file_io_helper, processing):
    if len(frames.root_pos.shape) == 3:
        frames = frames.get_idx(0)
    file_io_helper.save_ms_file(
        filepath=str(path),
        motion_frames=frames,
        fps=model._sequence_fps,
        loop_mode="CLAMP",
        terrain=terrain,
        hf_mask_inds=None,
        body_constraints=None,
        min_point_offset=None,
        path_nodes=path_nodes,
        obs=None,
        obs_shapes=None,
        cam_params=None,
        extra_misc_data=motion_metadata(processing),
    )
    return frames


def save_fk_world_terrain_diagnostics(path: Path, frames, terrain, model, np, torch):
    """Save FK body origins and their heightfield-relative clearances."""
    char_model = model._kin_char_model
    body_pos, body_rot = char_model.forward_kinematics(
        frames.root_pos.unsqueeze(0), frames.root_rot.unsqueeze(0), frames.joint_rot.unsqueeze(0)
    )
    body_pos, body_rot = body_pos[0], body_rot[0]
    body_xy = body_pos[..., :2]
    terrain_z = terrain.get_hf_val_from_points(body_xy)
    raw_grid = torch.round((body_xy - terrain.min_point) / terrain.dxdy).to(torch.int64)
    in_bounds = (
        (raw_grid[..., 0] >= 0) & (raw_grid[..., 0] < terrain.hf.shape[0])
        & (raw_grid[..., 1] >= 0) & (raw_grid[..., 1] < terrain.hf.shape[1])
    )
    clearance = body_pos[..., 2] - terrain_z
    contacts = frames.contacts if frames.contacts is not None else torch.zeros(
        (body_pos.shape[0], body_pos.shape[1]), dtype=torch.float32
    )
    names = char_model.get_body_names()
    hand_ids = [i for i, name in enumerate(names) if "hand" in name.lower() and i < contacts.shape[-1]]
    hand_frames = int((contacts[:, hand_ids] > 0.5).any(dim=-1).sum()) if hand_ids else 0
    np.savez_compressed(
        path,
        root_pos=frames.root_pos.detach().cpu().numpy(),
        root_rot=frames.root_rot.detach().cpu().numpy(),
        body_world_pos=body_pos.detach().cpu().numpy(),
        body_world_rot=body_rot.detach().cpu().numpy(),
        body_names=np.asarray(names),
        terrain_height_at_body_xy=terrain_z.detach().cpu().numpy(),
        body_origin_clearance=clearance.detach().cpu().numpy(),
        body_center_in_heightfield_bounds=in_bounds.detach().cpu().numpy(),
        body_contacts=contacts.detach().cpu().numpy(),
    )
    return {
        "fk_body_count": len(names),
        "root_displacement_xyz": (frames.root_pos[-1] - frames.root_pos[0]).detach().cpu().tolist(),
        "body_origin_clearance_min_m": float(clearance.min().detach().cpu()),
        "body_origin_clearance_max_m": float(clearance.max().detach().cpu()),
        "body_centers_outside_heightfield_samples": int((~in_bounds).sum().detach().cpu()),
        "hand_contact_body_ids": hand_ids,
        "hand_contact_frames": hand_frames,
        "diagnostic_scope": "FK body origins sampled against the heightfield; not mesh/surface collision certification",
    }


def unique_run_directory(output_root: Path) -> Path:
    from datetime import datetime, timezone

    output_root.mkdir(parents=True, exist_ok=True)
    stem = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / stem
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def install_observation_capture(mdm_path, terrain_util, torch_util, observations):
    """Capture the exact finite local heightfields passed to the official MDM."""
    import torch

    from parc.motion_generator import gen_util

    original = gen_util.gen_mdm_motion

    def captured(*args, **kwargs):
        values = dict(zip(("target_world_pos", "prev_frames", "terrain", "mdm_model", "mdm_settings"), args))
        values.update(kwargs)
        frames = values["prev_frames"]
        terrain = values["terrain"]
        model = values["mdm_model"]
        batch = frames.root_pos.shape[0]
        idx = model._num_prev_states - 1
        center_xy = frames.root_pos[:, idx:idx + 1, 0:2]
        heading = torch_util.calc_heading(frames.root_rot[:, idx:idx + 1])
        local_hf = terrain_util.sample_hf_z_on_terrain(
            terrain=terrain,
            center_xy=center_xy,
            heading=heading,
            dx=model._dx,
            dy=model._dy,
            num_x_neg=model._num_x_neg,
            num_x_pos=model._num_x_pos,
            num_y_neg=model._num_y_neg,
            num_y_pos=model._num_y_pos,
        )
        target = values.get("target_world_pos")
        if target is not None:
            target = target.expand(batch, *target.shape[1:])
            delta = target[:, 0:2] - center_xy[:, 0, :]
            target_dir = torch_util.rotate_2d_vec(delta, -heading[:, 0])
            target_dir = torch.nn.functional.normalize(target_dir, dim=-1)
        else:
            target_dir = values.get("target_dir")
        for row in range(batch):
            observations.append({
                "heightmap": local_hf[row].detach().cpu().numpy().astype("float32"),
                "root_pos": frames.root_pos[row, idx].detach().cpu().numpy().astype("float32"),
                "root_heading_rad": float(heading[row, 0].detach().cpu()),
                "target_world_pos": None if target is None else target[row].detach().cpu().numpy().astype("float32"),
                "target_dir_local": None if target_dir is None else target_dir[row].detach().cpu().numpy().astype("float32"),
                "batch_row": row,
            })
        return original(*args, **kwargs)

    gen_util.gen_mdm_motion = captured
    return original


def run_stair_case(name, direction, model, output_dir, settings, modules, report):
    import numpy as np
    import torch

    from parc.motion_synthesis.procgen import mdm_path
    from parc.motion_generator import gen_util
    from parc.util import torch_util

    terrain, path_nodes = build_stairs(modules["terrain_util"], direction)
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    save_terrain(case_dir / "terrain_and_route.npz", terrain, path_nodes, np)
    (case_dir / "input_condition.json").write_text(json.dumps({
        "case": name,
        "terrain_representation": "2.5D heightfield, 32x20 cells at 0.4m spacing",
        "terrain_input_to_model": "Only official root/heading-centered local heightmap; no global map tensor",
        "local_heightmap_shape": [31, 31],
        "route_nodes_shape": list(path_nodes.shape),
        "intent": direction,
        "candidate_batch_size": settings.mdm_batch_size,
        "conditioning_previous_frames": model._num_prev_states,
        "physics_corrected": False,
    }, indent=2))

    observations = []
    original = install_observation_capture(
        mdm_path, modules["terrain_util"], torch_util, observations
    )
    started = time.time()
    try:
        candidates, candidate_terrains, info = mdm_path.generate_frames_until_end_of_path(
            path_nodes=path_nodes,
            terrain=terrain,
            char_model=model._kin_char_model,
            mdm_model=model,
            prev_frames=None,
            mdm_path_settings=settings,
            mdm_gen_settings=gen_util.MDMGenSettings(),
            add_noise_to_loss=False,
            verbose=False,
            slice_terrain=False,
            first_heading_mode="auto",
        )
    finally:
        gen_util.gen_mdm_motion = original
    elapsed = time.time() - started

    if not candidates:
        raise RuntimeError(f"official PARC generator returned no candidates for {name}")
    candidate = candidates[0]
    out_motion = save_motion(
        case_dir / "generated_raw_kinematic.pkl", candidate,
        candidate_terrains[0], path_nodes, model, modules["file_io_helper"],
        processing=(
            "Official autoregressive path generation; PARC contact/penetration/path-completion heuristic computed and candidates sorted; "
            "batch size 1, so no multi-candidate ranking; no threshold filter, kinematic optimization, hesitation removal, or physics correction"
        ),
    )
    diagnostics = save_fk_world_terrain_diagnostics(
        case_dir / "kinematic_world_terrain_diagnostics.npz",
        out_motion, candidate_terrains[0], model, np, torch,
    )
    np.savez_compressed(
        case_dir / "local_model_observations.npz",
        heightmaps=np.stack([item["heightmap"] for item in observations]),
        root_pos=np.stack([item["root_pos"] for item in observations]),
        root_heading_rad=np.asarray([item["root_heading_rad"] for item in observations], dtype=np.float32),
        target_world_pos=np.stack([item["target_world_pos"] for item in observations]),
        target_dir_local=np.stack([item["target_dir_local"] for item in observations]),
    )

    target_pos = path_nodes[-1, 0:3]
    body_pos, _ = model._kin_char_model.forward_kinematics(
        out_motion.root_pos[-1:], out_motion.root_rot[-1:], out_motion.joint_rot[-1:]
    )
    lf = model._kin_char_model.get_body_id("left_foot")
    rf = model._kin_char_model.get_body_id("right_foot")
    avg_foot = (body_pos[0, lf] + body_pos[0, rf]) / 2.0
    endpoint_distance = float(torch.linalg.norm(avg_foot - target_pos).detach().cpu())
    status = "complete" if endpoint_distance < 0.5 else "generated_but_path_not_completed"
    record = {
        "case": name,
        "status": status,
        "failure_reason": None if status == "complete" else "candidate did not finish within the official path generator horizon",
        "candidate_file": str((case_dir / "generated_raw_kinematic.pkl").resolve()),
        "terrain_file": str((case_dir / "terrain_and_route.npz").resolve()),
        "candidate_frames": int(out_motion.root_pos.shape[0]),
        "fps": int(model._sequence_fps),
        "root_start": out_motion.root_pos[0].detach().cpu().tolist(),
        "root_end": out_motion.root_pos[-1].detach().cpu().tolist(),
        "goal_average_foot_distance_m": endpoint_distance,
        "official_heuristic_total_loss": jsonable(info["losses"]),
        "official_heuristic_contact_loss": jsonable(info["contact_losses"]),
        "official_heuristic_penetration_loss": jsonable(info["pen_losses"]),
        "captured_local_observations": len(observations),
        "local_observation_shape": list(observations[0]["heightmap"].shape) if observations else None,
        "local_observation_min_max": [
            float(np.min([x["heightmap"].min() for x in observations])) if observations else None,
            float(np.max([x["heightmap"].max() for x in observations])) if observations else None,
        ],
        "elapsed_seconds": elapsed,
        "motion_kind": "raw_kinematic",
        "physics_corrected": False,
        "processing": "Official path-generation heuristic scoring/sorting ran; no threshold filtering or kinematic optimization",
        "world_terrain_diagnostics": diagnostics,
    }
    (case_dir / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    report["cases"].append(record)


def run_hand_contact_continuation(source_path, model, output_dir, modules, report, scripted_local_direction):
    import numpy as np
    import torch

    from parc.motion_generator import gen_util
    from parc.util import motion_util, terrain_util, torch_util

    case_dir = output_dir / "hand_contact_continuation"
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, case_dir / "source_climb_motion_and_terrain.pkl")
    source = modules["file_io_helper"].load_ms_file(str(source_path), device="cpu")
    if source.motion_data is None or source.terrain_data is None:
        raise RuntimeError("climbing source is missing motion or terrain")
    frames = motion_util.MotionFrames.from_ms_motion_data(source.motion_data, device="cpu")
    terrain = terrain_util.SubTerrain.from_ms_terrain_data(source.terrain_data, device="cpu")
    contact_labels = frames.contacts
    if contact_labels is None:
        raise RuntimeError("climbing source has no contact labels")
    hand_ids = []
    for body_name in ("left_hand", "right_hand", "left_hand_end", "right_hand_end"):
        try:
            body_id = model._kin_char_model.get_body_id(body_name)
        except (AssertionError, KeyError, ValueError):
            continue
        if body_id not in hand_ids and body_id < contact_labels.shape[-1]:
            hand_ids.append(body_id)
    if not hand_ids:
        raise RuntimeError("official humanoid model exposes no hand contact body IDs")
    hand_contact_frames = torch.nonzero(
        torch.any(contact_labels[:, hand_ids] > 0.5, dim=-1), as_tuple=False
    ).flatten()
    if hand_contact_frames.numel() == 0:
        raise RuntimeError("climbing source has no labeled hand-contact frames")

    contact_flags = torch.zeros(contact_labels.shape[0], dtype=torch.bool)
    contact_flags[hand_contact_frames] = True
    try:
        frame_idx = first_contact_with_history(
            contact_flags.tolist(), model._num_prev_states
        )
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    prev = frames.get_slice(slice(frame_idx - 1, frame_idx + 1)).unsqueeze(0)
    prev.contacts = prev.contacts.to(dtype=torch.float32)
    current_heading = float(torch_util.calc_heading(prev.root_rot[:, -1:])[0, 0])
    local_xy, world_xy = transform_local_intent(scripted_local_direction, current_heading)
    local_dir = torch.tensor([local_xy], dtype=torch.float32)
    world_dir = torch.tensor(world_xy, dtype=torch.float32)
    target_world = torch.zeros((1, 3), dtype=torch.float32)
    target_world[0, :2] = prev.root_pos[0, -1, :2] + world_dir * 2.0
    target_world[0, 2] = prev.root_pos[0, -1, 2]

    observations = []
    from parc.motion_synthesis.procgen import mdm_path

    original = install_observation_capture(
        mdm_path, terrain_util, torch_util, observations
    )
    from parc.motion_generator import gen_util as gen_module
    started = time.time()
    try:
        settings = gen_module.MDMGenSettings()
        settings.use_prev_state = True
        settings.use_cfg = True
        settings.cfg_scale = 0.65
        settings.use_ddim = True
        settings.ddim_stride = max(1, model._diffusion_timesteps // 4)
        generated = gen_module.gen_mdm_motion(
            target_world_pos=target_world,
            prev_frames=prev,
            terrain=terrain,
            mdm_model=model,
            char_model=model._kin_char_model,
            mdm_settings=settings,
        )
    finally:
        gen_module.gen_mdm_motion = original
    elapsed = time.time() - started
    if not observations:
        raise RuntimeError("official MDM call did not expose a local heightmap condition")

    out_frames = save_motion(
        case_dir / "generated_raw_kinematic.pkl", generated,
        terrain, None, model, modules["file_io_helper"],
        processing="Single official MDM local continuation from current/history frames; explicit scripted local intent; no full-path ranking or kinematic optimization",
    )
    source_diagnostics = save_fk_world_terrain_diagnostics(
        case_dir / "source_kinematic_world_terrain_diagnostics.npz",
        frames, terrain, model, np, torch,
    )
    generated_diagnostics = save_fk_world_terrain_diagnostics(
        case_dir / "generated_kinematic_world_terrain_diagnostics.npz",
        out_frames, terrain, model, np, torch,
    )
    np.savez_compressed(
        case_dir / "local_model_observations.npz",
        heightmaps=np.stack([item["heightmap"] for item in observations]),
        root_pos=np.stack([item["root_pos"] for item in observations]),
        root_heading_rad=np.asarray([item["root_heading_rad"] for item in observations], dtype=np.float32),
        target_world_pos=np.stack([item["target_world_pos"] for item in observations]),
        target_dir_local=np.stack([item["target_dir_local"] for item in observations]),
    )
    record = {
        "case": "hand_contact_local_continuation",
        "status": "generation_complete",
        "failure_reason": None,
        "behavior_success_not_assessed": True,
        "source_motion": str((case_dir / "source_climb_motion_and_terrain.pkl").resolve()),
        "source_frame_index": int(frame_idx),
        "hand_contact_body_ids": hand_ids,
        "hand_contact_labels_at_condition": contact_labels[frame_idx, hand_ids].tolist(),
        "scripted_intent_local_xy": list(local_xy),
        "scripted_intent_world_xy": list(world_xy),
        "root_heading_at_condition_rad": current_heading,
        "previous_frame_count": int(model._num_prev_states),
        "conditioning_observation_shape": list(observations[0]["heightmap"].shape),
        "generated_candidate_file": str((case_dir / "generated_raw_kinematic.pkl").resolve()),
        "candidate_frames": int(out_frames.root_pos.shape[0]),
        "generated_hand_contact_frames": generated_diagnostics["hand_contact_frames"],
        "root_displacement_xyz": generated_diagnostics["root_displacement_xyz"],
        "source_world_terrain_diagnostics": source_diagnostics,
        "generated_world_terrain_diagnostics": generated_diagnostics,
        "elapsed_seconds": elapsed,
        "motion_kind": "raw_kinematic",
        "physics_corrected": False,
    }
    (case_dir / "result.json").write_text(json.dumps(record, indent=2, allow_nan=False))
    report["cases"].append(record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stairs-source", type=Path, default=DEFAULT_STAIRS_SOURCE)
    parser.add_argument("--climb-source", type=Path, default=DEFAULT_CLIMB_SOURCE)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--threads", type=int, default=8)
    case_group = parser.add_mutually_exclusive_group()
    case_group.add_argument("--stairs-only", action="store_true")
    case_group.add_argument("--hand-only", action="store_true")
    parser.add_argument(
        "--hand-intent-local", nargs=2, type=float, metavar=("X", "Y"),
        default=(1.0, 0.0),
        help="scripted intent in the current PARC root frame; default is forward",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    checkpoint_path = args.checkpoint.resolve()
    stairs_source = args.stairs_source.resolve()
    climb_source = args.climb_source.resolve()
    output_root = args.output.resolve()
    output_dir = unique_run_directory(output_root)
    if not (repo / "PARC" / "motion_generator" / "mdm.py").is_file():
        raise FileNotFoundError(f"not an official PARC source checkout: {repo}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    # The official checkout stores the package as PARC/ but imports it as
    # parc. Windows resolves that case mismatch; Linux needs this alias.
    if (repo / "PARC").is_dir() and not (repo / "parc").exists():
        sys.modules.setdefault("parc", importlib.import_module("PARC"))

    import numpy as np
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from parc.util import file_io_helper, terrain_util
    from parc.util import motion_util
    from parc.util import torch_util
    from parc.motion_synthesis.procgen import mdm_path

    model = load_official_model(repo, checkpoint_path, args.device)
    settings = mdm_path.MDMPathSettings()
    # PARC expands a one-element conditioning flag then mutates it in-place;
    # batch size >1 fails on the installed official PyTorch runtime.
    settings.mdm_batch_size = 1
    settings.top_k = 1
    settings.next_node_lookahead = 7
    settings.rewind_num_frames = 5
    settings.max_motion_length = 5.0
    settings.w_contact = 0.1
    settings.w_pen = 0.1

    config = model._cfg
    local_observation_shape = [
        config["heightmap"]["local_grid"]["num_x_neg"] + config["heightmap"]["local_grid"]["num_x_pos"] + 1,
        config["heightmap"]["local_grid"]["num_y_neg"] * 2 + 1,
    ]
    if local_observation_shape != [31, 31]:
        raise RuntimeError(f"checkpoint local observation shape differs from pilot contract: {local_observation_shape}")

    report = {
        "status": "running",
        "source_repository": "https://github.com/mshoe/PARC",
        "source_commit": "846ed98bd526c1dd93ddc1701b601036d553cd2f",
        "executed_script": str(Path(__file__).resolve()),
        "executed_script_sha256": sha256(Path(__file__).resolve()),
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_config": {
            "device": args.device,
            "character_xml": str((repo / "data" / "assets" / "humanoid.xml").resolve()),
            "cnn": config.get("cnn"),
            "num_layers": config.get("num_layers"),
            "d_model": config.get("d_model"),
            "diffusion_timesteps": config.get("diffusion_timesteps"),
            "sequence_fps": model._sequence_fps,
            "sequence_duration": model._sequence_duration,
            "num_prev_states": model._num_prev_states,
            "target_type": str(model._target_type),
            "horizontal_scale": model._dx,
            "local_observation_shape": local_observation_shape,
            "relative_z_style": str(model._relative_z_style),
        },
        "seed": args.seed,
        "requested_cases": (
            ["hand_contact_local_continuation"] if args.hand_only else
            (["stairs_up", "stairs_down"] if args.stairs_only else
             ["stairs_up", "stairs_down", "hand_contact_local_continuation"])
        ),
        "gpu_used": False,
        "motion_kind": "raw_kinematic",
        "physics_corrected": False,
        "coordinate_frame": {
            "stored_motion": "PARC right-handed XYZ, up=+Z, forward at initialization=+X",
            "prospective_mb_mapping_not_applied": "Proper axis rotation PARC (x,y,z)->MB (x,z,-y) maps PARC +X to MB +X, not MB +Z; aligning PARC canonical forward +X with MB semantic forward +Z requires a separate heading/frame transform. Terrain grid origin and heading transforms are also not applied.",
            "soma30_retarget": "not approved or performed",
        },
        "cases": [],
    }
    if stairs_source.is_file():
        source_dir = output_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_copy = source_dir / "parc_stairs_reference_motion_and_terrain.pkl"
        shutil.copy2(stairs_source, source_copy)
        config_source = output_root / "mdm_config.yaml"
        if config_source.is_file():
            shutil.copy2(config_source, source_dir / "mdm_config.yaml")
        source_data = file_io_helper.load_ms_file(str(stairs_source), device="cpu")
        if source_data.motion_data is not None and source_data.terrain_data is not None:
            source_frames = motion_util.MotionFrames.from_ms_motion_data(
                source_data.motion_data, device="cpu"
            )
            source_terrain = terrain_util.SubTerrain.from_ms_terrain_data(
                source_data.terrain_data, device="cpu"
            )
            source_metrics = save_fk_world_terrain_diagnostics(
                source_dir / "stairs_source_fk_world_terrain_diagnostics.npz",
                source_frames, source_terrain, model, np, torch,
            )
            report["stairs_source_geometry_diagnostics"] = source_metrics
    elif not args.hand_only:
        report["stairs_source_geometry_diagnostics"] = {
            "failure_reason": f"stairs source file not found: {stairs_source}"
        }
    try:
        if not args.hand_only:
            for direction in ("up", "down"):
                run_stair_case(
                    f"stairs_{direction}", direction, model, output_dir,
                    settings, {"terrain_util": terrain_util, "file_io_helper": file_io_helper}, report
                )
        if not args.stairs_only:
            if not climb_source.is_file():
                report["cases"].append({
                    "case": "hand_contact_local_continuation",
                    "status": "failed",
                    "failure_reason": f"climbing source not found: {climb_source}",
                })
            else:
                try:
                    run_hand_contact_continuation(
                        climb_source, model, output_dir,
                        {"terrain_util": terrain_util, "file_io_helper": file_io_helper}, report,
                        args.hand_intent_local,
                    )
                except Exception as error:
                    report["cases"].append({
                        "case": "hand_contact_local_continuation",
                        "status": "failed",
                        "failure_reason": f"{type(error).__name__}: {error}",
                    })
        report["status"] = "complete" if all(
            item["status"] in {"complete", "generation_complete"} for item in report["cases"]
        ) else "partial"
        report["failure_reason"] = None if report["status"] == "complete" else "one or more pilot cases were incomplete; see per-case status and failure_reason"
    except Exception as error:
        report["status"] = "failed"
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output_dir / "pilot_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
