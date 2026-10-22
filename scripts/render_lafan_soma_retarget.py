"""Render accepted Geno-to-SOMA joint trajectories without modifying motion."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np


SOMA30_NAMES = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm", "RightForeArm",
    "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd", "LeftLeg", "LeftShin",
    "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)
SOMA30_PARENTS = np.asarray((
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19,
    0, 22, 23, 24, 0, 26, 27, 28,
), dtype=np.int64)
SOLE_NAMES = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")


def _load_clip(path):
    with np.load(path, allow_pickle=False) as archive:
        required = ("source_positions", "source_global_rot_mats", "source_names", "source_parents",
                    "local_rot_mats", "root_positions", "target_positions", "target_fps", "uniform_scale")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"Retarget NPZ is missing fields: {', '.join(missing)}")
        data = {name: archive[name].copy() for name in required}
        if "sole_heights" in archive.files:
            data["sole_heights"] = archive["sole_heights"].copy()

    source = data["source_positions"]
    source_rot = data["source_global_rot_mats"]
    target = data["target_positions"]
    local_rot = data["local_rot_mats"]
    roots = data["root_positions"]
    names, parents = data["source_names"], data["source_parents"]
    if source.ndim != 3 or source.shape[1:] != (75, 3):
        raise ValueError("source_positions must have shape [T,75,3]")
    frames = len(source)
    if frames < 1 or source_rot.shape != (frames, 75, 3, 3):
        raise ValueError("source_global_rot_mats must have shape [T,75,3,3]")
    if names.shape != (75,) or names.dtype.kind not in "US" or len(set(names.tolist())) != 75:
        raise ValueError("source_names must contain 75 unique strings")
    if parents.shape != (75,) or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError("source_parents must contain 75 integer parent indices")
    _validate_parents(parents, 75, "source_parents")
    if local_rot.shape != (frames, 30, 3, 3):
        raise ValueError("local_rot_mats must have shape [T,30,3,3]")
    if roots.shape != (frames, 3) or target.shape != (frames, 30, 3):
        raise ValueError("root_positions/target_positions must have shapes [T,3] and [T,30,3]")
    fps, scale = data["target_fps"], data["uniform_scale"]
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) != 60.0:
        raise ValueError("target_fps must be the scalar 60")
    if scale.size != 1 or not np.isfinite(scale).all() or float(scale.reshape(-1)[0]) <= 0:
        raise ValueError("uniform_scale must be a finite positive scalar")
    for key in ("source_positions", "source_global_rot_mats", "local_rot_mats", "root_positions",
                "target_positions"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if "sole_heights" in data:
        if data["sole_heights"].shape != (frames, 4) or not np.isfinite(data["sole_heights"]).all():
            raise ValueError("sole_heights must be finite with shape [T,4]")
    return data


def _validate_parents(parents, count, field):
    if parents[0] != -1 or np.count_nonzero(parents == -1) != 1:
        raise ValueError(f"{field} must describe one root at joint zero")
    if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, count)):
        raise ValueError(f"{field} must reference an earlier joint for every non-root joint")


def camera_domain(source, target):
    """Fixed, full-segment X/Z/Y bounds in the renderer's Y-up coordinate convention."""
    radius = max(float(np.abs(source[..., (0, 2)]).max()),
                 float(np.abs(target[..., (0, 2)]).max()), 0.4) * 1.08
    low_y = min(0.0, float(source[..., 1].min()), float(target[..., 1].min())) - 0.08
    high_y = max(float(source[..., 1].max()), float(target[..., 1].max())) + 0.08
    return {"x": (-radius, radius), "z": (-radius, radius), "y": (low_y, high_y)}


def _display_positions(data, start, stop):
    source = data["source_positions"][start:stop].astype(np.float64) * float(data["uniform_scale"].reshape(-1)[0])
    target = data["target_positions"][start:stop].astype(np.float64).copy()
    # Follow the scaled source root while retaining source-target root error in the display.
    followed_root_xz = source[:, 0][:, (0, 2)].copy()
    for positions in (source, target):
        positions[..., 0] -= followed_root_xz[:, None, 0]
        positions[..., 2] -= followed_root_xz[:, None, 1]
    return source, target


def _root_tracking_report(data, start, stop):
    scale = float(data["uniform_scale"].reshape(-1)[0])
    source = data["source_positions"][start:stop, 0][:, (0, 2)].astype(np.float64) * scale
    target = data["root_positions"][start:stop][:, (0, 2)].astype(np.float64)
    error = target - source
    return {
        "coordinate_axes": ["X", "Z"],
        "source_root_xz_m": source.tolist(),
        "target_root_xz_m": target.tolist(),
        "target_minus_source_root_xz_m": error.tolist(),
        "rmse_m": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
        "max_error_m": float(np.max(np.linalg.norm(error, axis=1))),
    }


def render_retarget(input_path, output_path, *, behavior="walk", start=0, duration_frames=360,
                    frame_stride=2, gif_fps=30):
    if behavior not in {"walk", "run"}:
        raise ValueError("behavior must be walk or run")
    if start < 0 or duration_frames <= 0 or frame_stride <= 0 or gif_fps <= 0:
        raise ValueError("start, duration, frame stride, and GIF fps must be positive")
    data = _load_clip(input_path)
    stop = start + duration_frames
    if stop > len(data["source_positions"]):
        raise ValueError(f"Requested {duration_frames}-frame segment exceeds {len(data['source_positions'])}-frame clip")
    source, target = _display_positions(data, start, stop)
    bounds = camera_domain(source, target)
    root_tracking = _root_tracking_report(data, start, stop)
    has_soles = "sole_heights" in data

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("Output path must end in .gif")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing animation: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(12, 7.4 if has_soles else 6.5))
    grid = figure.add_gridspec(2 if has_soles else 1, 2, height_ratios=(3.0, 1.0) if has_soles else None)
    axes = [figure.add_subplot(grid[0, column], projection="3d") for column in range(2)]
    source_edges = [(int(parent), joint) for joint, parent in enumerate(data["source_parents"]) if parent >= 0]
    target_edges = [(int(parent), joint) for joint, parent in enumerate(SOMA30_PARENTS) if parent >= 0]
    panels = []
    for axis, points, edges, title, color in (
            (axes[0], source, source_edges, "Geno scaled", "#cf762d"),
            (axes[1], target, target_edges, "SOMA30 retarget", "#168078")):
        radius = bounds["x"][1]
        x, z = np.meshgrid(np.linspace(-radius, radius, 5), np.linspace(-radius, radius, 5))
        axis.plot_surface(x, z, np.zeros_like(x), color="#d9d9d9", alpha=0.16, linewidth=0)
        lines = [axis.plot([], [], [], color=color, linewidth=1.5, solid_capstyle="round")[0]
                 for _ in edges]
        joints = axis.scatter([], [], [], color=color, s=7, depthshade=False)
        axis.set(xlim=bounds["x"], ylim=bounds["z"], zlim=bounds["y"],
                 xlabel="X (m)", ylabel="Z (m)", zlabel="Y (m)")
        axis.set_box_aspect((bounds["x"][1] - bounds["x"][0],
                             bounds["z"][1] - bounds["z"][0],
                             bounds["y"][1] - bounds["y"][0]))
        axis.view_init(elev=12, azim=-48)
        axis.set_proj_type("ortho")
        axis.set_title(title, fontsize=12)
        axis.text2D(0.03, 0.97, "Reference y=0 plane: calibration only, not physical ground truth",
                    transform=axis.transAxes, fontsize=7, va="top", color="#555555")
        panels.append((axis, points, edges, lines, joints))

    sole_axis = None
    sole_cursor = None
    if has_soles:
        sole_axis = figure.add_subplot(grid[1, :])
        times = np.arange(duration_frames) / 60.0
        sole_colors = ("#cf762d", "#e4a66d", "#168078", "#65aaa4")
        for index, (name, color) in enumerate(zip(SOLE_NAMES, sole_colors)):
            sole_axis.plot(times, data["sole_heights"][start:stop, index], label=name,
                           color=color, linewidth=1.0)
        sole_axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        sole_axis.set(xlim=(0, (duration_frames - 1) / 60), xlabel="Segment time (s)",
                      ylabel="Target sole height (m)")
        sole_axis.legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
        sole_cursor = sole_axis.axvline(0, color="#333333", linewidth=0.8)
    caption = "offline GMR retarget | no runtime/post-hoc repair"
    title = figure.suptitle(
        f"{behavior.title()} | start frame {start} | root-following display only; input/scoring unchanged\n{caption}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    rendered = 0
    try:
        with PillowWriter(fps=gif_fps).saving(figure, str(output_path), dpi=90) as writer:
            for frame in range(0, duration_frames, frame_stride):
                for axis, points, edges, lines, joints in panels:
                    current = points[frame]
                    for line, (parent, joint) in zip(lines, edges):
                        pair = current[[parent, joint]]
                        line.set_data_3d(pair[:, 0], pair[:, 2], pair[:, 1])
                    joints._offsets3d = (current[:, 0], current[:, 2], current[:, 1])
                if sole_cursor is not None:
                    sole_cursor.set_xdata([frame / 60.0, frame / 60.0])
                title.set_text(f"{behavior.title()} | {frame / 60:.2f} s | {caption}")
                writer.grab_frame()
                rendered += 1
    finally:
        plt.close(figure)
    if rendered <= 0 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Animation writer produced no usable frames")

    metadata_written = False
    try:
        metadata = {
            "behavior": behavior, "start_frame": start, "source_fps": 60,
            "segment_duration_seconds": duration_frames / 60,
            "rendered_frame_count": rendered, "frame_stride": frame_stride,
            "playback_fps": gif_fps, "uniform_scale": float(data["uniform_scale"].reshape(-1)[0]),
            "caption": caption,
            "display_transform": (
                "Per-frame scaled source-root XZ follow applied identically to source and target for display only; "
                "input motion is unchanged and transformed coordinates are not scored."
            ),
            "root_tracking_original": root_tracking,
            "camera_domain_xyz": {key: list(value) for key, value in bounds.items()},
            "floor_plane": "y=0 calibration reference only; not asserted physical ground truth",
            "retarget_method": "offline GMR retarget", "runtime_posthoc_repair": False,
            "sole_heights_included": has_soles,
            "metrics": {"source_joint_count": 75, "target_joint_count": 30,
                        "source_y_range_m": [float(source[..., 1].min()), float(source[..., 1].max())],
                        "target_y_range_m": [float(target[..., 1].min()), float(target[..., 1].max())]},
        }
        output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        metadata_written = True
    except Exception:
        # A useful GIF remains deliverable even if optional report metadata cannot be written.
        metadata_written = False
    return {"gif": str(output_path), "rendered_frame_count": rendered,
            "metadata_written": metadata_written, "camera_domain_xyz": bounds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Accepted retarget NPZ")
    parser.add_argument("--output", type=Path, required=True, help="New GIF output path")
    parser.add_argument("--behavior", choices=("walk", "run"), default="walk")
    parser.add_argument("--start", type=int, default=0, help="First 60 Hz frame in the six-second segment")
    args = parser.parse_args(argv)
    result = render_retarget(args.input, args.output, behavior=args.behavior, start=args.start)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
