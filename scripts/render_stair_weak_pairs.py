"""Render a small generic review set of planar weak-support evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np


CLIP_SECONDS = 6.0
MAX_GIF_FRAMES = 120
REVIEW_SELECTION = (("train", "walk"), ("train", "run"), ("validation", "walk"))
SIDES = ("left", "right")


def _select_rows(records):
    selected = []
    for split, behavior in REVIEW_SELECTION:
        record = next((row for row in records if row.get("execution_status") == "processed"
                       and row.get("split") == split and row.get("behavior") == behavior), None)
        if record is None:
            raise ValueError(f"No processed {split} {behavior} item found in manifest")
        selected.append(record)
    return selected


def _segment(root_positions, fps, count):
    length = min(count, max(1, int(round(CLIP_SECONDS * fps))))
    if length == count:
        return 0, count
    horizontal_speed = np.linalg.norm(np.diff(root_positions[:, [0, 2]], axis=0), axis=-1) * fps
    window = max(1, int(round(0.5 * fps)))
    prefix = np.concatenate(([0.0], np.cumsum(horizontal_speed, dtype=np.float64)))
    mean_speed = (prefix[window:] - prefix[:-window]) / window if len(prefix) > window else np.zeros(0)
    moving = np.flatnonzero(mean_speed >= 0.18)
    start = int(moving[0]) if len(moving) else 0
    start = min(start, count - length)
    return start, start + length


def _sample_frames(start, end, fps):
    stride = max(1, int(round(float(fps) / 15.0)))
    frames = np.arange(start, end, stride, dtype=np.int64)
    if len(frames) > MAX_GIF_FRAMES:
        frames = frames[:MAX_GIF_FRAMES]
    return frames


def _edges(parents):
    return np.asarray([(int(parent), child) for child, parent in enumerate(parents) if int(parent) >= 0],
                      dtype=np.int64)


def render_item(dataset, output, record):
    item_dir = dataset / record["item_relative_path"]
    report = json.loads((item_dir / "report.json").read_text(encoding="utf-8"))
    with np.load(item_dir / "pair.npz", allow_pickle=False) as shard:
        data = {key: shard[key].copy() for key in shard.files}
    joints = data.get("posed_joints")
    if joints is None or joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("pair.npz must contain posed_joints [T,J,3]")
    count = len(joints)
    fps = float(np.asarray(data.get("fps", 60.0)).reshape(()))
    confidence = {side: data[f"support_confidence_{side}"] for side in SIDES}
    root_positions = data["root_positions"]
    if root_positions.shape != (count, 3):
        raise ValueError("pair.npz root_positions must be [T,3]")
    start, stop = _segment(root_positions, fps, count)
    frames = _sample_frames(start, stop, fps)
    parents = report.get("skeleton_parents")
    if parents is None or len(parents) != joints.shape[1]:
        raise ValueError("report.json skeleton_parents do not match posed_joints")
    edges = _edges(parents)

    fig = plt.figure(figsize=(11, 5.2))
    ax_body = fig.add_subplot(1, 2, 1, projection="3d")
    ax_support = fig.add_subplot(1, 2, 2, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.80, wspace=0.08)
    for axis in (ax_body, ax_support):
        axis.view_init(elev=22, azim=-58)
        axis.set_xlabel("X (m)", labelpad=-1)
        axis.set_ylabel("Z (m)", labelpad=-1)
        axis.set_zlabel("Y (m)", labelpad=-1)
        axis.tick_params(labelsize=7, pad=0)
    body_lines = Line3DCollection(np.zeros((len(edges), 2, 3)), colors="#34495e", linewidths=1.7)
    ax_body.add_collection3d(body_lines)
    ax_body.set_title("Accepted SOMA motion", fontsize=10)
    ax_support.set_title("Official-LBS sole regions and weak patches", fontsize=10)

    segment_sole_heights = np.concatenate([
        data[f"sole_vertices_{side}"][start:stop, :, 1].reshape(-1) for side in SIDES])
    for axis in (ax_body, ax_support):
        axis.set_box_aspect((1, 1, 0.8))
    plane_y = report.get("recording_plane_y_m")
    if plane_y is None:
        plane_y = report.get("support", {}).get("summary", {}).get("recording_plane_y_m")
    plane_y = float(plane_y) if plane_y is not None and np.isfinite(plane_y) else float(np.median(segment_sole_heights))
    y_low = min(float(segment_sole_heights.min()), plane_y)
    y_high = max(float(segment_sole_heights.max()), plane_y)
    y_pad = max(0.015, (y_high - y_low) * 0.06)
    fixed_support_y_limits = (y_low - y_pad, y_high + y_pad)
    ax_support.set_zlim(*fixed_support_y_limits)

    sole_artists = []
    for side_index, side in enumerate(SIDES):
        vertices = data[f"sole_vertices_{side}"]
        patch_ids = data[f"patch_id_{side}"]
        scatter = ax_support.scatter([], [], [], s=3, alpha=0.7, c="#a8b0b8")
        sole_artists.append((scatter, vertices, patch_ids, side_index))
    patch_artists = []
    support = report.get("support", {})
    events = support.get("events", [])
    active_events = {event["id"]: event for event in events
                     if int(event["start_frame"]) < stop and int(event["end_frame_exclusive"]) > start}
    patches = support.get("patches", [])
    patches = [patch for patch in patches if patch.get("status") != "rejected"]
    visible_patches = []
    for patch in patches:
        polygon = np.asarray(patch.get("polygon_xz_m", []), dtype=np.float64)
        event_ids = set(patch.get("event_ids", [])) & active_events.keys()
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2 or not event_ids:
            continue
        near_segment = False
        for event_id in event_ids:
            event = active_events[event_id]
            first = max(start, int(event["start_frame"]))
            last = min(stop, int(event["end_frame_exclusive"]))
            centers = root_positions[first:last][:, [0, 2]]
            if not len(centers):
                continue
            near_segment = np.any(
                (centers[:, 0] >= polygon[:, 0].min() - 1.7)
                & (centers[:, 0] <= polygon[:, 0].max() + 1.7)
                & (centers[:, 1] >= polygon[:, 1].min() - 1.7)
                & (centers[:, 1] <= polygon[:, 1].max() + 1.7))
            if near_segment:
                break
        if near_segment:
            visible_patches.append(patch)
    patches = visible_patches
    patches.sort(key=lambda patch: float(patch.get("confidence", 0.0)), reverse=True)
    plane_outlines = []
    for patch in patches[:24]:
        polygon = np.asarray(patch.get("polygon_xz_m", []), dtype=np.float64)
        height = float(patch.get("height_m", np.nan))
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2 or not np.isfinite(height):
            continue
        surface_xyz = np.column_stack((polygon[:, 0], polygon[:, 1], np.full(len(polygon), height)))
        plane_xyz = np.column_stack((polygon[:, 0], polygon[:, 1], np.full(len(polygon), plane_y)))
        surface = Poly3DCollection([surface_xyz], alpha=0.28, facecolor="#e69f00",
                                   edgecolor="#9a6700", linewidth=0.8)
        ax_support.add_collection3d(surface)
        patch_artists.append(surface)
        closed = np.vstack((plane_xyz, plane_xyz[:1]))
        segments = np.stack((closed[:-1], closed[1:]), axis=1)
        for axis in (ax_body, ax_support):
            outline = Line3DCollection(segments, colors="#455a64", linewidths=1.0, linestyles="--")
            axis.add_collection3d(outline)
            plane_outlines.append(outline)

    pair_status = report.get("pair_status", "unknown")
    ax_body.set_title("Accepted SOMA motion", fontsize=10)
    ax_support.set_title(f"Weak support | pair: {pair_status} | plane {plane_y:.3f} m\n"
                         "confidence is evidence quality, not load",
                         fontsize=9)
    title = fig.suptitle("", fontsize=9, y=0.98)

    def update(frame_index):
        frame = int(frames[frame_index])
        points = joints[frame][:, [0, 2, 1]]
        body_lines.set_segments(points[edges])
        for scatter, vertices, patch_ids, side_index in sole_artists:
            pts = vertices[frame][:, [0, 2, 1]]
            scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            color = "#d97706" if side_index == 0 else "#0f766e"
            scatter.set_color([color if int(patch_ids[frame]) >= 0 else "#a8b0b8"] * len(pts))
        root = root_positions[frame]
        ax_body.set_xlim(root[0] - 1.2, root[0] + 1.2)
        ax_body.set_ylim(root[2] - 1.2, root[2] + 1.2)
        ax_body.set_zlim(root[1] - 1.05, root[1] + 1.05)
        ax_support.set_xlim(root[0] - 1.7, root[0] + 1.7)
        ax_support.set_ylim(root[2] - 1.7, root[2] + 1.7)
        title.set_text(f"{record['filename']} | {record['split']} {record['behavior']} | "
                       f"{frame / fps:.2f}s | confidence L {confidence['left'][frame]:.2f}, "
                       f"R {confidence['right'][frame]:.2f}")
        return [body_lines, title, *(row[0] for row in sole_artists), *patch_artists, *plane_outlines]

    gif_name = f"{record['filename']}_{record['split']}.gif"
    update(0)
    fig.canvas.draw()
    canvas_renderer = fig.canvas.get_renderer()
    title_bounds = title.get_window_extent(canvas_renderer)
    if any(title_bounds.overlaps(axis.title.get_window_extent(canvas_renderer)) for axis in (ax_body, ax_support)):
        raise ValueError("Review heading overlaps a panel title")
    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / 15, blit=False)
    animation.save(output / gif_name, writer=PillowWriter(fps=15))
    plt.close(fig)

    plot_name = f"{record['filename']}_{record['split']}_confidence.png"
    time_axis = np.arange(start, stop) / fps
    fig, axis = plt.subplots(figsize=(8, 2.8), constrained_layout=True)
    axis.plot(time_axis - start / fps, confidence["left"][start:stop], color="#d97706", label="Left")
    axis.plot(time_axis - start / fps, confidence["right"][start:stop], color="#0f766e", label="Right")
    axis.set(xlim=(0, max((stop - start - 1) / fps, 0.01)), ylim=(0, 0.85),
             xlabel="Segment time (s)", ylabel="Weak confidence", title=f"{record['filename']} | selected window")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper right", frameon=False)
    fig.savefig(output / plot_name, dpi=130)
    plt.close(fig)
    return {
        "filename": record["filename"], "split": record["split"], "behavior": record["behavior"],
        "recording_group": record.get("recording_group"),
        "source_shard": record["source_shard"], "source_shard_sha256": record["source_shard_sha256"],
        "frame_start": start, "frame_end_exclusive": stop, "fps": fps,
        "gif": gif_name, "confidence_plot": plot_name,
        "gif_frames": len(frames), "segment_seconds": (stop - start) / fps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Built weak-support sidecar directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        parser.error(f"Refusing to overwrite non-empty render directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected = _select_rows(manifest.get("records", []))
    renders = [render_item(dataset, output, record) for record in selected]
    result = {
        "schema_version": manifest.get("schema_version"),
        "selection": [list(value) for value in REVIEW_SELECTION],
        "max_segment_seconds": CLIP_SECONDS,
        "segment_selection_policy": "first moving six-second segment by 0.5s root horizontal speed >= 0.18m/s; otherwise clip start",
        "max_gif_frames": MAX_GIF_FRAMES,
        "gif_fps": 15,
        "renders": renders,
        "scope": "generic motion-derived planar weak support review; not physical contact or load truth",
    }
    (output / "render_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"rendered": len(renders), "files": [row["gif"] for row in renders]}, sort_keys=True))


if __name__ == "__main__":
    main()
