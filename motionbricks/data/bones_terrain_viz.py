"""Review images and viewer data for fitted BONES terrain.

Matplotlib (Agg) renders a contact sheet (3D view plus a side elevation) and an
optional GIF; viewer_payload() exports a compact JSON for the browser viewer in
web/terrain_fit_review. Input motion is never changed; frames are only sampled.
"""

from __future__ import annotations

import numpy as np

#: Stick-figure segments over SOMA77 joint names; missing names are skipped.
SEGMENTS = (
    ("Hips", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Chest"), ("Chest", "Neck1"), ("Neck1", "Head"),
    ("Chest", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"), ("LeftHand", "LeftHandMiddleEnd"),
    ("Chest", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"), ("RightHand", "RightHandMiddleEnd"),
    ("Hips", "LeftLeg"), ("LeftLeg", "LeftShin"), ("LeftShin", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "RightLeg"), ("RightLeg", "RightShin"), ("RightShin", "RightFoot"), ("RightFoot", "RightToeBase"),
)
VIEW_JOINTS = tuple(dict.fromkeys(name for pair in SEGMENTS for name in pair))
COLORS = {"ground": "#9aa0a6", "support": "#e8710a", "maximal": "#5f6368", "body": "#1a73e8",
          "trail": "#188038", "evidence": "#d93025"}


def box_corners(box):
    center = np.asarray(box["center_xyz_m"], dtype=np.float64)
    half = np.asarray(box["half_extent_xyz_m"], dtype=np.float64)
    rotation = np.asarray(box["rotation_local_to_world"], dtype=np.float64)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    return (signs * half) @ rotation.T + center


_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def _plot_xyz(points):
    """World (x, y-up, z) -> matplotlib (x, z, y) so the vertical axis is up."""
    points = np.asarray(points)
    return points[..., 0], points[..., 2], points[..., 1]


def _draw_boxes(ax, boxes, maximal=None, ground_extent=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for box in boxes:
        corners = box_corners(box)
        if box["role"] == "ground":
            if ground_extent is not None:
                (x0, z0), (x1, z1) = ground_extent
                top = box["top_m"]
                quad = np.array([[x0, top, z0], [x1, top, z0], [x1, top, z1], [x0, top, z1]])
                xs, ys, zs = _plot_xyz(quad)
                ax.add_collection3d(Poly3DCollection([list(zip(xs, ys, zs))], facecolor=COLORS["ground"],
                                                     alpha=0.15, edgecolor="none"))
            continue
        faces = [[tuple(_plot_xyz(corners[i])) for i in face] for face in _FACES]
        ax.add_collection3d(Poly3DCollection(faces, facecolor=COLORS["support"], alpha=0.35,
                                             edgecolor=COLORS["support"], linewidth=0.6))
    for box in maximal or ():
        if box["role"] == "ground":
            continue
        corners = box_corners(box)
        for a, b in _EDGES:
            xs, ys, zs = _plot_xyz(corners[[a, b]])
            ax.plot(xs, ys, zs, color=COLORS["maximal"], linewidth=0.6, linestyle="--")


def _draw_body(ax, joints, frame, alpha, color=COLORS["body"]):
    for a, b in SEGMENTS:
        if a in joints and b in joints:
            xs, ys, zs = _plot_xyz(np.stack([joints[a][frame], joints[b][frame]]))
            ax.plot(xs, ys, zs, color=color, linewidth=1.4, alpha=alpha)


def _set_equal(ax, points):
    points = np.asarray(points).reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    center, radius = (low + high) / 2, max((high - low).max() / 2, 0.5)
    xs, ys, zs = _plot_xyz(np.stack([center - radius, center + radius]))
    ax.set_xlim(*sorted(xs)), ax.set_ylim(*sorted(ys)), ax.set_zlim(*sorted(zs))
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x (m)"), ax.set_ylabel("z (m)"), ax.set_zlabel("y up (m)")


def keyframes(frames, count):
    return np.unique(np.linspace(0, frames - 1, max(2, count)).round().astype(int))


def render_contact_sheet(joints, fit, path, *, title="", count=7, variant="minimal", azim=-60, elev=18):
    """3D overlay of keyframes with fitted boxes, plus a side elevation along travel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    joints = {name: np.asarray(value, dtype=np.float64) for name, value in joints.items()}
    frames = len(next(iter(joints.values())))
    picks = keyframes(frames, count)
    boxes, maximal = fit["boxes"][variant], fit["boxes"]["maximal"]
    everything = np.concatenate([np.stack(list(joints.values()), axis=1).reshape(-1, 3)] +
                                [box_corners(b) for b in boxes if b["role"] != "ground"] +
                                [box_corners(b) for b in maximal if b["role"] != "ground"])
    figure = plt.figure(figsize=(12, 5.2), dpi=110)
    ax = figure.add_subplot(1, 2, 1, projection="3d")
    xz = everything[:, [0, 2]]
    _draw_boxes(ax, boxes, maximal, ground_extent=(xz.min(axis=0) - 0.3, xz.max(axis=0) + 0.3))
    for rank, frame in enumerate(picks):
        _draw_body(ax, joints, frame, alpha=0.25 + 0.75 * rank / max(1, len(picks) - 1))
    if "Hips" in joints:
        xs, ys, _ = _plot_xyz(joints["Hips"])
        ax.plot(xs, ys, np.full(frames, fit["ground_m"]), color=COLORS["trail"], linewidth=1.0)
    _set_equal(ax, everything)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"3D: {variant} boxes solid, maximal dashed", fontsize=9)

    # Side elevation: horizontal axis = principal travel direction of the hips.
    side = figure.add_subplot(1, 2, 2)
    hips = joints.get("Hips", np.stack(list(joints.values()), axis=1).mean(axis=1))
    travel = hips[:, [0, 2]] - hips[0, [0, 2]]
    direction = travel[-1] if np.linalg.norm(travel[-1]) > 0.2 else np.array([0.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    origin = hips[0, [0, 2]]

    def along(points):
        points = np.asarray(points)
        return (points[..., [0, 2]] - origin) @ direction

    for box in boxes + [b for b in maximal if b["role"] != "ground"]:
        corners = box_corners(box)
        s = along(corners)
        top = box["top_m"]
        bottom = top if box["role"] == "ground" else corners[:, 1].min()
        dashed = box in maximal and box not in boxes
        if box["role"] == "ground":
            side.axhline(top, color=COLORS["ground"], linewidth=1.0)
        else:
            side.add_patch(plt.Rectangle((s.min(), bottom), np.ptp(s), top - bottom,
                                         fill=not dashed, alpha=0.3 if not dashed else 1.0,
                                         facecolor=COLORS["support"], edgecolor=COLORS["maximal"] if dashed else COLORS["support"],
                                         linestyle="--" if dashed else "-", linewidth=0.8))
    for rank, frame in enumerate(picks):
        alpha = 0.25 + 0.75 * rank / max(1, len(picks) - 1)
        for a, b in SEGMENTS:
            if a in joints and b in joints:
                pair = np.stack([joints[a][frame], joints[b][frame]])
                side.plot(along(pair), pair[:, 1], color=COLORS["body"], alpha=alpha, linewidth=1.2)
    for level in fit["levels_m"]:
        side.axhline(fit["ground_m"] + level, color=COLORS["evidence"], linewidth=0.5, linestyle=":")
    side.set_aspect("equal", adjustable="datalim")
    side.set_xlabel("along travel (m)"), side.set_ylabel("height (m)")
    side.set_title("side elevation; dotted = support levels", fontsize=9)
    metrics = fit["metrics"][variant]
    if "skin" in metrics:
        source, peak, over = "skin", metrics["skin"]["max_m"], metrics["skin"]["frames_over_report"]
    else:
        source, peak, over = "landmarks", metrics["penetration_max_m"], metrics["penetration_frames_over_report"]
    figure.suptitle(f"{title}\nlevels {[round(v, 3) for v in fit['levels_m']]} m | "
                    f"{source} penetration max {peak:.3f} m, frames>2cm {over} | "
                    f"flags {fit['flags']}", fontsize=9)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def render_gif(joints, fit, path, *, fps_in=30.0, fps_out=10.0, variant="minimal", azim=-60, elev=18, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    joints = {name: np.asarray(value, dtype=np.float64) for name, value in joints.items()}
    frames = len(next(iter(joints.values())))
    stride = max(1, int(round(fps_in / fps_out)))
    picks = np.arange(0, frames, stride)
    boxes, maximal = fit["boxes"][variant], fit["boxes"]["maximal"]
    everything = np.concatenate([np.stack(list(joints.values()), axis=1).reshape(-1, 3)] +
                                [box_corners(b) for b in boxes if b["role"] != "ground"])
    figure = plt.figure(figsize=(6, 5), dpi=90)
    ax = figure.add_subplot(projection="3d")
    xz = everything[:, [0, 2]]

    def draw(index):
        ax.cla()
        _draw_boxes(ax, boxes, maximal, ground_extent=(xz.min(axis=0) - 0.3, xz.max(axis=0) + 0.3))
        _draw_body(ax, joints, picks[index], alpha=1.0)
        _set_equal(ax, everything)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{title}  t={picks[index] / fps_in:.2f}s", fontsize=8)

    animation = FuncAnimation(figure, draw, frames=len(picks))
    animation.save(path, writer=PillowWriter(fps=fps_out))
    plt.close(figure)


def viewer_payload(joints, fit, *, fps_in=30.0, fps_out=15.0, meta=None):
    """Compact JSON for web/terrain_fit_review: millimetre joints, both box variants."""
    names = [name for name in VIEW_JOINTS if name in joints]
    stride = max(1, int(round(fps_in / fps_out)))
    stacked = np.stack([np.asarray(joints[name]) for name in names], axis=1)[::stride]
    return {
        "schema_version": "bones-terrain-view-v1", "fps": fps_in / stride, "joint_names": names,
        "segments": [[names.index(a), names.index(b)] for a, b in SEGMENTS if a in names and b in names],
        "frames_mm": np.round(stacked * 1000).astype(int).tolist(),
        "boxes": fit["boxes"], "ground_m": fit["ground_m"], "levels_m": fit["levels_m"],
        "metrics": fit["metrics"], "flags": fit["flags"], "meta": meta or {},
    }
