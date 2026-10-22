"""Complete raw-SOMA rollout videos plus the worst official-skinned surface excerpt."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import torch

from motionbricks.data.bones_seed import build_representation
from motionbricks.data.soma_skin_reference import load_official_skinner, skinned_vertices


def pose_goal_trial_inputs(folder, output, rep):
    """Adapt completed raw trial files; no output motion is changed."""
    from scripts.diagnose_soma_continuity import command
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
    from motionbricks.data.sole_region_contact import sole_region_indices
    if output.exists():
        raise ValueError("Use a new pose-goal render folder")
    output.mkdir(parents=True)
    skinner = load_official_skinner(rep.skeleton)
    surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
    indices = np.concatenate([sole_region_indices(surfaces, side) for side in ("Left", "Right")])
    entries = []
    for name, sub in (("Original", "review_original"), ("No goal A", "review_A_v3"),
                      ("Train pose goal B", "review_B_library")):
        with np.load(folder / sub / "delay_0.npz", allow_pickle=False) as data:
            features = torch.from_numpy(data["global_features"].copy())
        with torch.inference_mode():
            pose = rep.inverse(features, False, return_all=False)
            height, horizontal = region_geometry_from_motion(pose["local_joint_rots"],
                                                              pose["posed_joints"][:, 0], skinner, indices)
        vertices = torch.stack((horizontal[..., 0], height, horizontal[..., 1]), -1).numpy()
        peak = float(np.maximum(-vertices[..., 1], 0).max())
        entries.append({"seed": name, "sole_penetration_m": {"peak": peak}})
        np.savez_compressed(output / f"rollout_{name}.npz", joints=pose["posed_joints"].numpy(),
            local_rot_mats=pose["local_joint_rots"].numpy(), global_features=features.numpy(),
            command_velocity_xz=np.asarray([command(frame)[0] for frame in range(len(features))]),
            sole_vertices=vertices)
    return {"rollouts": entries}


def decoder_trial_inputs(folder, output, rep):
    """Raw 60-second comparison; arm names are labels, not random seeds."""
    return native_trial_inputs([(arm, folder / ("review_" + arm) / "long_0.npz")
                                for arm in ("original", "A", "B", "C")], output, rep)


def joint_worst_angular_event(folder, arms=("Original", "A", "B")):
    candidates = []
    for arm in arms:
        metrics = json.loads((folder / ("review_" + arm) / "metrics.json").read_text())
        for history in range(3):
            event = metrics["rollouts"][str(history)]["worst_angular_step"]
            candidates.append({"arm": arm, "history": history, **event})
    return max(candidates, key=lambda event: event["deg"])


def joint_trial_inputs(folder, output, rep, worst_angular=False):
    summary = json.loads((folder / "summary.json").read_text())
    if not summary.get("matched_scoring_verified") or summary["post_processing"] is not None:
        raise ValueError("A verified raw joint trial is required")
    event = joint_worst_angular_event(folder) if worst_angular else None
    history = event["history"] if event else 0
    frame_range = None
    if event:
        with np.load(folder / "review_Original" / f"long_{history}.npz", allow_pickle=False) as data:
            count = len(data["global_features"])
            if int(data["fps"]) != 60 or not 0 < event["destination_frame"] < count:
                raise ValueError("Angular event must identify an adjacent-frame step on the 60-Hz clock")
        start = max(0, min(event["destination_frame"] - 60, count - 120))
        frame_range = (start, min(start + 120, count))
    report = native_trial_inputs([(arm, folder / ("review_" + arm) / f"long_{history}.npz")
                                 for arm in ("Original", "A", "B")], output, rep, frame_range)
    report["history"] = history
    report["angular_event"] = event
    report["source_frame_range"] = frame_range
    from motionbricks.data.soma_free_geometry import observed_sole_geometry
    geometry = {}
    for rollout in report["rollouts"]:
        arm = rollout["seed"]
        with np.load(output / f"rollout_{arm}.npz", allow_pickle=False) as data:
            vertices = torch.from_numpy(data["sole_vertices"].copy())
        left = report["region_vertex_counts"]["Left"]
        geometry[arm] = {side: observed_sole_geometry(region, 60) for side, region in
                         (("Left", vertices[:, :left]), ("Right", vertices[:, left:]))}
    (output / "free_geometry.json").write_text(json.dumps({"arms": geometry,
        "scope": f"initial history {history}; source frames {frame_range or [0, 3600]}; not support truth",
        "post_processing": None, "quality_approved": False}, indent=2, allow_nan=False))
    return report


def planner_trial_inputs(folder, output, rep):
    report = json.loads((folder / "metrics.json").read_text())
    if not report.get("complete") or report["post_processing"] is not None or report["fps"] != 60:
        raise ValueError("A completed raw replanning comparison is required")
    return native_trial_inputs([(str(interval), folder / f"interval_{interval}_history_0.npz")
                                for interval in report["intervals"]], output, rep)


def supervision_trial_inputs(folder, output, rep, worst_angular=False):
    summary = json.loads((folder / "summary.json").read_text())
    if not summary.get("verified_matched") or summary["post_processing"] is not None:
        raise ValueError("A matched raw supervision trial is required")
    arms = tuple(summary["metrics"])
    event = joint_worst_angular_event(folder, arms) if worst_angular else None
    history = event["history"] if event else 0
    frame_range = None
    if event:
        with np.load(folder / "review_original" / f"long_{history}.npz", allow_pickle=False) as data:
            count = len(data["global_features"])
        start = max(0, min(event["destination_frame"] - 60, count - 120))
        frame_range = (start, start + 120)
    report = native_trial_inputs([(arm, folder / f"review_{arm}" / f"long_{history}.npz")
                                 for arm in arms], output, rep, frame_range)
    report.update(history=history, angular_event=event, source_frame_range=frame_range)
    return report


def native_trial_inputs(entries, output, rep, frame_range=None):
    from scripts.evaluate_soma_decoder_refinement import command_at
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
    from motionbricks.data.sole_region_contact import sole_region_indices
    if output.exists():
        raise ValueError("Use a fresh decoder render directory")
    output.mkdir(parents=True)
    skinner = load_official_skinner(rep.skeleton)
    surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
    regions = {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}
    indices = np.concatenate(list(regions.values()))
    rollouts = []
    for arm, path in entries:
        with np.load(path, allow_pickle=False) as data:
            raw = data["global_features"]
            start, stop = frame_range or (0, len(raw))
            if not 0 <= start < stop <= len(raw):
                raise ValueError("Requested source frames must exist without padding")
            features = torch.from_numpy(raw[start:stop].copy())
        with torch.inference_mode():
            pose = rep.inverse(features, False, return_all=False)
            heights, horizontal = [], []
            for batch_start in range(0, len(features), 128):
                height, positions = region_geometry_from_motion(pose["local_joint_rots"][batch_start:batch_start + 128],
                    pose["posed_joints"][batch_start:batch_start + 128, 0], skinner, indices)
                heights.append(height)
                horizontal.append(positions)
        height, positions = torch.cat(heights), torch.cat(horizontal)
        vertices = torch.stack((positions[..., 0], height, positions[..., 1]), -1).numpy()
        rollouts.append({"seed": arm, "label": arm, "sole_penetration_m": {"peak": float(np.maximum(-vertices[..., 1], 0).max())}})
        np.savez_compressed(output / f"rollout_{arm}.npz", joints=pose["posed_joints"].numpy(),
            local_rot_mats=pose["local_joint_rots"].numpy(), global_features=features.numpy(),
            command_velocity_xz=np.asarray([command_at(frame)[1] for frame in range(start, stop)]),
            sole_vertices=vertices)
    return {"rollouts": rollouts, "labels_are_arms": True,
            "region_vertex_counts": {side: len(indices) for side, indices in regions.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--pose-goal-trial", type=Path,
                        help="Convert completed Original/A/B raw trials into this fresh --review folder")
    parser.add_argument("--decoder-trial", type=Path)
    parser.add_argument("--planner-trial", type=Path)
    parser.add_argument("--joint-trial", type=Path)
    parser.add_argument("--supervision-trial", type=Path)
    parser.add_argument("--joint-worst-angular", action="store_true",
                        help="Render a matched two-second excerpt around the worst step across all three histories")
    parser.add_argument("--skip-surface-video", action="store_true",
                        help="Render full native-rate skeleton comparison; keep surface media in the overview")
    args = parser.parse_args()
    torch.set_num_threads(4)
    rep = build_representation()
    if sum(value is not None for value in (args.pose_goal_trial, args.decoder_trial, args.planner_trial, args.joint_trial, args.supervision_trial)) > 1:
        raise ValueError("Select one trial")
    if args.joint_worst_angular and (not (args.joint_trial or args.supervision_trial) or args.stride != 1):
        raise ValueError("Angular excerpts require a joint/supervision trial and --stride 1 to preserve adjacent frames")
    report = supervision_trial_inputs(args.supervision_trial, args.review, rep, args.joint_worst_angular) if args.supervision_trial else \
        joint_trial_inputs(args.joint_trial, args.review, rep, args.joint_worst_angular) if args.joint_trial else \
        planner_trial_inputs(args.planner_trial, args.review, rep) if args.planner_trial else \
        decoder_trial_inputs(args.decoder_trial, args.review, rep) if args.decoder_trial else \
        pose_goal_trial_inputs(args.pose_goal_trial, args.review, rep) if args.pose_goal_trial else \
        json.loads((args.review / "metrics.json").read_text())
    prefix = "arm" if report.get("labels_are_arms") or args.pose_goal_trial else "seed"
    parents = rep.skeleton.joint_parents.tolist()
    edges = np.array([(parent, child) for child, parent in enumerate(parents) if parent >= 0])
    arrays = []
    for rollout in report["rollouts"]:
        with np.load(args.review / f"rollout_{rollout['seed']}.npz", allow_pickle=False) as data:
            arrays.append({name: data[name].copy() for name in
                           ("joints", "local_rot_mats", "global_features", "command_velocity_xz", "sole_vertices")})
    if not arrays or len({len(item["joints"]) for item in arrays}) != 1:
        raise ValueError("Need completed rollouts sharing a frame count")
    count = len(arrays[0]["joints"])
    source_start = (report.get("source_frame_range") or (0, count))[0]
    figure = plt.figure(figsize=(4.2 * len(arrays), 5))
    artists = []
    for column, (rollout, data) in enumerate(zip(report["rollouts"], arrays)):
        axis = figure.add_subplot(1, len(arrays), column + 1, projection="3d")
        axis.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-0.2, 2.1))
        axis.set_box_aspect((1, 1, 1.15))
        axis.view_init(elev=16, azim=-50)
        axis.set_axis_off()
        for tick in np.arange(-1, 1.1, 0.25):
            axis.plot([-1, 1], [tick, tick], [0, 0], color="#cccccc", linewidth=0.6)
            axis.plot([tick, tick], [-1, 1], [0, 0], color="#cccccc", linewidth=0.6)
        lines = Line3DCollection(np.zeros((len(edges), 2, 3)), colors="#187fac", linewidths=2.2)
        axis.add_collection3d(lines)
        label = axis.set_title(str(rollout["seed"]), fontsize=9)
        artists.append((data, lines, label, rollout["seed"]))
    figure.suptitle("Executed-history + commands | raw model output | no IK, foot locking or root correction", fontsize=10)
    figure.tight_layout()

    def update(frame):
        for data, lines, label, seed in artists:
            origin = data["joints"][frame, 0].copy()
            origin[1] = 0
            points = data["joints"][frame] - origin
            lines.set_segments(points[:, [0, 2, 1]][edges])
            speed = np.linalg.norm(data["command_velocity_xz"][frame])
            actual = np.linalg.norm(data["global_features"][frame, [0, 2]] -
                                    data["global_features"][max(frame - 1, 0), [0, 2]]) * 60
            label.set_text(f"{prefix} {seed} | {(frame + source_start) / 60:.2f}s | intent {speed:.2f}, actual {actual:.2f} m/s\n"
                           f"root height {data['joints'][frame, 0, 1]:.2f} m")
        return [item[1] for item in artists]

    update(min(60, count - 1))
    figure.savefig(args.review / "rollouts.png", dpi=110)
    animation = FuncAnimation(figure, update, frames=range(0, count, args.stride), interval=args.stride / 60 * 1000)
    video = args.review / "rollouts_full.mp4"
    animation.save(video, writer=FFMpegWriter(fps=60 / args.stride, codec="libx264",
                                             extra_args=["-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(figure)
    if args.skip_surface_video:
        (args.review / "render_contract.json").write_text(json.dumps({
            "source_fps": 60, "render_fps": 60 / args.stride, "stride": args.stride,
            "frames": count, "complete_duration_seconds": count / 60, "post_processing": None,
            "camera": "XZ follows each root; world Y and ground stay fixed",
            "surface_video": "not rendered in this native skeleton comparison",
            "complete_video": str(video)}, indent=2))
        print(str(video), flush=True)
        return

    event = report.get("angular_event")
    worst = next(index for index, item in enumerate(report["rollouts"]) if item["seed"] == event["arm"]) if event else \
        max(range(len(arrays)), key=lambda index: report["rollouts"][index]["sole_penetration_m"]["peak"])
    data = arrays[worst]
    penetration = np.maximum(-data["sole_vertices"][..., 1], 0).max(-1)
    peak = event["destination_frame"] - source_start if event else int(penetration.argmax())
    start = max(0, min(peak - 180, count - 360))
    stop = min(start + 360, count)
    selected = np.arange(start, stop, args.stride)
    skinner = load_official_skinner(rep.skeleton)
    with torch.inference_mode():
        vertices = skinned_vertices(skinner, torch.from_numpy(data["local_rot_mats"][selected]),
                                    torch.from_numpy(data["joints"][selected])).numpy()
    faces = skinner.faces.numpy()
    figure = plt.figure(figsize=(6, 5))
    axis = figure.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(np.zeros((1, 3, 3)), linewidth=0, edgecolors="none")
    axis.add_collection3d(mesh)
    axis.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(-0.2, 2.1))
    axis.set_box_aspect((1, 1, 1.15))
    axis.view_init(elev=12, azim=-50)
    axis.set_axis_off()
    for tick in np.arange(-1, 1.1, 0.25):
        axis.plot([-1, 1], [tick, tick], [0, 0], color="#bbbbbb", linewidth=0.6)
        axis.plot([tick, tick], [-1, 1], [0, 0], color="#bbbbbb", linewidth=0.6)
    title = axis.set_title("Worst raw surface excerpt", fontsize=10)
    light = np.array([0.4, -0.7, 1.0])
    light /= np.linalg.norm(light)

    def update_mesh(index):
        frame = selected[index]
        origin = data["joints"][frame, 0].copy()
        origin[1] = 0
        triangles = (vertices[index] - origin)[:, [0, 2, 1]][faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-10)
        brightness = 0.45 + 0.55 * np.maximum(normals @ light, 0)
        mesh.set_verts(triangles)
        mesh.set_facecolor(brightness[:, None] * np.array([0.23, 0.58, 0.70]))
        detail = f"{event['joint']} step {event['deg']:.1f} deg at source frame {event['destination_frame']}" if event else \
            f"Raw output; penetration {penetration[frame] * 1000:.1f} mm"
        title.set_text(f"Official SOMA skin | worst {prefix} {report['rollouts'][worst]['seed']} | {(frame + source_start) / 60:.2f}s\n"
                       + detail)
        return [mesh]

    update_mesh(int(np.flatnonzero(selected >= peak)[0]) if (selected >= peak).any() else len(selected) - 1)
    figure.savefig(args.review / "worst_skinned.png", dpi=110)
    if event:
        for name, frame in (("before_flip", peak - 1), ("after_flip", peak)):
            update_mesh(int(np.flatnonzero(selected == frame)[0]))
            figure.savefig(args.review / f"{name}.png", dpi=110)
    animation = FuncAnimation(figure, update_mesh, frames=range(len(selected)), interval=args.stride / 60 * 1000)
    animation.save(args.review / "worst_skinned.mp4", writer=FFMpegWriter(
        fps=60 / args.stride, codec="libx264", extra_args=["-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]))
    plt.close(figure)
    (args.review / "render_contract.json").write_text(json.dumps({
        "source_fps": 60, "render_fps": 60 / args.stride, "stride": args.stride, "frames": count,
        "complete_duration_seconds": count / 60, "post_processing": None,
        "camera": "XZ follows each generated root; world Y and ground stay fixed; no output alignment",
        "worst_seed": report["rollouts"][worst]["seed"], "worst_frame": peak,
        "worst_excerpt_frames": [start, stop], "complete_video": str(video),
        "source_frame_range": report.get("source_frame_range"), "initial_history": report.get("history"),
        "angular_event": event,
    }, indent=2))
    print(str(video), flush=True)


if __name__ == "__main__":
    main()
