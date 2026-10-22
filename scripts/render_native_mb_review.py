"""Render saved held-out reconstructions without modifying any predicted motion."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


def _load_reconstruction(path):
    with np.load(path, allow_pickle=False) as data:
        source = np.asarray(data["source"])
        prediction = np.asarray(data["prediction"])
        fps = float(data["fps"])
    if (source.ndim != 3 or source.shape[-1] != 3 or len(source) == 0
            or source.shape != prediction.shape):
        raise ValueError(f"Invalid reconstruction arrays: {path}")
    if source.shape[1] != len(SOMASkeleton30().joint_parents):
        raise ValueError(f"Expected full-body SOMA joint arrays: {path}")
    if not np.isfinite(source).all() or not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite reconstruction arrays: {path}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid reconstruction FPS: {path}")
    return source, prediction, fps


def _render_motion_set(path, motions, fps, output_dir, title, three_panel=False,
                       panel_labels=None):
    parents = SOMASkeleton30().joint_parents.tolist()
    all_points = np.concatenate(motions, axis=0)
    lower, upper = all_points.min((0, 1)), all_points.max((0, 1))
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) / 2 + 0.1, 1.0)
    figure = plt.figure(figsize=(13, 5) if three_panel else (9, 5))
    lines = []
    panel_count = len(motions)
    colors = ("#555555", "#d17a00", "#087d95") if three_panel else ("#555555", "#087d95")
    for index, (motion, color) in enumerate(zip(motions, colors), 1):
        axis = figure.add_subplot(1, panel_count, index, projection="3d")
        z_limits = ((center[1] - radius, center[1] + radius) if three_panel else
                    (min(-0.1, lower[1] - 0.1), max(2.0, upper[1] + 0.1)))
        labels = panel_labels or ("Source motion", "Native MB reconstruction")
        axis.set(xlim=(center[0] - radius, center[0] + radius),
                 ylim=(center[2] - radius, center[2] + radius),
                 zlim=z_limits,
                 xlabel="X (m)", ylabel="Z (m)", zlabel="Y (m)",
                 title=labels[index - 1])
        axis.set_box_aspect((1, 1, 1))
        axis.tick_params(labelsize=8)
        axis.view_init(elev=15, azim=-55)
        axis.plot([center[0] - radius, center[0] + radius], [center[2]] * 2,
                  [0, 0], color="#aaaaaa", linewidth=0.7)
        for joint, parent in enumerate(parents):
            if parent >= 0:
                line, = axis.plot([], [], [], color=color, linewidth=2)
                lines.append((line, motion, joint, parent))
    figure.suptitle(title)
    figure.tight_layout(rect=(0.01, 0.02, 0.94, 0.84 if "\n" in title else 0.91))

    def update(frame):
        for line, motion, joint, parent in lines:
            pair = motion[frame, [parent, joint]]
            line.set_data_3d(pair[:, 0], pair[:, 2], pair[:, 1])
        return [entry[0] for entry in lines]

    update(min(24, len(motions[0]) - 1))
    image_path = output_dir / path.with_suffix(".png").name
    gif_path = output_dir / path.with_suffix(".gif").name
    figure.savefig(image_path, dpi=120)
    animation = FuncAnimation(figure, update, frames=range(len(motions[0])), interval=1000 / fps)
    animation.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return gif_path.name


def _render_reference_review(review, reference_review, output_dir, pilot_label, expanded_label):
    review, reference_review, output_dir = map(Path, (review, reference_review, output_dir))
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    current_root, reference_root = review.resolve(), reference_review.resolve()
    if output_dir.exists():
        raise ValueError(f"Reference review output directory already exists: {output_dir}")
    if output_dir == current_root or output_dir == reference_root:
        raise ValueError("Reference review output directory must be separate from both inputs")
    if current_root in output_dir.parents or reference_root in output_dir.parents:
        raise ValueError("Reference review output directory cannot be inside either input directory")

    current_metrics = json.loads((current_root / "metrics.json").read_text())
    reference_metrics = json.loads((reference_root / "metrics.json").read_text())
    current_paths = {p.name: p for p in current_root.glob("*_reconstruction.npz")}
    reference_paths = {p.name: p for p in reference_root.glob("*_reconstruction.npz")}
    if not current_paths or current_paths.keys() != reference_paths.keys():
        raise ValueError("Pilot and expanded reviews must contain the same behavior reconstructions")

    prepared = []
    for name in sorted(current_paths):
        behavior = name.removesuffix("_reconstruction.npz")
        try:
            current_identity = current_metrics["examples"][behavior]
            reference_identity = reference_metrics["examples"][behavior]
            current_key = tuple(current_identity[k] for k in ("shard", "start", "conditioning"))
            reference_key = tuple(reference_identity[k] for k in ("shard", "start", "conditioning"))
        except (KeyError, TypeError) as error:
            raise ValueError(f"Missing example identity for {behavior}") from error
        if current_key != reference_key:
            raise ValueError(f"Pilot and expanded example identity differs for {behavior}")

        current_source, expanded, current_fps = _load_reconstruction(current_paths[name])
        reference_source, pilot, reference_fps = _load_reconstruction(reference_paths[name])
        if current_source.shape != reference_source.shape or current_source.shape != expanded.shape or current_source.shape != pilot.shape:
            raise ValueError(f"Pilot and expanded shapes differ for {behavior}")
        if current_fps != reference_fps:
            raise ValueError(f"Pilot and expanded FPS differs for {behavior}")
        if not np.allclose(current_source, reference_source, atol=1e-5, rtol=0):
            raise ValueError(f"Source arrays differ beyond 1e-5 m for {behavior}")
        prepared.append((current_paths[name], current_source, pilot, expanded, current_fps))

    output_dir.mkdir(parents=True)
    outputs = []
    for path, source, pilot, expanded, fps in prepared:
        outputs.append(_render_motion_set(
            path, (source, pilot, expanded), fps, output_dir,
            f"{path.stem.removesuffix('_reconstruction')} | {pilot_label} / {expanded_label} | "
            "source-encoded reconstruction | no postprocessing",
            three_panel=True,
            panel_labels=("Source motion", f"{pilot_label} reconstruction",
                          f"{expanded_label} reconstruction"),
        ))
    return {"task": current_metrics["task"], "quality_approved": False, "gifs": outputs}


def _two_panel_caption(behavior, metrics):
    if metrics.get("stage") != "pose":
        return (behavior + " | source-encoded tokens, context4 | no postprocessing",
                ("Source motion", "Native MB reconstruction"))

    mode = metrics.get("render_mode") or metrics.get("examples", {}).get(
        behavior, {}).get("conditioning_mode")
    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("Pose review requires render_mode or example conditioning_mode")
    descriptions = {
        "context4_stress_probe": "Context-4 stress probe",
        "inbetween_endpoint_pose": "Endpoint-pose in-between",
        "native_keyframe1_inbetween": "Native keyframe-1 in-between",
    }
    description = descriptions.get(mode, mode.replace("_", " "))
    root_source = metrics.get("root_condition_source", "source")
    if root_source not in {"source", "root_model"}:
        raise ValueError(f"Unknown root condition source: {root_source}")
    root_caption = "model-predicted root" if root_source == "root_model" else "known source root"
    return (f"{behavior} | generated tokens + {root_caption}\n"
            f"{description} [{mode}]\nno WASD | no postprocessing",
            ("Source", "Generated pose"))


def render(review, reference_review=None, output_dir=None,
           pilot_label="Pilot", expanded_label="Expanded"):
    if reference_review is not None:
        if output_dir is None:
            raise ValueError("--output-dir is required for a reference review")
        return _render_reference_review(
            review, reference_review, output_dir, pilot_label, expanded_label
        )

    review = Path(review)
    metrics = json.loads((review / "metrics.json").read_text())
    outputs = []
    for path in sorted(review.glob("*_reconstruction.npz")):
        source, prediction, fps = _load_reconstruction(path)
        title, panel_labels = _two_panel_caption(path.stem.removesuffix("_reconstruction"), metrics)
        outputs.append(_render_motion_set(
            path, (source, prediction), fps, review,
            title, panel_labels=panel_labels,
        ))
    if not outputs:
        raise ValueError("No reconstruction windows to render")
    return {"task": metrics["task"], "quality_approved": False, "gifs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True,
                        help="Expanded/current validation review directory")
    parser.add_argument("--reference-review", type=Path,
                        help="Pilot review directory to compare against")
    parser.add_argument("--output-dir", type=Path,
                        help="New output directory for three-panel reference assets")
    parser.add_argument("--pilot-label", default="Pilot")
    parser.add_argument("--expanded-label", default="Expanded")
    args = parser.parse_args()
    print(json.dumps(render(args.review, args.reference_review, args.output_dir,
                            args.pilot_label, args.expanded_label), indent=2))
