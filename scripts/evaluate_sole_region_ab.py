"""Fixed-window comparison of the two tokenizer arms on held-out validation clips.

Source, A and B are scored with the same official skinning, the same source-derived
labels and the same scoring masks, and all three skip the same four context frames.
Nothing here re-picks frames from the predictions, applies IK or locks a foot.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from motionbricks.data.bones_models import BonesTokenizer
from motionbricks.data.bones_seed import build_representation, official_sole_surfaces, deterministic_window_starts
from motionbricks.data.bones_contract import dataset_snapshot, require_same_snapshot
from motionbricks.data.bones_evaluation import rotation_error_degrees
from motionbricks.data.sole_geometry_losses import frame_has_evidence, region_geometry_from_motion
from motionbricks.data.sole_region_contact import (FRAME_STATES, INTERVAL_STATES, SOLE_LABEL_THRESHOLDS,
                                                   material_point_slip, region_gap_m, sole_region_indices,
                                                   sole_region_heights, sole_region_labels)
from motionbricks.data.soma_skin_reference import load_official_skinner, skinned_vertices

RENDER_VERTICES = {}
RENDER_LIMITS = ((-1.0, 1.0), (0.0, 0.2))
RENDER_GAP_LIMITS = (-15.0, 80.0)
RENDER_SKINNER = None
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

CONTEXT_FRAMES = 4
WINDOWS = (("walk", "walk_ff_loop_360_R_very_slow_001__A443"),
           ("run", "jog_ff_start_270_R_fast_001__A451"),
           ("turn", "Turn_Start_Walk_0360_001__A020"),
           ("idle", "idle_right_R_003__A235"),
           ("backward", "walk_backward_stop_002__A030"))


def source_labels(vertices_m, indices, fps):
    """Labels and masks from the source motion; every arm is scored on these."""
    labels, geometry = {}, {}
    for side in ("Left", "Right"):
        key = side.lower()
        height = sole_region_heights(vertices_m, indices[side], 0.0)
        horizontal = vertices_m.index_select(
            1, torch.as_tensor(np.asarray(indices[side]), dtype=torch.long))[..., [0, 2]]
        gap = region_gap_m(height)
        slip = material_point_slip(height, horizontal, fps,
                                   contact_clearance_m=SOLE_LABEL_THRESHOLDS["contact_clearance_m"],
                                   min_run_frames=SOLE_LABEL_THRESHOLDS["min_contact_run_frames"],
                                   rest_vertical_speed_mps=SOLE_LABEL_THRESHOLDS["rest_vertical_speed_mps"])
        tagging = sole_region_labels(gap["region_gap_m"], slip,
                                     contact_clearance_m=SOLE_LABEL_THRESHOLDS["contact_clearance_m"],
                                     clearance_m=SOLE_LABEL_THRESHOLDS["clearance_m"],
                                     slip_speed_mps=SOLE_LABEL_THRESHOLDS["slip_speed_mps"])
        labels[key] = dict(tagging, vertex_measured=slip["vertex_measured"])
        geometry[key] = {"slip": slip, "gap": gap}
    return labels, geometry


def scoring_masks(frame_state, measured, frames):
    """Scoring range after the shared context skip.

    Frame labels cover all 128 frames; interval labels cover 127. An interval t spans
    frames t and t+1, so requiring both endpoints inside [CONTEXT_FRAMES, frames) means
    interval indices from CONTEXT_FRAMES to frames - 2. Everything inside the context
    window is excluded from every reported statistic, not only from MPJPE.
    """
    frame_scoring = torch.zeros(frames, dtype=torch.bool)
    frame_scoring[CONTEXT_FRAMES:] = True
    interval_scoring = torch.zeros(max(frames - 1, 0), dtype=torch.bool)
    interval_scoring[CONTEXT_FRAMES:] = True
    assert frame_state.shape[0] == frames
    assert measured.shape[0] == frames - 1
    return frame_scoring, interval_scoring


def motion_metrics(motion_rots, root_positions, skinner, indices, labels, fps):
    """Sole-region and reconstruction metrics for one motion under the shared masks."""
    out = {}
    penetration, contact_height, slip_mean, slip_peak = [], [], [], []
    measured_intervals = contact_frames = measured_vertices = 0
    for side in ("Left", "Right"):
        key = side.lower()
        height, horizontal = region_geometry_from_motion(motion_rots, root_positions, skinner, indices[side])
        lowest = height.amin(dim=-1)
        gap = region_gap_m(height)["region_gap_m"]
        tagging = labels[key]
        frame_state = tagging["frame_state"]
        stable = tagging["interval_state"] == INTERVAL_STATES.index("stable_contact")
        supervised = stable & tagging["measured_interval"]
        trusted = (frame_state == FRAME_STATES.index("contact")) & frame_has_evidence(tagging["measured_interval"])
        step = (horizontal[1:] - horizontal[:-1]).norm(dim=-1) * fps
        per_vertex = torch.where(tagging["vertex_measured"], step, torch.zeros_like(step))
        region_slip = per_vertex.amax(dim=-1)
        frame_scoring, interval_scoring = scoring_masks(frame_state, tagging["measured_interval"],
                                                        lowest.shape[0])
        trusted = trusted & frame_scoring
        supervised = supervised & interval_scoring
        penetration.append(torch.relu(-lowest)[frame_scoring])
        if int(trusted.sum()):
            contact_height.append(lowest[trusted].abs())
        if int(supervised.sum()):
            slip_mean.append(region_slip[supervised])
            slip_peak.append(float(region_slip[supervised].max()))
        measured_intervals += int(supervised.sum())
        contact_frames += int(trusted.sum())
        measured_vertices += int((tagging["vertex_measured"] & supervised[..., None]).sum())
    penetration = torch.cat(penetration)
    outcome = {
        "region_penetration_m": {"mean": float(penetration.mean()), "p95": float(torch.quantile(penetration, 0.95)),
                                 "peak": float(penetration.max())},
        "contact_height_abs_m": None if not contact_height else
        {"median": float(torch.cat(contact_height).median()),
         "p95": float(torch.quantile(torch.cat(contact_height), 0.95))},
        "measured_slip_mps": None if not slip_mean else
        {"mean": float(torch.cat(slip_mean).mean()), "p95": float(torch.quantile(torch.cat(slip_mean), 0.95)),
         "peak": max(slip_peak)},
        "valid_samples": {"measured_stable_intervals": measured_intervals,
                          "trusted_contact_frames": contact_frames,
                          "measured_vertices": measured_vertices},
    }
    return outcome


def render(folder, behavior, motions, indices, labels, fps, joints):
    """Source, A and B sole regions, region gaps, and the full body skeleton."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    if not motions or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Rendering requires motions and a finite positive FPS")
    frames = int(next(iter(motions.values()))[0].shape[0])
    if frames < 1 or any(value.shape[0] != frames for motion in motions.values() for value in motion):
        raise ValueError("Rendered motions must share a nonempty frame axis")
    colors = {"source": "#777777", "A0": "#25883c", "A": "#007fa3", "B": "#d1691a"}
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for number, arm in enumerate(motions):
        colors.setdefault(arm, palette[number % len(palette)])
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    time = np.arange(frames) / fps
    parents = joints["skeleton"].joint_parents.tolist()
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    gap = {}
    for arm, (rots, root, posed) in motions.items():
        gap[arm] = {}
        for side in ("Left", "Right"):
            height, _ = region_geometry_from_motion(rots, root, RENDER_SKINNER, indices[side])
            gap[arm][side] = height.amin(dim=-1).numpy() * 1000
    clearance = SOLE_LABEL_THRESHOLDS["contact_clearance_m"] * 1000
    band = SOLE_LABEL_THRESHOLDS["clearance_m"] * 1000
    highest = max(float(series.max()) for arm in gap for series in gap[arm].values())
    gap_limits = (-15.0, max(highest + 10.0, 40.0))
    all_joints = torch.cat([posed for _, _, posed in motions.values()])
    joint_limits = ((float(all_joints[..., 0].min()) - 0.2, float(all_joints[..., 0].max()) + 0.2),
                    (float(all_joints[..., 1].min()) - 0.1, float(all_joints[..., 1].max()) + 0.1))

    def draw(frame):
        for axis in axes:
            axis.clear()
        for column, side in enumerate(("Left", "Right")):
            axis = axes[column]
            for arm, (rots, root, posed) in motions.items():
                points = RENDER_VERTICES[arm][side][frame].numpy()
                axis.scatter(points[:, 0], points[:, 1], s=10, color=colors[arm], alpha=0.75, label=arm)
            axis.axhline(0.0, color="black", linewidth=1.3)
            axis.axhspan(0, SOLE_LABEL_THRESHOLDS["contact_clearance_m"], color="#cfe8cf", alpha=0.6)
            axis.set(xlim=RENDER_LIMITS[0], ylim=RENDER_LIMITS[1], xlabel="world X (m)", ylabel="world Y (m)")
            axis.set_title(f"{behavior} | {side} sole region | frame {frame}", fontsize=9)
            axis.legend(fontsize=7, loc="upper left")
        axis = axes[2]
        for arm in motions:
            axis.plot(time, gap[arm]["Left"], color=colors[arm], linewidth=1.0, linestyle="-")
            axis.plot(time, gap[arm]["Right"], color=colors[arm], linewidth=1.0, linestyle=":")
        axis.axhspan(RENDER_GAP_LIMITS[0], clearance, color="#cfe8cf", alpha=0.6)
        axis.axhspan(clearance, band, color="#f2e2c4", alpha=0.6)
        axis.axvline(time[frame], color="black", linewidth=0.8)
        axis.set(xlim=(0, time[-1]), ylim=gap_limits, xlabel="time (s)", ylabel="region gap (mm)")
        axis.set_title("solid: left foot, dotted: right foot", fontsize=9)
        axis = axes[3]
        for arm, (rots, root, posed) in motions.items():
            points = posed[frame].detach().cpu().numpy()
            labeled = False
            for joint, parent in enumerate(parents):
                if parent >= 0:
                    axis.plot(points[[parent, joint], 0], points[[parent, joint], 1],
                              color=colors[arm], linewidth=1.2, alpha=0.85,
                              label=arm if not labeled else None)
                    labeled = True
        axis.axhline(0.0, color="black", linewidth=1.3)
        axis.set(xlim=joint_limits[0], ylim=joint_limits[1], xlabel="world X (m)", ylabel="world Y (m)")
        axis.set_title("whole body skeleton", fontsize=9)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(fontsize=7, loc="upper left")
        figure.tight_layout()
        return []

    stem = f"{behavior}_" + "_".join(motions)
    gif = folder / f"{stem}.gif"
    try:
        # Preflight the first frame so a draw error is not masked by Pillow's empty-frame error.
        draw(0)
        animation = FuncAnimation(figure, draw, frames=range(frames), interval=1000 / 30)
        animation.save(gif, writer=PillowWriter(fps=30))
    finally:
        plt.close(figure)
    output = {"gif": gif.name, "source_frames": frames, "source_fps": fps,
              "gif_writer_fps": 30, "gif_playback": "slow motion, GIF timing is quantized"}
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mp4 = folder / f"{stem}.mp4"
        # Restore physical frame timestamps rather than inherit the slow GIF's timestamps.
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ignore_loop", "1", "-i", str(gif),
                        "-vf", f"setpts=N/({fps}*TB),fps={fps}", "-frames:v", str(frames),
                        "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(mp4)], check=True)
        output["mp4"] = mp4.name
    return output


def run_record(run):
    """Keep reported run evidence separate from measurements computed by this evaluator."""
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "pilot_result.json").read_text())
    checkpoint = run / "checkpoints/final.ckpt"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"run_directory": str(run), "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest.hexdigest(), "steps": result["steps"],
            "elapsed_seconds": result["elapsed_seconds"],
            "peak_gpu_memory_mb": result.get("peak_gpu_memory_mb"),
            "initial_weights_sha256": contract.get("initial_weights_sha256"),
            "seed": contract.get("seed"), "sampling_order": contract.get("sampling_order"),
            "train_clips": contract.get("train_clips"),
            "validation_windows": contract.get("validation_windows"),
            "parameters": contract.get("parameters"),
            "dataset_snapshot_sha256": contract["dataset_snapshot"]["sha256"],
            "config_sha256": contract["config_sha256"],
            "training_source_sha256": contract["training_source_sha256"],
            "loss_overrides": contract.get("loss_overrides"),
            "validation": result.get("validation"),
            "initial_state_verification": "recorded" if contract.get("initial_weights_sha256")
                                          else "not recorded in saved run contract"}


def build_model(run, representation):
    conf = OmegaConf.load(run / "upstream_config_overrides.yaml")
    network = instantiate(conf.model.pose_vqvae_network, motion_rep=representation.dual_rep.local_motion_rep)
    model = BonesTokenizer(pose_vqvae_network=network, root_vqvae_network=None, motion_rep=representation,
                           optimizer=None, scheduler=None,
                           args=OmegaConf.to_container(conf.model.args, resolve=True))
    checkpoint = torch.load(run / "checkpoints/final.ckpt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, network


def reconstruct(model, network, representation, features):
    with torch.inference_mode():
        motion = representation.normalize(features)[None]
        local = representation.dual_rep.global_to_local(motion, True, True,
                                                        lengths=torch.tensor([len(features)]))[:, :128]
        external = extract_feature_from_motion_rep(local, model.local_motion_rep,
                                                   network.decoder_external_cond_feature_mode)
        context = torch.zeros(local.shape[:2], dtype=torch.bool)
        context[:, :CONTEXT_FRAMES] = True
        state = network(local, target_cond=local * context[..., None], has_target_cond=context,
                        external_cond=external)["recon_state"]
        return model.local_motion_rep.inverse(state, True, return_all=True), local


def evaluation_windows(records, all_windows=False):
    if not all_windows:
        return [(behavior, filename, 0) for behavior, filename in WINDOWS]
    windows = []
    for filename, record in sorted(records.items()):
        if record["split"] != "validation" or record["frames_60hz"] < 129:
            continue
        for start in deterministic_window_starts(record["frames_60hz"], 128, 3):
            windows.append((record["behavior"], filename, int(start)))
    if not windows:
        raise ValueError("No held-out validation windows")
    return windows


def aggregate_region_results(entries):
    """Explicit window means and count-weighted slip, never averages labelled as pooled P95."""
    arms = next(iter(entries.values()))["arms"]
    result = {}
    for arm in arms:
        rows = [entry["arms"][arm] for entry in entries.values()]
        slip = [row for row in rows if row["measured_slip_mps"] is not None]
        count = sum(row["valid_samples"]["measured_stable_intervals"] for row in slip)
        result[arm] = {
            "windows": len(rows), "mpjpe_m": None if arm == "source" else float(np.mean([r["mpjpe_m"] for r in rows])),
            "penetration_mean_window_m": float(np.mean([r["region_penetration_m"]["mean"] for r in rows])),
            "penetration_peak_m": max(r["region_penetration_m"]["peak"] for r in rows),
            "measured_slip_count_weighted_mean_mps": None if not count else sum(
                r["measured_slip_mps"]["mean"] * r["valid_samples"]["measured_stable_intervals"] for r in slip) / count,
            "mean_window_slip_p95_mps": None if not slip else float(np.mean([r["measured_slip_mps"]["p95"] for r in slip])),
            "measured_slip_peak_mps": None if not slip else max(r["measured_slip_mps"]["peak"] for r in slip),
            "measured_stable_intervals": count,
        }
        for key in ("joint_acceleration_error_mps2", "local_rotation_error_deg", "fk_vs_position_feature_m"):
            values = [row[key] for row in rows if row.get(key) is not None]
            result[arm][key] = float(np.mean(values)) if values else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, help="Legacy argument; scoring labels are recomputed from source skin")
    parser.add_argument("--runs", required=True,
                        help="Comma-separated label=run_directory pairs, in reporting order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--all-windows", action="store_true", help="Score all three deterministic windows per validation clip")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    skeleton = build_representation().skeleton
    surfaces, _, _, _ = official_sole_surfaces(skeleton)
    indices = {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}
    skinner = load_official_skinner(skeleton)
    globals()["RENDER_SKINNER"] = skinner
    records = {record["filename"]: record for record in json.loads((args.dataset / "accepted.json").read_text())}
    report = json.loads((args.dataset / "quality_report.json").read_text())
    representation = build_representation(args.dataset / report.get("stats_folder", "stats"))
    snapshot = dataset_snapshot(args.dataset)
    arms = []
    run_records = {}
    for entry in args.runs.split(","):
        label, _, path = entry.partition("=")
        label = label.strip()
        if not label or label == "source" or label in run_records or not path:
            raise ValueError("Runs must have unique non-source labels and a run directory")
        model, network = build_model(Path(path), representation)
        contract = json.loads((Path(path) / "run_contract.json").read_text())
        require_same_snapshot(contract["dataset_snapshot"], snapshot)
        arms.append((label, model, network))
        run_records[label] = run_record(Path(path))
    first_model = arms[0][1]
    result = {"context_frames_skipped": CONTEXT_FRAMES, "fps": 60.0,
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scoring": {"frames": "[4,128)", "intervals": "[4,127), both endpoints scored"},
              "dataset_snapshot_sha256": snapshot["sha256"], "all_validation_windows": args.all_windows,
              "run_records": run_records, "behaviors": {}}
    render_jobs = []
    for behavior, filename, start in evaluation_windows(records, args.all_windows):
        record = records[filename]
        with np.load(args.dataset / record["shard"], allow_pickle=False) as shard:
            features = torch.from_numpy(shard["global_features"][start:start + 129].copy())
        root_index = skeleton.root_idx
        motion = representation.normalize(features)[None]
        local_source = representation.dual_rep.global_to_local(motion, True, True,
                                                               lengths=torch.tensor([len(features)]))[:, :128]
        source_pose = first_model.local_motion_rep.inverse(local_source, True, return_all=True)
        source_rots_local = source_pose["local_joint_rots"][0]
        posed_source = source_pose["posed_joints"][0]
        source_root = posed_source[:, root_index]
        source_vertices = skinned_vertices(skinner, source_rots_local, posed_source)
        labels, _ = source_labels(source_vertices, indices, 60.0)
        entry = {"filename": filename, "behavior": behavior, "start_frame": start, "arms": {}}
        predictions = {"source": (source_rots_local, source_root, posed_source)}
        entry["arms"]["source"] = motion_metrics(source_rots_local, source_root, skinner, indices, labels, 60.0)
        for name, model, network in arms:
            pred, _ = reconstruct(model, network, representation, features)
            rots = pred["local_joint_rots"][0][:128]
            root = pred["posed_joints"][0, :128, root_index]
            entry["arms"][name] = motion_metrics(rots, root, skinner, indices, labels, 60.0)
            entry["arms"][name]["mpjpe_m"] = float(
                (pred["posed_joints"][0, CONTEXT_FRAMES:128] - posed_source[CONTEXT_FRAMES:]).norm(dim=-1).mean())
            predicted_joints = pred["posed_joints"][0, CONTEXT_FRAMES:128]
            target_joints = posed_source[CONTEXT_FRAMES:]
            acceleration_error = ((predicted_joints[2:] - 2 * predicted_joints[1:-1] + predicted_joints[:-2])
                                  - (target_joints[2:] - 2 * target_joints[1:-1] + target_joints[:-2])) * 60.0 ** 2
            entry["arms"][name]["joint_acceleration_error_mps2"] = float(acceleration_error.norm(dim=-1).mean())
            entry["arms"][name]["local_rotation_error_deg"] = float(rotation_error_degrees(
                rots[CONTEXT_FRAMES:], source_rots_local[CONTEXT_FRAMES:]).mean())
            entry["arms"][name]["fk_vs_position_feature_m"] = float(
                (predicted_joints - pred["all_posed_joints"]["ric_data"][0, CONTEXT_FRAMES:]).norm(dim=-1).mean())
            predictions[name] = (rots, root, pred["posed_joints"][0, :128])
        key = f"{behavior}__{filename}__{start}" if args.all_windows else behavior
        result["behaviors"][key] = entry
        if args.render and not args.all_windows:
            render_jobs.append((behavior, predictions, labels))
        print(json.dumps({behavior: {arm: {k: v for k, v in data.items() if k != "valid_samples"}
                                     for arm, data in entry["arms"].items()}}, indent=1), flush=True)
    result["aggregate"] = aggregate_region_results(result["behaviors"])
    metrics_file = args.output / "sole_region_ab_metrics.json"
    metrics_file.write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.output / "sole_region_ab_summary.json").write_text(
        json.dumps(run_records, indent=2, allow_nan=False))
    print(json.dumps({"metrics_saved": str(metrics_file)}, indent=1), flush=True)
    render_status = {}
    if args.render:
        for behavior, predictions, labels in render_jobs:
            global RENDER_VERTICES, RENDER_LIMITS, RENDER_GAP_LIMITS
            RENDER_VERTICES = {}
            for arm, (rots, root, posed) in predictions.items():
                RENDER_VERTICES[arm] = {side: skinned_vertices(skinner, rots, posed).index_select(
                    1, torch.as_tensor(indices[side])) for side in ("Left", "Right")}
            stacked = torch.cat([value for arm in RENDER_VERTICES for value in RENDER_VERTICES[arm].values()])
            RENDER_LIMITS = ((float(stacked[..., 0].min()) - 0.1, float(stacked[..., 0].max()) + 0.1),
                             (float(stacked[..., 1].min()) - 0.02, max(float(stacked[..., 1].max()) + 0.03, 0.1)))
            try:
                output = render(args.output, behavior, predictions, indices, labels, 60.0,
                                {"skeleton": skeleton})
                render_status[behavior] = {"status": "complete", **output}
            except Exception as error:
                render_status[behavior] = {"status": "failed", "error": repr(error)}
                (args.output / "render_status.json").write_text(json.dumps(render_status, indent=2))
                raise
            (args.output / "render_status.json").write_text(json.dumps(render_status, indent=2))
            print(json.dumps({"render_complete": behavior, **output}), flush=True)


if __name__ == "__main__":
    main()
