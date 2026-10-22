"""Held-out stair skill probe through original MB sparse-keyframe generation.

Weak support patches are comparison geometry, never future network inputs. This
is not a proof of arbitrary terrain-aware WASD control or measured ground truth.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.stair_support import query_support


START, LAST, ENDPOINT, GLOBAL_FRAMES = 4, 62, 63, 65
SIDES = ("left", "right")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_shard(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def support_report(dataset, record):
    path = dataset / (record.get("support_report") or record["report_path"])
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("support", report)


def select_windows(dataset, maximum_per_direction=3):
    """Select by held-out SOURCE height travel/coverage, never prediction quality."""
    dataset = Path(dataset)
    records = json.loads((dataset / "accepted.json").read_text(encoding="utf-8"))
    candidates = {"up": [], "down": []}
    for record in records:
        if record["split"] != "validation" or not ({"support_report", "report_path"} & record.keys()):
            continue
        data = read_shard(dataset / record["shard"])
        features = data["global_features"]
        report = support_report(dataset, record)
        if report["summary"]["pair_status"] != "weak_pair_candidate":
            continue
        for start in range(0, len(features) - GLOBAL_FRAMES + 1, 16):
            delta = float(features[start + LAST, 1] - features[start + START, 1])
            if abs(delta) < .18:
                continue
            ids = np.concatenate([data[f"patch_id_{side}"][start + START:start + LAST + 1]
                                  for side in SIDES])
            known = ids >= 0
            levels = {p["level_id"] for p in report["patches"] if p["id"] in set(ids[known])}
            evidence = {side: int((data[f"patch_id_{side}"][start + START:start + LAST + 1] >= 0).sum())
                        for side in SIDES}
            if len(levels) < 2 or min(evidence.values()) < 5:
                continue
            direction = "up" if delta > 0 else "down"
            candidates[direction].append({**record, "start": start, "direction": direction,
                "source_interior_root_height_change_m": delta, "source_evidence_frames": evidence,
                "source_levels": len(levels), "selection_score": int(known.sum())})
    selected = []
    for direction, options in candidates.items():
        options.sort(key=lambda r: (-r["selection_score"], r["shard"], r["start"]))
        chosen = []
        for row in options:
            if any(row["shard"] == other["shard"] and abs(row["start"] - other["start"]) < 64
                   for other in chosen):
                continue
            chosen.append(row)
            if len(chosen) == maximum_per_direction:
                break
        selected.extend(chosen)
    if {row["direction"] for row in selected} != {"up", "down"}:
        raise ValueError("Need independent held-out ascending AND descending windows with bilateral support evidence")
    return selected


def transform_support(patches, rotation, origin):
    result = copy.deepcopy(patches)
    for patch in result:
        polygon = np.asarray(patch["polygon_xz_m"], dtype=np.float64)
        points = np.column_stack((polygon[:, 0], np.full(len(polygon), patch["height_m"]), polygon[:, 1]))
        points = (points - origin) @ rotation.T
        patch["polygon_xz_m"] = points[:, [0, 2]].tolist()
        patch["height_m"] = float(points[:, 1].mean())
    return result


def stats(value):
    value = np.asarray(value, dtype=np.float64)
    value = value[np.isfinite(value)]
    return {"count": int(value.size), "mean": float(value.mean()) if value.size else None,
            "p95": float(np.quantile(value, .95)) if value.size else None,
            "max": float(value.max()) if value.size else None}


def sole_scores(points, source_data, patches, start, fps=30.0):
    """Use immutable source evidence; query unknown spatial coverage explicitly."""
    if set(points) != set(SIDES):
        raise ValueError("Both sole regions are required")
    heights = {p["id"]: p["height_m"] for p in patches if p["status"] == "weak_support"}
    result = {}
    for side in SIDES:
        vertices = np.asarray(points[side])
        if vertices.shape != (64, 98, 3) or not np.isfinite(vertices).all():
            raise ValueError("Require finite 64-frame, 98-material-vertex sole regions")
        ids = source_data[f"patch_id_{side}"][start:start + 64]
        source_mask = source_data[f"vertex_measured_{side}"][start:start + 63]
        if ids.shape != (64,) or source_mask.shape != (63, 98):
            raise ValueError("Source frame/interval evidence is not aligned")
        expected = np.array([heights.get(int(i), np.nan) for i in ids])
        contact_gap = vertices[START:LAST + 1, :, 1].min(-1) - expected[START:LAST + 1]
        speed = np.linalg.norm(np.diff(vertices[..., [0, 2]], axis=0), axis=-1) * fps
        mask = source_mask[START:LAST].astype(bool)
        spatial_height, _, _ = query_support(vertices[START:LAST + 1, :, [0, 2]], patches)
        spatial_gap = vertices[START:LAST + 1, :, 1] - spatial_height
        # Missing patch coverage is unknown, not a zero penetration sample.
        penetration = np.where(np.isfinite(spatial_gap), np.maximum(-spatial_gap, 0), np.nan)
        result[side] = {
            "expected_support_abs_gap_m": stats(abs(contact_gap)),
            "expected_support_below_m": stats(np.maximum(-contact_gap, 0)),
            "source_mask_material_slip_mps": stats(speed[START:LAST][mask]),
            "known_patch_penetration_m": stats(penetration),
            "spatial_query_unknown_vertex_fraction": float((~np.isfinite(spatial_gap)).mean()),
            "source_measured_vertex_intervals": int(mask.sum()),
            "source_measured_intervals": int(mask.any(-1).sum()),
            "source_mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
        }
    return result


def root_scores(joints, source, root_index):
    root, source_root = joints[:, root_index], source[:, root_index]
    delta = root[LAST] - root[START]
    target = source_root[LAST] - source_root[START]
    return {"interior_displacement_xyz_m": delta.tolist(),
            "source_interior_displacement_xyz_m": target.tolist(),
            "height_change_error_m": float(abs(delta[1] - target[1])),
            "horizontal_travel_error_m": float(np.linalg.norm(delta[[0, 2]] - target[[0, 2]])),
            "height_direction_matches": bool(delta[1] * target[1] > 0),
            "root_height_mae_m": float(abs(root[START:LAST + 1, 1] - source_root[START:LAST + 1, 1]).mean()),
            "endpoint_error_m": float(np.linalg.norm(root[ENDPOINT] - source_root[ENDPOINT])),
            "fk_mpjpe_m": float(np.linalg.norm(joints[START:LAST + 1] - source[START:LAST + 1], axis=-1).mean())}


def skill_gate(scores, source_scores, root):
    """Declared engineering probe tolerances, not paper or biological constants."""
    failures = []
    if not root["height_direction_matches"] or root["height_change_error_m"] > .08:
        failures.append("root_vertical_progress")
    if root["horizontal_travel_error_m"] > .15:
        failures.append("root_horizontal_progress")
    for side in SIDES:
        for metric, tolerance in (("expected_support_abs_gap_m", .03),
                                  ("known_patch_penetration_m", .03)):
            value = scores[side][metric]["p95"]
            if value is None or value > tolerance:
                failures.append(side + ":" + metric)
        source_slip = source_scores[side]["source_mask_material_slip_mps"]["p95"]
        slip = scores[side]["source_mask_material_slip_mps"]["p95"]
        if slip is None or source_slip is None or slip > source_slip + .05:
            failures.append(side + ":excess_slip")
    return {"passed": not failures, "failed_checks": failures}


def stair_constraints(infer, raw):
    """Original four-frame context plus one target, without future context velocity."""
    import torch
    from scripts.evaluate_native_mb_inbetween import inbetween_constraints
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    inputs, source = inbetween_constraints(infer, raw)
    canonical = infer.global_motion_rep.change_first_heading(raw, 0., False, False)
    local = infer.motion_rep.dual_rep.global_to_local(
        canonical, False, False, lengths=torch.tensor([65], device=raw.device))
    values = {
        "global_root_values": extract_feature_from_motion_rep(canonical, infer.global_motion_rep, "root"),
        "local_root_values": extract_feature_from_motion_rep(local, infer.local_motion_rep, "root"),
        "local_poses": extract_feature_from_motion_rep(local, infer.local_motion_rep, infer.EXTERNAL_POSE_FEATURE_MODE),
    }
    for key, value in values.items():
        count = 3 if key == "local_root_values" else 4
        inputs[key][:, :4] = 0
        inputs[key][:, :count] = value[:, :count]
        inputs["has_" + key][:, :4] = False
        inputs["has_" + key][:, :count] = True
    return inputs, source


def stair_prediction(infer, raw):
    import torch
    from scripts.evaluate_native_mb_inbetween import PREDICT_CONFIG
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation

    with full_precision_evaluation(), torch.inference_mode():
        inputs, source = stair_constraints(infer, raw)
        prediction, count = infer.predict(**inputs, config=dict(PREDICT_CONFIG), info={})
        if int(count.item()) != 16 or prediction.shape != source.shape or not torch.isfinite(prediction).all():
            raise ValueError("Invalid original MB stair prediction")
    return source, prediction


def _skin(motion, skeleton, surfaces, skinner):
    from scripts.prepare_native_mb_stairs import _skin_sole_regions
    return _skin_sole_regions(motion["local_joint_rots"][0].cpu(),
                             motion["posed_joints"][0].cpu(), skinner, surfaces, 64)[0]


def render_example(path, arrays, patches, parents, direction, condition_caption="context 0-3, target 63"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    names = list(arrays)
    frame_count = len(arrays[names[0]])
    if frame_count < 2 or any(len(value) != frame_count for value in arrays.values()):
        raise ValueError("Rendered motions must share at least two frames")
    fig = plt.figure(figsize=(5 * len(names), 5.2))
    fig.subplots_adjust(left=.03, right=.97, top=.82, bottom=.04, wspace=.13)
    edge = np.asarray([(int(p), i) for i, p in enumerate(parents) if p >= 0])
    all_joints = np.concatenate(list(arrays.values()), 0)
    lower, upper = all_joints.min((0, 1)), all_joints.max((0, 1))
    axes, artists = [], []
    for i, name in enumerate(names):
        axis = fig.add_subplot(1, len(names), i + 1, projection="3d")
        axis.view_init(elev=20, azim=-52)
        axis.set(xlim=(lower[0] - .25, upper[0] + .25), ylim=(lower[2] - .25, upper[2] + .25),
                 zlim=(min(lower[1], 0) - .15, upper[1] + .2), xlabel="X (m)", ylabel="Z (m)", zlabel="Y (m)")
        axis.set_box_aspect((max(.7, upper[0] - lower[0]), max(.7, upper[2] - lower[2]), upper[1] - min(lower[1], 0) + .35))
        axis.xaxis.set_major_locator(MaxNLocator(2))
        axis.yaxis.set_major_locator(MaxNLocator(4))
        axis.zaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(labelsize=8)
        axis.set_title(name, fontsize=11)
        for patch in patches:
            if patch["status"] != "weak_support":
                continue
            polygon = np.asarray(patch["polygon_xz_m"])
            center = polygon.mean(0)
            if np.any(center < lower[[0, 2]] - .5) or np.any(center > upper[[0, 2]] + .5):
                continue
            polygon3d = np.column_stack((polygon, np.full(len(polygon), patch["height_m"])))
            axis.add_collection3d(Poly3DCollection([polygon3d], facecolors="#909a92", edgecolors="#506051", alpha=.48))
        line = Line3DCollection(np.zeros((len(edge), 2, 3)), colors=("#252f38" if i == 0 else "#168b8f"), linewidths=2.4)
        axis.add_collection3d(line)
        axes.append(axis)
        artists.append(line)
    title = fig.suptitle("", fontsize=12)
    frame_ids = np.arange(0, frame_count, 2)
    def update(index):
        frame = int(frame_ids[index])
        for name, artist in zip(names, artists):
            artist.set_segments(arrays[name][frame][:, [0, 2, 1]][edge])
        title.set_text(f"{direction} | {frame / 30:.2f} s | {condition_caption}\n"
                       "Weak support patches only; no IK, foot lock or root correction")
        return artists + [title]
    update(0)
    fig.canvas.draw()
    fig.savefig(path.with_suffix(".png"), dpi=110)
    FuncAnimation(fig, update, frames=len(frame_ids), interval=1000 / 15, blit=False).save(
        path.with_suffix(".gif"), writer=PillowWriter(fps=15))
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer-run", type=Path, required=True)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--windows-per-direction", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh review directory")
    if args.windows_per_direction < 1:
        parser.error("Need at least one window per direction")
    from scripts.train_native_mb import select_gpu, verify_dataset
    gpu = select_gpu(args.gpu)
    manifest = json.loads((args.dataset / "native_manifest.json").read_text())
    verify_dataset(args.dataset, manifest)
    windows = select_windows(args.dataset, args.windows_per_direction)

    import torch
    from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint
    from scripts.evaluate_native_mb_pipeline import load_bundle
    from scripts.evaluate_native_mb_inbetween import PREDICT_CONFIG
    from scripts.review_native_mb_lafan_system import tokenizer_reconstruction
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.soma_skin_reference import load_official_skinner
    from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    np.random.seed(2333)
    infer, model_dataset, bundle = load_bundle(args.root_run, args.pose_run, "cuda")
    tokenizer, metadata, _ = load_native_checkpoint(args.tokenizer_run, device="cuda")
    if metadata["checkpoint_sha256"] != bundle["tokenizer"]["checkpoint_sha256"]:
        raise ValueError("Tokenizer must be the exact frozen Pose dependency")
    for filename in ("mean.npy", "std.npy"):
        if digest(model_dataset / "stats" / filename) != digest(args.dataset / "stats" / filename):
            raise ValueError("Probe and model normalization must be identical")
    rep, skeleton = infer.global_motion_rep, infer.motion_rep.skeleton
    skin_skeleton = build_native_representation().skeleton
    surfaces, _, _, mesh_hash = official_sole_surfaces(skin_skeleton)
    skinner = load_official_skinner(skin_skeleton)
    args.output.mkdir(parents=True)
    report = {"task": "held-out stair keyframe skill probe", "bundle": bundle,
              "model_dataset": str(model_dataset), "probe_dataset_sha256": digest(args.dataset / "native_manifest.json"),
              "seed": 2333, "gpu_at_launch": gpu, "fps": 30, "mesh_sha256": mesh_hash,
              "conditions": {"pose_frames": [0, 1, 2, 3, 63], "global_root_frames": [0, 1, 2, 3, 63], "local_root_frames": [0, 1, 2, 63],
                             "future_root_trajectory": False, "future_source_tokens": False, "terrain_network_input": False},
              "height_condition_route": "Original Pose receives global XZ and heading; decoder receives local XZ velocity. Neither root condition carries hip height; height enters sparse body poses and decoded motion, not a hard Root-height trajectory constraint",
              "generation_config": PREDICT_CONFIG, "postprocessing": "none", "root_overwrite": False,
              "scoring_frames_inclusive": [START, LAST], "scoring_interval_starts_inclusive": [START, LAST - 1],
              "gate": {"status": "engineering probe, not paper metric or calibrated truth", "height_change_error_m": .08,
                       "horizontal_travel_error_m": .15, "support_gap_p95_m": .03, "penetration_p95_m": .03,
                       "slip_p95_above_source_mps": .05},
              "limitations": ["Sparse keyframe skill, not arbitrary terrain-aware navigation.",
                              "Source-derived support hulls do not reveal risers, full tread extents or unvisited space.",
                              "Weak motion-derived evidence is not independent measured terrain or force ground truth.",
                              "Tokenizer reconstruction uses source interior/root; never counted as generated skill.",
                              "Endpoint keyframe accuracy is excluded from interior skill metrics."], "windows": []}
    for i, row in enumerate(windows):
        data = read_shard(args.dataset / row["shard"])
        start = row["start"]
        raw = torch.as_tensor(data["global_features"][start:start + GLOBAL_FRAMES], device="cuda")[None]
        with full_precision_evaluation(), torch.inference_mode():
            source_features, prediction = stair_prediction(infer, raw)
            motions = {"source": rep.inverse(source_features, False, return_all=True),
                       "generated": rep.inverse(prediction, False, return_all=True)}
            _, motions["tokenizer_reconstruction"] = tokenizer_reconstruction(tokenizer, raw)
            _, angle = rep.compute_root_pos_and_rot(raw, return_angle=True, return_quat=False)
            rotation = angle_to_Y_rotation_matrix(-angle[:, 0])[0].cpu().numpy()
            origin = raw[0, 0, :3].cpu().numpy().copy()
            origin[1] = 0
        patches = transform_support(support_report(args.dataset, row)["patches"], rotation, origin)
        soles = {name: _skin(motion, skeleton, surfaces, skinner) for name, motion in motions.items()}
        parity = {}
        for side in SIDES:
            transformed = (data[f"sole_vertices_{side}"][start:start + 64] - origin) @ rotation.T
            parity[side] = float(abs(transformed - soles["source"][side]).max())
            if parity[side] > .003:
                raise ValueError(f"Source feature/official sole coordinate mismatch: {parity}")
        joints = {name: motion["posed_joints"][0].detach().cpu().numpy() for name, motion in motions.items()}
        scores = {name: sole_scores(points, data, patches, start) for name, points in soles.items()}
        roots = {name: root_scores(value, joints["source"], skeleton.root_idx) for name, value in joints.items()}
        gate = skill_gate(scores["generated"], scores["source"], roots["generated"])
        name = f"{i:02d}_{row['direction']}_start{start:06d}"
        np.savez_compressed(args.output / (name + ".npz"), **joints,
                            generated_global_features=prediction[0].cpu().numpy(), fps=np.float32(30))
        result = {"source": row, "source_parity_max_error_m": parity, "sole_scores": scores,
                  "root_scores": roots, "generated_skill_gate": gate, "artifact": name}
        report["windows"].append(result)
        # Write metrics before rendering: a visualization failure cannot discard scores.
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if args.render:
            render_example(args.output / name, {"Source": joints["source"], "Generated": joints["generated"]},
                           patches, skeleton.joint_parents.detach().cpu().numpy(), row["direction"])
        print(json.dumps({"artifact": name, "root": roots["generated"], "gate": gate}), flush=True)
    report["summary"] = {direction: {"windows": sum(r["source"]["direction"] == direction for r in report["windows"]),
        "passed": sum(r["source"]["direction"] == direction and r["generated_skill_gate"]["passed"] for r in report["windows"])}
        for direction in ("up", "down")}
    report["stair_skill_proven"] = all(v["windows"] > 0 and v["windows"] == v["passed"] for v in report["summary"].values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
