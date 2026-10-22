"""Original-MB sparse stair intent experiments, with generated-only history.

Source-authored landing keyframes are explicit inputs, not evidence of autonomous
terrain perception. Free frames are never supplied as root trajectories or codes.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.review_native_mb_stairs import (
    SIDES, _skin, digest, read_shard, render_example, select_windows,
    support_report, transform_support,
)
from scripts.evaluate_native_mb_inbetween import PREDICT_CONFIG
from motionbricks.data.stair_sequence_metrics import stair_sequence_metrics


def training_condition_counts(infer, bundle):
    """Read actual original curriculum, not just the configured final maxima."""
    result = {}
    for stage in ("root", "pose"):
        model = getattr(infer, f"_{stage}_model")
        probe = SimpleNamespace(trainer=SimpleNamespace(global_step=bundle[f"{stage}_step"]), args=model.args)
        result[stage] = {}
        for end in ("start", "end"):
            probabilities = type(model)._construct_keyframe_prob(
                probe, model.args[f"max_num_{end}_keyframes"], model.args[f"no_{end}_keyframe_prob"])
            result[stage][end] = np.flatnonzero(probabilities).tolist()
    return result


def landing_endpoints(events, start, minimum_tokens, maximum_tokens):
    """Choose a single interior landing from source evidence, on the token grid."""
    candidates = []
    minimum_new = minimum_tokens * 4 - 4
    for event in events:
        if event["status"] != "weak_support":
            continue
        for frame in range(7, 63, 4):
            if (event["start_frame"] <= start + frame < event["end_frame_exclusive"]
                    and frame - 3 >= minimum_new and 63 - frame >= minimum_new
                    and (frame + 1) // 4 <= maximum_tokens
                    and (67 - frame) // 4 <= maximum_tokens):
                center = (event["start_frame"] + event["end_frame_exclusive"] - 1) / 2 - start
                candidates.append((abs(frame - 31), abs(frame - center), frame))
    if not candidates:
        raise ValueError("No evidenced intermediate landing fits the native duration range")
    return [min(candidates)[2], 63]


def common_scoring_masks(endpoints, frames=64):
    mask = np.ones(frames, dtype=bool)
    mask[:4] = False
    for endpoint in endpoints:
        mask[endpoint - 3:endpoint + 1] = False
        if endpoint + 1 < frames:
            mask[endpoint + 1] = False  # Authored endpoint velocity reveals the next root.
    return mask, mask[:-1] & mask[1:]


def all_landing_endpoints(events, start, minimum_tokens, maximum_tokens, frames=64):
    """Author one goal per available stationary event, subject to native timing.

    Event centers determine choices, never generated quality. Events too close to
    the previous goal or to the final goal are skipped and explicitly recorded.
    """
    minimum_new, maximum_new = minimum_tokens * 4 - 4, maximum_tokens * 4 - 4
    if frames % 4 or frames < minimum_tokens * 4:
        raise ValueError("Sequence frame count must fit the native token grid")
    final, previous, selected, skipped = frames - 1, 3, [], []
    ordered = sorted((e for e in events if e['status'] == 'weak_support'),
                     key=lambda e: (e['start_frame'] + e['end_frame_exclusive'], e['side']))
    for event in ordered:
        center = (event['start_frame'] + event['end_frame_exclusive'] - 1) / 2 - start
        if center <= 3 or center >= final:
            continue
        choices = [f for f in range(7, final, 4)
                   if event['start_frame'] <= f + start < event['end_frame_exclusive']
                   and minimum_new <= f - previous <= maximum_new and final - f >= minimum_new]
        if not choices:
            skipped.append({'side': event['side'], 'source_event_center_frame': center,
                            'reason': 'no_remaining_native_duration_grid_candidate'})
            continue
        previous = min(choices, key=lambda f: (abs(f - center), f))
        selected.append(previous)
    if not minimum_new <= final - previous <= maximum_new:
        raise ValueError("Source contact events cannot cover the sequence with native durations")
    return selected + [final], skipped


def intent_constraints(infer, history, goal_block, tokens, context_count=4, goal_count=1):
    """Pack only played history and an explicit 4-frame goal plus its derivative."""
    import torch
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    rep, local_rep = infer.global_motion_rep, infer.local_motion_rep
    dimension = len(rep.indices["all"])
    if (history.shape != (1, 4, dimension) or goal_block.shape != (1, 5, dimension)
            or not torch.isfinite(history).all() or not torch.isfinite(goal_block).all()):
        raise ValueError("Need finite four history frames and five authored goal frames")
    minimum, maximum = infer._args["min_tokens"], infer._args["max_tokens"]
    if not minimum <= tokens <= maximum or context_count not in (1, 4) or goal_count not in (1, 4):
        raise ValueError("Intent duration or keyframe count outside native contract")

    def features(global_motion):
        local = infer.motion_rep.dual_rep.global_to_local(
            global_motion, False, False,
            lengths=torch.tensor([global_motion.shape[1]], device=history.device))
        return {
            "global_root_values": extract_feature_from_motion_rep(global_motion, rep, "root"),
            "local_root_values": extract_feature_from_motion_rep(local, local_rep, "root"),
            "local_poses": extract_feature_from_motion_rep(local, local_rep, infer.EXTERNAL_POSE_FEATURE_MODE),
        }

    past, goal = features(history), features(goal_block)
    inputs = {}
    for name in past:
        values = history.new_zeros((1, 8, past[name].shape[-1]))
        values[:, :4] = past[name][:, :4]
        values[:, 4:] = goal[name][:, :4]
        mask = torch.zeros((1, 8), dtype=torch.bool, device=history.device)
        if name == "local_root_values":
            mask[:, :3] = context_count == 4
            mask[:, 2] = True
            values[:, 3] = 0  # No future frame exists after the played history.
        else:
            mask[:, :4] = context_count == 4
            mask[:, 3] = True
        mask[:, 4:] = goal_count == 4
        mask[:, 7] = True
        # Slot zero is also the original runtime's world-coordinate reference.
        inputs[name], inputs["has_" + name] = values, mask
    allowed = torch.zeros((1, maximum - minimum + 1), dtype=torch.long, device=history.device)
    allowed[0, tokens - minimum] = 1
    inputs.update(num_tokens=None, allowed_pred_num_tokens=allowed)
    return inputs


def official_intent_constraints(infer, history, goal_block, tokens, context_count=1, goal_count=1):
    """Invoke the shipped runtime packer on FK transforms, without executing a model.

    Official target velocity uses the last measurable BACKWARD interval of its
    four-frame clip. The unused fifth goal frame is not read by this path.
    """
    import torch
    from motionbricks.data.native_mb_locomotion import _InferenceInputs
    from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent

    if history.shape[1] != 4 or goal_block.shape[1] != 5:
        raise ValueError("Need four history and five goal-block frames")
    if not infer._args['min_tokens'] <= tokens <= infer._args['max_tokens']:
        raise ValueError("Intent duration outside native contract")
    captured = {}
    def capture(*args, **kwargs):
        names = ('global_root_values', 'has_global_root_values', 'local_root_values',
                 'has_local_root_values', 'local_poses', 'has_local_poses', 'num_tokens')
        captured.update(zip(names, args))
        captured['allowed_pred_num_tokens'] = kwargs['allowed_pred_num_tokens']
        return None, torch.tensor([tokens], device=history.device)
    proxy = SimpleNamespace(_root_model=infer._root_model, local_motion_rep=infer.local_motion_rep,
                            _args=infer._args, predict=capture)
    packer = SimpleNamespace(_inferencer=_InferenceInputs(proxy, goal_count, tokens, context_count),
        NUM_FRAMES_PER_TOKEN=4, _device=history.device, _target_root_realignment=True,
        SKIP_ENDING_TARGET_COND=False)
    past = infer.global_motion_rep.inverse(history, False, return_quat=False, return_all=True)
    target = infer.global_motion_rep.inverse(goal_block[:, :4], False, return_quat=False, return_all=True)
    root_planar = target['posed_joints'][:, :, 0] * history.new_tensor([1., 0., 1.])
    full_navigation_agent._predict_inbetween_motion_features(packer, {
        'context_global_joint_positions': past['posed_joints'],
        'context_global_joint_rotations': past['global_joint_rots'],
        'target_global_root_positions': root_planar,
        'target_global_joint_positions': target['posed_joints'] - root_planar[:, :, None],
        'target_global_joint_rotations': target['global_joint_rots'],
    })
    return captured


def generate_sequence(infer, seed, goals, *, context_count, goal_count, iterations, official_inputs=False):
    """Goals are (absolute endpoint frame, authored five-frame block) pairs.

    Only this initial seed and explicit goal blocks can access source features.
    Generated context predictions are discarded; no played future is modified.
    """
    import torch
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation

    history, pieces, plans = seed.clone(), [seed.clone()], []
    played = 4
    config = {**PREDICT_CONFIG, "num_inference_step": iterations}
    with full_precision_evaluation(), torch.inference_mode():
        for endpoint, goal in goals:
            new_frames = endpoint + 1 - played
            if new_frames < 1 or (new_frames + 4) % 4:
                raise ValueError("Each intent must advance time by a positive token-aligned interval")
            tokens = (new_frames + 4) // 4
            pack = official_intent_constraints if official_inputs else intent_constraints
            inputs = pack(infer, history, goal, tokens, context_count, goal_count)
            if history.is_cuda:
                torch.cuda.synchronize()
            began = time.perf_counter()
            prediction, count = infer.predict(**inputs, config=config, info={})
            if history.is_cuda:
                torch.cuda.synchronize()
            duration = time.perf_counter() - began
            if int(count.item()) != tokens or not torch.isfinite(prediction).all():
                raise ValueError("Original MB returned invalid sequence prediction")
            committed = prediction[:, 4:tokens * 4].clone()
            if committed.shape[1] != new_frames:
                raise ValueError("Original MB returned fewer frames than the requested intent")
            plans.append({"first_generated_frame": played, "endpoint_frame": endpoint,
                          "tokens": tokens, "latency_seconds": duration,
                          "context_source": "initial_seed" if played == 4 else "generated_only",
                          "authored_pose_frames": list(range(endpoint - 3, endpoint + 1)) if goal_count == 4 else [endpoint],
                          "input_geometry": "official FK runtime" if official_inputs else "raw feature values",
                          "authored_derivative_interval": [endpoint - 1, endpoint] if official_inputs else [endpoint, endpoint + 1]})
            pieces.append(committed)
            history = torch.cat((history, committed), dim=1)[:, -4:].clone()
            played += new_frames
    return torch.cat(pieces, dim=1), plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--whole-clips", action="store_true",
                        help="Roll out entire held-out recordings with every feasible authored landing")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output directory")
    from scripts.train_native_mb import select_gpu
    gpu = select_gpu(args.gpu)
    import torch
    from scripts.evaluate_native_mb_pipeline import load_bundle
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.soma_skin_reference import load_official_skinner
    from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    np.random.seed(2333)
    infer, model_dataset, bundle = load_bundle(args.root_run, args.pose_run, "cuda")
    for filename in ("mean.npy", "std.npy"):
        if digest(model_dataset / "stats" / filename) != digest(args.dataset / "stats" / filename):
            raise ValueError("Model and review normalization must match exactly")
    rep, skeleton = infer.global_motion_rep, infer.motion_rep.skeleton
    skin_skeleton = build_native_representation().skeleton
    surfaces, _, _, mesh_hash = official_sole_surfaces(skin_skeleton)
    skinner = load_official_skinner(skin_skeleton)
    variants = [(mode, context, target, iterations)
                for mode in ("endpoint", "sequence")
                for context, target in ((1, 1), (4, 1), (4, 4))
                for iterations in (1, 4)]
    variants += [("landings", 1, 1, iterations) for iterations in (1, 4)]
    if args.whole_clips:
        variants = [("landings", 1, 1, iterations) for iterations in (1, 4)]
        variants += [("official_landings", 1, 1, iterations) for iterations in (1, 4)]
    report = {"task": "original MB Smart Object sparse intent ablation", "bundle": bundle,
              "dataset_manifest_sha256": digest(args.dataset / "native_manifest.json"),
              "mesh_sha256": mesh_hash, "gpu_at_launch": gpu, "seed": 2333,
              "min_tokens": infer._args["min_tokens"], "max_tokens": infer._args["max_tokens"],
              "postprocessing": "none", "root_overwrite": False, "terrain_network_input": False,
              "goal_provenance": "Held-out source authored sparse landing poses and terminal derivatives. Not independent terrain navigation or novel-goal generalization.",
              "subsequent_history": "generated_only, no teacher forcing",
              "dense_source_root_trajectory_or_source_tokens": False,
              "authored_interior_root_keyframes": True, "quality_approved": False,
              "scoring": "Same free-frame mask for every variant; exclude all supplied pose/root goal frames, derivative endpoints and initial context; interval endpoints both free.",
              "candidate_contact_tolerance_m": .02, "windows": []}
    report["scope"] = "full held-out recordings" if args.whole_clips else "fixed 64-frame ablation"
    report["training_condition_counts_at_saved_step"] = training_condition_counts(infer, bundle)
    report["source_sha256"] = {str(Path(__file__).name): digest(__file__),
        "stair_sequence_metrics.py": digest(Path(__file__).parents[1] / "motionbricks/data/stair_sequence_metrics.py"),
        "official_full_agent.py": digest(Path(__file__).parents[1] / "motionbricks/motion_backbone/demo/full_agent.py"),
        "official_motion_inference.py": digest(Path(__file__).parents[1] / "motionbricks/motion_backbone/inference/motion_inference.py")}
    args.output.mkdir(parents=True)
    seen_shards = set()
    for index, row in enumerate(select_windows(args.dataset)):
        if args.whole_clips:
            if row['shard'] in seen_shards:
                continue
            seen_shards.add(row['shard'])
            row = {**row, 'selection_window_start': row['start'], 'start': 0}
        data = read_shard(args.dataset / row["shard"])
        start = row["start"]
        frames = ((len(data['global_features']) - 1) // 4) * 4 if args.whole_clips else 64
        support = support_report(args.dataset, row)
        endpoints = [] if args.whole_clips else landing_endpoints(
            support["events"], start, infer._args["min_tokens"], infer._args["max_tokens"])
        all_endpoints, skipped = all_landing_endpoints(
            support["events"], start, infer._args["min_tokens"], infer._args["max_tokens"], frames=frames)
        # Four-pose conditions exist only in the two-intent diagnostic arms;
        # the per-landing arm supplies one pose each, not four per landing.
        frame_mask, _ = common_scoring_masks(endpoints, frames=frames)
        frame_mask[all_endpoints] = False
        frame_mask[[e + 1 for e in all_endpoints if e + 1 < frames]] = False
        if args.whole_clips:
            frame_mask[[e - 1 for e in all_endpoints]] = False
        interval_mask = frame_mask[:-1] & frame_mask[1:]
        raw = torch.as_tensor(data["global_features"][start:start + frames + 1], device="cuda")[None]
        with full_precision_evaluation(), torch.inference_mode():
            canonical = rep.change_first_heading(raw, 0., False, False)
            _, angles = rep.compute_root_pos_and_rot(raw, return_angle=True, return_quat=False)
            rotation = angle_to_Y_rotation_matrix(-angles[:, 0])[0].cpu().numpy()
            origin = raw[0, 0, :3].cpu().numpy().copy()
            origin[1] = 0
        patches = transform_support(support["patches"], rotation, origin)
        generated, plans = {"source": canonical[:, :frames]}, {}
        for mode, context, target, iterations in variants:
            name = f"{mode}_c{context}_t{target}_i{iterations}"
            chosen = all_endpoints if mode.endswith("landings") else endpoints if mode == "sequence" else [63]
            goals = [(end, canonical[:, end - 3:end + 2].clone()) for end in chosen]
            generated[name], plans[name] = generate_sequence(
                infer, canonical[:, :4], goals, context_count=context, goal_count=target, iterations=iterations,
                official_inputs=mode.startswith("official_"))
        scores, joints = {}, {}
        masks = {side: data[f"vertex_measured_{side}"][start:start + frames - 1].astype(bool) for side in SIDES}
        for name, features in generated.items():
            with full_precision_evaluation(), torch.inference_mode():
                motion = rep.inverse(features, False, return_all=True)
                soles = _skin(motion, skeleton, surfaces, skinner)
            joints[name] = motion["posed_joints"][0].cpu().numpy()
            if name == "source":
                for side in SIDES:
                    expected = (data[f"sole_vertices_{side}"][start:start + frames] - origin) @ rotation.T
                    if abs(expected - soles[side]).max() > .003:
                        raise ValueError("Source official sole geometry mismatch")
            geometry = stair_sequence_metrics(soles, 30., patches, near_surface_tolerance_m=.02,
                closest_contact_tolerance_m=.02, expected_source_material_masks=masks,
                scoring_frame_mask=frame_mask, scoring_interval_mask=interval_mask)
            delta = joints[name] - joints["source"]
            velocity = np.diff(joints[name], axis=0) * 30
            acceleration = np.diff(velocity, axis=0) * 30
            scores[name] = {"world_mpjpe_m": float(np.linalg.norm(delta[frame_mask], axis=-1).mean()),
                "root_height_mae_m": float(abs(delta[frame_mask, skeleton.root_idx, 1]).mean()),
                "joint_acceleration_p95_mps2_all_frames": float(np.quantile(np.linalg.norm(acceleration, axis=-1), .95)),
                "geometry": geometry}
        name = f"{index:02d}_{row['direction']}_start{start:06d}"
        result = {"source": row, "landing_endpoints": endpoints, "all_landing_endpoints": all_endpoints,
                  "skipped_contact_events": skipped, "support_patches": patches, "plans": plans, "scores": scores,
                  "frames": frames, "generated_frames": frames - 4,
                  "free_frame_indices": np.flatnonzero(frame_mask).tolist(), "artifact": name}
        report["windows"].append(result)
        np.savez_compressed(args.output / f"{name}.npz", **joints,
                            **{f"features_{key}": value[0].cpu().numpy() for key, value in generated.items()}, fps=np.float32(30))
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"artifact": name, "goals": endpoints,
                          "scores": {key: {k: v for k, v in value.items() if k != 'geometry'} for key, value in scores.items()}}), flush=True)
        if args.render:
            views = ({"Source": joints["source"], "Raw feature history": joints["landings_c1_t1_i1"],
                      "Official runtime inputs": joints["official_landings_c1_t1_i1"]} if args.whole_clips else
                     {"Source": joints["source"], "Old 4-context input": joints["endpoint_c4_t1_i1"],
                      "Matched sparse input": joints["endpoint_c1_t1_i1"],
                      "Each landing intent": joints["landings_c1_t1_i1"]})
            render_example(args.output / name, views, patches,
                skeleton.joint_parents.cpu().numpy(), row["direction"],
                condition_caption=f"seed 0-3; sparse authored landing endpoints {all_endpoints}; generated history")
    report["status"] = "completed_not_quality_approved"
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
