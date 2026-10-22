"""Deterministic SOMA reconstruction review on recording-held-out windows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from motionbricks.data.bones_evaluation import reconstruction_metrics, summarize_windows
from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_models import BonesTokenizer, apply_root_velocity_condition
from motionbricks.data.bones_seed import build_representation, deterministic_window_starts
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


def evaluate_numerical_screen(summary, limits):
    if limits is None:
        return None
    bounds = (("upper_bounds", lambda value, limit: value <= limit),
              ("lower_bounds", lambda value, limit: value >= limit))
    checks = {}
    for kind, compare in bounds:
        for name, limit in limits.get(kind, {}).items():
            value = summary.get(name)
            checks[name] = value is not None and np.isfinite(value) and bool(compare(value, limit))
    if not checks:
        raise ValueError("A declared numerical screen must contain bounds; an empty screen cannot pass")
    return {"passed": all(checks.values()), "checks": checks, "limits": limits, "not_quality_approval": True}


def render_review(examples, skeleton, folder, prediction_label="raw reconstruction"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    parents = skeleton.joint_parents.tolist()
    figure = plt.figure(figsize=(12, 3 * len(examples)))
    for row, (behavior, sample) in enumerate(sorted(examples.items())):
        prediction, target = sample["prediction"], sample["target"]
        for column, frame in enumerate((0, 32, 64, 96)):
            axis = figure.add_subplot(len(examples), 4, 4 * row + column + 1, projection="3d")
            origin = target[frame, skeleton.root_idx].copy()
            origin[1] = 0
            for points, color in ((target[frame] - origin, "#777777"),
                                  (prediction[frame] - origin, "#007fa3")):
                for joint, parent in enumerate(parents):
                    if parent >= 0:
                        axis.plot(points[[parent, joint], 0], points[[parent, joint], 2],
                                  points[[parent, joint], 1], color=color, linewidth=1.3)
            axis.set(xlim=(-0.9, 0.9), ylim=(-0.9, 0.9), zlim=(-0.1, 2))
            axis.set_box_aspect((1, 1, 1.3))
            axis.view_init(elev=12, azim=-45)
            axis.set_axis_off()
            axis.set_title(f"{behavior} | {frame / 60:.2f}s", fontsize=10)
    figure.suptitle(f"SOMA30 | source: gray | {prediction_label}: blue | worst reviewed window per behavior")
    figure.tight_layout()
    figure.savefig(folder / "reconstruction_review.png", dpi=120)
    plt.close(figure)
    for behavior in ("walk", "turn"):
        if behavior not in examples:
            continue
        sample = examples[behavior]
        prediction, target = sample["prediction"], sample["target"]
        figure = plt.figure(figsize=(7, 6))
        axis = figure.add_subplot(111, projection="3d")
        points = np.concatenate((prediction, target), axis=0)
        center = target[:, skeleton.root_idx].mean(0)
        radius = max(float(np.ptp(points[..., 0])) / 2, float(np.ptp(points[..., 2])) / 2, 1.0)
        axis.set(xlim=(center[0] - radius, center[0] + radius),
                 ylim=(center[2] - radius, center[2] + radius), zlim=(-0.1, 2.1))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=15, azim=-45)
        axis.set(xlabel="X (m)", ylabel="Z (m)", zlabel="Y (m)")
        axis.plot([center[0] - radius, center[0] + radius], [center[2], center[2]], [0, 0], color="gray")
        axis.set_title(f"{behavior} | gray: source | blue: {prediction_label}")
        lines = [(axis.plot([], [], [], color=color)[0], joints, joint, parent)
                 for joints, color in ((target, "#777777"), (prediction, "#007fa3"))
                 for joint, parent in enumerate(parents) if parent >= 0]

        def update(frame):
            for line, joints, joint, parent in lines:
                pair = joints[frame, [parent, joint]]
                line.set_data_3d(pair[:, 0], pair[:, 2], pair[:, 1])
            return [line for line, *_ in lines]

        animation = FuncAnimation(figure, update, frames=range(0, len(target), 2), interval=1000 / 30)
        animation.save(folder / f"{behavior}_reconstruction.gif", writer=PillowWriter(fps=30))
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--windows-per-clip", type=int, default=3)
    parser.add_argument("--conditioning", choices=("none", "context"), default="context")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--diagnose-quantization", action="store_true", help="Compare continuous latents as an offline diagnostic, not a runtime replacement")
    args = parser.parse_args()
    if args.windows_per_clip < 1:
        parser.error("windows-per-clip must be positive")
    torch.set_num_threads(4)
    run = args.checkpoint.resolve().parent.parent
    contract = json.loads((run / "run_contract.json").read_text())
    if contract["stage"] != "tokenizer" or contract["dataset"] != str(args.dataset.resolve()):
        raise ValueError("Checkpoint stage/dataset mismatch")
    content_bound = "dataset_snapshot" in contract
    if content_bound:
        require_same_snapshot(contract["dataset_snapshot"], dataset_snapshot(args.dataset))
    if contract.get("geometry_preflight_sha256"):
        if file_sha256(run / "geometry_preflight.json") != contract["geometry_preflight_sha256"]:
            raise ValueError("Checkpoint source geometry preflight changed")
    report = json.loads((args.dataset / "quality_report.json").read_text())
    stats = args.dataset / report.get("stats_folder", "stats")
    digest = hashlib.sha256(b"".join((stats / name).read_bytes() for name in ("mean.npy", "std.npy"))).hexdigest()
    if digest != contract["stats_sha256"]:
        raise ValueError("Checkpoint normalization mismatch")
    rep = build_representation(stats)
    config_path = run / "upstream_config_overrides.yaml"
    if contract.get("config_sha256") and file_sha256(config_path) != contract["config_sha256"]:
        raise ValueError("Checkpoint network/training configuration changed")
    config = OmegaConf.load(config_path)
    network = instantiate(config.model.pose_vqvae_network, motion_rep=rep.dual_rep.local_motion_rep)
    model = BonesTokenizer(pose_vqvae_network=network, root_vqvae_network=None, motion_rep=rep,
                           optimizer=None, scheduler=None, args=OmegaConf.to_container(config.model.args, resolve=True))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    frames = config.model.args.max_tokens * model.get_num_frames_per_code()
    records = [record for record in json.loads((args.dataset / "accepted.json").read_text())
               if record["split"] == args.split and record["frames_60hz"] >= frames + 1]
    if not records:
        raise ValueError("No eligible evaluation records")
    windows, raw_windows, examples, code_counts = [], [], {}, torch.zeros(network.quantizer.num_heads,
                                                       network.quantizer.nb_code_per_head, dtype=torch.long)
    maximum_token_feature_error, maximum_token_fk_error = 0.0, 0.0
    continuous_windows, latent_errors = [], []
    with torch.inference_mode():
        for record in records:
            with np.load(args.dataset / record["shard"], allow_pickle=False) as shard:
                features = torch.from_numpy(shard["global_features"].copy())
                confidence = torch.from_numpy(shard["contact_confidence"].copy()).bool()
            starts = deterministic_window_starts(len(features), frames, args.windows_per_clip)
            for start in starts:
                motion = rep.normalize(features[start:start + frames + 1])[None]
                local = rep.dual_rep.global_to_local(motion, True, True,
                                                     lengths=torch.tensor([frames + 1]))[:, :frames]
                external = extract_feature_from_motion_rep(local, model.local_motion_rep,
                                                           network.decoder_external_cond_feature_mode)
                context_mask = torch.zeros(local.shape[:2], dtype=torch.bool)
                context_mask[:, :4] = True
                output = network(local, target_cond=local * context_mask[..., None] if args.conditioning == "context" else None,
                                     has_target_cond=context_mask if args.conditioning == "context" else None,
                                     external_cond=external)
                prediction = output["recon_state"]
                if not torch.isfinite(prediction).all():
                    raise ValueError("Nonfinite reconstruction")
                metrics, pred, gt = reconstruction_metrics(prediction, local, model.local_motion_rep,
                                                           model.sole_offsets, confidence[start:start + frames][None],
                                                           skip_frames=4 if args.conditioning == "context" else 0)
                if "raw_recon_state" in output:
                    raw_metrics, _, _ = reconstruction_metrics(output["raw_recon_state"], local, model.local_motion_rep,
                                                               model.sole_offsets, confidence[start:start + frames][None],
                                                               skip_frames=4 if args.conditioning == "context" else 0)
                    raw_windows.append({"filename": record["filename"], "start_frame": int(start), "metrics": raw_metrics})
                indices = network.encode_into_idx(local, fetch_overall_indices=False)
                decoded = network.forward_decoder(indices,
                                                   target_cond=local * context_mask[..., None] if args.conditioning == "context" else None,
                                                   has_target_cond=context_mask if args.conditioning == "context" else None,
                                                   external_cond=external, use_overall_indices=False)["recon_state"]
                feature_error = float((decoded - prediction).abs().max())
                decoded_joints = model.local_motion_rep.inverse(decoded, True)["posed_joints"]
                fk_error = float((decoded_joints - pred["posed_joints"]).norm(dim=-1).max())
                if not np.isfinite(feature_error) or not np.isfinite(fk_error) or feature_error > 1e-5 or fk_error > 1e-5:
                    raise ValueError("Official token-decode inference disagrees with training reconstruction")
                maximum_token_feature_error = max(maximum_token_feature_error, feature_error)
                maximum_token_fk_error = max(maximum_token_fk_error, fk_error)
                if args.diagnose_quantization:
                    encoded = network.encoder(network.extract_feature(local, network.encoder_input_feature_mode).permute(0, 2, 1))
                    target_features = (network.extract_feature(local * context_mask[..., None], network.decoder_target_cond_feature_mode)
                                       if args.conditioning == "context" else None)
                    continuous = network.decoder(encoded, external, target_features,
                                                 context_mask if args.conditioning == "context" else None).permute(0, 2, 1)
                    if getattr(network, "root_condition_policy", None):
                        continuous = apply_root_velocity_condition(continuous, external, model.local_motion_rep)
                    continuous_metrics, _, _ = reconstruction_metrics(
                        continuous, local, model.local_motion_rep, model.sole_offsets,
                        confidence[start:start + frames][None], skip_frames=4 if args.conditioning == "context" else 0)
                    continuous_windows.append({"metrics": continuous_metrics})
                    quantized = network.quantizer.dequantize(indices, use_overall_indices=False).permute(0, 2, 1)
                    latent_errors.append(float((encoded - quantized).square().mean()))
                for head in range(len(code_counts)):
                    code_counts[head] += torch.bincount(indices[..., head].flatten(), minlength=code_counts.shape[1])
                window = {"filename": record["filename"], "behavior": record["behavior"],
                          "start_frame": int(start), "metrics": metrics}
                windows.append(window)
                behavior = record["behavior"]
                if behavior not in examples or metrics["mpjpe_m"] > examples[behavior]["error"]:
                    examples[behavior] = {"prediction": pred["posed_joints"][0].numpy(),
                                          "target": gt["posed_joints"][0].numpy(),
                                          "pred_local_rot_mats": pred["local_joint_rots"][0].numpy(),
                                          "source_local_rot_mats": gt["local_joint_rots"][0].numpy(),
                                          "pred_contacts": pred["foot_contacts"][0].numpy(),
                                          "source_contacts": gt["foot_contacts"][0].numpy(),
                                          "source_confidence": confidence[start:start + frames].numpy(),
                                          "error": metrics["mpjpe_m"], "window": window}
    probabilities = code_counts.float() / code_counts.sum(-1, keepdim=True).clamp(min=1)
    protocol_path = run / "experiment_protocol.json"
    numerical_screen = None
    protocol = {}
    if contract.get("protocol_sha256"):
        if file_sha256(protocol_path) != contract["protocol_sha256"]:
            raise ValueError("Frozen experiment protocol changed")
        protocol = json.loads(protocol_path.read_text())
        summary = summarize_windows(windows)
        numerical_screen = evaluate_numerical_screen(summary, protocol.get("numerical_screen"))
    coverage = {split: sorted({record["behavior"] for record in json.loads((args.dataset / "accepted.json").read_text())
                               if record["split"] == split}) for split in ("train", "validation", "test")}
    clusters = network.quantizer.vq._codebook.cluster_size.detach().cpu()
    reset_size = network.quantizer.vq._codebook.reset_cluster_size
    effective_batch = protocol.get("batch_size")
    if effective_batch is not None:
        effective_batch *= config.model.args.batchsize_mul_factor
    result = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_step": checkpoint.get("global_step"),
        "dataset": str(args.dataset.resolve()), "stats_sha256": digest, "split": args.split,
        "checkpoint_sha256": file_sha256(args.checkpoint), "dataset_content_bound": content_bound,
        "dataset_snapshot_sha256": contract.get("dataset_snapshot", {}).get("sha256"),
        "protocol_sha256": contract.get("protocol_sha256"), "numerical_screen": numerical_screen,
        "geometry_preflight_sha256": contract.get("geometry_preflight_sha256"),
        "token_decoder_parity": {"passed": True, "maximum_normalized_feature_error": maximum_token_feature_error,
                                 "maximum_fk_error_m": maximum_token_fk_error, "absolute_tolerance": 1e-5},
        "evaluation_source_sha256": {path.name: file_sha256(path) for path in (
            Path(__file__), Path(__file__).resolve().parents[1] / "motionbricks/data/bones_evaluation.py")},
        "behavior_coverage_by_split": coverage,
        "frames_per_window": frames, "clips": len(records), "windows": len(windows),
        "conditioning": args.conditioning, "observed_context_frames": 4 if args.conditioning == "context" else 0,
        "root_condition_policy": getattr(network, "root_condition_policy", "decoded"),
        "encoder_observes_entire_source_window": True,
        "root_condition_source": "recorded full-window planar velocities; not generated or player-driven",
        "generated_locomotion_evaluated": False,
        "raw_decoder_aggregate": summarize_windows(raw_windows) if raw_windows else None,
        "scored_frames_per_window": frames - (4 if args.conditioning == "context" else 0),
        "aggregate": summarize_windows(windows),
        "per_behavior": {behavior: summarize_windows([window for window in windows if window["behavior"] == behavior])
                         for behavior in sorted({window["behavior"] for window in windows})},
        "codebook_used_per_head": (code_counts > 0).sum(-1).tolist(),
        "codebook_perplexity_per_head": (-(probabilities * probabilities.clamp(min=1e-10).log()).sum(-1)).exp().tolist(),
        "codebook_ema_snapshot": {
            "reset_cluster_size": reset_size,
            "entries_at_reset_size_per_head": (clusters == reset_size).sum(-1).tolist(),
            "mean_cluster_size_per_head": clusters.mean(-1).tolist(),
            "minimum_cluster_size_per_head": clusters.min(-1).values.tolist(),
            "maximum_cluster_size_per_head": clusters.max(-1).values.tolist(),
            "training_vectors_per_head_range": ([effective_batch * config.model.args.min_tokens,
                                                  effective_batch * config.model.args.max_tokens]
                                                 if effective_batch is not None else None),
            "entries_per_head": network.quantizer.nb_code_per_head,
            "note": "An EMA snapshot, not a measured reset rate. Small batches can have fewer assignments than entries.",
        },
        "quantization_diagnostic": ({"continuous_latent_aggregate": summarize_windows(continuous_windows),
                                     "mean_latent_quantization_mse": sum(latent_errors) / len(latent_errors),
                                     "scope": "Unquantized teacher encoder output through the unchanged decoder; diagnostic only, not deployable token inference"}
                                    if args.diagnose_quantization else None),
        "all_windows": windows,
        "review_examples": {behavior: sample["window"] for behavior, sample in examples.items()},
        "manual_contact_ground_truth": False, "quality_approved": False,
        "evaluation_notes": "Equal-window pose aggregate; skating weighted by source stance samples; peaks use maximum. "
                            "Contact scores use automatic labels and confidence masks. "
                            "Oracle-root substitution is diagnostic only. Provided-velocity policy, when enabled, replaces only supplied root-rate channels, "
                            "not height, body rotations or feet; train and token inference share this policy. No IK or foot locking applied. "
                            "Recording-held-out, not actor-held-out. No generated locomotion evaluated.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    for behavior, sample in examples.items():
        np.savez_compressed(args.output / f"{behavior}_review_window.npz",
                            pred_positions_m=sample["prediction"], source_positions_m=sample["target"],
                            pred_local_rot_mats=sample["pred_local_rot_mats"],
                            source_local_rot_mats=sample["source_local_rot_mats"],
                            pred_contacts=sample["pred_contacts"], source_contacts=sample["source_contacts"],
                            source_confidence=sample["source_confidence"], fps=np.array(rep.fps))
    if args.render:
        label = "command-root reconstruction" if getattr(network, "root_condition_policy", None) else "raw reconstruction"
        render_review(examples, rep.skeleton, args.output, label)
    print(json.dumps({key: value for key, value in result.items() if key not in
                     ("all_windows", "review_examples", "per_behavior")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
