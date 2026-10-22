"""Read-only mechanism probes; oracle decompositions are not runtime corrections."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_seed import build_representation, official_sole_surfaces
from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
from motionbricks.data.sole_region_contact import INTERVAL_STATES, sole_region_indices
from motionbricks.data.soma_skin_reference import load_official_skinner
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.evaluate_sole_region_ab import build_model, evaluation_windows, motion_metrics, source_labels


def contact_velocity_decomposition(predicted, target, measured, fps):
    """Split measured velocity error into common translation and shape change.

    Orthogonality gives total squared error = common + residual. The hypothetical
    common correction uses source contacts and is not available to a controller.
    A large common component does not identify root prediction as its cause.
    """
    error = ((predicted[1:] - predicted[:-1]) - (target[1:] - target[:-1])) * fps
    mask = measured.detach().bool()
    weights = mask.to(error.dtype)
    count = weights.sum(-1)
    mean = (error * weights[..., None]).sum(-2) / count.clamp_min(1)[..., None]
    residual = error - mean[:, None]
    total = (error.square().sum(-1) * weights).sum()
    common = (mean.square().sum(-1) * count).sum()
    remaining = (residual.square().sum(-1) * weights).sum()
    drift = torch.cat([torch.zeros_like(mean[:1]), (-mean / fps).cumsum(0)])
    return {
        "measured_vertices": int(mask.sum()), "squared_error_sum_m2ps2": float(total),
        "common_squared_error_sum_m2ps2": float(common),
        "residual_squared_error_sum_m2ps2": float(remaining),
        "orthogonality_error": float((total - common - remaining).abs()),
        "unconstrained_oracle_root_offset_peak_m": float(drift.norm(dim=-1).max()),
        "unconstrained_oracle_root_offset_endpoint_m": float(drift[-1].norm()),
    }


def selected_vertices(pose, skinner, indices, root_idx):
    height, horizontal = region_geometry_from_motion(
        pose["local_joint_rots"][0], pose["posed_joints"][0, :, root_idx], skinner, indices)
    return torch.stack([horizontal[..., 0], height, horizontal[..., 1]], -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise ValueError("Use a new output file")
    report = json.loads((args.dataset / "quality_report.json").read_text())
    rep = build_representation(args.dataset / report["stats_folder"])
    snapshot = dataset_snapshot(args.dataset)
    require_same_snapshot(json.loads((args.run / "run_contract.json").read_text())["dataset_snapshot"], snapshot)
    model, network = build_model(args.run, rep)
    skinner = load_official_skinner(rep.skeleton)
    surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
    indices = {side: sole_region_indices(surfaces, side) for side in ("Left", "Right")}
    all_indices = np.concatenate(list(indices.values()))
    widths = [len(x) for x in indices.values()]
    compact_indices = {"Left": np.arange(widths[0]), "Right": np.arange(widths[0], sum(widths))}
    records = {r["filename"]: r for r in json.loads((args.dataset / "accepted.json").read_text())}
    root_idx = rep.skeleton.root_idx
    rows = []
    with torch.inference_mode():
        for behavior, filename, start in evaluation_windows(records, True):
            with np.load(args.dataset / records[filename]["shard"], allow_pickle=False) as shard:
                features = torch.from_numpy(shard["global_features"][start:start + 129].copy())
            global_motion = rep.normalize(features)[None]
            local = rep.dual_rep.global_to_local(global_motion, True, True, lengths=torch.tensor([129]))[:, :128]
            source = model.local_motion_rep.inverse(local, True, return_all=True)
            vertices_source = selected_vertices(source, skinner, all_indices, root_idx)
            labels, _ = source_labels(vertices_source, compact_indices, rep.fps)
            masks = []
            for side in ("Left", "Right"):
                label = labels[side.lower()]
                valid = (label["interval_state"] == INTERVAL_STATES.index("stable_contact")) & label["measured_interval"]
                valid[:4] = False
                masks.append(label["vertex_measured"] & valid[:, None])
            measured = torch.cat(masks, -1)
            context = torch.zeros(local.shape[:2], dtype=torch.bool)
            context[:, :4] = True
            target = local * context[..., None]
            external = network.extract_feature(local, network.decoder_external_cond_feature_mode)
            encoded = network.encoder(network.extract_feature(local, network.encoder_input_feature_mode).permute(0, 2, 1))
            quantized, _, perplexity = network.quantizer(encoded)
            rows_for_modes = {}
            for name, latent in (("quantized", quantized), ("continuous_probe", encoded)):
                decoded = network.decoder(latent, external,
                                          network.extract_feature(target, network.decoder_target_cond_feature_mode), context)
                state = decoded.permute(0, 2, 1)
                if hasattr(network, "_apply_condition"):
                    state = network._apply_condition({"recon_state": state}, external)["recon_state"]
                pose = model.local_motion_rep.inverse(state, True, return_all=True)
                points = pose["posed_joints"][0]
                vertices = selected_vertices(pose, skinner, all_indices, root_idx)
                metrics = motion_metrics(pose["local_joint_rots"][0], points[:, root_idx],
                                         skinner, indices, labels, rep.fps)
                metrics["mpjpe_m"] = float((points[4:] - source["posed_joints"][0, 4:]).norm(dim=-1).mean())
                metrics["fk_vs_position_feature_m"] = float(
                    (points[4:] - pose["all_posed_joints"]["ric_data"][0, 4:]).norm(dim=-1).mean())
                metrics["contact_error_decomposition"] = contact_velocity_decomposition(
                    vertices[..., [0, 2]], vertices_source[..., [0, 2]], measured, rep.fps)
                velocity_error = ((points[1:] - points[:-1]) -
                                  (source["posed_joints"][0, 1:] - source["posed_joints"][0, :-1])) * rep.fps
                metrics["velocity_error_by_token_interval_mps"] = [
                    float(velocity_error[4:][torch.arange(4, 127) % 4 == phase].norm(dim=-1).mean())
                    for phase in range(4)]
                rows_for_modes[name] = metrics
            global_target = rep.dual_rep.global_motion_rep.inverse(
                global_motion[:, :128], True, joint_positions_from="ric_data")["posed_joints"]
            oracle_velocity = ((global_target[:, 1:] - global_target[:, :-1]) -
                               (source["posed_joints"][:, 1:] - source["posed_joints"][:, :-1])) * rep.fps
            rows.append({"filename": filename, "behavior": behavior, "start": start,
                         "modes": rows_for_modes, "perplexity": float(perplexity),
                         "oracle_target_velocity_disagreement_mps": float(oracle_velocity[:, 4:].norm(dim=-1).mean())})
    aggregate = {}
    for mode in ("quantized", "continuous_probe"):
        entries = [r["modes"][mode] for r in rows]
        slip = [e for e in entries if e["measured_slip_mps"] is not None]
        n = sum(e["valid_samples"]["measured_stable_intervals"] for e in slip)
        energy = sum(e["contact_error_decomposition"]["squared_error_sum_m2ps2"] for e in entries)
        aggregate[mode] = {
            "mpjpe_m": float(np.mean([e["mpjpe_m"] for e in entries])),
            "slip_mps": sum(e["measured_slip_mps"]["mean"] * e["valid_samples"]["measured_stable_intervals"] for e in slip) / n if n else None,
            "common_translation_energy_fraction": sum(e["contact_error_decomposition"]["common_squared_error_sum_m2ps2"] for e in entries) / energy if energy else None,
            "fk_vs_position_feature_m": float(np.mean([e["fk_vs_position_feature_m"] for e in entries])),
            "velocity_error_by_token_interval_mps": np.mean([e["velocity_error_by_token_interval_mps"] for e in entries], 0).tolist(),
        }
    output = {
        "schema": "motionbricks.tokenizer_mechanism_probe.v1", "windows": len(rows),
        "dataset_sha256": snapshot["sha256"], "checkpoint_sha256": file_sha256(args.run / "checkpoints/final.ckpt"),
        "script_sha256": file_sha256(Path(__file__)), "aggregate": aggregate,
        "oracle_target_velocity_disagreement_mean_mps": float(np.mean([r["oracle_target_velocity_disagreement_mps"] for r in rows])),
        "mean_perplexity": float(np.mean([r["perplexity"] for r in rows])),
        "limitations": ["The continuous-latent decoder sees out-of-training-distribution inputs; this is a probe, not a deployable baseline.",
                        "Common translation is an unconstrained oracle decomposition using source labels, not proof that root authority caused the error.",
                        "No training or runtime correction is performed. All scored intervals retain the source masks."],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
