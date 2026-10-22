"""Read-only parent/child tokenizer and Pose token-semantics audit."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REMOTE_ROOT = Path("${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks")
STAIR_REPORT = Path("out/native_mb_stairs_20261007/review-ft5000-v1/report.json")
FLAT_REPORT = Path("out/native_mb_stairs_20261007/flat-regression-ft5000-v2/report.json")
DATASET = Path("${PRIVATE_STORAGE}/motionbricks-bones/native-stairs-50k-v3")
PARENT_TOKENIZER = Path("out/native_mb_lafan_paper_20261006/stage50k-v1/tokenizer-50000")
CHILD_TOKENIZER = Path("out/native_mb_stairs_20261007/ft5000-v1/tokenizer")
PARENT_ROOT = Path("out/native_mb_lafan_paper_20261006/stage50k-v1/root-50000")
CHILD_ROOT = Path("out/native_mb_stairs_20261007/ft5000-v1/root")
PARENT_POSE = Path("out/native_mb_lafan_paper_20261006/pose-efficiency-v1/training/pose-50000")
CHILD_POSE = Path("out/native_mb_stairs_20261007/ft5000-v1/pose")
OUTPUT = Path("out/native_mb_stairs_20261007/token-semantic-audit-v2")
POSE_KEYS = {
    "input_embedding": "backbone_net._pose_token_emb.weight",
    "output_weight": "backbone_net._proj_pose_output_logit.weight",
    "output_bias": "backbone_net._proj_pose_output_logit.bias",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report_windows(stair_report, flat_report):
    stairs = json.loads(Path(stair_report).read_text(encoding="utf-8"))
    flats = json.loads(Path(flat_report).read_text(encoding="utf-8"))
    stair_rows = []
    for row in stairs["windows"]:
        source = row["source"]
        stair_rows.append({
            "group": "stairs",
            "behavior": "up" if source["direction"] == "up" else "down",
            "shard": source["shard"],
            "start": int(source["start"]),
            "split": source["split"],
            "filename": source["filename"],
            "source_sha256": source["source_sha256"],
        })
    flat_rows = [{
        "group": "flat",
        "behavior": row["behavior"],
        "shard": row["shard"],
        "start": int(row["start"]),
        "split": row["split"],
        "filename": row["filename"],
        "source_shard_sha256": row["source_shard_sha256"],
    } for row in flats["windows"]]
    validate_window_rows(stair_rows, flat_rows)
    return stair_rows + flat_rows


def validate_window_rows(stairs, flats):
    identities = [(row["group"], row["shard"], row["start"]) for row in stairs + flats]
    if len(set(identities)) != 12:
        raise ValueError("Selected source windows must be unique")
    if (len(stairs) != 6 or len(flats) != 6
            or sum(row.get("behavior") == "up" for row in stairs) != 3
            or sum(row.get("behavior") == "down" for row in stairs) != 3
            or sum(row.get("behavior") == "run" for row in flats) != 3
            or sum(row.get("behavior") == "walk" for row in flats) != 3):
        raise ValueError("Expected report-bound 3 up/3 down stairs and 3 run/3 walk flat windows")
    if any(row.get("split") not in (None, "validation") for row in stairs + flats):
        raise ValueError("All comparison windows must be held out")


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def compare_codebooks(parent, child):
    parent = np.asarray(parent, dtype=np.float64)
    child = np.asarray(child, dtype=np.float64)
    if parent.shape != child.shape or parent.ndim != 3:
        raise ValueError("Expected matching [head, code, dimension] codebooks")
    if not np.isfinite(parent).all() or not np.isfinite(child).all():
        raise ValueError("Codebooks must be finite")
    pnorm = np.linalg.norm(parent, axis=-1)
    cnorm = np.linalg.norm(child, axis=-1)
    denom = np.maximum(pnorm * cnorm, 1e-30)
    cosine = np.sum(parent * child, axis=-1) / denom
    distance = np.linalg.norm(child - parent, axis=-1)
    relative = distance / np.maximum(pnorm, 1e-30)
    rows = []
    for head in range(parent.shape[0]):
        old = parent[head]
        new = child[head]
        old_unit = old / np.maximum(np.linalg.norm(old, axis=-1, keepdims=True), 1e-30)
        new_unit = new / np.maximum(np.linalg.norm(new, axis=-1, keepdims=True), 1e-30)
        nearest_cosine = (new_unit @ old_unit.T).argmax(axis=1)
        nearest_l2 = np.linalg.norm(new[:, None, :] - old[None, :, :], axis=-1).argmin(axis=1)
        rows.append({
            "head": head,
            "same_id_cosine": _summary(cosine[head]),
            "same_id_l2": _summary(distance[head]),
            "same_id_relative_l2": _summary(relative[head]),
            "nearest_old_by_cosine_same_id_fraction": float(np.mean(nearest_cosine == np.arange(parent.shape[1]))),
            "nearest_old_by_cosine_old_id_counts": np.bincount(nearest_cosine, minlength=parent.shape[1]).tolist(),
            "nearest_old_by_l2_same_id_fraction": float(np.mean(nearest_l2 == np.arange(parent.shape[1]))),
            "nearest_old_by_l2_old_id_counts": np.bincount(nearest_l2, minlength=parent.shape[1]).tolist(),
        })
    return {"shape_head_code_dim": list(parent.shape), "heads": rows,
            "same_id_cosine_all": _summary(cosine),
            "same_id_l2_all": _summary(distance),
            "same_id_relative_l2_all": _summary(relative)}


def compare_token_ids(parent, child, windows):
    parent, child = np.asarray(parent), np.asarray(child)
    if parent.shape != child.shape or parent.ndim != 3:
        raise ValueError("Expected matching [window, token, head] token arrays")
    if parent.shape[0] != len(windows):
        raise ValueError("Token windows do not align with the selected window manifest")
    if not np.issubdtype(parent.dtype, np.integer) or not np.issubdtype(child.dtype, np.integer):
        raise ValueError("Token IDs must be integer arrays")
    equal = parent == child
    result = {"shape_window_token_head": list(parent.shape),
              "equal_id_fraction_per_head": equal.mean(axis=(0, 1)).tolist(),
              "equal_id_fraction_all": float(equal.mean()), "windows": []}
    for i, row in enumerate(windows):
        result["windows"].append({
            "group": row["group"], "behavior": row["behavior"],
            "shard": row["shard"], "start": row["start"],
            "equal_id_fraction_per_head": equal[i].mean(axis=0).tolist(),
            "equal_id_fraction_all": float(equal[i].mean()),
            "changed_id_counts_per_head": (parent[i] != child[i]).sum(axis=0).astype(int).tolist(),
        })
    for group in ("stairs", "flat"):
        mask = np.asarray([row["group"] == group for row in windows])
        result[f"{group}_equal_id_fraction_per_head"] = equal[mask].mean(axis=(0, 1)).tolist()
        result[f"{group}_equal_id_fraction_all"] = float(equal[mask].mean())
    return result


def checkpoint_runtime(saved):
    groups = []
    for optimizer in saved.get("optimizer_states", []):
        groups.append([{"lr": float(group["lr"]), "initial_lr": (
            float(group["initial_lr"]) if "initial_lr" in group else None)}
            for group in optimizer.get("param_groups", [])])
    schedulers = []
    for scheduler in saved.get("lr_schedulers", saved.get("lr_scheduler_states", [])):
        if isinstance(scheduler, dict):
            schedulers.append({key: scheduler[key] for key in
                               ("last_epoch", "_step_count", "_last_lr", "base_lrs") if key in scheduler})
    return {"global_step": saved.get("global_step"), "optimizer_learning_rates": groups,
            "scheduler_state": schedulers,
            "optimizer_state_present": bool(saved.get("optimizer_states")),
            "scheduler_state_present": bool(saved.get("lr_schedulers", saved.get("lr_scheduler_states")))}


def tokenizer_keyframe_curriculum(config, global_step):
    args = config["model"]["args"]
    warmup = int(args["keyframe_num_warmup_steps"])
    if warmup < 1 or global_step < 0:
        raise ValueError("Invalid Tokenizer keyframe schedule position")
    result = {"global_step": int(global_step), "warmup_steps": warmup, "modules": {}}
    for module in ("root", "pose"):
        maximum = int(args[f"{module}_vqvae_max_num_keyframes"])
        no_probability = float(args[f"{module}_vqvae_no_keyframe_prob"])
        if maximum < 1 or not 0 <= no_probability < 1:
            raise ValueError("Invalid Tokenizer keyframe curriculum config")
        scheduled_maximum = int(max(1, min(maximum, maximum * global_step / warmup)))
        probabilities = [no_probability] + [
            (1 - no_probability) / scheduled_maximum if count <= scheduled_maximum else 0.0
            for count in range(1, maximum + 1)
        ]
        result["modules"][module] = {
            "scheduled_maximum": scheduled_maximum,
            "count_probabilities": probabilities,
            "expected_target_keyframes_per_sample": float(sum(
                count * probability for count, probability in enumerate(probabilities))),
            "configured_no_keyframe_probability": no_probability,
        }
    return result


def backbone_keyframe_curriculum(config, stage, global_step):
    args = config["model"]["args"]
    warmup = int(args["keyframe_num_warmup_steps"])
    if warmup < 1 or global_step < 0:
        raise ValueError("Invalid backbone keyframe schedule position")
    result = {"stage": stage, "global_step": int(global_step), "warmup_steps": warmup,
              "components": {}}
    for component in (("start", "end") if stage == "root" else ("start", "end", "middle")):
        maximum = int(args[f"max_num_{component}_keyframes"])
        no_probability = float(args[f"no_{component}_keyframe_prob"])
        scheduled = int(max(1, min(maximum, maximum * global_step / warmup)))
        probabilities = [no_probability] + [
            (1 - no_probability) / scheduled if count <= scheduled else 0.0
            for count in range(1, maximum + 1)
        ]
        result["components"][component] = {
            "scheduled_maximum": scheduled,
            "count_probabilities": probabilities,
            "expected_keyframes": float(sum(count * probability
                                             for count, probability in enumerate(probabilities))),
        }
    return result


def checkpoint_state_report(run_dir, saved):
    run_dir = Path(run_dir)
    training = json.loads((run_dir / "training_result.json").read_text(encoding="utf-8"))
    runtime = checkpoint_runtime(saved)
    runtime.update({
        "checkpoint_sha256": sha256(run_dir / "checkpoints/final.ckpt"),
        "training_result_optimizer_updates": training.get("optimizer_updates"),
        "checkpoint_step_matches_training_result": saved.get("global_step") == training.get("optimizer_updates"),
    })
    return runtime


def _tensor_summary(parent, child):
    a = np.asarray(parent.detach().cpu().numpy() if hasattr(parent, "detach") else parent,
                   dtype=np.float64).reshape(-1)
    b = np.asarray(child.detach().cpu().numpy() if hasattr(child, "detach") else child,
                   dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return {"parent_shape": list(parent.shape), "child_shape": list(child.shape),
                "shape_match": False}
    delta = b - a
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cosine = np.dot(a, b) / max(na * nb, 1e-30)
    exact_equal = np.array_equal(
        parent.detach().cpu().numpy() if hasattr(parent, "detach") else parent,
        child.detach().cpu().numpy() if hasattr(child, "detach") else child)
    return {"shape": list(parent.shape), "shape_match": True,
            "exact_equal": bool(exact_equal),
            "delta_l2": float(np.linalg.norm(delta)),
            "delta_l2_relative_to_parent": float(np.linalg.norm(delta) / max(na, 1e-30)),
            "parent_l2": float(na), "child_l2": float(nb), "flattened_cosine": float(cosine)}


def compare_pose_checkpoints(parent_saved, child_saved):
    parent_state, child_state = parent_saved.get("state_dict", {}), child_saved.get("state_dict", {})
    result = {"parent_runtime": checkpoint_runtime(parent_saved),
              "child_runtime": checkpoint_runtime(child_saved), "tensors": {}}
    for name, key in POSE_KEYS.items():
        if key not in parent_state or key not in child_state:
            raise ValueError(f"Pose checkpoint is missing expected tensor {key}")
        result["tensors"][name] = _tensor_summary(parent_state[key], child_state[key])
    return result


def compare_embedding_codebook_alignment(saved, codebook):
    state = saved.get("state_dict", {})
    key = POSE_KEYS["input_embedding"]
    if key not in state:
        raise ValueError(f"Pose checkpoint is missing expected tensor {key}")
    tensor = state[key]
    embedding = np.asarray(tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor,
                           dtype=np.float64)
    codebook = np.asarray(codebook, dtype=np.float64)
    heads, codes, dim = codebook.shape
    stride = codes + 1
    if embedding.shape != (heads * stride, dim):
        raise ValueError("Pose input table shape does not match tokenizer head/code dimensions")
    result = []
    for head in range(heads):
        rows = embedding[head * stride:head * stride + codes]
        vectors = codebook[head]
        cosine = np.sum(rows * vectors, axis=-1) / np.maximum(
            np.linalg.norm(rows, axis=-1) * np.linalg.norm(vectors, axis=-1), 1e-30)
        distance = np.linalg.norm(rows - vectors, axis=-1)
        result.append({"head": head, "same_id_cosine": _summary(cosine),
                       "same_id_l2": _summary(distance),
                       "mask_embedding_l2": float(np.linalg.norm(embedding[head * stride + codes]))})
    return {"mapping": "Pose input row head*(num_codes+1)+code_id compared to same-ID tokenizer row; no remapping",
            "heads": result}


def _pose_run_binding(run_dir, tokenizer_info, tokenizer_state, saved):
    run_dir = Path(run_dir)
    contract = json.loads((run_dir / "run_contract.json").read_text(encoding="utf-8"))
    training = json.loads((run_dir / "training_result.json").read_text(encoding="utf-8"))
    dependency = contract.get("tokenizer") or {}
    recorded_state = contract.get("tokenizer_state_sha256")
    result_state = training.get("tokenizer_state_sha256")
    return {
        "tokenizer_checkpoint_sha256_recorded": dependency.get("checkpoint_sha256"),
        "tokenizer_checkpoint_sha256_loaded": tokenizer_info.get("checkpoint_sha256"),
        "tokenizer_checkpoint_hash_matches": dependency.get("checkpoint_sha256") == tokenizer_info.get("checkpoint_sha256"),
        "tokenizer_tensor_state_sha256_loaded": tokenizer_state,
        "tokenizer_tensor_state_sha256_contract": recorded_state,
        "tokenizer_tensor_state_sha256_training_result": result_state,
        "tokenizer_tensor_state_matches_contract": tokenizer_state == recorded_state,
        "tokenizer_tensor_state_matches_training_result": tokenizer_state == result_state,
        "checkpoint_global_step": saved.get("global_step"),
        "training_result_optimizer_updates": training.get("optimizer_updates"),
        "pose_update_count_matches_checkpoint": saved.get("global_step") == training.get("optimizer_updates"),
    }


def _manifest_window_digest(dataset, row, artifacts):
    shard = row["shard"]
    expected = artifacts.get(shard)
    path = (dataset / shard).resolve()
    if expected is None or dataset.resolve() not in path.parents or sha256(path) != expected:
        raise ValueError(f"Selected shard is missing or not bound by the dataset manifest: {shard}")
    return expected


def run_audit(args):
    import torch
    from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint
    from scripts.review_native_mb_lafan_system import state_digest
    from scripts.train_native_mb import select_gpu

    gpu = select_gpu(args.gpu)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing audit output: {output}")
    dataset = Path(args.dataset).resolve()
    manifest_path = dataset / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != "36e82bba18447ebfe83576343aa368160dfd32a18a2b53c2da059ab70834fc9c":
        raise ValueError("Stair dataset manifest differs from the authorized audit input")
    windows = _report_windows(args.stair_report, args.flat_report)
    artifact_hashes = manifest.get("artifact_sha256", {})
    for name in ("mean.npy", "std.npy"):
        if sha256(dataset / "stats" / name) != artifact_hashes.get(f"stats/{name}"):
            raise ValueError(f"Dataset manifest does not bind stats/{name}")
    shard_hashes = {f"{row['group']}:{row['shard']}:{row['start']}":
                    _manifest_window_digest(dataset, row, artifact_hashes) for row in windows}

    torch.set_num_threads(4)
    torch.manual_seed(2333)
    np.random.seed(2333)
    # Each tokenizer is loaded in turn so only one model occupies GPU1 at a time.
    parent, parent_info, parent_binding = load_native_checkpoint(args.parent_tokenizer, device="cuda")
    if (parent_binding["dataset_dir"] / "stats/mean.npy").read_bytes() != (dataset / "stats/mean.npy").read_bytes() or (
            parent_binding["dataset_dir"] / "stats/std.npy").read_bytes() != (dataset / "stats/std.npy").read_bytes():
        raise ValueError("Parent tokenizer normalization differs from stair/flat comparison data")
    parent_codebook = parent.pose_net.get_codebook().detach().cpu().numpy().copy()
    parent_rep = parent.motion_rep
    local_by_window, ids_parent = [], []
    for row in windows:
        with np.load(dataset / row["shard"], allow_pickle=False) as archive:
            raw = archive["global_features"][row["start"]:row["start"] + 65].copy()
        if raw.shape[0] != 65 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid 65-frame source window: {row}")
        raw_tensor = torch.from_numpy(raw).to("cuda")[None]
        canonical = parent_rep.change_first_heading(raw_tensor, 0.0, False, False)
        normalized = parent_rep.normalize(canonical)
        local = parent_rep.dual_rep.global_to_local(
            normalized, True, True, lengths=torch.tensor([65], device="cuda"))[:, :64]
        with torch.inference_mode():
            ids = parent.pose_net.encode_into_idx(local, fetch_overall_indices=False)
        local_by_window.append(local.detach().cpu())
        ids_parent.append(ids.detach().cpu().numpy()[0])
    parent_identity = {"checkpoint": parent_info, "dataset_manifest_sha256": parent_binding["dataset_manifest_sha256"],
                       "tensor_state_sha256": state_digest(parent.pose_net)}
    del parent
    torch.cuda.empty_cache()

    child, child_info, child_binding = load_native_checkpoint(args.child_tokenizer, device="cuda")
    if child_binding["dataset_dir"].resolve() != dataset:
        raise ValueError("Child tokenizer is not bound to the selected stair/flat dataset")
    if (parent_binding["dataset_dir"] / "stats/mean.npy").read_bytes() != (dataset / "stats/mean.npy").read_bytes() or (
            parent_binding["dataset_dir"] / "stats/std.npy").read_bytes() != (dataset / "stats/std.npy").read_bytes():
        raise ValueError("Parent tokenizer normalization changed during audit")
    child_codebook = child.pose_net.get_codebook().detach().cpu().numpy().copy()
    ids_child = []
    for local in local_by_window:
        with torch.inference_mode():
            ids_child.append(child.pose_net.encode_into_idx(local.to("cuda"), fetch_overall_indices=False).cpu().numpy()[0])
    child_identity = {"checkpoint": child_info, "dataset_manifest_sha256": child_binding["dataset_manifest_sha256"],
                      "tensor_state_sha256": state_digest(child.pose_net)}
    tokenizer_state_change = parent_identity["tensor_state_sha256"] != child_identity["tensor_state_sha256"]
    token_comparison = compare_token_ids(np.stack(ids_parent), np.stack(ids_child), windows)
    codebook_comparison = compare_codebooks(parent_codebook, child_codebook)
    # Check the exact same recorded windows reconstruct with both shipped code paths.
    reconstruction_check = {"windows": len(local_by_window), "source_global_frames": 65,
                            "canonical_local_frames": 64, "all_windows_encoded_by_both": True,
                            "parent_child_tokenizer_state_changed": tokenizer_state_change}
    del child
    torch.cuda.empty_cache()

    parent_pose_saved = torch.load(Path(args.parent_pose) / "checkpoints/final.ckpt",
                                   map_location="cpu", weights_only=False)
    child_pose_saved = torch.load(Path(args.child_pose) / "checkpoints/final.ckpt",
                                  map_location="cpu", weights_only=False)
    parent_root_saved = torch.load(Path(args.parent_root) / "checkpoints/final.ckpt",
                                   map_location="cpu", weights_only=False)
    child_root_saved = torch.load(Path(args.child_root) / "checkpoints/final.ckpt",
                                  map_location="cpu", weights_only=False)
    pose_comparison = compare_pose_checkpoints(parent_pose_saved, child_pose_saved)
    pose_comparison["parent_embedding_to_parent_codebook"] = compare_embedding_codebook_alignment(
        parent_pose_saved, parent_codebook)
    pose_comparison["parent_embedding_to_child_codebook"] = compare_embedding_codebook_alignment(
        parent_pose_saved, child_codebook)
    pose_comparison["child_embedding_to_parent_codebook"] = compare_embedding_codebook_alignment(
        child_pose_saved, parent_codebook)
    pose_comparison["child_embedding_to_child_codebook"] = compare_embedding_codebook_alignment(
        child_pose_saved, child_codebook)
    # The loader identity is the tensor digest of the exact tokenizer module used by Pose.
    parent_for_digest, parent_pose_tokenizer_info, _ = load_native_checkpoint(args.parent_tokenizer, device="cpu")
    child_for_digest, child_pose_tokenizer_info, _ = load_native_checkpoint(args.child_tokenizer, device="cpu")
    parent_pose_identity = _pose_run_binding(args.parent_pose, parent_pose_tokenizer_info,
                                             state_digest(parent_for_digest.pose_net), parent_pose_saved)
    child_pose_identity = _pose_run_binding(args.child_pose, child_pose_tokenizer_info,
                                            state_digest(child_for_digest.pose_net), child_pose_saved)
    import yaml
    parent_config = yaml.safe_load(parent_binding["config_path"].read_text(encoding="utf-8"))
    child_config = yaml.safe_load(child_binding["config_path"].read_text(encoding="utf-8"))
    parent_root_config_path = Path(args.parent_root) / "native_config.yaml"
    child_root_config_path = Path(args.child_root) / "native_config.yaml"
    parent_root_config = yaml.safe_load(parent_root_config_path.read_text(encoding="utf-8"))
    child_root_config = yaml.safe_load(child_root_config_path.read_text(encoding="utf-8"))
    parent_pose_config = yaml.safe_load((Path(args.parent_pose) / "native_config.yaml").read_text(encoding="utf-8"))
    child_pose_config = yaml.safe_load((Path(args.child_pose) / "native_config.yaml").read_text(encoding="utf-8"))
    parent_tokenizer_schedule = tokenizer_keyframe_curriculum(parent_config, parent_info["global_step"])
    child_tokenizer_schedule = tokenizer_keyframe_curriculum(child_config, child_info["global_step"])
    model_curricula = {
        "parent_root": backbone_keyframe_curriculum(
            parent_root_config, "root", parent_root_saved.get("global_step")),
        "child_root": backbone_keyframe_curriculum(
            child_root_config, "root", child_root_saved.get("global_step")),
        "parent_pose": backbone_keyframe_curriculum(
            parent_pose_config, "pose", parent_pose_saved.get("global_step")),
        "child_pose": backbone_keyframe_curriculum(
            child_pose_config, "pose", child_pose_saved.get("global_step")),
    }

    output.mkdir(parents=True, exist_ok=False)
    result = {
        "task": "read-only parent/child tokenizer and Pose token-semantics audit",
        "device": gpu,
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(manifest_path),
        "normalization_sha256": {name: sha256(dataset / "stats" / name) for name in ("mean.npy", "std.npy")},
        "windows": [{**row, "shard_sha256": shard_hashes[
            f"{row['group']}:{row['shard']}:{row['start']}" ]} for row in windows],
        "tokenizers": {"parent": parent_identity, "child": child_identity,
                       "tokenizer_state_changed": tokenizer_state_change,
                       "training_keyframe_curriculum": {"parent_at_saved_step": parent_tokenizer_schedule,
                                                         "child_at_saved_step": child_tokenizer_schedule},
                       "token_id_comparison_no_remapping": token_comparison,
                       "codebook_rows_no_remapping": codebook_comparison,
                       "conversion": reconstruction_check},
        "pose_checkpoints": {"parent": {"checkpoint_sha256": sha256(Path(args.parent_pose) / "checkpoints/final.ckpt"),
                                          "frozen_tokenizer_binding": parent_pose_identity},
                             "child": {"checkpoint_sha256": sha256(Path(args.child_pose) / "checkpoints/final.ckpt"),
                                         "frozen_tokenizer_binding": child_pose_identity},
                             "parent_child_tensors": pose_comparison},
        "root_checkpoints": {"parent": checkpoint_state_report(args.parent_root, parent_root_saved),
                              "child": checkpoint_state_report(args.child_root, child_root_saved)},
        "model_keyframe_curricula": model_curricula,
        "interpretation_limit": "Numeric changes establish changed tensors/assignments only; this audit does not attribute cause or infer model-quality effects.",
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--stair-report", type=Path, default=STAIR_REPORT)
    parser.add_argument("--flat-report", type=Path, default=FLAT_REPORT)
    parser.add_argument("--parent-tokenizer", type=Path, default=PARENT_TOKENIZER)
    parser.add_argument("--child-tokenizer", type=Path, default=CHILD_TOKENIZER)
    parser.add_argument("--parent-root", type=Path, default=PARENT_ROOT)
    parser.add_argument("--child-root", type=Path, default=CHILD_ROOT)
    parser.add_argument("--parent-pose", type=Path, default=PARENT_POSE)
    parser.add_argument("--child-pose", type=Path, default=CHILD_POSE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_audit(args)
    print(json.dumps({"output": str(Path(args.output) / "metrics.json"),
                      "gpu": report["device"],
                      "all_token_id_agreement": report["tokenizers"]["token_id_comparison_no_remapping"]["equal_id_fraction_all"],
                      "same_id_codebook_cosine": report["tokenizers"]["codebook_rows_no_remapping"]["same_id_cosine_all"],
                      "pose_tensor_deltas": report["pose_checkpoints"]["parent_child_tensors"]["tensors"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
