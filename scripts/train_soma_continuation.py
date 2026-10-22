"""Bounded native decoder or Pose continuation trials; old bundles remain intact."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_control import state_digest
from motionbricks.data.soma_continuation import decoder_training_loss
from scripts.portable_bones_control import load_bundle


def training_records(records):
    result = [r for r in records if r["split"] == "train" and r["frames_60hz"] >= 129]
    if not result:
        raise ValueError("No complete training clips")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--train-part", choices=("decoder", "pose"), default="decoder")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 2000:
        raise ValueError("Fresh bounded experiment required")
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    rng = np.random.default_rng(2333)
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    snapshot = dataset_snapshot(args.dataset)
    require_same_snapshot(bundle["contracts"]["pose"]["dataset_snapshot"], snapshot)
    gpu = str(torch.cuda.get_device_properties(0).uuid)
    gpu = gpu if gpu.startswith("GPU-") else "GPU-" + gpu
    if gpu not in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","):
        raise ValueError("Training device was not explicitly selected")
    records = training_records(json.loads((args.dataset / "accepted.json").read_text()))
    clips = []
    for record in records:
        with np.load(args.dataset / record["shard"], allow_pickle=False) as data:
            clips.append((data["global_features"].copy(), data["contact_confidence"].copy()))
    pose, rep = inference._pose_model, inference.motion_rep
    frozen = {"root": state_digest(inference._root_model),
              "encoder": state_digest(pose.pose_net.encoder), "quantizer": state_digest(pose.pose_net.quantizer)}
    frozen_part = pose.backbone_net if args.train_part == "decoder" else pose.pose_net.decoder
    frozen["other_part"] = state_digest(frozen_part)
    trained = pose.pose_net.decoder if args.train_part == "decoder" else pose.backbone_net
    trained.requires_grad_(True)
    optimizer = torch.optim.Adam(trained.parameters(), lr=2e-5)
    args.output.mkdir(parents=True)
    contract = {"schema": "motionbricks.soma_continuation_repair.v1", "arm": args.arm, "train_part": args.train_part,
                "dataset_sha256": snapshot["sha256"], "train_clips": len(records), "seed": 2333,
                "initial_bundle_sha256": file_sha256(args.bundle), "steps": args.steps, "batch": 4,
                "learning_rate": 2e-5, "gpu_uuid": gpu, "frozen": frozen, "post_processing": None,
                "history": "four observed source frames; boundary uses actual history, not reconstructed prefix",
                "tokens": "alternate oracle and frozen Pose argmax; future body never conditions Pose",
                "scope": "bounded native continuation adaptation, not a full-training replacement",
                "gradient_estimator": "argmax forward / softmax straight-through backward (biased)" if args.train_part == "pose" else None,
                "contact_geometry_objective": "not changed/reintroduced; evaluate regressions before any promotion",
                "training_source_sha256": {p: file_sha256(Path(p)) for p in
                    ("scripts/train_soma_continuation.py", "motionbricks/data/soma_continuation.py")}}
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    for relative in contract["training_source_sha256"]:
        destination = args.output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(relative).read_bytes())
    sampling = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        # Sample once for motion AND contact confidence; neither arm picks easier windows.
        frames = int(rng.choice((64, 128)))
        chosen = rng.integers(len(clips), size=4)
        starts = [int(rng.integers(len(clips[index][0]) - frames)) for index in chosen]
        sampling.update(np.asarray([frames, *chosen, *starts], dtype=np.int64).tobytes())
        raw = torch.from_numpy(np.stack([clips[index][0][start:start + frames + 1]
                                        for index, start in zip(chosen, starts)])).cuda()
        confidence = torch.from_numpy(np.stack([clips[index][1][start:start + frames]
                                               for index, start in zip(chosen, starts)])).cuda()
        yaw = torch.from_numpy(rng.uniform(0, 2 * np.pi, 4).astype(np.float32)).cuda()
        with torch.no_grad():
            global_motion = rep.change_first_heading(raw, yaw, False, True)
            local = rep.dual_rep.global_to_local(global_motion, True, True,
                        lengths=torch.full((4,), frames + 1, device=raw.device))[:, :frames]
        optimizer.zero_grad(set_to_none=True)
        loss, terms, weights = decoder_training_loss(pose, local, global_motion[:, :frames], confidence,
                         {"enabled": args.arm == "B", "predicted_tokens": step % 2 == 1,
                          "train_pose": args.train_part == "pose"})
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite continuation loss")
        loss.backward()
        parameters = list(trained.parameters())
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("Nonfinite continuation gradient")
        torch.nn.utils.clip_grad_norm_(parameters, .5)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            event = {"step": step + 1, "loss": float(loss.detach()),
                     **{key: float(value.detach()) for key, value in terms.items()}}
            with (args.output / "training.jsonl").open("a") as output:
                output.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event), flush=True)
    final_frozen = {"root": state_digest(inference._root_model), "other_part": state_digest(frozen_part),
                   "encoder": state_digest(pose.pose_net.encoder), "quantizer": state_digest(pose.pose_net.quantizer)}
    if final_frozen != frozen:
        raise RuntimeError("Frozen network/codebook changed")
    adapted = copy.copy(bundle)
    adapted["pose_state"] = {key: value.detach().cpu() for key, value in pose.state_dict().items()}
    adapted["continuation_adaptation"] = {**contract, "sampling_sha256": sampling.hexdigest(), "weights": weights}
    adapted["quality_approved"] = False
    torch.save(adapted, args.output / "backend.pt")
    torch.save({"step": args.steps, "train_part": args.train_part, "model": trained.state_dict(), "optimizer": optimizer.state_dict(),
                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, args.output / "training_state.pt")
    result = {"steps": args.steps, "elapsed_seconds": time.perf_counter() - started,
              "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
              "sampling_sha256": sampling.hexdigest(), "frozen_unchanged": True,
              "bundle_sha256": file_sha256(args.output / "backend.pt"), "quality_approved": False}
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
