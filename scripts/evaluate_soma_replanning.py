"""Paired, raw 60-fps replanning ablation; no weights or output poses changed."""

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_rollout import SomaLocomotionBackend
from motionbricks.data.bones_seed import validation_window_records
from motionbricks.data.soma_primitive_scheduler import NativePrimitiveScheduler
from scripts.diagnose_soma_continuity import motion_metrics, stats
from scripts.evaluate_soma_decoder_refinement import command_at
from scripts.portable_bones_control import load_bundle


@torch.inference_mode()
def evaluate(inference, initial, interval, frames=3600):
    backend = SomaLocomotionBackend(inference, initial, argmax=True, commit_frames=8)
    scheduler = NativePrimitiveScheduler(backend, interval)
    chunks, requests, times = [], [], []
    for frame in range(0, frames, 8):
        name, velocity, heading = command_at(frame)
        result = scheduler.step([velocity], [heading])
        chunks.append(result["motion"])
        if result["replanned"]:
            requests.append({"frame": frame, "reason": result["replan_reason"], "command": name})
            times.append(result["generation_seconds"] * 1000)
    motion = torch.cat(chunks, 1)
    boundaries = [request["frame"] for request in requests if request["frame"]]
    measured = motion_metrics(motion, inference.motion_rep, boundaries)
    measured["published_block_boundaries"] = motion_metrics(motion, inference.motion_rep, range(8, frames, 8))["boundary_angular_step_deg"]
    speed = motion[0, :, [0, 2]].diff(dim=0) * inference.motion_rep.fps
    requested = speed.new_tensor([command_at(frame)[1] for frame in range(frames - 1)])
    errors = (speed - requested).norm(dim=-1)
    measured["root_command_velocity_error_mps"] = stats(errors)
    measured["per_command"] = {command_at(start)[0]: {
        "velocity_error_mps": stats(errors[start:min(start + 299, len(errors))]),
        "speed_mps": stats(speed[start:min(start + 299, len(speed))].norm(dim=-1))}
        for start in range(0, frames, 300)}
    measured["model_generation_ms"] = stats(times)
    measured["model_calls"] = scheduler.model_calls
    measured["replan_requests"] = requests
    return motion, boundaries, measured


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh ablation directory")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True)
    selected = None
    for row in inventory.strip().splitlines():
        index, uuid, memory, utilization = [value.strip() for value in row.split(",")]
        if int(index) == args.physical_gpu:
            if int(memory) > 1024 or int(utilization) > 5:
                raise RuntimeError("The requested evaluation GPU is not idle at launch")
            selected = uuid
    if selected is None:
        raise ValueError("Requested GPU does not exist")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    if inference.motion_rep.fps != 60:
        raise ValueError("This ablation contract is 60 fps")
    snapshot = dataset_snapshot(args.dataset)
    require_same_snapshot(bundle["contracts"]["pose"]["dataset_snapshot"], snapshot)
    records = validation_window_records([record for record in json.loads((args.dataset / "accepted.json").read_text())
                                         if record["split"] == "validation"], 128, 3)
    histories = []
    for record in records:
        if record.get("behavior") == "idle" and len(histories) < 3:
            path, start = args.dataset / record["shard"], record["window_start_frame"]
            with np.load(path, allow_pickle=False) as data:
                initial = torch.from_numpy(data["global_features"][start:start + 4].copy())[None].cuda()
            histories.append((initial, record["filename"], start, file_sha256(path)))
    if len(histories) != 3:
        raise ValueError("Three held-out initial histories are required")
    args.output.mkdir(parents=True)
    (args.output / "gpu_at_launch.txt").write_text(inventory)
    report = {"bundle_sha256": file_sha256(args.bundle), "dataset_sha256": snapshot["sha256"],
              "source_sha256": {"evaluation": file_sha256(Path(__file__)),
                                "scheduler": file_sha256(Path(__file__).parents[1] / "motionbricks/data/soma_primitive_scheduler.py"),
                                "backend": file_sha256(Path(__file__).parents[1] / "motionbricks/data/bones_rollout.py")},
              "source": "https://arxiv.org/html/2604.24833v1 (Algorithm 1, Appendix C)",
              "fps": 60, "publish_frames": 8, "intervals": [8, 16, 56], "test_used": False,
              "gpu_uuid": selected,
              "scope": "three matched held-out initial histories, not three training seeds; command changes interrupt at the next unpublished eight-frame block",
              "interpretation": "16 frames lies in the paper's 30-fps 3--9 frame time range; 56 frames is a long-consumption experiment, not the recommended default",
              "post_processing": None, "quality_approved": False, "ue_deployed": False, "rollouts": {}}
    for interval in report["intervals"]:
        rows = []
        for index, (initial, filename, start, sha) in enumerate(histories):
            motion, boundaries, measured = evaluate(inference, initial.clone(), interval)
            rows.append({"initial_filename": filename, "initial_frame": start, "input_sha256": sha, **measured})
            np.savez_compressed(args.output / f"interval_{interval}_history_{index}.npz",
                                global_features=motion[0].cpu().numpy(), boundaries=np.asarray(boundaries), fps=np.asarray(60))
            report["rollouts"][str(interval)] = rows
            (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps({"interval": interval, "history": index, "model_calls": measured["model_calls"],
                              "angular_step_p95_deg": measured["angular_step_deg"]["p95"],
                              "worst_step_deg": measured["angular_step_deg"]["max"],
                              "command_error_mps": measured["root_command_velocity_error_mps"]["mean"]}), flush=True)
    report["complete"] = True
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
