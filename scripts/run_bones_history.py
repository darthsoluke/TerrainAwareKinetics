"""Paired continuation from the same Pose checkpoint, with distinct history policies."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256
from scripts.train_bones_locomotion import select_idle_gpus


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "tokenizer-run", "root-run", "pose-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpus", default="0,3")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new paired history experiment directory")
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise ValueError("Two distinct idle GPUs are required")
    select_idle_gpus(gpus)
    snapshot = dataset_snapshot(args.dataset)
    tokenizer_sha = file_sha256(args.tokenizer_run / "checkpoints/final.ckpt")
    root_sha = file_sha256(args.root_run / "checkpoints/final.ckpt")
    pose_sha = file_sha256(args.pose_run / "checkpoints/final.ckpt")
    args.output.mkdir(parents=True)
    jobs, logs = {}, []
    for policy, gpu in zip(("teacher", "generated"), gpus):
        protocol = {"experiment_kind": "bounded_soma_control", "stage": "pose", "max_steps": 10000,
                    "dataset_snapshot_sha256": snapshot["sha256"], "tokenizer_sha256": tokenizer_sha,
                    "initial_checkpoint_sha256": pose_sha, "root_checkpoint_sha256": root_sha,
                    "generator_pose_checkpoint_sha256": pose_sha, "history_policy": policy,
                    "history_ramp_steps": 5000, "history_max_probability": 0.5,
                    "batch_size": 16, "scheduler_warmup_steps": 500, "validation_interval": 1000,
                    "validation_windows_per_clip": 3, "seed": 2333, "post_processing": None}
        path = args.output / (policy + "_protocol.json")
        path.write_text(json.dumps(protocol, indent=2))
        command = [sys.executable, "scripts/train_bones_locomotion.py", "--control-pilot", "--stage", "pose",
                   "--dataset", str(args.dataset), "--tokenizer-run", str(args.tokenizer_run),
                   "--control-init-run", str(args.pose_run), "--history-policy", policy,
                   "--history-root-run", str(args.root_run), "--history-generator-pose-run", str(args.pose_run),
                   "--output", str(args.output / policy), "--gpus", str(gpu), "--max-steps", "10000",
                   "--batch-size", "16", "--workers", "2", "--scheduler-warmup-steps", "500",
                   "--validation-interval", "1000", "--protocol", str(path)]
        env = dict(os.environ)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        log = (args.output / (policy + ".log")).open("w")
        logs.append(log)
        jobs[policy] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(json.dumps({"started": policy, "pid": jobs[policy].pid, "gpu": gpu}), flush=True)
    results = {name: job.wait() for name, job in jobs.items()}
    for log in logs:
        log.close()
    (args.output / "training_status.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"completed": results}), flush=True)
    if any(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
