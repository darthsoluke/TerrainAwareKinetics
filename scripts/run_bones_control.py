"""Freeze and run the authorized SOMA Root/Pose pilot on two idle GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256
from scripts.train_bones_locomotion import select_idle_gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,2")
    parser.add_argument("--root-steps", type=int, default=10000)
    parser.add_argument("--pose-steps", type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new experiment directory")
    devices = [int(value) for value in args.gpus.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("Use two distinct idle GPUs")
    select_idle_gpus(devices)
    snapshot = dataset_snapshot(args.dataset)
    tokenizer_sha = file_sha256(args.tokenizer_run / "checkpoints/final.ckpt")
    if tokenizer_sha != "134ed76c3bc2fcdcbe087c945aac15d42ba4735f7ec785cb1e4f71c8415435f5":
        raise ValueError("This pilot is bound to the selected Control tokenizer")
    args.output.mkdir(parents=True)
    jobs, logs = {}, {}
    for stage, gpu, steps in (("root", devices[0], args.root_steps), ("pose", devices[1], args.pose_steps)):
        protocol = {"experiment_kind": "bounded_soma_control", "stage": stage, "max_steps": steps,
                    "dataset_snapshot_sha256": snapshot["sha256"], "tokenizer_sha256": tokenizer_sha,
                    "batch_size": 16, "scheduler_warmup_steps": 500, "validation_interval": 1000,
                    "validation_windows_per_clip": 3, "seed": 2333,
                    "history_policy": "teacher", "post_processing": None,
                    "goals": "future root endpoint/velocity; no future body or height",
                    "deployment_approved": False}
        path = args.output / f"{stage}_protocol.json"
        path.write_text(json.dumps(protocol, indent=2))
        command = [sys.executable, "scripts/train_bones_locomotion.py", "--control-pilot", "--stage", stage,
                   "--dataset", str(args.dataset), "--tokenizer-run", str(args.tokenizer_run),
                   "--output", str(args.output / stage), "--gpus", str(gpu), "--max-steps", str(steps),
                   "--batch-size", "16", "--workers", "2", "--scheduler-warmup-steps", "500",
                   "--validation-interval", "1000", "--protocol", str(path)]
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "4"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("CUDA_VISIBLE_DEVICES", None)
        logs[stage] = (args.output / f"{stage}.log").open("w")
        jobs[stage] = subprocess.Popen(command, stdout=logs[stage], stderr=subprocess.STDOUT, env=env)
        print(json.dumps({"started": stage, "pid": jobs[stage].pid, "gpu": gpu}), flush=True)
    while any(job.poll() is None for job in jobs.values()):
        time.sleep(1)
    results = {name: job.returncode for name, job in jobs.items()}
    for log in logs.values():
        log.close()
    (args.output / "training_status.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"completed": results}), flush=True)
    if any(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
