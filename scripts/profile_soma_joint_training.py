"""Bounded two-update CUDA profile, isolated from the formal paired runs."""

import argparse
import functools
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from scripts import train_soma_pose_rollout as training


def named_call(function, name):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name):
            return function(*args, **kwargs)
    return wrapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parents[1])
    if args.profile_dir.exists():
        raise ValueError("Use a fresh profiling output")
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    memory, utilization = [int(v.strip()) for v in subprocess.check_output([
        "nvidia-smi", "-i", gpu, "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")]
    if memory > 1000 or utilization > 5:
        raise RuntimeError("Profiling GPU is not idle")
    args.profile_dir.mkdir(parents=True)
    training.deployed_reference = named_call(training.deployed_reference, "native_reference")
    training.joint_rollout_loss = named_call(training.joint_rollout_loss, "joint_loss_and_parity")
    sys.argv = ["train_soma_pose_rollout", "--dataset", "${PRIVATE_STORAGE}/motionbricks-bones/pilot-v7",
        "--labels", "out/sole_region_repair_v3/labels_train", "--bundle", "out/soma_control_v3/baseline_backend.pt",
        "--tokenizer-config", "${PRIVATE_STORAGE}/motionbricks-preparation/runs/locomotion-consistency-20261005/control/upstream_config_overrides.yaml",
        "--output", str(args.profile_dir / "smoke_B"), "--arm", "B", "--coupled-root-trial", "--steps", "2", "--smoke-test"]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA]) as profile:
        training.main()
    events = profile.key_averages()
    (args.profile_dir / "cpu_table.txt").write_text(events.table(sort_by="self_cpu_time_total", row_limit=30))
    (args.profile_dir / "cuda_table.txt").write_text(events.table(sort_by="self_device_time_total", row_limit=30))
    rows = [{"name": e.key, "calls": e.count, "self_cpu_ms": e.self_cpu_time_total / 1000,
             "total_cpu_ms": e.cpu_time_total / 1000, "self_device_ms": e.self_device_time_total / 1000,
             "total_device_ms": e.device_time_total / 1000} for e in events]
    (args.profile_dir / "events.json").write_text(json.dumps({"events": rows,
        "scope": "two updates including initialization and checkpoint writing; profiler adds overhead; nested times must not be added",
        "gpu_uuid": gpu, "gpu_at_launch": {"memory_mib": memory, "utilization_percent": utilization},
        "formal_runs_changed": False}, indent=2))
    print((args.profile_dir / "cuda_table.txt").read_text(), flush=True)


if __name__ == "__main__":
    main()
