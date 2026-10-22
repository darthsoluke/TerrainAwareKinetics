"""Launch matched decoder trials only on freshly checked idle remote GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "labels", "bundle", "tokenizer-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpus", default="2,3,4")
    parser.add_argument("--stage", choices=("decoder", "pose"), default="decoder")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh comparison directory")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    available = {}
    for row in inventory.strip().splitlines():
        index, uuid, memory, utilization = [value.strip() for value in row.split(",")]
        if int(memory) <= 1024 and int(utilization) <= 5:
            available[int(index)] = uuid
    requested = [int(value) for value in args.gpus.split(",")]
    if len(requested) != 3 or len(set(requested)) != 3 or not set(requested) <= available.keys():
        raise RuntimeError("Three requested GPUs must be idle at launch")
    args.output.mkdir(parents=True)
    (args.output / "gpu_at_launch.txt").write_text(inventory)
    jobs = []
    try:
        for arm, index in zip(("A", "B", "C"), requested):
            module = "scripts.train_soma_pose_rollout" if args.stage == "pose" else "scripts.train_soma_decoder_refinement"
            command = [sys.executable, "-m", module, "--arm", arm]
            for name in ("dataset", "labels", "bundle", "tokenizer_config"):
                command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
            command += ["--output", str(args.output / arm)]
            log = (args.output / f"{arm}.log").open("w")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=available[index]))
            jobs.append((arm, process, log))
            print(json.dumps({"launched": arm, "pid": process.pid, "gpu_uuid": available[index]}), flush=True)
        while any(process.poll() is None for _, process, _ in jobs):
            time.sleep(30)
            progress = {}
            for arm, process, _ in jobs:
                log_path = args.output / arm / "training.jsonl"
                lines = log_path.read_text().splitlines() if log_path.exists() else []
                progress[arm] = {"exit": process.poll(), "last": json.loads(lines[-1]) if lines else None}
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"Arm {arm} failed; inspect its retained log")
            print(json.dumps({"progress": {arm: {"exit": item["exit"],
                "step": item["last"]["step"] if item["last"] else 0} for arm, item in progress.items()}}), flush=True)
        status = {arm: process.wait() for arm, process, _ in jobs}
        (args.output / "training_status.json").write_text(json.dumps(status, indent=2))
        if any(status.values()):
            raise RuntimeError("Training comparison incomplete")
        results = [json.loads((args.output / arm / "result.json").read_text()) for arm in ("A", "B", "C")]
        if len({result["sampling_sha256"] for result in results}) != 1:
            raise RuntimeError("Paired sampling differs")
        print(json.dumps({"completed": status, "sampling_equal": True}), flush=True)
    finally:
        for _, process, log in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()


if __name__ == "__main__":
    main()
