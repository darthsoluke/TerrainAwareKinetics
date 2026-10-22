"""Complete evaluation and media after all fixed-budget decoder trials finish."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.summarize_soma_decoder_refinement import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--gpus", default="2,3,4")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stage", choices=("decoder", "pose"), default="decoder")
    parser.add_argument("--original-bundle", type=Path)
    args = parser.parse_args()
    if json.loads((args.trial / "training_status.json").read_text()) != {"A": 0, "B": 0, "C": 0}:
        raise ValueError("All three fixed-budget trials must finish first")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    available = {}
    for row in inventory.strip().splitlines():
        index, uuid, memory, utilization = [value.strip() for value in row.split(",")]
        if int(memory) <= 1024 and int(utilization) <= 5:
            available[int(index)] = uuid
    requested = [int(value) for value in args.gpus.split(",")]
    if len(requested) != 3 or len(set(requested)) != 3 or not set(requested) <= available.keys():
        raise RuntimeError("Evaluation GPUs must be idle at launch")
    (args.trial / "gpu_at_review.txt").write_text(inventory)
    if args.original_bundle is not None:
        with (args.trial / "review_original.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "scripts.evaluate_soma_decoder_refinement",
                "--dataset", str(args.dataset), "--bundle", str(args.original_bundle),
                "--output", str(args.trial / "review_original"), "--commit-frames", "8" if args.stage == "pose" else "4"], stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=available[requested[0]]), check=True)
        print(json.dumps({"original_evaluation_completed": True}), flush=True)
    jobs = []
    try:
        for arm, index in zip(("A", "B", "C"), requested):
            command = [sys.executable, "-m", "scripts.evaluate_soma_decoder_refinement",
                "--dataset", str(args.dataset), "--bundle", str(args.trial / arm / "backend.pt"),
                "--output", str(args.trial / ("review_" + arm)), "--commit-frames", "8" if args.stage == "pose" else "4"]
            log = (args.trial / f"review_{arm}.log").open("w")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=available[index]))
            jobs.append((arm, process, log))
            print(json.dumps({"evaluation_launched": arm, "pid": process.pid}), flush=True)
        while any(process.poll() is None for _, process, _ in jobs):
            time.sleep(30)
            print(json.dumps({"evaluation": {arm: process.poll() for arm, process, _ in jobs}}), flush=True)
            if any(process.poll() not in (None, 0) for _, process, _ in jobs):
                raise RuntimeError("Evaluation failed; retained per-arm log has diagnostics")
        if any(process.wait() for _, process, _ in jobs):
            raise RuntimeError("Evaluation incomplete")
    finally:
        for _, process, log in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()
    summary = summarize(args.trial, args.stage)
    print(json.dumps({"decision": summary["deployment_decision"], "arms": summary["arms"]}), flush=True)
    if args.render:
        subprocess.run([sys.executable, "-m", "scripts.render_bones_control", "--decoder-trial", str(args.trial),
                        "--review", str(args.trial / "render"), "--stride", "4"], check=True)


if __name__ == "__main__":
    main()
