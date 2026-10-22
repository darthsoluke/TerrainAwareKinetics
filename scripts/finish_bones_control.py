"""Finish the repaired Pose/history experiments without stopping at preparation."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_bones_locomotion import select_idle_gpus


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "tokenizer-run", "root-run", "pose-run", "history-output", "review-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reuse-teacher", action="store_true", help="Continue orchestration after transport loss; do not restart teacher")
    parser.add_argument("--review-only", action="store_true", help="Re-evaluate completed arms; never train again")
    args = parser.parse_args()
    for name in ("dataset", "tokenizer_run", "root_run", "pose_run", "history_output", "review_output"):
        setattr(args, name, getattr(args, name).resolve())
    os.chdir(Path(__file__).resolve().parents[1])
    while not (args.pose_run / "pilot_result.json").exists():
        time.sleep(5)
    result = json.loads((args.pose_run / "pilot_result.json").read_text())
    contract = json.loads((args.pose_run / "run_contract.json").read_text())
    if (result["steps"] != 20000 or not result["frozen_tokenizer_unchanged"]
            or not contract.get("pose_target_tokens", "").startswith("all positions")):
        raise ValueError("The repaired all-token Pose baseline did not complete")
    if args.review_only:
        for name in ("teacher", "generated"):
            previous = json.loads((args.history_output / name / "pilot_result.json").read_text())
            if previous["steps"] != 10000 or not previous["frozen_tokenizer_unchanged"]:
                raise ValueError("Review-only requires both completed frozen-tokenizer arms")
    elif args.reuse_teacher:
        teacher_result = args.history_output / "teacher/pilot_result.json"
        deadline = time.monotonic() + 1800
        while not teacher_result.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Teacher did not complete; do not duplicate its training")
            time.sleep(5)
        previous = json.loads(teacher_result.read_text())
        if previous["steps"] != 10000 or not previous["frozen_tokenizer_unchanged"]:
            raise ValueError("Reused teacher did not complete its exact frozen-tokenizer budget")
        if (args.history_output / "generated").exists():
            raise ValueError("Generated arm already exists; refuse to duplicate or overwrite it")
        select_idle_gpus([3])
        protocol = json.loads((args.history_output / "teacher_protocol.json").read_text())
        protocol["history_policy"] = "generated"
        path = args.history_output / "generated_protocol.json"
        path.write_text(json.dumps(protocol, indent=2))
        with (args.history_output / "generated.log").open("w") as log:
            subprocess.run([sys.executable, "scripts/train_bones_locomotion.py", "--control-pilot", "--stage", "pose",
                            "--dataset", str(args.dataset), "--tokenizer-run", str(args.tokenizer_run),
                            "--control-init-run", str(args.pose_run), "--history-policy", "generated",
                            "--history-root-run", str(args.root_run), "--history-generator-pose-run", str(args.pose_run),
                            "--output", str(args.history_output / "generated"), "--gpus", "3", "--max-steps", "10000",
                            "--batch-size", "16", "--workers", "2", "--scheduler-warmup-steps", "500",
                            "--validation-interval", "1000", "--protocol", str(path)], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        (args.history_output / "orchestration_recovery.json").write_text(json.dumps({
            "teacher": "reused completed run, no extra updates, process exit not observed",
            "generated_exit_code": 0, "reason": "SSH transport lost after teacher launch"}, indent=2))
    else:
        subprocess.run([sys.executable, "scripts/run_bones_history.py", "--dataset", str(args.dataset),
                        "--tokenizer-run", str(args.tokenizer_run), "--root-run", str(args.root_run),
                        "--pose-run", str(args.pose_run), "--output", str(args.history_output), "--gpus", "0,3"], check=True)
    select_idle_gpus([0, 2, 3])
    arms = {"baseline": args.pose_run, "teacher": args.history_output / "teacher",
            "generated": args.history_output / "generated"}
    jobs = []
    for (name, run), gpu in zip(arms.items(), (2, 0, 3)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        jobs.append(subprocess.Popen([sys.executable, "scripts/evaluate_bones_control.py",
                                      "--dataset", str(args.dataset), "--root-run", str(args.root_run),
                                      "--pose-run", str(run), "--output", str(args.review_output / name)], env=env))
    evaluation_status = [job.wait() for job in jobs]
    if any(evaluation_status):
        raise RuntimeError("A repaired control evaluation failed: " + str(evaluation_status))
    subprocess.run([sys.executable, "scripts/summarize_bones_control.py", "--dataset", str(args.dataset),
                    "--control-runs", str(args.root_run.parent), "--pose-baseline-run", str(args.pose_run),
                    "--history-runs", str(args.history_output), "--review", str(args.review_output)], check=True)
    jobs = [subprocess.Popen([sys.executable, "scripts/render_bones_control.py", "--review", str(args.review_output / name)])
            for name in arms]
    rendering_status = [job.wait() for job in jobs]
    if any(rendering_status):
        raise RuntimeError("A repaired rollout rendering failed: " + str(rendering_status))
    subprocess.run([sys.executable, "scripts/portable_bones_control.py", "export", "--dataset", str(args.dataset),
                    "--root-run", str(args.root_run), "--pose-run", str(args.pose_run),
                    "--output", str(args.review_output / "baseline_backend.pt")], check=True)
    (args.review_output / "runner_status.json").write_text(json.dumps({
        "evaluation_exit_codes": evaluation_status, "rendering_exit_codes": rendering_status,
        "completed": True, "quality_approved": False}, indent=2))


if __name__ == "__main__":
    main()
