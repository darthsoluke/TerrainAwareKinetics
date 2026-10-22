"""Evaluate immutable paired native milestones after their training completes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.continue_native_mb_control import MILESTONES
from scripts.train_native_mb import select_gpu, sha256, write_json


def paired_milestone_ready(base, step):
    """Require both finished reviews; a partial JSON write is not completion."""
    for stage in ("root", "pose"):
        path = base / f"{stage}-{step}" / "run_contract.json"
        try:
            contract = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if contract.get("status") != "completed":
            return False
        if contract.get("stage") != "native_" + stage or contract.get("stop_steps") != step:
            raise ValueError("Unexpected paired native milestone identity")
    return True


def training_failure(base):
    for stage in ("root", "pose"):
        try:
            state = json.loads((base / f"{stage}_state.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if state.get("status") == "failed":
            return state.get("error", f"{stage} training failed")
    return None


def evaluation_commands(base, step, gpu):
    common = ["--root-run", str(base / f"root-{step}"),
              "--pose-run", str(base / f"pose-{step}")]
    review = base / f"review-{step}"
    return [
        [sys.executable, "-u", "scripts/evaluate_native_mb_inbetween.py", *common,
         "--output", str(review / "inbetween"), "--device", "cuda", "--gpu", str(gpu)],
        [sys.executable, "-u", "scripts/evaluate_native_mb_smart_locomotion.py", *common,
         "--output", str(review / "smart-locomotion"), "--device", "cuda", "--gpu", str(gpu),
         "--include-fixed8"],
        [sys.executable, "-u", "scripts/render_native_mb_review.py", "--review", str(review / "inbetween")],
    ]


def review_milestones(base, gpu, timeout_seconds=43200, poll_seconds=30):
    base = Path(base).resolve()
    code_root = Path(__file__).resolve().parents[1]
    state_path = base / "review_state.json"
    if state_path.exists():
        raise FileExistsError("Review orchestration already exists")
    state = {"status": "waiting_for_training", "milestones": list(MILESTONES),
             "evaluation_gpu_index": gpu, "completed": [], "started_at_unix": time.time(),
             "source_sha256": sha256(Path(__file__)), "quality_approved": False}
    deadline = time.monotonic() + timeout_seconds
    write_json(state_path, state)
    try:
        for step in MILESTONES:
            state.update(status="waiting_for_training", target_step=step)
            write_json(state_path, state)
            while not paired_milestone_ready(base, step):
                failure = training_failure(base)
                if failure:
                    raise RuntimeError(failure)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Waiting for paired native training timed out")
                time.sleep(poll_seconds)
            commands = evaluation_commands(base, step, gpu)
            review = base / f"review-{step}"
            review.mkdir(exist_ok=False)
            for index, command in enumerate(commands):
                if "--gpu" in command:
                    state.update(status="waiting_for_idle_evaluation_gpu", command=command)
                    write_json(state_path, state)
                    while True:
                        try:
                            state["gpu_at_evaluation"] = select_gpu(gpu)
                            break
                        except RuntimeError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Waiting for an idle evaluation GPU timed out")
                            time.sleep(poll_seconds)
                state.update(status="evaluating", command=command, log=str(review / f"command-{index}.log"))
                write_json(state_path, state)
                with (review / f"command-{index}.log").open("xb") as stream:
                    result = subprocess.run(command, cwd=code_root, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Evaluation exited {result.returncode}; see {state['log']}")
            smart = json.loads((review / "smart-locomotion/report.json").read_text())
            inbetween = json.loads((review / "inbetween/metrics.json").read_text())
            if smart["bundle"]["root_step"] != step or smart["bundle"]["pose_step"] != step:
                raise ValueError("Evaluator loaded the wrong milestone")
            summary = {"step": step, "inbetween": inbetween["summary"],
                       "smart_locomotion": smart["summary"], "quality_approved": False,
                       "notes": ["No output repair or root replacement", "A completed evaluation is not quality approval",
                                 "In-between videos use generated tokens, not source-encoded tokens",
                                 "No foot-slip/penetration or UE/RTX3080 acceptance claim"]}
            write_json(review / "summary.json", summary)
            state["completed"].append({"step": step, "output": str(review)})
            state.update(status="milestone_reviewed", updated_at_unix=time.time())
            write_json(state_path, state)
        state.update(status="completed", completed_at_unix=time.time())
    except Exception as error:
        state.update(status="failed", error=type(error).__name__ + ": " + str(error), failed_at_unix=time.time())
        raise
    finally:
        write_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu < 0 or not args.output_parent.is_dir():
        parser.error("Requires an existing continuation directory and a nonnegative GPU index")
    print(json.dumps(review_milestones(args.output_parent, args.gpu), indent=2), flush=True)


if __name__ == "__main__":
    main()
