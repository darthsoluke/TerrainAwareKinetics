"""Run audited native training milestones without changing the training method."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_native_mb import sha256, write_json
from scripts.train_native_mb_control import validate_resume


MILESTONES = (100000, 220000)


def native_command(stage, initial, resume, output, gpu, step):
    command = [sys.executable, "-u", "scripts/train_native_mb_control.py",
               "--stage", stage, "--dataset", initial["dataset"],
               "--output", str(output), "--resume-run", str(resume),
               "--steps", str(step),
               "--schedule-steps", str(initial["original_schedule_steps"]),
               "--batch-size", str(initial["batch_size"]), "--seed", str(initial["seed"]),
               "--workers", "2", "--gpu", str(gpu)]
    if stage == "pose":
        command += ["--tokenizer-run", initial["tokenizer"]["run"]]
    return command


def verify_completed(run, stage, step, initial, previous_step):
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    if (contract["status"] != "completed" or contract["stage"] != "native_" + stage
            or result["optimizer_updates"] != step
            or result["updates_this_run"] != step - previous_step
            or result["skipped_batches"] != 0):
        raise ValueError("Native milestone did not complete its training and review")
    for field in ("dataset_manifest_sha256", "config_sha256", "seed", "batch_size", "original_schedule_steps"):
        if contract[field] != initial[field]:
            raise ValueError(f"Native continuation changed {field}")
    if not contract["original_training_step"] or contract["custom_losses"]:
        raise ValueError("Continuation changed the original training method")
    if sha256(run / "checkpoints/final.ckpt") != result["final_checkpoint_sha256"]:
        raise ValueError("Native milestone checkpoint hash mismatch")
    if stage == "pose":
        if (contract["tokenizer"]["checkpoint_sha256"] != initial["tokenizer"]["checkpoint_sha256"]
                or result["tokenizer_state_sha256"] != initial["tokenizer_state_sha256"]):
            raise ValueError("Native continuation changed the frozen tokenizer")
    return result


def continue_stage(stage, resume_run, output_parent, gpu):
    code_root = Path(__file__).resolve().parents[1]
    resume_run, output_parent = Path(resume_run).resolve(), Path(output_parent).resolve()
    initial = json.loads((resume_run / "run_contract.json").read_text())
    if initial.get("stage") != "native_" + stage or initial.get("original_schedule_steps") != 2000001:
        raise ValueError("Requires the existing native stage and unchanged official schedule")
    expected = {**initial, "stop_steps": MILESTONES[-1]}
    validation = validate_resume(resume_run, expected, code_root)
    if validation["saved_step"] >= MILESTONES[0]:
        raise ValueError("This execution continues the existing pre-100k checkpoints")
    dataset = Path(initial["dataset"]).resolve()
    if (dataset == output_parent or dataset in output_parent.parents
            or output_parent in dataset.parents):
        raise ValueError("Training output must be outside the dataset")
    state_path = output_parent / f"{stage}_state.json"
    if state_path.exists() or any((output_parent / f"{stage}-{step}").exists() for step in MILESTONES):
        raise FileExistsError("Continuation state or milestone already exists")
    output_parent.mkdir(parents=True, exist_ok=True)
    state = {"stage": stage, "status": "preparing", "pid": os.getpid(), "gpu_index": gpu,
             "initial_resume": validation, "milestones": list(MILESTONES),
             "source_sha256": sha256(Path(__file__)), "completed": [],
             "training_method": "unchanged official Root/Pose training_step, sampler and losses",
             "quality_approved": False, "ue_deployment": False,
             "started_at_unix": time.time()}
    write_json(state_path, state)
    previous_step = validation["saved_step"]
    try:
        for step in MILESTONES:
            output = output_parent / f"{stage}-{step}"
            log = output_parent / f"{stage}-{step}.log"
            command = native_command(stage, initial, resume_run, output, gpu, step)
            state.update(status="running", target_step=step, resume_run=str(resume_run),
                         output=str(output), log=str(log), command=command)
            write_json(state_path, state)
            with log.open("xb") as stream:
                process = subprocess.Popen(command, cwd=code_root, stdout=stream, stderr=subprocess.STDOUT)
                state["training_pid"] = process.pid
                write_json(state_path, state)
                returncode = process.wait()
            if returncode != 0:
                raise RuntimeError(f"Native {stage} milestone {step} exited {returncode}; see {log}")
            result = verify_completed(output, stage, step, initial, previous_step)
            state["completed"].append({"step": step, "run": str(output),
                                       "checkpoint_sha256": result["final_checkpoint_sha256"],
                                       "elapsed_seconds": result["elapsed_seconds"],
                                       "updates_this_run": result["updates_this_run"]})
            state.update(status="milestone_completed", updated_at_unix=time.time())
            write_json(state_path, state)
            previous_step, resume_run = step, output
        state.update(status="completed", completed_at_unix=time.time())
    except Exception as error:
        state.update(status="failed", error=type(error).__name__ + ": " + str(error), failed_at_unix=time.time())
        raise
    finally:
        write_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("root", "pose"), required=True)
    parser.add_argument("--resume-run", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    print(json.dumps(continue_stage(args.stage, args.resume_run, args.output_parent, args.gpu), indent=2), flush=True)


if __name__ == "__main__":
    main()
