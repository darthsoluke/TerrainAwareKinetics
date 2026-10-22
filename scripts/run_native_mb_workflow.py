"""Schedule the complete native curriculum and dependent reviews without human gates."""

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_workflow import (
    FULL_SCHEDULE_STEPS, native_stage_command, verify_workflow_run,
)
from scripts.review_native_mb_continuation import evaluation_commands
from scripts.train_native_mb import (
    select_gpu, sha256, source_lf_sha256, validate_native_resume,
    verify_resume_training_sources,
)
from scripts.train_native_mb_control import SOURCE_FILES, validate_resume


def save_state(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_pending(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def ready_to_launch(stage, stages, pilot):
    if stage == "tokenizer":
        return True
    if stage == "pose":
        return stages["tokenizer"]["status"] == "completed"
    if stage != "root":
        raise ValueError(f"Unknown workflow stage: {stage}")
    upstream = read_pending(pilot / "root_state.json")
    if upstream and upstream.get("status") == "failed":
        raise RuntimeError(f"Upstream Root failed: {upstream.get('error')}")
    contract = read_pending(pilot / "root-220000/run_contract.json")
    if not contract or contract.get("status") != "completed":
        return False
    if contract.get("stage") != "native_root" or contract.get("stop_steps") != 220000:
        raise ValueError("Unexpected upstream Root checkpoint identity")
    return True


def load_initial(run, stage):
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    if (contract.get("status") != "completed" or contract.get("stage") != "native_" + stage
            or contract.get("original_schedule_steps") != FULL_SCHEDULE_STEPS
            or not 0 < result.get("optimizer_updates", 0) < FULL_SCHEDULE_STEPS
            or sha256(run / "native_config.yaml") != contract.get("config_sha256")
            or sha256(run / "checkpoints/final.ckpt") != result.get("final_checkpoint_sha256")):
        raise ValueError(f"Requires an intact completed native {stage} template")
    return contract, result


def validate_common_identity(templates):
    reference = templates["tokenizer"]
    for stage, contract in templates.items():
        for key in ("dataset", "dataset_manifest_sha256", "seed", "batch_size", "original_schedule_steps"):
            if contract.get(key) != reference.get(key):
                raise ValueError(f"Incompatible {stage} template: {key}")
    dataset = Path(reference["dataset"])
    if sha256(dataset / "native_manifest.json") != reference["dataset_manifest_sha256"]:
        raise ValueError("Workflow dataset changed")


def frozen_tokenizer_identity(run, target_steps=FULL_SCHEDULE_STEPS):
    from scripts.evaluate_native_mb_checkpoint import load_native_checkpoint
    from scripts.train_native_mb_control import state_digest

    model, metadata, _ = load_native_checkpoint(run, device="cpu")
    if metadata["global_step"] != target_steps or metadata["run_status_at_review"] != "completed":
        raise ValueError("Pose requires the fully trained and reviewed tokenizer")
    return {"checkpoint_sha256": metadata["checkpoint_sha256"],
            "tensor_state_sha256": state_digest(model.pose_net), "run": str(run)}


def assert_sources(root, expected):
    for relative, digest in expected.items():
        if source_lf_sha256(root / relative) != digest:
            raise ValueError(f"Workflow source changed: {relative}")


def stop_owned(processes):
    for process, stream in processes.values():
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        stream.close()


def launch(command, code_root, log):
    stream = log.open("xb")
    try:
        process = subprocess.Popen(command, cwd=code_root, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except Exception:
        stream.close()
        raise
    return process, stream


def run_final_review(output, gpu, code_root, source_hashes, deadline, poll_seconds, state,
                     step=FULL_SCHEDULE_STEPS):
    review = output / f"review-{step}"
    review.mkdir(exist_ok=False)
    for index, command in enumerate(evaluation_commands(output, step, gpu)):
        assert_sources(code_root, source_hashes)
        if "--gpu" in command:
            state.update(status="waiting_for_idle_evaluation_gpu", command=command)
            save_state(output / "workflow_state.json", state)
            while True:
                try:
                    state["gpu_at_evaluation"] = select_gpu(gpu)
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Waiting for evaluation GPU timed out")
                    time.sleep(poll_seconds)
        state.update(status="evaluating", command=command, log=str(review / f"command-{index}.log"))
        save_state(output / "workflow_state.json", state)
        process, stream = launch(command, code_root, Path(state["log"]))
        owned = {"evaluation": (process, stream)}
        try:
            returncode = process.wait(timeout=max(1, deadline - time.monotonic()))
            if returncode:
                raise RuntimeError(f"Evaluation exited {returncode}; see {state['log']}")
        finally:
            stop_owned(owned)
    smart = json.loads((review / "smart-locomotion/report.json").read_text())
    inbetween = json.loads((review / "inbetween/metrics.json").read_text())
    if (smart["bundle"]["root_step"] != step or smart["bundle"]["pose_step"] != step
            or smart["bundle"]["tokenizer"]["global_step"] != step):
        raise ValueError("Final review loaded a different native bundle")
    summary = {"inbetween": inbetween["summary"], "smart_locomotion": smart["summary"],
               "quality_approved": False,
               "notes": ["Generated tokens and predicted roots; no output repairs",
                         "Completion and reference errors are not naturalness approval",
                         "No foot-contact, UE PIE or RTX3080 performance acceptance claim"]}
    save_state(review / "summary.json", summary)
    bundle = {"schema": "native_mb_soma_runtime_candidate_v1", "status": "trained_evaluated_not_approved",
              "motion_fps": 30, "skeleton": "SOMA30", "ue_pie_target_fps": 30,
              "dataset": state["dataset"], "dataset_manifest_sha256": state["dataset_manifest_sha256"],
              "stages": {stage: {key: info[key] for key in ("output", "checkpoint_sha256")}
                         for stage, info in state["stages"].items()},
              "frozen_tokenizer": state["frozen_tokenizer"], "review": str(review),
              "quality_approved": False, "ue_deployment": False,
              "notes": ["Checkpoint binding manifest, not a packaged UE project",
                        "No IK, foot locking, smoothing, root overwrite or custom physics losses"]}
    save_state(output / "candidate_bundle.json", bundle)
    return str(review)


def run_workflow(args):
    code_root = Path(__file__).resolve().parents[1]
    output, pilot = args.output.resolve(), args.pilot.resolve()
    template_runs = {"tokenizer": args.tokenizer_resume.resolve(),
                     "root": pilot / "root-100000", "pose": pilot / "pose-100000"}
    templates = {stage: load_initial(run, stage)[0] for stage, run in template_runs.items()}
    validate_common_identity(templates)
    dataset = Path(templates["tokenizer"]["dataset"]).resolve()
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Workflow output must be outside the dataset")
    verify_resume_training_sources(templates["tokenizer"], code_root)
    tokenizer_resume = validate_native_resume(template_runs["tokenizer"],
        templates["tokenizer"]["dataset_manifest_sha256"], templates["tokenizer"]["config_sha256"],
        templates["tokenizer"]["seed"], templates["tokenizer"]["batch_size"], FULL_SCHEDULE_STEPS)
    validate_resume(template_runs["root"], {**templates["root"], "stop_steps": FULL_SCHEDULE_STEPS}, code_root)
    extra_sources = ("scripts/run_native_mb_workflow.py", "motionbricks/data/native_mb_workflow.py",
                     "scripts/review_native_mb_continuation.py", "scripts/continue_native_mb_control.py",
                     "scripts/evaluate_native_mb_inbetween.py", "scripts/evaluate_native_mb_smart_locomotion.py",
                     "scripts/render_native_mb_review.py", "motionbricks/data/native_mb_locomotion.py",
                     "motionbricks/data/native_mb_control_review.py", "motionbricks/vqvae/models/motion_vqvae.py")
    source_hashes = {name: source_lf_sha256(code_root / name) for name in (*SOURCE_FILES, *extra_sources)}
    for name in ("motionbricks_vqvae", "motionbricks_root", "motionbricks_pose"):
        relative = f"out/{name}/version_1/hparams.yaml"
        source_hashes[relative] = source_lf_sha256(code_root / relative)
    output.mkdir(parents=True, exist_ok=False)
    stages = {stage: {"status": "queued", "gpu_index": getattr(args, stage + "_gpu"),
                      "output": str(output / f"{stage}-{FULL_SCHEDULE_STEPS}")}
              for stage in ("tokenizer", "root", "pose")}
    state = {"status": "training", "pid": os.getpid(), "started_at_unix": time.time(),
             "dataset": str(dataset), "dataset_manifest_sha256": templates["tokenizer"]["dataset_manifest_sha256"],
             "source_lf_sha256": source_hashes, "stages": stages, "manual_gates": False,
             "native_target_steps": FULL_SCHEDULE_STEPS, "quality_approved": False, "ue_deployment": False,
             "training_strategy": "Tokenizer and Root independent; fresh Pose after final frozen tokenizer",
             "existing_pilot": str(pilot), "initial_tokenizer_resume": tokenizer_resume}
    state_path = output / "workflow_state.json"
    save_state(state_path, state)
    processes = {}
    deadline = time.monotonic() + args.timeout_hours * 3600
    try:
        while not all(info["status"] == "completed" for info in stages.values()):
            if time.monotonic() >= deadline:
                raise TimeoutError("Continuous native training exceeded its workflow deadline")
            assert_sources(code_root, source_hashes)
            for stage, info in stages.items():
                if info["status"] == "completed":
                    continue
                if stage in processes:
                    process, stream = processes[stage]
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    stream.close()
                    del processes[stage]
                    if returncode:
                        raise RuntimeError(f"Native {stage} exited {returncode}; see {info['log']}")
                    contract, result = verify_workflow_run(Path(info["output"]), stage, templates[stage],
                        info["previous_step"], tokenizer_checkpoint_sha256=(
                            state["frozen_tokenizer"]["checkpoint_sha256"] if stage == "pose" else None))
                    info.update(status="completed", checkpoint_sha256=result["final_checkpoint_sha256"],
                                elapsed_seconds=result["elapsed_seconds"], completed_at_unix=time.time())
                    continue
                if not ready_to_launch(stage, stages, pilot):
                    info["status"] = "waiting_for_dependency"
                    continue
                try:
                    gpu = select_gpu(info["gpu_index"])
                except RuntimeError as error:
                    info.update(status="waiting_for_idle_gpu", wait_reason=str(error))
                    continue
                resume, tokenizer = None, None
                if stage == "tokenizer":
                    resume, previous_step = template_runs[stage], tokenizer_resume["saved_step"]
                elif stage == "root":
                    resume = pilot / "root-220000"
                    checked = validate_resume(resume, {**templates[stage], "stop_steps": FULL_SCHEDULE_STEPS}, code_root)
                    previous_step = checked["saved_step"]
                else:
                    tokenizer = Path(stages["tokenizer"]["output"])
                    state["frozen_tokenizer"] = frozen_tokenizer_identity(tokenizer)
                    templates[stage] = copy.deepcopy(templates[stage])
                    templates[stage]["tokenizer_state_sha256"] = state["frozen_tokenizer"]["tensor_state_sha256"]
                    previous_step = 0
                command = native_stage_command(stage, templates[stage], info["output"], info["gpu_index"],
                                               resume_run=resume, tokenizer_run=tokenizer)
                info.update(status="launching", command=command, gpu_at_launch=gpu, previous_step=previous_step,
                            log=str(output / f"{stage}.log"), launched_at_unix=time.time())
                save_state(state_path, state)
                processes[stage] = launch(command, code_root, Path(info["log"]))
                info.update(status="running", training_pid=processes[stage][0].pid)
                save_state(state_path, state)
            state["updated_at_unix"] = time.time()
            save_state(state_path, state)
            if not all(info["status"] == "completed" for info in stages.values()):
                time.sleep(args.poll_seconds)
        state["training_completed_at_unix"] = time.time()
        state["review"] = run_final_review(output, args.review_gpu, code_root, source_hashes,
                                           deadline, args.poll_seconds, state)
        state.update(status="completed", completed_at_unix=time.time())
    except BaseException as error:
        state.update(status="failed", error=type(error).__name__ + ": " + str(error), failed_at_unix=time.time())
        for stage in processes:
            stages[stage]["status"] = "stopped_after_workflow_failure"
        raise
    finally:
        stop_owned(processes)
        save_state(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-resume", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-gpu", type=int, required=True)
    parser.add_argument("--root-gpu", type=int, required=True)
    parser.add_argument("--pose-gpu", type=int, required=True)
    parser.add_argument("--review-gpu", type=int, required=True)
    parser.add_argument("--timeout-hours", type=float, default=96)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    gpus = (args.tokenizer_gpu, args.root_gpu, args.pose_gpu, args.review_gpu)
    if min(gpus) < 0 or len(set(gpus)) != len(gpus) or min(args.timeout_hours, args.poll_seconds) <= 0:
        parser.error("Requires four distinct nonnegative GPUs and positive wait settings")
    print(json.dumps(run_workflow(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
