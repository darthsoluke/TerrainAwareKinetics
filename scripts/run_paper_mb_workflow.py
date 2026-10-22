"""Run approved flat-locomotion data and paper optimization without manual milestones."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_paper_protocol import PAPER_BATCH_PER_GPU, PAPER_STEPS, profile_notes
from motionbricks.data.native_mb_workflow import verify_workflow_run
from scripts.run_native_mb_workflow import (
    assert_sources, frozen_tokenizer_identity, launch, read_pending, run_final_review, save_state, stop_owned,
)
from scripts.train_native_mb import select_gpu, sha256, source_lf_sha256
from scripts.train_native_mb_control import SOURCE_FILES


def paper_command(stage, dataset, output, gpu, *, tokenizer_run=None):
    if stage not in ("tokenizer", "root", "pose") or (stage == "pose") != (tokenizer_run is not None):
        raise ValueError("Only Pose requires a completed frozen tokenizer")
    script = "scripts/train_native_mb.py" if stage == "tokenizer" else "scripts/train_native_mb_control.py"
    command = [sys.executable, "-u", script]
    if stage != "tokenizer":
        command += ["--stage", stage]
    command += ["--training-profile", "paper", "--dataset", str(dataset), "--output", str(output),
                "--steps", str(PAPER_STEPS), "--schedule-steps", str(PAPER_STEPS),
                "--batch-size", str(PAPER_BATCH_PER_GPU), "--seed", "7", "--workers", "2", "--gpu", str(gpu)]
    if tokenizer_run is not None:
        command += ["--tokenizer-run", str(tokenizer_run)]
    return command


def capture_stage_identity(run, stage, dataset, dataset_hash, tokenizer_hash=None):
    contract = read_pending(run / "run_contract.json")
    if contract is None:
        return None
    if (contract.get("stage") != "native_" + stage or contract.get("dataset") != str(dataset)
            or contract.get("dataset_manifest_sha256") != dataset_hash
            or contract.get("batch_size") != PAPER_BATCH_PER_GPU or contract.get("seed") != 7
            or contract.get("stop_steps") != PAPER_STEPS or contract.get("original_schedule_steps") != PAPER_STEPS
            or contract.get("training_profile") != profile_notes(stage)):
        raise ValueError(f"{stage} did not execute the approved paper profile")
    if tokenizer_hash is not None and (contract.get("tokenizer") or {}).get("checkpoint_sha256") != tokenizer_hash:
        raise ValueError("Pose loaded a different tokenizer")
    return {key: contract[key] for key in (
        "dataset_manifest_sha256", "config_sha256", "seed", "batch_size", "original_schedule_steps",
    )} | {"tokenizer_state_sha256": contract.get("tokenizer_state_sha256")}


def require_technical_checks(path):
    checks = read_pending(path)
    if checks is None:
        return False
    if checks.get("status") == "failed":
        raise RuntimeError(f"Paper batch-256 technical check failed: {checks.get('error')}")
    if checks.get("status") != "completed":
        return False
    for stage in ("tokenizer", "root", "pose"):
        item = checks["stages"][stage]
        run = Path(item["run"])
        contract = json.loads((run / "run_contract.json").read_text())
        result = json.loads((run / "training_result.json").read_text())
        if (contract.get("status") not in ({"completed"} if stage == "tokenizer" else {"completed", "trained_not_reviewed"})
                or contract.get("training_profile") != profile_notes(stage)
                or contract.get("batch_size") != 256 or result.get("optimizer_updates") != 200
                or result.get("optimizer_class") != "torch.optim.adam.Adam"
                or result.get("updates_this_run") != 200 or result.get("skipped_batches") != 0
                or result.get("observed_input_batch_sizes") != [256]
                or contract.get("sampling_contract", {}).get("input_batch_size") != 256
                or sha256(run / "checkpoints/final.ckpt") != item["checkpoint_sha256"]):
            raise ValueError("Paper technical-check artifact identity mismatch")
    return True


def run_workflow(args):
    code_root = Path(__file__).resolve().parents[1]
    source, dataset, output = args.source.resolve(), args.dataset.resolve(), args.output.resolve()
    if output == dataset or dataset in output.parents or source in output.parents:
        raise ValueError("Outputs must not overlap the input datasets")
    # Freeze executable training, data and inference sources across the queued stages.
    files = [code_root / name for name in SOURCE_FILES]
    for name in ("vqvae", "motion_backbone/models", "motion_backbone/neural_modules",
                 "motion_backbone/inference", "motionlib"):
        files += list((code_root / "motionbricks" / name).rglob("*.py"))
    files += [code_root / "motionbricks/data" / name for name in (
        "native_mb_data.py", "native_mb_models.py", "native_mb_evaluation.py",
        "native_mb_control_review.py", "native_mb_locomotion.py", "native_mb_paper_protocol.py",
        "native_mb_workflow.py")]
    files += [code_root / "scripts" / name for name in (
        "train_native_mb.py", "train_native_mb_control.py", "build_native_mb_dataset.py",
        "run_paper_mb_workflow.py", "run_native_mb_workflow.py", "evaluate_native_mb_checkpoint.py",
        "evaluate_native_mb_inbetween.py", "evaluate_native_mb_smart_locomotion.py",
        "render_native_mb_review.py", "review_native_mb_continuation.py")]
    files += [code_root / "out" / name / "version_1/hparams.yaml" for name in (
        "motionbricks_vqvae", "motionbricks_root", "motionbricks_pose")]
    source_hashes = {path.relative_to(code_root).as_posix(): source_lf_sha256(path) for path in files}
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "workflow_state.json"
    stages = {stage: {"status": "queued", "gpu_index": getattr(args, stage + "_gpu"),
                      "output": str(output / f"{stage}-{PAPER_STEPS}")}
              for stage in ("tokenizer", "root", "pose")}
    state = {"status": "waiting_for_data", "pid": os.getpid(), "started_at_unix": time.time(),
             "source_dataset": str(source), "dataset": str(dataset), "stages": stages,
             "profile": profile_notes("tokenizer"), "manual_gates": False,
             "source_lf_sha256": source_hashes, "quality_approved": False, "ue_deployment": False,
             "technical_checks": str(args.technical_checks.resolve()),
             "notes": ["Fresh paper-optimizer training; no legacy optimizer state",
                       "Fewer physical GPUs than paper; not an equivalent 8192-sample global batch",
                       "LAFAN not included until SOMA conversion is verified"]}
    processes = {}
    deadline = time.monotonic() + args.timeout_days * 86400

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError("Paper workflow deadline exceeded")
        assert_sources(code_root, source_hashes)
        state["updated_at_unix"] = time.time()
        save_state(state_path, state)

    try:
        save_state(state_path, state)
        while True:
            tick()
            report = read_pending(source / "expansion_report.json")
            if report and report.get("status") == "completed":
                state["training_data_summary"] = {key: report[key] for key in (
                    "split_counts", "train_duration_hours", "train_behaviors", "rejected")}
                break
            if not Path(f"/proc/{args.preparation_pid}").exists():
                raise RuntimeError("Source preparation exited before publishing its completed report")
            time.sleep(args.poll_seconds)
        state["status"] = "converting_native_data"
        command = [sys.executable, "-u", "scripts/build_native_mb_dataset.py",
                   "--source", str(source), "--output", str(dataset)]
        processes["data"] = launch(command, code_root, output / "native-data.log")
        while processes["data"][0].poll() is None:
            tick()
            time.sleep(args.poll_seconds)
        process, stream = processes.pop("data")
        stream.close()
        if process.returncode:
            raise RuntimeError("Native data conversion failed; see native-data.log")
        state["dataset_manifest_sha256"] = sha256(dataset / "native_manifest.json")
        state["status"] = "waiting_for_technical_checks"
        while not require_technical_checks(args.technical_checks):
            tick()
            time.sleep(args.poll_seconds)
        state["status"] = "training"
        while not all(info["status"] == "completed" for info in stages.values()):
            tick()
            if sha256(dataset / "native_manifest.json") != state["dataset_manifest_sha256"]:
                raise ValueError("Approved native data manifest changed")
            for stage, info in stages.items():
                if info["status"] == "completed":
                    continue
                if stage in processes:
                    process, stream = processes[stage]
                    expected = capture_stage_identity(Path(info["output"]), stage, dataset,
                        state["dataset_manifest_sha256"], state.get("frozen_tokenizer", {}).get(
                            "checkpoint_sha256") if stage == "pose" else None)
                    if expected is not None:
                        if info.get("initial_identity", expected) != expected:
                            raise ValueError("A running training stage changed its configuration")
                        info["initial_identity"] = expected
                    if process.poll() is None:
                        continue
                    stream.close()
                    del processes[stage]
                    if process.returncode or expected is None:
                        raise RuntimeError(f"Native paper {stage} failed; see {info['log']}")
                    if stage == "pose":
                        expected["tokenizer_state_sha256"] = state["frozen_tokenizer"]["tensor_state_sha256"]
                    _, result = verify_workflow_run(info["output"], stage, expected, 0,
                        target_steps=PAPER_STEPS, tokenizer_checkpoint_sha256=(
                            state["frozen_tokenizer"]["checkpoint_sha256"] if stage == "pose" else None))
                    if result.get("optimizer_class") != "torch.optim.adam.Adam":
                        raise ValueError("Completed paper stage did not use Adam")
                    info.update(status="completed", checkpoint_sha256=result["final_checkpoint_sha256"],
                                elapsed_seconds=result["elapsed_seconds"], completed_at_unix=time.time())
                    continue
                if stage == "pose" and stages["tokenizer"]["status"] != "completed":
                    info["status"] = "waiting_for_frozen_tokenizer"
                    continue
                try:
                    gpu = select_gpu(info["gpu_index"])
                except RuntimeError as error:
                    info.update(status="waiting_for_idle_gpu", wait_reason=str(error))
                    continue
                tokenizer = None
                if stage == "pose":
                    tokenizer = Path(stages["tokenizer"]["output"])
                    state["frozen_tokenizer"] = frozen_tokenizer_identity(tokenizer, PAPER_STEPS)
                command = paper_command(stage, dataset, info["output"], info["gpu_index"], tokenizer_run=tokenizer)
                info.update(status="launching", command=command, gpu_at_launch=gpu,
                            log=str(output / f"{stage}.log"), launched_at_unix=time.time())
                save_state(state_path, state)
                processes[stage] = launch(command, code_root, Path(info["log"]))
                info.update(status="running", training_pid=processes[stage][0].pid)
            save_state(state_path, state)
            if not all(info["status"] == "completed" for info in stages.values()):
                time.sleep(args.poll_seconds)
        state["training_completed_at_unix"] = time.time()
        state["review"] = run_final_review(output, args.review_gpu, code_root, source_hashes,
            deadline, args.poll_seconds, state, step=PAPER_STEPS)
        state.update(status="completed", completed_at_unix=time.time())
    except BaseException as error:
        state.update(status="failed", error=type(error).__name__ + ": " + str(error), failed_at_unix=time.time())
        for stage in processes:
            if stage in stages:
                stages[stage]["status"] = "stopped_after_workflow_failure"
        raise
    finally:
        stop_owned(processes)
        save_state(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation-pid", type=int, required=True)
    parser.add_argument("--technical-checks", type=Path, required=True)
    parser.add_argument("--tokenizer-gpu", type=int, required=True)
    parser.add_argument("--root-gpu", type=int, required=True)
    parser.add_argument("--pose-gpu", type=int, required=True)
    parser.add_argument("--review-gpu", type=int, required=True)
    parser.add_argument("--timeout-days", type=float, default=45)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if min(args.preparation_pid, args.timeout_days, args.poll_seconds) <= 0:
        parser.error("Positive PID and wait settings required")
    if min(args.tokenizer_gpu, args.root_gpu, args.pose_gpu, args.review_gpu) < 0:
        parser.error("Nonnegative GPU indices required")
    if args.tokenizer_gpu == args.root_gpu or args.review_gpu in (args.tokenizer_gpu, args.root_gpu, args.pose_gpu):
        parser.error("Independent training and review must not reserve the same GPU")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Workflow terminated")))
    print(json.dumps(run_workflow(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
