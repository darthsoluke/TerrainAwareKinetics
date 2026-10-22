"""Train the official MotionBricks paper profile only from accepted LAFAN SOMA data."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motionbricks.data.native_mb_workflow import verify_workflow_run
from scripts.run_native_mb_workflow import (
    assert_sources, frozen_tokenizer_identity, launch, read_pending,
    save_state, stop_owned,
)
from scripts.train_native_mb import select_gpu, sha256, source_lf_sha256, verify_dataset
from scripts.train_native_mb_control import SOURCE_FILES

PAPER_STEPS = 2_000_000


def paper_command(*args, **kwargs):
    from scripts.run_paper_mb_workflow import paper_command as build_command

    command = build_command(*args, **kwargs)
    stage = args[0] if args else kwargs.get("stage")
    if stage in {"root", "pose"}:
        command = [*command, "--skip-review"]
    return command


def capture_stage_identity(*args, **kwargs):
    from scripts.run_paper_mb_workflow import capture_stage_identity as capture

    return capture(*args, **kwargs)


def require_technical_checks(path):
    from scripts.run_paper_mb_workflow import require_technical_checks as require

    return require(path)


def _source_acceptance(source, review_path):
    """Validate the receipt and all immutable source identities before any output is made."""
    source = Path(source).resolve()
    accepted_path = source / "accepted.json"
    review = read_pending(review_path)
    if review is None:
        return None
    accepted_hash = sha256(accepted_path)
    if review.get("status") != "passed" or review.get("training_eligible") is not True:
        raise ValueError("Retarget review is not machine-approved for training")
    if review.get("accepted_contains_failed_clips") is not False:
        raise ValueError("Retarget review must certify that accepted.json contains no failed clips")
    if review.get("source_accepted_sha256") != accepted_hash:
        raise ValueError("Retarget review does not bind this accepted.json")
    heldout = review.get("heldout_subjects")
    if heldout != {"validation": ["subject4"], "test": ["subject5"]}:
        raise ValueError("Retarget review must reserve subject4 for validation and subject5 for test")
    limitations = review.get("assessment_limitations", review.get("limitations"))
    if not isinstance(limitations, (list, dict)) or not limitations:
        raise ValueError("Retarget review must record assessment limitations")

    records = json.loads(accepted_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("accepted.json must contain accepted shard records")
    shard_paths = []
    subjects = {split: set() for split in ("train", "validation", "test")}
    split_counts = {split: 0 for split in subjects}
    for row in records:
        assessment = row.get("retarget_assessment")
        if not isinstance(assessment, dict) or assessment.get("status") != "passed":
            raise ValueError("Every accepted row must contain a passed retarget_assessment")
        split, subject, relative = row.get("split"), row.get("subject_id"), row.get("shard")
        if split not in subjects or not isinstance(subject, str) or not subject or not isinstance(relative, str):
            raise ValueError("Each accepted record needs a valid split, subject_id, and shard path")
        path = (source / relative).resolve()
        if source not in path.parents or not path.is_file():
            raise ValueError(f"Accepted shard is missing or escapes the source directory: {relative}")
        shard_paths.append(Path(relative).as_posix())
        subjects[split].add(subject)
        split_counts[split] += 1
    if len(set(shard_paths)) != len(shard_paths):
        raise ValueError("accepted.json contains duplicate shard paths")
    if any(count == 0 or not subjects[split] for split, count in split_counts.items()):
        raise ValueError("Train, validation, and test must each contain accepted shards and subjects")
    if subjects["validation"] != {"subject4"} or subjects["test"] != {"subject5"}:
        raise ValueError("Accepted validation/test subjects do not match the reviewed held-outs")
    if subjects["train"] & (subjects["validation"] | subjects["test"]):
        raise ValueError("Training and held-out subject IDs must be disjoint")

    declared = review.get("accepted_shard_sha256")
    if not isinstance(declared, dict) or set(declared) != set(shard_paths):
        raise ValueError("Retarget review must hash every accepted shard and no others")
    for relative, expected in declared.items():
        path = (source / relative).resolve()
        if source not in path.parents or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Accepted shard hash mismatch: {relative}")
    return {"accepted_sha256": accepted_hash, "shard_sha256": dict(declared),
            "split_counts": split_counts, "subject_ids": {key: sorted(value) for key, value in subjects.items()},
            "assessment_limitations": limitations,
            "source_inventory_quality_status": review.get("source_inventory_quality_status")}


def _preparation_alive(pid):
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        return False
    except PermissionError:
        alive = True
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except FileNotFoundError:
            return False
        except OSError:
            return alive
        fields_after_command = stat.rsplit(")", 1)[-1].strip().split()
        if fields_after_command and fields_after_command[0] in {"Z", "X"}:
            return False
    return alive


def _raise_preparation_failure(source):
    path = Path(source) / "preparation_failure.json"
    if path.exists():
        failure = read_pending(path)
        detail = failure.get("error") if isinstance(failure, dict) else None
        raise RuntimeError("Source preparation published preparation_failure.json"
                           + (f": {detail}" if detail else ""))


def _source_hashes(code_root):
    files = [code_root / name for name in SOURCE_FILES]
    files += [code_root / name for name in (
        "scripts/run_verified_lafan_workflow.py", "scripts/run_paper_mb_workflow.py",
        "scripts/run_native_mb_workflow.py", "scripts/build_native_mb_dataset.py",
        "scripts/evaluate_native_mb_checkpoint.py", "scripts/evaluate_native_mb_inbetween.py",
        "scripts/evaluate_native_mb_smart_locomotion.py", "scripts/render_native_mb_review.py",
        "scripts/check_paper_mb_training.py",
        "scripts/evaluate_native_mb_lafan.py", "scripts/render_lafan_soma_retarget.py",
        "motionbricks/data/native_mb_data.py", "motionbricks/data/native_mb_models.py",
        "motionbricks/data/native_mb_evaluation.py", "motionbricks/data/native_mb_control_review.py",
        "motionbricks/data/native_mb_locomotion.py", "motionbricks/data/native_mb_paper_protocol.py",
        "motionbricks/data/native_mb_workflow.py")]
    for name in ("vqvae", "motion_backbone/models", "motion_backbone/neural_modules",
                 "motion_backbone/inference", "motionlib"):
        files += list((code_root / "motionbricks" / name).rglob("*.py"))
    files += [code_root / "out" / name / "version_1/hparams.yaml" for name in (
        "motionbricks_vqvae", "motionbricks_root", "motionbricks_pose")]
    return {path.relative_to(code_root).as_posix(): source_lf_sha256(path) for path in files}


def run_lafan_final_review(source, dataset, output, gpu, code_root, source_hashes,
                           deadline, poll_seconds, state, step=PAPER_STEPS):
    review = Path(output) / f"lafan-review-{step}"
    command = [sys.executable, "-u", "scripts/evaluate_native_mb_lafan.py",
               "--source", str(source), "--dataset", str(dataset),
               "--tokenizer-run", state["stages"]["tokenizer"]["output"],
               "--root-run", state["stages"]["root"]["output"],
               "--pose-run", state["stages"]["pose"]["output"],
               "--output", str(review)]
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for LAFAN review GPU")
        try:
            gpu_identity = select_gpu(gpu)
            break
        except RuntimeError as error:
            state.update(status="waiting_for_idle_lafan_review_gpu", review_gpu_wait_reason=str(error))
            save_state(Path(output) / "workflow_state.json", state)
            time.sleep(poll_seconds)
    assert_sources(code_root, source_hashes)
    if sha256(Path(source) / "accepted.json") != state["acceptance"]["accepted_sha256"]:
        raise ValueError("Accepted source manifest changed before LAFAN review")
    for relative, expected in state["acceptance"]["shard_sha256"].items():
        if sha256(Path(source) / relative) != expected:
            raise ValueError(f"Accepted source shard changed before LAFAN review: {relative}")
    native_manifest = json.loads((Path(dataset) / "native_manifest.json").read_text(encoding="utf-8"))
    verify_dataset(Path(dataset), native_manifest)
    if sha256(Path(dataset) / "native_manifest.json") != state["dataset_manifest_sha256"]:
        raise ValueError("Native dataset manifest changed before LAFAN review")
    if review.exists():
        raise FileExistsError(f"LAFAN review output already exists: {review}")
    state.update(status="evaluating_lafan_validation", review_command=command,
                 review_gpu_at_launch=gpu_identity, review_log=str(Path(output) / "lafan-review.log"))
    save_state(Path(output) / "workflow_state.json", state)
    owned = {"lafan_review": launch(command, code_root, Path(state["review_log"]))}
    try:
        while owned["lafan_review"][0].poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("LAFAN validation review exceeded its deadline")
            assert_sources(code_root, source_hashes)
            if sha256(Path(source) / "accepted.json") != state["acceptance"]["accepted_sha256"]:
                raise ValueError("Accepted source manifest changed during LAFAN review")
            if sha256(Path(dataset) / "native_manifest.json") != state["dataset_manifest_sha256"]:
                raise ValueError("Native dataset manifest changed during LAFAN review")
            time.sleep(poll_seconds)
        process = owned["lafan_review"][0]
        if process.returncode:
            raise RuntimeError(f"LAFAN validation review failed; see {state['review_log']}")
        report_path = review / "report.json"
        report = read_pending(report_path)
        candidate_path = review / "candidate_bundle.json"
        candidate = read_pending(candidate_path)
        if (report is None or report.get("status") != "completed" or report.get("quality_approved") is not False
                or candidate is None or candidate.get("quality_approved") is not False
                or candidate.get("status") != "trained_evaluated_not_approved"
                or candidate.get("paper_steps") != step
                or report.get("source_accepted_sha256") != state["acceptance"]["accepted_sha256"]
                or report.get("dataset_manifest_sha256") != state["dataset_manifest_sha256"]
                or candidate.get("dataset_manifest_sha256") != state["dataset_manifest_sha256"]
                or report.get("subject_id") != "subject4"
                or report.get("behaviors") != ["walk", "run"]
                or report.get("wasd_continuous_review", {}).get("status") != "not_assessed"
                or not isinstance(report.get("windows"), list) or not report["windows"]
                or any(row.get("subject_id") != "subject4" or row.get("split") != "validation"
                       or row.get("behavior") not in {"walk", "run"}
                       or state["acceptance"]["shard_sha256"].get(row.get("source_shard"))
                          != row.get("source_shard_sha256")
                       for row in report["windows"])):
            raise ValueError("LAFAN review did not publish the expected unapproved candidate and report")
        for stage, info in state["stages"].items():
            candidate_stage = candidate.get("stages", {}).get(stage, {})
            if (candidate_stage.get("checkpoint_sha256") != info["checkpoint_sha256"]
                    or candidate_stage.get("optimizer_updates") != step):
                raise ValueError(f"LAFAN candidate does not bind the completed {stage} checkpoint")
        frozen = candidate.get("frozen_tokenizer") or {}
        if (frozen.get("checkpoint_sha256") != state["frozen_tokenizer"]["checkpoint_sha256"]
                or frozen.get("tensor_state_sha256") != state["frozen_tokenizer"]["tensor_state_sha256"]):
            raise ValueError("LAFAN candidate does not bind the frozen tokenizer tensors")
        return {"report": str(report_path), "candidate_bundle": str(candidate_path),
                "window_count": report["window_count"], "quality_approved": False,
                "wasd_continuous_review": report["wasd_continuous_review"]}
    finally:
        stop_owned(owned)


def run_lafan_technical_check(source, dataset, output, gpu, code_root, source_hashes,
                              deadline, poll_seconds, state):
    check_output = Path(output) / "technical-check"
    checks_path = check_output / "checks.json"
    log_path = Path(output) / "technical-check.log"
    command = [sys.executable, "-u", "scripts/check_paper_mb_training.py",
               "--dataset", str(dataset), "--output", str(check_output), "--gpu", str(gpu)]
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for LAFAN technical-check GPU")
        try:
            gpu_identity = select_gpu(gpu)
            break
        except RuntimeError as error:
            state.update(status="waiting_for_idle_technical_check_gpu",
                         technical_check_gpu_wait_reason=str(error))
            save_state(Path(output) / "workflow_state.json", state)
            time.sleep(poll_seconds)

    assert_sources(code_root, source_hashes)
    if sha256(Path(source) / "accepted.json") != state["acceptance"]["accepted_sha256"]:
        raise ValueError("Accepted source manifest changed before technical check")
    for relative, expected in state["acceptance"]["shard_sha256"].items():
        if sha256(Path(source) / relative) != expected:
            raise ValueError(f"Accepted source shard changed before technical check: {relative}")
    manifest_path = Path(dataset) / "native_manifest.json"
    if sha256(manifest_path) != state["dataset_manifest_sha256"]:
        raise ValueError("Native dataset manifest changed before technical check")
    if check_output.exists():
        raise FileExistsError(f"Technical-check output already exists: {check_output}")
    state.update(status="running_technical_check", technical_check={
        "status": "running", "command": command, "gpu_at_launch": gpu_identity,
        "output": str(check_output), "checks": str(checks_path), "log": str(log_path),
        "production_checkpoint": False})
    save_state(Path(output) / "workflow_state.json", state)
    owned = {"technical_check": launch(command, code_root, log_path)}
    try:
        while owned["technical_check"][0].poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("LAFAN technical check exceeded its deadline")
            assert_sources(code_root, source_hashes)
            if sha256(Path(source) / "accepted.json") != state["acceptance"]["accepted_sha256"]:
                raise ValueError("Accepted source manifest changed during technical check")
            for relative, expected in state["acceptance"]["shard_sha256"].items():
                if sha256(Path(source) / relative) != expected:
                    raise ValueError(f"Accepted source shard changed during technical check: {relative}")
            if sha256(manifest_path) != state["dataset_manifest_sha256"]:
                raise ValueError("Native dataset manifest changed during technical check")
            time.sleep(poll_seconds)
        process, stream = owned["technical_check"]
        if process.returncode:
            raise RuntimeError(f"LAFAN paper-profile technical check failed; see {log_path}")
        checks = read_pending(checks_path)
        if (checks is None or checks.get("status") != "completed"
                or checks.get("production_checkpoint") is not False
                or set(checks.get("stages", {})) != {"tokenizer", "root", "pose"}):
            raise ValueError("Technical check did not publish complete isolated 200-update results")
        production_runs = [Path(info["output"]).resolve() for info in state["stages"].values()]
        for item in checks["stages"].values():
            check_run = Path(item["run"]).resolve()
            if check_output.resolve() not in check_run.parents:
                raise ValueError("Technical-check stage output escaped its isolated directory")
            if any(check_run == run or check_run in run.parents or run in check_run.parents
                   for run in production_runs):
                raise ValueError("Technical-check weights overlap a production checkpoint path")
        if not require_technical_checks(checks_path):
            raise ValueError("Paper-profile technical check did not pass artifact verification")
        stream.close()
        check = state["technical_check"]
        check.update(status="passed", checks_sha256=sha256(checks_path),
                     completed_at_unix=time.time(), production_checkpoint=False)
        state["status"] = "training"
        return check
    finally:
        stop_owned(owned)


def _validate_paths(source, dataset, output):
    source, dataset, output = (Path(path).resolve() for path in (source, dataset, output))
    if not source.is_dir():
        raise FileNotFoundError(f"Prepared source directory does not exist: {source}")
    if dataset.exists() or output.exists():
        raise FileExistsError("Dataset and run output must both be fresh directories")
    if (source == dataset or source in dataset.parents or dataset in source.parents
            or source == output or source in output.parents or output in source.parents
            or dataset == output or dataset in output.parents or output in dataset.parents):
        raise ValueError("Source, dataset, and run output must be separate non-overlapping paths")
    return source, dataset, output


def run_workflow(args):
    code_root = Path(__file__).resolve().parents[1]
    source, dataset, output = _validate_paths(args.source, args.dataset, args.output)
    review_path = args.review.resolve()
    deadline = time.monotonic() + args.timeout_days * 86400

    while True:
        _raise_preparation_failure(source)
        acceptance = _source_acceptance(source, review_path)
        if acceptance is not None:
            _raise_preparation_failure(source)
            break
        if args.preparation_pid is not None and not _preparation_alive(args.preparation_pid):
            raise RuntimeError("Preparation exited before publishing an accepted retarget review")
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the accepted retarget review")
        time.sleep(args.poll_seconds)

    source_hashes = _source_hashes(code_root)
    assert_sources(code_root, source_hashes)
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "workflow_state.json"
    stages = {stage: {"status": "queued", "gpu_index": getattr(args, stage + "_gpu"),
                      "output": str(output / f"{stage}-{PAPER_STEPS}")}
              for stage in ("tokenizer", "root", "pose")}
    state = {"status": "converting_native_data", "pid": os.getpid(), "started_at_unix": time.time(),
             "source_dataset": str(source), "retarget_review": str(review_path), "acceptance": acceptance,
             "dataset": str(dataset), "stages": stages, "paper_steps": PAPER_STEPS,
             "source_lf_sha256": source_hashes, "manual_gates": False,
             "training_strategy": "fresh Tokenizer and Root; fresh Pose after frozen completed Tokenizer",
             "quality_approved": False, "ue_deployment": False,
             "limitations": ["Source acknowledgement is 64 minutes total, not all training data.",
                             "No exact-paper architecture/global-batch equivalence or quality claim.",
                             "Continuous WASD review is not assessed; no LAFAN command/reference bank exists.",
                             "No UE deployment or performance claim."]}
    processes = {}

    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError("Verified LAFAN workflow deadline exceeded")
        assert_sources(code_root, source_hashes)
        if sha256(source / "accepted.json") != acceptance["accepted_sha256"]:
            raise ValueError("Accepted source manifest changed during workflow")
        for relative, expected in acceptance["shard_sha256"].items():
            if sha256(source / relative) != expected:
                raise ValueError(f"Accepted source shard changed during workflow: {relative}")
        state["updated_at_unix"] = time.time()
        save_state(state_path, state)

    try:
        save_state(state_path, state)
        tick()
        converter = [sys.executable, "-u", "scripts/build_native_mb_dataset.py",
                    "--source", str(source), "--output", str(dataset)]
        processes["data"] = launch(converter, code_root, output / "native-data.log")
        while processes["data"][0].poll() is None:
            tick()
            time.sleep(args.poll_seconds)
        process, stream = processes.pop("data")
        stream.close()
        if process.returncode:
            raise RuntimeError("Native data conversion failed; see native-data.log")
        manifest = json.loads((dataset / "native_manifest.json").read_text(encoding="utf-8"))
        verify_dataset(dataset, manifest)
        assert_sources(code_root, source_hashes)
        state["dataset_manifest_sha256"] = sha256(dataset / "native_manifest.json")
        run_lafan_technical_check(source, dataset, output, args.tokenizer_gpu,
            code_root, source_hashes, deadline, args.poll_seconds, state)
        state["status"] = "training"
        save_state(state_path, state)

        while not all(info["status"] == "completed" for info in stages.values()):
            tick()
            if sha256(dataset / "native_manifest.json") != state["dataset_manifest_sha256"]:
                raise ValueError("Native dataset manifest changed during training")
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
                            raise ValueError(f"{stage} configuration changed during training")
                        info["initial_identity"] = expected
                    if process.poll() is None:
                        continue
                    stream.close()
                    del processes[stage]
                    if process.returncode or expected is None:
                        raise RuntimeError(f"Native paper {stage} failed; see {info['log']}")
                    if stage == "pose":
                        expected["tokenizer_state_sha256"] = state["frozen_tokenizer"]["tensor_state_sha256"]
                    _, result = verify_workflow_run(Path(info["output"]), stage, expected, 0,
                        target_steps=PAPER_STEPS, tokenizer_checkpoint_sha256=(
                            state["frozen_tokenizer"]["checkpoint_sha256"] if stage == "pose" else None))
                    if result.get("optimizer_class") != "torch.optim.adam.Adam":
                        raise ValueError(f"Paper {stage} did not use Adam")
                    info.update(status="completed", checkpoint_sha256=result["final_checkpoint_sha256"],
                                elapsed_seconds=result["elapsed_seconds"], completed_at_unix=time.time(),
                                optimizer_updates=result["optimizer_updates"])
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
                command = paper_command(stage, dataset, info["output"], info["gpu_index"],
                                        tokenizer_run=tokenizer)
                info.update(status="launching", command=command, gpu_at_launch=gpu,
                            log=str(output / f"{stage}.log"), launched_at_unix=time.time())
                save_state(state_path, state)
                processes[stage] = launch(command, code_root, Path(info["log"]))
                info.update(status="running", training_pid=processes[stage][0].pid)
            save_state(state_path, state)
            if not all(info["status"] == "completed" for info in stages.values()):
                time.sleep(args.poll_seconds)
        state["training_completed_at_unix"] = time.time()
        state["review"] = run_lafan_final_review(source, dataset, output, args.review_gpu,
            code_root, source_hashes, deadline, args.poll_seconds, state, step=PAPER_STEPS)
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
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation-pid", type=int)
    parser.add_argument("--tokenizer-gpu", type=int, required=True)
    parser.add_argument("--root-gpu", type=int, required=True)
    parser.add_argument("--pose-gpu", type=int, required=True)
    parser.add_argument("--review-gpu", type=int, required=True)
    parser.add_argument("--timeout-days", type=float, default=45)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    gpus = (args.tokenizer_gpu, args.root_gpu, args.pose_gpu, args.review_gpu)
    if min(gpus) < 0 or len({args.tokenizer_gpu, args.root_gpu, args.review_gpu}) != 3:
        parser.error("Requires nonnegative GPUs; Tokenizer/Root/Review must be distinct")
    if args.pose_gpu in (args.root_gpu, args.review_gpu):
        parser.error("Pose may reuse Tokenizer GPU, but not Root or Review GPU")
    if min(args.timeout_days, args.poll_seconds) <= 0 or (
            args.preparation_pid is not None and args.preparation_pid <= 0):
        parser.error("Positive timeout, poll interval, and optional preparation PID required")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Workflow terminated")))
    print(json.dumps(run_workflow(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
