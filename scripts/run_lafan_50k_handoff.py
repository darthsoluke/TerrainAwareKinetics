"""Finish requested 50k snapshots, train official Pose50k, then review held-out LAFAN."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_native_mb_snapshot import export_snapshot, file_hash, write_json


def process_state(identity):
    """Check UID, process start time and command before signaling an adopted process."""
    proc = Path("/proc") / str(identity["pid"])
    if not proc.exists():
        return None
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    if stat[0] == "Z":
        return None
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    if (proc.stat().st_uid != os.getuid() or stat[19] != str(identity["proc_start_ticks"])
            or os.getpgid(identity["pid"]) != identity["pgid"] or command != identity["command"].strip()):
        raise ValueError("Adopted process identity changed; refusing signal")
    return stat[0]


def signal_verified(identity, sig):
    if process_state(identity) is not None:
        os.killpg(identity["pgid"], sig)


def terminate_verified(identity):
    if process_state(identity) is None:
        return
    signal_verified(identity, signal.SIGTERM)
    signal_verified(identity, signal.SIGCONT)
    deadline = time.monotonic() + 40
    while process_state(identity) is not None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process_state(identity) is not None:
        signal_verified(identity, signal.SIGKILL)


def stable_file(path):
    if not path.is_file():
        return False
    first = (path.stat().st_size, path.stat().st_mtime_ns)
    time.sleep(1)
    return first[0] > 0 and path.is_file() and first == (path.stat().st_size, path.stat().st_mtime_ns)


def verify_pose_result(run, steps, tokenizer):
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    if (contract["status"] not in {"completed", "trained_not_reviewed"}
            or contract["stage"] != "native_pose" or result["optimizer_updates"] != steps
            or result["updates_this_run"] != steps or result["skipped_batches"] != 0
            or contract["original_schedule_steps"] != 2_000_000
            or contract["tokenizer"]["checkpoint_sha256"] != tokenizer["checkpoint_sha256"]
            or contract["dataset_manifest_sha256"] != tokenizer["dataset_manifest_sha256"]
            or not contract.get("tokenizer_state_sha256")
            or result["tokenizer_state_sha256"] != contract["tokenizer_state_sha256"]
            or file_hash(run / "checkpoints/final.ckpt") != result["final_checkpoint_sha256"]):
        raise ValueError("Pose final updates/frozen dependency/checkpoint binding failed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--pose-gpu", type=int, required=True)
    parser.add_argument("--review-gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50_000)
    args = parser.parse_args(argv)
    if args.steps != 50_000:
        parser.error("This handoff is explicitly authorized for 50,000 updates")
    request = json.loads(args.request.read_text())
    old = Path(request["source_workflow"])
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output / "workflow_state.json"
    state = {"status": "selecting_tokenizer", "pid": os.getpid(), "started_at_unix": time.time(),
             "source_workflow": str(old), "requested_stage_steps": args.steps,
             "schedule_steps": 2_000_000, "quality_approved": False,
             "paper_training_completed": False, "stages": {}}
    child = None
    stream = None

    def save():
        write_json(state_path, state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        if process_state(request["manager"]) != "T":
            raise ValueError("Old workflow must remain suspended during handoff")
        if process_state(request["tokenizer"]) is not None:
            raise ValueError("Old tokenizer must be stopped before immutable selection")
        tokenizer = export_snapshot(old / "tokenizer-2000000", request["tokenizer_checkpoint"],
                                    args.output / "tokenizer-50000", args.steps)
        state["stages"]["tokenizer"] = {"status": "checkpoint_selected", **tokenizer}
        state["status"] = "waiting_for_root_50000"
        save()
        checkpoint = old / "root-2000000/checkpoints/native-step=50000.ckpt"
        deadline = time.monotonic() + 3600
        while not stable_file(checkpoint):
            if process_state(request["root"]) is None:
                raise RuntimeError("Root stopped before its exact 50k checkpoint was saved")
            if time.monotonic() > deadline:
                raise TimeoutError("Root checkpoint wait exceeded one hour")
            time.sleep(1)
        signal_verified(request["root"], signal.SIGSTOP)
        state["status"] = "selecting_root"
        save()
        root = export_snapshot(old / "root-2000000", checkpoint, args.output / "root-50000", args.steps)
        state["stages"]["root"] = {"status": "checkpoint_selected", **root}
        terminate_verified(request["root"])
        terminate_verified(request["manager"])
        for name in ("tokenizer", "root"):
            path = old / f"{name}-2000000/run_contract.json"
            contract = json.loads(path.read_text())
            contract.update(status="stopped_at_user_request", selected_checkpoint_steps=args.steps,
                            bounded_snapshot=state["stages"][name]["run"],
                            quality_approved=False, paper_training_completed=False)
            write_json(path, contract)
        previous = json.loads((old / "workflow_state.json").read_text())
        previous.update(status="stopped_at_user_request", superseded_by=str(args.output),
                        requested_stage_steps=args.steps, paper_training_completed=False)
        write_json(old / "workflow_state.json", previous)
        dataset = json.loads((Path(tokenizer["run"]) / "run_contract.json").read_text())["dataset"]
        code_root = Path(__file__).resolve().parents[1]
        from scripts.train_native_mb import select_gpu
        state["pose_gpu_at_launch"] = select_gpu(args.pose_gpu)
        command = [sys.executable, "-u", "scripts/train_native_mb_control.py", "--stage", "pose",
                   "--training-profile", "paper", "--dataset", dataset,
                   "--output", str(args.output / "pose-50000"), "--tokenizer-run", tokenizer["run"],
                   "--steps", str(args.steps), "--schedule-steps", "2000000", "--batch-size", "256",
                   "--workers", "2", "--seed", "7", "--gpu", str(args.pose_gpu), "--skip-review"]
        stream = (args.output / "pose.log").open("xb")
        child = subprocess.Popen(command, cwd=code_root, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        state["status"] = "training_pose"
        state["stages"]["pose"] = {"status": "running", "pid": child.pid, "command": command,
                                     "run": str(args.output / "pose-50000")}
        save()
        if child.wait() != 0:
            raise RuntimeError("Pose training failed; see pose.log")
        child = None
        stream.close()
        stream = None
        result = verify_pose_result(args.output / "pose-50000", args.steps, tokenizer)
        state["stages"]["pose"].update(status="trained_not_reviewed", result=result)
        state["status"] = "reviewing"
        state["review_gpu_at_launch"] = select_gpu(args.review_gpu)
        save()
        command = [sys.executable, "-u", "scripts/evaluate_native_mb_lafan.py",
                   "--source", str(args.source_dataset), "--dataset", dataset,
                   "--tokenizer-run", tokenizer["run"], "--root-run", root["run"],
                   "--pose-run", str(args.output / "pose-50000"), "--output", str(args.output / "review"),
                   "--expected-stage-steps", str(args.steps)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.review_gpu)
        stream = (args.output / "review.log").open("xb")
        child = subprocess.Popen(command, cwd=code_root, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        state["review_command"] = command
        state["review_pid"] = child.pid
        save()
        if child.wait() != 0:
            raise RuntimeError("LAFAN evaluation failed; trained checkpoints are preserved")
        child = None
        state.update(status="trained_evaluated_not_approved", finished_at_unix=time.time())
        save()
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}", failed_at_unix=time.time())
        save()
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for name in ("root", "manager"):
            try:
                terminate_verified(request[name])
            except (OSError, ValueError) as cleanup_error:
                state.setdefault("cleanup_errors", []).append(str(cleanup_error))
        save()
        raise
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
