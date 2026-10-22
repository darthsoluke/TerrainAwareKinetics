"""Guarded Pose DDP continuation and fixed held-out native MB inference review."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_native_mb_snapshot import file_hash, write_json


def idle_gpus():
    busy = {line.split(",")[0].strip() for line in subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True).splitlines()
        if line.strip()}
    rows = []
    for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True).splitlines():
        index, uuid, memory, utilization = [value.strip() for value in line.split(",")]
        if uuid not in busy and int(memory) < 1024 and int(utilization) <= 5:
            rows.append({"index": int(index), "uuid": uuid, "used_mib": int(memory),
                         "utilization_percent": int(utilization), "checked_at_unix": time.time()})
    return rows


def verify_run(run, start, stop, world):
    from scripts.train_native_mb_ddp import verify_rank_receipts
    run = Path(run)
    contract = json.loads((run / "run_contract.json").read_text())
    result = json.loads((run / "training_result.json").read_text())
    if (contract["status"] != "trained_not_reviewed" or result["world_size"] != world
            or result["optimizer_updates"] != stop or result["updates_this_run"] != stop - start
            or result["skipped_batches"] != 0 or result["optimizer_class"] != "torch.optim.adam.Adam"
            or contract["original_schedule_steps"] != 2_000_000
            or file_hash(run / "checkpoints/final.ckpt") != result["final_checkpoint_sha256"]):
        raise ValueError("DDP final checkpoint/update/optimizer budget contract failed")
    verify_rank_receipts(result["rank_receipts"], start, stop, world, contract["tokenizer_state_sha256"])
    if contract.get("batch_profile"):
        plan = contract["batch_profile"]
        numerics = contract["runtime_numerics"]
        if (result.get("batch_profile") != contract["batch_profile"]
                or plan["per_gpu"] * world != plan["global"]
                or (plan["profile"] == "fixed_global_batch" and plan["global"] != 256)
                or contract["batch_size"] != plan["per_gpu"]
                or any(x["batch_size"] != plan["per_gpu"]
                       or x["observed_input_batch_sizes"] != [plan["per_gpu"]] for x in result["rank_receipts"])
                or result.get("runtime_numerics") != numerics
                or numerics["trainer_precision"] != 32 or numerics["bf16"]
                or (not contract.get("benchmark_only") and numerics["effective_matmul_precision"] != "high")
                or any(x.get("runtime_numerics") != numerics for x in result["rank_receipts"])):
            raise ValueError("DDP batch/numerical execution receipts disagree")
    return result


def select_batch_gpus(gpus, profile, global_batch_size=None):
    if profile == "paper":
        if global_batch_size is not None or len(gpus) < 2:
            raise ValueError("Paper DDP requires at least two idle GPUs and no global batch override")
        return gpus, 256
    if profile != "fixed_global_batch" or global_batch_size != 256 or len(gpus) < 4:
        raise ValueError("Efficiency continuation requires four idle GPUs and global batch256")
    return gpus[:4], 64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--resume-run", type=Path, help="Use a completed, verified DDP checkpoint instead of the single-GPU snapshot")
    parser.add_argument("--runner-change-reason")
    parser.add_argument("--batch-profile", choices=("paper", "fixed_global_batch"), default="paper")
    parser.add_argument("--global-batch-size", type=int)
    parser.add_argument("--speed-benchmarks", action="store_true",
                        help="Run isolated single-rank fixed-batch highest/high timing forks before the four-rank check")
    args = parser.parse_args()
    if args.speed_benchmarks and args.batch_profile != "fixed_global_batch":
        parser.error("Speed benchmarks require the explicitly authorized fixed-global-batch profile")
    prepared = json.loads((args.prepared / "prepare_state.json").read_text())
    if prepared["status"] != "ready_for_ddp":
        raise ValueError("A preserved, fully stopped single-GPU checkpoint is required")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output / "workflow_state.json"
    snapshot = prepared["snapshot"]
    resume_run = Path(snapshot["run"])
    if args.resume_run is not None:
        run = args.resume_run.resolve()
        result = json.loads((run / "training_result.json").read_text())
        contract = json.loads((run / "run_contract.json").read_text())
        if (contract["status"] not in {"completed", "trained_not_reviewed"}
                or contract["stage"] != "native_pose"
                or file_hash(run / "checkpoints/final.ckpt") != result["final_checkpoint_sha256"]):
            raise ValueError("DDP continuation requires a complete bound checkpoint")
        resume_run = run
        snapshot = {"run": str(run), "optimizer_updates": result["optimizer_updates"],
                    "checkpoint_sha256": result["final_checkpoint_sha256"]}
    resume_contract = json.loads((resume_run / "run_contract.json").read_text())
    tokenizer = resume_contract["tokenizer"]
    state = {"status": "checking_idle_gpus", "pid": os.getpid(), "started_at_unix": time.time(),
             "total_pose_updates": 50000, "schedule_steps": 2000000,
             "source_single_gpu_snapshot": prepared["snapshot"], "resume_checkpoint": snapshot,
             "quality_approved": False,
             "model_and_losses_changed": False, "stages": {}}
    state["batch_profile"] = args.batch_profile
    state["global_batch_size"] = args.global_batch_size
    child = None
    stream = None
    code_root = Path(__file__).resolve().parents[1]

    def save():
        write_json(state_path, state)

    def interrupt(*_):
        raise KeyboardInterrupt("DDP workflow terminated")

    signal.signal(signal.SIGTERM, interrupt)
    save()

    def execute(command, name, env):
        nonlocal child, stream
        stream = (args.output / f"{name}.log").open("xb")
        child = subprocess.Popen(command, cwd=code_root, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        state["stages"][name] = {"status": "running", "pid": child.pid, "command": command}
        save()
        while child.poll() is None:
            state["updated_at_unix"] = time.time()
            save()
            time.sleep(10)
        status = child.returncode
        stream.close()
        stream = None
        child = None
        if status:
            raise RuntimeError(f"{name} exited {status}; preserved checkpoints remain available")
        state["stages"][name]["status"] = "processed"
        save()

    try:
        gpus = idle_gpus()
        gpus, per_gpu_batch = select_batch_gpus(gpus, args.batch_profile, args.global_batch_size)
        state["gpu_at_launch"] = gpus
        world = len(gpus)
        state["world_size"] = world
        report_path = args.output / "gpu_launch_report.json"
        write_json(report_path, {"gpus": gpus})
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(row["uuid"] for row in gpus)
        env["PYTHONPATH"] = str(code_root) + ":${PRIVATE_STORAGE}/motionbricks-preparation/kimodo"
        env["OMP_NUM_THREADS"] = "4"
        env["MPLBACKEND"] = "Agg"
        start = snapshot["optimizer_updates"]
        technical_stop = start + 200
        if technical_stop >= 50000:
            raise ValueError("Not enough remaining budget for the technical check")
        if args.speed_benchmarks:
            baseline_env = env.copy()
            baseline_env["CUDA_VISIBLE_DEVICES"] = gpus[0]["uuid"]
            baseline_report = args.output / "single_rank_gpu_report.json"
            write_json(baseline_report, {"gpus": gpus[:1]})
            state["status"] = "benchmarking_fixed_global_batch"
            save()
            for precision in ("highest", "high"):
                available = {row["uuid"] for row in idle_gpus()}
                if gpus[0]["uuid"] not in available:
                    raise RuntimeError("Benchmark GPU became occupied; no other jobs stopped")
                name = f"benchmark-single-{precision}"
                run = args.output / name
                command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                           "--nproc_per_node=1", "--max-restarts=0", "scripts/train_native_mb_ddp.py",
                           "--dataset", resume_contract["dataset"], "--output", str(run),
                           "--tokenizer-run", tokenizer["run"], "--resume-run", str(resume_run),
                           "--gpu-launch-report", str(baseline_report), "--steps", str(technical_stop),
                           "--schedule-steps", "2000000", "--batch-size", "256", "--workers", "2", "--seed", "7",
                           "--batch-profile", "fixed_global_batch", "--global-batch-size", "256",
                           "--benchmark-only", "--benchmark-matmul-precision", precision]
                if args.runner_change_reason:
                    command.extend(("--runner-change-reason", args.runner_change_reason))
                execute(command, name, baseline_env)
                result = verify_run(run, start, technical_stop, 1)
                state["stages"][name].update(status="timing_fork_not_selected", result=result, run=str(run))
                save()
        for name, stop in (("ddp-technical", technical_stop), ("pose-50000", 50000)):
            state["status"] = "testing_ddp" if name == "ddp-technical" else "training_pose_ddp"
            save()
            available = {row["uuid"] for row in idle_gpus()}
            if not {row["uuid"] for row in gpus}.issubset(available):
                raise RuntimeError("Selected GPUs became occupied before continuation; no other jobs stopped")
            run = args.output / name
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                       f"--nproc_per_node={world}", "--max-restarts=0", "scripts/train_native_mb_ddp.py",
                       "--dataset", resume_contract["dataset"], "--output", str(run),
                       "--tokenizer-run", tokenizer["run"], "--resume-run", str(resume_run),
                       "--gpu-launch-report", str(report_path), "--steps", str(stop),
                       "--schedule-steps", "2000000", "--batch-size", str(per_gpu_batch), "--workers", "2", "--seed", "7",
                       "--batch-profile", args.batch_profile]
            if args.global_batch_size is not None:
                command.extend(("--global-batch-size", str(args.global_batch_size)))
            if args.find_unused_parameters:
                command.append("--find-unused-parameters")
            if args.runner_change_reason:
                command.extend(("--runner-change-reason", args.runner_change_reason))
            execute(command, name, env)
            result = verify_run(run, start, stop, world)
            state["stages"][name].update(status="trained_not_reviewed", result=result, run=str(run))
            if name == "ddp-technical" and args.speed_benchmarks:
                base = state["stages"]["benchmark-single-highest"]["result"]
                high = state["stages"]["benchmark-single-high"]["result"]
                comparison = {"global_batch_size": 256, "same_start_checkpoint_sha256": snapshot["checkpoint_sha256"],
                              "single_highest": base["timing"], "single_high": high["timing"],
                              "four_high": result["timing"],
                              "matmul_speedup": high["timing"]["updates_per_second"] / base["timing"]["updates_per_second"],
                              "four_rank_speedup_over_single_high": result["timing"]["updates_per_second"] / high["timing"]["updates_per_second"],
                              "combined_speedup": result["timing"]["updates_per_second"] / base["timing"]["updates_per_second"],
                              "quality_claim": False, "baseline_checkpoints_selected": False,
                              "single_precision_length_histograms_equal": base["rank_receipts"][0]["token_length_counts"] == high["rank_receipts"][0]["token_length_counts"],
                              "four_rank_length_histograms_match_single": all(x["token_length_counts"] == base["rank_receipts"][0]["token_length_counts"] for x in result["rank_receipts"]),
                              "scope": "200-update timing forks; 25 warmup updates excluded; equal global input batch; native variable lengths retained"}
                write_json(args.output / "speed_comparison.json", comparison)
                state["speed_comparison"] = comparison
            resume_run = run
            start = stop
            save()
        state["status"] = "reviewing"
        save()
        review_gpus = idle_gpus()
        if not review_gpus:
            raise RuntimeError("No idle GPU for inference review; trained checkpoints preserved")
        env["CUDA_VISIBLE_DEVICES"] = review_gpus[0]["uuid"]
        command = [sys.executable, "-u", "scripts/evaluate_native_mb_lafan.py", "--source", str(args.source_dataset),
                   "--dataset", resume_contract["dataset"], "--tokenizer-run", tokenizer["run"],
                   "--root-run", str(args.root_run), "--pose-run", str(resume_run),
                   "--output", str(args.output / "review"), "--expected-stage-steps", "50000"]
        execute(command, "review", env)
        state.update(status="trained_evaluated_not_approved", finished_at_unix=time.time())
        save()
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=40)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
