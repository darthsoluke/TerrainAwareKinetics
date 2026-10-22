"""Generate a finite extension batch from an immutable official PARC snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import deploy_parc_remote_batch as deploy


def run(output, repeats=2, original_jobs=None):
    if repeats not in (1, 2):
        raise ValueError("Use one or two seeds per extension layout")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / "workspace"
    entries = []
    for source in deploy.source_files(deploy.ROOT):
        relative = source.relative_to(deploy.ROOT)
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        before = deploy.digest(source)
        if relative.as_posix() == deploy.CHECKPOINT:
            os.link(source, target)
        else:
            shutil.copy2(source, target)
        if deploy.digest(target) != before:
            raise RuntimeError(f"Snapshot changed: {relative}")
        entries.append({"path": relative.as_posix(), "sha256": before, "bytes": target.stat().st_size})
    snapshot = {"files": entries, "source_sha256": hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode()).hexdigest()}
    deploy.write_json(work / "_bundle.json", snapshot, exclusive=True)
    jobs = []
    if original_jobs is not None:
        known = {j["id"]: j for host in ("g2", "g3") for j in deploy.make_jobs(host)}
        if not original_jobs or len(original_jobs) != len(set(original_jobs)) or set(original_jobs) - known.keys():
            raise ValueError("Recovery needs unique original job IDs")
        jobs = [dict(known[key], state="queued") for key in original_jobs]
    else:
        for repeat in range(repeats):
            for variant in range(3):
                seed = 20263008 + repeat * 3 + variant
                jobs.append({"id": f"extension_v{variant}_s{seed}", "kind": "extension",
                             "variant": variant, "seed": seed, "state": "queued"})
    status = {"state": "running", "source_sha256": snapshot["source_sha256"], "jobs": jobs,
              "weight_selection": "ema_model", "ddim_stride": 5, "batch_size": 8,
              "physics_verified": False, "training_approved": False}
    deploy.write_json(output / "queue_status.json", status)
    env = os.environ.copy()
    env.update(WANDB_MODE="disabled", WANDB_SILENT="true", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    for job in jobs:
        started = time.monotonic()
        # Wait for memory rather than closing the user's graphics applications.
        while True:
            free = int(deploy.checked_run(["nvidia-smi", "--id=0",
                "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).strip())
            if free >= 3500:
                break
            if time.monotonic() - started > 600:
                job.update(state="deferred_low_memory", free_mib=free)
                break
            time.sleep(15)
        if job["state"] == "deferred_low_memory":
            continue
        command = [sys.executable, "-B", "-u", str(work / "scripts/generate_parc_long_stairs.py"),
            "--repo", str(work / deploy.VENDOR), "--checkpoint", str(work / deploy.CHECKPOINT),
            "--output", str(output / "results" / job["id"]), "--device", "cuda:0",
            "--kind", job["kind"], "--variant", str(job["variant"]), "--seed", str(job["seed"]),
            "--batch-size", "8", "--max-seconds", "45", "--threads", "4",
            "--score-frame-chunk", "4"]
        job.update(state="running", started_at=deploy.stamp(), command=command)
        deploy.write_json(output / "queue_status.json", status)
        print(json.dumps({"started": job["id"], "free_mib": free}), flush=True)
        try:
            with (output / (job["id"] + ".log")).open("x", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=work, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=7200)
            job.update(returncode=result.returncode,
                       state="completed" if result.returncode == 0 else "failed")
        except (subprocess.TimeoutExpired, OSError) as error:
            job.update(state="failed", error=f"{type(error).__name__}: {error}", returncode=None)
        job.update(finished_at=deploy.stamp(), elapsed_seconds=time.monotonic() - started)
        deploy.write_json(output / "queue_status.json", status)
        print(json.dumps({"finished": job["id"], "state": job["state"]}), flush=True)
    status["state"] = "completed" if all(j["state"] == "completed" for j in jobs) else "finished_with_errors"
    try:
        deploy.verify_workspace(work)
        status["source_snapshot_verified"] = True
    except Exception as error:
        status.update(state="finished_with_errors", source_snapshot_verified=False,
                      error=f"{type(error).__name__}: {error}")
    deploy.write_json(output / "queue_status.json", status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2, choices=(1, 2))
    parser.add_argument("--original-jobs", nargs="+")
    args = parser.parse_args()
    result = run(args.output, args.repeats, args.original_jobs)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(0 if result["state"] == "completed" else 1)
