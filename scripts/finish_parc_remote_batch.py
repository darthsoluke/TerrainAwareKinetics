"""Recover the finite PARC batch without changing generation or old outputs."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

try:
    import deploy_parc_remote_batch as local
except ModuleNotFoundError:
    sys.path.insert(0, "${PRIVATE_STORAGE}/parc-remote-batch-20261008/runs/ema_stride5_20261008_v1/workspace/scripts")
    import deploy_parc_remote_batch as local


ORIGINAL = "ema_stride5_20261008_v1"
BASE = Path("${PRIVATE_STORAGE}/parc-remote-batch-20261008")


def remote_library():
    original = BASE / "runs" / ORIGINAL
    launch = json.loads((original / "launch.json").read_text())
    if local.digest(Path(launch["queue_script"])) != launch["queue_script_sha256"]:
        raise ValueError("Queue bootstrap hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_deploy", launch["queue_script"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stop_idle_original(run_id=ORIGINAL):
    """Freeze/check only our exact queue PID before terminating an idle queue."""
    d = remote_library()
    run = d.owned_path(BASE / "runs" / run_id)
    launch = json.loads((run / "launch.json").read_text())
    pid = launch["queue_pid"]
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return {"state": "already_stopped"}
    cmd = (proc / "cmdline").read_bytes().split(b"\0")
    if proc.stat().st_uid != os.getuid() or b"remote-queue" not in cmd or str(run).encode() not in cmd:
        raise ValueError("Refusing to stop an unrecognized process")
    os.kill(pid, signal.SIGSTOP)
    try:
        for _ in range(100):
            if "\nState:\tT" in (proc / "status").read_text():
                break
            time.sleep(.02)
        else:
            raise RuntimeError("Queue did not suspend")
        state = json.loads((run / "queue_status.json").read_text())
        children = (proc / "task" / str(pid) / "children").read_text().strip()
        if children or any(j["state"] in ("running", "launched_awaiting_cuda") for j in state["jobs"]):
            raise RuntimeError("Queue has started work; refusing migration")
        os.kill(pid, signal.SIGTERM)
    finally:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    for _ in range(100):
        if not proc.exists() or "\nState:\tZ" in (proc / "status").read_text():
            break
        time.sleep(.05)
    else:
        raise RuntimeError("Idle queue has not exited")
    state.update(state="cancelled_for_reassignment", at=d.stamp(),
                 note="Unstarted jobs reassigned by finish_parc_remote_batch; no generated output removed")
    d.write_json(run / "queue_status.json", state)
    return {"state": state["state"], "jobs": [j["id"] for j in state["jobs"] if j["state"] != "completed"]}


def start_recovery(run_id, job_ids):
    d = remote_library()
    original = d.owned_path(BASE / "runs" / ORIGINAL)
    source = original / "workspace"
    snapshot = d.verify_workspace(source)
    jobs = {j["id"]: j for host in ("g2", "g3") for j in d.make_jobs(host)}
    if not job_ids or len(job_ids) != len(set(job_ids)) or set(job_ids) - jobs.keys():
        raise ValueError("Recovery needs a unique subset of the original fourteen jobs")
    run = d.owned_path(BASE / "runs" / run_id)
    run.mkdir(parents=True, exist_ok=False)
    work = run / "workspace"
    for entry in snapshot["files"] + [{"path": "_bundle.json"}]:
        relative = entry["path"]
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source / relative, target)
    d.verify_workspace(work)
    manifest = json.loads((original / "manifest.json").read_text())
    manifest.update(batch_id=run_id, created_at=d.stamp(), max_candidates=len(job_ids) * 8,
                    recovery_of=ORIGINAL, jobs=[dict(jobs[key]) for key in job_ids])
    for job in manifest["jobs"]:
        job["command"] = d.job_command(run, job)
    d.write_json(run / "manifest.json", manifest, exclusive=True)
    certificate = json.loads((original / "checkpoint_probe.json").read_text())
    d.write_json(run / "checkpoint_probe.json", certificate, exclusive=True)
    d.write_json(run / "queue_status.json", {"state": "prepared", "generation_started": False,
                                           "at": d.stamp()}, exclusive=True)
    return d.remote_launch(run)


def inventory():
    d = remote_library()
    runs = []
    for directory in sorted((BASE / "runs").iterdir()):
        if directory.name != ORIGINAL and not directory.name.startswith("complete_20261008_"):
            continue
        status_path = directory / "queue_status.json"
        if not status_path.is_file():
            continue
        status = json.loads(status_path.read_text())
        jobs = []
        for job in status.get("jobs", []):
            report_path = directory / "results" / job["id"] / "generation.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
            candidates = report.get("candidates", [])
            jobs.append({"id": job["id"], "state": job["state"], "gpu_index": job.get("gpu_index"),
                         "candidates": len(candidates), "report_status": report.get("status"),
                         "covered": sum(c.get("geometric_route_covered", False) for c in candidates),
                         "elapsed_seconds": report.get("elapsed_seconds"),
                         "directory": str(report_path.parent),
                         "duration_s": sum(c["duration_s"] for c in candidates)})
        runs.append({"run_id": directory.name, "state": status["state"], "at": status.get("at"),
                     "last_error": status.get("last_error"), "jobs": jobs})
    return {"at": d.stamp(), "gpu": d.gpu_sample(), "runs": runs}


def pack_results(run_id):
    d = remote_library()
    run = d.owned_path(BASE / "runs" / run_id)
    state = json.loads((run / "queue_status.json").read_text())
    jobs = [j for j in state.get("jobs", []) if j["state"] == "completed"]
    paths = []
    for job in jobs:
        root = run / "results" / job["id"]
        report = json.loads((root / "generation.json").read_text())
        if report.get("status") != "kinematic_candidates_generated_physics_pending" or len(report["candidates"]) != 8:
            raise ValueError("Incomplete candidate report")
        for row in report["candidates"]:
            file = root / f"candidate_{row['rank']:03d}" / "generated.pkl"
            if d.digest(file) != row["sha256"]:
                raise ValueError(f"Candidate hash mismatch: {file}")
        paths.extend(p for p in root.rglob("*") if p.is_file())
    if not paths:
        raise ValueError("No completed outputs to download")
    if any(p.is_symlink() for p in paths) or sum(p.stat().st_size for p in paths) >= 1_000_000_000:
        raise ValueError("Unexpected output size/symlink; use NAS for large transfers")
    name = "results_" + d.stamp().replace(":", "").replace("-", "") + ".tar"
    archive = run / name
    with tarfile.open(archive, "x") as stream:
        for path in paths:
            stream.add(path, arcname=path.relative_to(run), recursive=False)
    return {"archive": str(archive), "sha256": d.digest(archive), "bytes": archive.stat().st_size,
            "jobs": [j["id"] for j in jobs]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("stop-idle", "start", "inventory", "pack"))
    p.add_argument("--run-id", default=ORIGINAL)
    p.add_argument("--jobs", nargs="*")
    args = p.parse_args()
    if not local.re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.run_id):
        p.error("Invalid run ID")
    if args.action == "stop-idle":
        result = stop_idle_original(args.run_id)
    elif args.action == "start":
        result = start_recovery(args.run_id, args.jobs)
    elif args.action == "pack":
        result = pack_results(args.run_id)
    else:
        result = inventory()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
