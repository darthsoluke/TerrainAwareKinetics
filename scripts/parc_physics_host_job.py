"""Launch one isolated, idle-GPU-only official PARC physics job on HKU-4090D."""

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import time


def launch(job, run_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Noncanonical run id")
    base = Path.home() / "projects/motionbricks-parc-physics-20261009"
    nas = Path("${PRIVATE_STORAGE}/motionbricks-parc-physics-20261009")
    if base.is_symlink() or base.stat().st_uid != os.getuid():
        raise ValueError("Unexpected project owner")
    gpu = "GPU-d9729088-fda6-d94f-1809-486dca450d8d"
    with (base / "gpu1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(2):
            rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader"], text=True)
            if gpu in rows:
                raise RuntimeError("GPU 1 is occupied; do not displace its process")
            free, utilization = map(int, subprocess.check_output([
                "nvidia-smi", "--id=" + gpu, "--query-gpu=memory.free,utilization.gpu",
                "--format=csv,noheader,nounits"], text=True).strip().split(","))
            if free < 20000 or utilization > 10:
                raise RuntimeError("GPU 1 is not idle")
            time.sleep(2)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
            PATH=str(base / "env/bin") + ":" + env.get("PATH", ""),
            LD_LIBRARY_PATH=str(base / "env/lib") + ":" + env.get("LD_LIBRARY_PATH", ""),
            MAX_JOBS="4", TORCH_EXTENSIONS_DIR=str(base / "torch_extensions"),
            WANDB_MODE="disabled", WANDB_SILENT="true", MPLBACKEND="Agg",
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        if job == "smoke":
            command = [str(base / "env/bin/python"), "-u", str(base / "smoke_parc_isaacgym.py"),
                       "--output", str(nas / (run_id + ".json"))]
        else:
            command = [str(base / "env/bin/python"), "-u", str(base / "run_parc_physics_record.py"),
                "--bundle", str(nas / "inputs_v1"), "--output", str(nas / "runs" / run_id),
                "--selection", "all" if job == "all" else "representatives"]
            if job == "preflight":
                command.append("--preflight-only")
        (base / "logs").mkdir(exist_ok=True)
        with (base / "logs" / (run_id + ".log")).open("x") as log:
            print("Launching " + job + "; log " + log.name, flush=True)
            result = subprocess.run(command, env=env, cwd=base, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1200)
        print("Exit code: " + str(result.returncode), flush=True)
        return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", choices=("smoke", "preflight", "representatives", "all"), required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    raise SystemExit(launch(args.job, args.run_id))
