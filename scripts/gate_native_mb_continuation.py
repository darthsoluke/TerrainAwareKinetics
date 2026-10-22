"""Hold only the remote continuation wrappers; current native training finishes normally."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_native_mb import write_json


def verify_wrapper_identity(state, cmdline, stage, base):
    if state.get("stage") != stage or state.get("status") != "running" or state.get("target_step") != 100000:
        raise ValueError("The current stage must still be training its first 100k milestone")
    if not any(Path(argument).name == "continue_native_mb_control.py" for argument in cmdline):
        raise ValueError("Refusing to signal a process other than the continuation wrapper")
    for flag, expected in (("--stage", stage), ("--output-parent", str(base))):
        if flag not in cmdline or cmdline.index(flag) + 1 >= len(cmdline):
            raise ValueError("Missing continuation wrapper identity")
        if cmdline[cmdline.index(flag) + 1] != expected:
            raise ValueError("Continuation wrapper identity mismatch")
    if state.get("pid") == state.get("training_pid") or not isinstance(state.get("pid"), int):
        raise ValueError("Wrapper PID must be separate from the actual training process")
    return state["pid"]


def arm_gate(base):
    if sys.platform != "linux":
        raise RuntimeError("This process gate is only for the g3 Linux host")
    base = Path(base).resolve()
    path = base / "continuation_gate.json"
    if path.exists():
        raise FileExistsError("Continuation gate already exists")
    jobs = []
    for stage in ("root", "pose"):
        state = json.loads((base / f"{stage}_state.json").read_text())
        cmdline = (Path("/proc") / str(state["pid"]) / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
        pid = verify_wrapper_identity(state, cmdline, stage, base)
        os.kill(state["training_pid"], 0)
        jobs.append({"stage": stage, "wrapper_pid": pid, "training_pid_at_gate": state["training_pid"]})
    gate = {"status": "arming", "held_after_step": 100000, "jobs": jobs,
            "requested_policy": "Keep current configuration; review paired 100k before more long training",
            "training_config_changed": False, "quality_approved": False, "created_at_unix": time.time(),
            "notes": ["Signal only each wrapper PID, never the training child or process group",
                      "Current training, checkpoint saving and validation finish normally",
                      "Independent paired evaluation remains running",
                      "Wrapper state files stay at running while held; inspect final run contracts",
                      "Release requires reviewed milestone evidence; no automatic quality approval"]}
    write_json(path, gate)
    try:
        for job in jobs:
            os.kill(job["wrapper_pid"], signal.SIGSTOP)
            job["wrapper_held"] = True
            write_json(path, gate)
        time.sleep(.1)
        for job in jobs:
            status = (Path("/proc") / str(job["wrapper_pid"]) / "status").read_text()
            job["wrapper_process_state"] = next(line for line in status.splitlines() if line.startswith("State:"))
            if "T (stopped)" not in job["wrapper_process_state"]:
                raise RuntimeError("Continuation wrapper did not enter the held state")
            os.kill(job["training_pid_at_gate"], 0)
        gate["status"] = "armed_waiting_for_100k_review"
    except Exception as error:
        gate.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        write_json(path, gate)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(arm_gate(args.output_parent), indent=2), flush=True)
