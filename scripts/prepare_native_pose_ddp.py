"""Preserve a complete Pose checkpoint before replacing its verified launcher."""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_native_mb_snapshot import export_snapshot, write_json
from scripts.run_lafan_50k_handoff import process_state, signal_verified, stable_file, terminate_verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suspend-manager", action="store_true",
                        help="Suspend the verified old coordinator before waiting for its checkpoint")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "waiting_for_checkpoint", "request": request, "quality_approved": False}
    path = args.output / "prepare_state.json"
    write_json(path, state)
    try:
        if args.suspend_manager:
            signal_verified(request["manager"], signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while process_state(request["manager"]) != "T" and time.monotonic() < deadline:
                time.sleep(0.05)
        if process_state(request["manager"]) != "T":
            raise ValueError("Old manager must be suspended during launcher handoff")
        filename = request.get("checkpoint_name", f"native-step={request['target_step']}.ckpt")
        if Path(filename).name != filename:
            raise ValueError("Checkpoint filename must not escape its run directory")
        checkpoint = Path(request["pose_run"]) / "checkpoints" / filename
        deadline = time.monotonic() + 3600
        while not stable_file(checkpoint):
            if process_state(request["pose"]) is None or time.monotonic() > deadline:
                raise RuntimeError("Pose did not produce the requested complete checkpoint")
            time.sleep(1)
        signal_verified(request["pose"], signal.SIGSTOP)
        state["status"] = "exporting_checkpoint"
        write_json(path, state)
        snapshot = export_snapshot(request["pose_run"], checkpoint,
                                   args.output / "pose-resume", request["target_step"])
        state["snapshot"] = snapshot
        write_json(path, state)
        terminate_verified(request["pose"])
        terminate_verified(request["manager"])
        old_contract_path = Path(request["pose_run"]) / "run_contract.json"
        contract = json.loads(old_contract_path.read_text())
        contract.update(status="stopped_for_ddp_handoff", selected_checkpoint_steps=request["target_step"],
                        bounded_snapshot=snapshot["run"])
        write_json(old_contract_path, contract)
        old_state_path = Path(request["source_workflow"]) / "workflow_state.json"
        old_state = json.loads(old_state_path.read_text())
        old_state.update(status="superseded_by_ddp", ddp_handoff=str(args.output),
                         requested_stage_steps=50000, quality_approved=False)
        write_json(old_state_path, old_state)
        state.update(status="ready_for_ddp", finished_at_unix=time.time())
        write_json(path, state)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(path, state)
        for identity in (request["pose"], request["manager"]):
            if process_state(identity) == "T":
                signal_verified(identity, signal.SIGCONT)
        raise


if __name__ == "__main__":
    main()
