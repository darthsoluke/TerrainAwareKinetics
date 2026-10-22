"""Automatically check the real batch-256 training paths; never claim motion quality."""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_native_mb_workflow import launch, save_state, stop_owned
from scripts.run_paper_mb_workflow import paper_command, require_technical_checks


def check_training(dataset, output, gpu):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "stages": {}, "quality_approved": False,
             "production_checkpoint": False,
             "purpose": "Actual official training_step, declared Adam, full paper schedule, batch256, finite gradients"}
    owned = {}
    try:
        for stage in ("tokenizer", "root", "pose"):
            run = output / stage
            command = paper_command(stage, Path(dataset).resolve(), run, gpu,
                                    tokenizer_run=output / "tokenizer" if stage == "pose" else None)
            command[command.index("--steps") + 1] = "200"
            if stage != "tokenizer":
                command.append("--skip-review")
            state.update(current_stage=stage, command=command)
            save_state(output / "checks.json", state)
            process, stream = launch(command, root, output / f"{stage}.log")
            owned[stage] = process, stream
            result_code = process.wait(timeout=3600)
            stream.close()
            del owned[stage]
            if result_code:
                raise RuntimeError(f"{stage} exited {result_code}; see {stage}.log")
            result = json.loads((run / "training_result.json").read_text())
            state["stages"][stage] = {"run": str(run), "checkpoint_sha256": result["final_checkpoint_sha256"],
                                      "elapsed_seconds": result["elapsed_seconds"],
                                      "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                                      "observed_input_batch_sizes": result["observed_input_batch_sizes"],
                                      "optimizer_updates": result["optimizer_updates"],
                                      "skipped_batches": result["skipped_batches"]}
        state.update(status="completed", completed_at_unix=time.time())
        save_state(output / "checks.json", state)
        require_technical_checks(output / "checks.json")
    except BaseException as error:
        state.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        stop_owned(owned)
        save_state(output / "checks.json", state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("Check terminated")))
    print(json.dumps(check_training(args.dataset, args.output, args.gpu), indent=2), flush=True)
