"""Profile a short clone of the unchanged native control trainer, not a production run."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_native_mb import select_gpu, write_json


def profile_arguments(contract, saved_step, resume, output, gpu, workers, updates):
    if updates < 1 or workers < 0:
        raise ValueError("Positive updates and nonnegative workers are required")
    command = ["--stage", contract["stage"].removeprefix("native_"),
               "--dataset", contract["dataset"], "--resume-run", str(resume),
               "--output", str(output), "--gpu", str(gpu), "--workers", str(workers),
               "--steps", str(saved_step + updates),
               "--schedule-steps", str(contract["original_schedule_steps"]),
               "--batch-size", str(contract["batch_size"]), "--seed", str(contract["seed"]),
               "--skip-review"]
    if contract["stage"] == "native_pose":
        command += ["--tokenizer-run", contract["tokenizer"]["run"]]
    return command


def summarize_durations(durations, warmup=10):
    result = {}
    for name, values in durations.items():
        if not values:
            continue
        steady = values[warmup:] if len(values) > warmup else values
        result[name] = {"calls": len(values), "mean_ms": sum(values) / len(values) * 1000,
                        "steady_mean_ms": sum(steady) / len(steady) * 1000,
                        "steady_calls": len(steady)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--updates", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists() or args.gpu < 0 or args.workers < 0 or args.updates < 1:
        parser.error("Requires new output and valid GPU/workers/update count")
    gpu = select_gpu(args.gpu)
    import pytorch_lightning as pl
    from pytorch_lightning.profilers import SimpleProfiler
    from scripts import train_native_mb_control as runner

    contract = json.loads((args.resume_run / "run_contract.json").read_text())
    saved = json.loads((args.resume_run / "training_result.json").read_text())["optimizer_updates"]
    arguments = profile_arguments(contract, saved, args.resume_run, args.output,
                                  args.gpu, args.workers, args.updates)
    profiler = SimpleProfiler(dirpath=str(args.output / "profile"), filename="native")
    original_trainer = pl.Trainer

    def profiled_trainer(*positional, **keywords):
        return original_trainer(*positional, **keywords, profiler=profiler)

    # Only add Lightning's official timing instrumentation; leave the trainer intact.
    pl.Trainer = profiled_trainer
    try:
        runner.main(arguments)
    finally:
        pl.Trainer = original_trainer
    result = json.loads((args.output / "training_result.json").read_text())
    report = {"gpu_at_launch": gpu, "workers": args.workers, "batch_size": contract["batch_size"],
              "arguments": arguments, "training_result": result,
              "durations": summarize_durations(profiler.recorded_durations),
              "measurement": "CPU wall timings of a short full-state resumed native training clone",
              "quality_approved": False, "production_checkpoint": False,
              "limitations": ["Profiler adds overhead", "Single run, shared-server load",
                              "Loader worker count changes random sampling stream, not sampling rules",
                              "Timing regions overlap; do not sum them"]}
    write_json(args.output / "performance_report.json", report)
    print(json.dumps({"workers": args.workers, "updates": result["updates_this_run"],
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
