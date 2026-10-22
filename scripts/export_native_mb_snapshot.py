"""Export an exact saved update as an immutable, resumable bounded-stage artifact."""

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import time


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_training_evidence(saved, step, schedule_steps):
    import torch

    if type(step) is not int or step < 1 or saved.get("global_step") != step:
        raise ValueError("Checkpoint must contain the exact requested update")
    if not saved.get("state_dict") or not saved.get("optimizer_states") or not saved.get("lr_schedulers"):
        raise ValueError("Snapshot requires weights, optimizer and scheduler states")
    progress = saved["loops"]["fit_loop"]["epoch_loop.automatic_optimization.optim_progress"]
    completed = progress["optimizer"]["step"]["total"]["completed"]
    if completed != step:
        raise ValueError("Optimizer loop updates do not match the checkpoint")
    for value in saved["state_dict"].values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains nonfinite model state")
    optimizer_steps = []
    declared_parameter_count = 0
    uninitialized_parameter_count = 0
    for optimizer in saved["optimizer_states"]:
        if not optimizer.get("state"):
            raise ValueError("Missing optimizer parameter states")
        parameter_ids = {parameter for group in optimizer["param_groups"] for parameter in group["params"]}
        if not set(optimizer["state"]).issubset(parameter_ids):
            raise ValueError("Optimizer state references an undeclared parameter")
        # Adam creates moments lazily; never-used parameters legitimately lack state.
        declared_parameter_count += len(parameter_ids)
        uninitialized_parameter_count += len(parameter_ids - set(optimizer["state"]))
        if any(not math.isfinite(float(group["lr"])) for group in optimizer["param_groups"]):
            raise ValueError("Optimizer learning rate is nonfinite")
        for state in optimizer["state"].values():
            if not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
                raise ValueError("Adam moment states are incomplete")
            value = state.get("step")
            if value is None or float(value) != step:
                raise ValueError("Adam parameter update count differs from requested step")
            optimizer_steps.append(float(value))
            for tensor in state.values():
                if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                    raise ValueError("Optimizer state is nonfinite")
    for scheduler in saved["lr_schedulers"]:
        if scheduler.get("last_epoch") != step or scheduler.get("num_training_steps") != schedule_steps:
            raise ValueError("Learning-rate schedule identity/update count changed")
        if not scheduler.get("_last_lr") or any(not math.isfinite(float(lr)) for lr in scheduler["_last_lr"]):
            raise ValueError("Scheduler learning rate is nonfinite")
    return {"global_step": step, "optimizer_loop_completed": completed,
            "adam_parameter_states": len(optimizer_steps), "adam_parameter_step": step,
            "optimizer_parameter_count": declared_parameter_count,
            "uninitialized_optimizer_parameter_count": uninitialized_parameter_count,
            "schedule_steps": schedule_steps, "full_optimizer_and_scheduler_preserved": True,
            "weights_finite": True,
            "batch_completed_note": "Batch-end checkpoint may precede increment of batch completed counter."}


def export_snapshot(source_run, checkpoint, output, step):
    import torch

    source_run, checkpoint, output = map(lambda p: Path(p).resolve(), (source_run, checkpoint, output))
    if checkpoint.parent != source_run / "checkpoints" or not checkpoint.is_file():
        raise ValueError("Snapshot checkpoint must belong to the source run")
    if output.exists() or output == source_run or source_run in output.parents:
        raise FileExistsError("Snapshot output must be a fresh run outside the source run")
    contract_path = source_run / "run_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("stage") not in {"native_tokenizer", "native_root", "native_pose"}:
        raise ValueError("Expected a native Tokenizer, Root or Pose snapshot")
    if contract["stage"] == "native_pose":
        dependency = contract.get("tokenizer") or {}
        if not {"checkpoint", "checkpoint_sha256"}.issubset(dependency):
            raise ValueError("Pose snapshot requires a complete frozen tokenizer dependency")
        tokenizer_checkpoint = Path(dependency["checkpoint"])
        if (file_hash(tokenizer_checkpoint) != dependency["checkpoint_sha256"]
                or dependency.get("dataset_manifest_sha256") != contract["dataset_manifest_sha256"]
                or not contract.get("tokenizer_state_sha256")):
            raise ValueError("Pose snapshot frozen tokenizer binding changed")
    dataset = Path(contract["dataset"])
    if (file_hash(source_run / "native_config.yaml") != contract["config_sha256"]
            or file_hash(dataset / "native_manifest.json") != contract["dataset_manifest_sha256"]):
        raise ValueError("Source configuration or dataset binding changed")
    manifest = json.loads((dataset / "native_manifest.json").read_text(encoding="utf-8"))
    for relative, digest in manifest["artifact_sha256"].items():
        path = (dataset / relative).resolve()
        if dataset.resolve() not in path.parents or file_hash(path) != digest:
            raise ValueError(f"Dataset artifact changed: {relative}")
    code_root = Path(__file__).resolve().parents[1]
    for relative, digest in contract.get("source_lf_sha256", {}).items():
        data = (code_root / relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Training source changed: {relative}")
    checkpoint_hash = file_hash(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    evidence = checkpoint_training_evidence(saved, step, contract["original_schedule_steps"])
    if step > contract["stop_steps"]:
        raise ValueError("Requested snapshot exceeds source training budget")
    resume_step = (contract.get("resume") or {}).get("saved_step", 0)
    if step <= resume_step:
        raise ValueError("Snapshot must be after the source initialization")
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    shutil.copy2(checkpoint, output / "checkpoints/final.ckpt")
    if file_hash(output / "checkpoints/final.ckpt") != checkpoint_hash:
        raise ValueError("Snapshot byte-copy hash mismatch")
    shutil.copy2(source_run / "native_config.yaml", output / "native_config.yaml")
    shutil.copy2(contract_path, output / "source_run_contract.json")
    provenance = {"artifact_kind": "selected_checkpoint_export", "requested_stop_steps": step,
                  "source_run": str(source_run), "source_checkpoint": str(checkpoint),
                  "source_contract_sha256": file_hash(contract_path),
                  "source_original_stop_steps": contract["stop_steps"],
                  "exported_at_unix": time.time(), "checkpoint_sha256": checkpoint_hash,
                  "original_long_run_completed": False, "quality_approved": False,
                  "note": "Completed means this bounded checkpoint export, not completed paper training or quality approval."}
    snapshot = copy.deepcopy(contract)
    snapshot.update(status="completed", stop_steps=step, snapshot_provenance=provenance,
                    checkpoint_training_evidence=evidence, quality_approved=False, ue_deployment=False)
    result = {"optimizer_updates": step, "updates_this_run": step - resume_step,
              "final_checkpoint_sha256": checkpoint_hash,
              "tokenizer_state_sha256": contract.get("tokenizer_state_sha256"),
              "elapsed_seconds": None, "peak_gpu_memory_mb": None,
              "skipped_batches": None, "quality_approved": False,
              "evidence": evidence,
              "metrics_note": "Timing/skipped batch callback metrics were not recorded for this selected checkpoint."}
    write_json(output / "training_result.json", result)
    write_json(output / "run_contract.json", snapshot)
    return {"run": str(output), "checkpoint": str(output / "checkpoints/final.ckpt"),
            "optimizer_updates": step, "checkpoint_sha256": checkpoint_hash,
            "dataset_manifest_sha256": contract["dataset_manifest_sha256"], "provenance": provenance}
