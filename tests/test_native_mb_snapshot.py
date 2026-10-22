import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import export_native_mb_snapshot as snapshot


STEP = 50_000
SCHEDULE_STEPS = 2_000_000
OPTIM_PROGRESS = "epoch_loop.automatic_optimization.optim_progress"


def saved_checkpoint():
    return {
        "global_step": STEP,
        "state_dict": {"weight": torch.ones(2, 3), "bias": torch.zeros(2)},
        "optimizer_states": [{
            "state": {
                index: {
                    "step": torch.tensor(float(STEP)),
                    "exp_avg": torch.full_like(parameter, .1),
                    "exp_avg_sq": torch.full_like(parameter, .01),
                }
                for index, parameter in enumerate((torch.ones(2, 3), torch.zeros(2)))
            },
            "param_groups": [{"params": [0, 1], "lr": 1e-4}],
        }],
        "lr_schedulers": [{
            "last_epoch": STEP, "num_training_steps": SCHEDULE_STEPS,
            "_last_lr": [1e-4], "base_lrs": [1e-4],
        }],
        "loops": {"fit_loop": {
            OPTIM_PROGRESS: {"optimizer": {"step": {"total": {"completed": STEP}}}},
            "epoch_loop.batch_progress": {"total": {"completed": STEP - 1}},
        }},
        "callbacks": {"ModelCheckpoint": {"best_model_score": torch.tensor(float("inf"))}},
    }


def test_evidence_accepts_exact_50k_with_callback_inf_and_batch_counter_one_behind():
    saved = saved_checkpoint()

    evidence = snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)

    assert evidence["global_step"] == evidence["optimizer_loop_completed"] == STEP
    assert evidence["adam_parameter_states"] == 2
    assert evidence["optimizer_parameter_count"] == 2
    assert evidence["uninitialized_optimizer_parameter_count"] == 0
    assert evidence["adam_parameter_step"] == STEP
    assert evidence["schedule_steps"] == SCHEDULE_STEPS
    assert evidence["full_optimizer_and_scheduler_preserved"] is True
    assert evidence["weights_finite"] is True
    assert torch.isinf(saved["callbacks"]["ModelCheckpoint"]["best_model_score"])
    assert saved["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"]["completed"] == STEP - 1


@pytest.mark.parametrize("requested", [STEP - 1, STEP + 1, 0, True, float(STEP)])
def test_evidence_rejects_wrong_or_noninteger_requested_step(requested):
    with pytest.raises(ValueError, match="exact requested update"):
        snapshot.checkpoint_training_evidence(saved_checkpoint(), requested, SCHEDULE_STEPS)


@pytest.mark.parametrize("field", ["state_dict", "optimizer_states", "lr_schedulers"])
def test_evidence_requires_weights_optimizer_and_scheduler(field):
    saved = saved_checkpoint()
    saved[field] = {} if field == "state_dict" else []
    with pytest.raises(ValueError, match="weights, optimizer and scheduler"):
        snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)


def test_evidence_rejects_optimizer_loop_one_update_behind():
    saved = saved_checkpoint()
    saved["loops"]["fit_loop"][OPTIM_PROGRESS]["optimizer"]["step"]["total"]["completed"] -= 1
    with pytest.raises(ValueError, match="Optimizer loop updates"):
        snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)


@pytest.mark.parametrize("state_index", [0, 1])
def test_evidence_checks_every_adam_parameter_step(state_index):
    saved = saved_checkpoint()
    saved["optimizer_states"][0]["state"][state_index]["step"] -= 1
    with pytest.raises(ValueError, match="Adam parameter update count"):
        snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)


@pytest.mark.parametrize("field", ["last_epoch", "num_training_steps"])
def test_evidence_rejects_changed_schedule_identity_or_update(field):
    saved = saved_checkpoint()
    saved["lr_schedulers"][0][field] -= 1
    with pytest.raises(ValueError, match="Learning-rate schedule"):
        snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)


@pytest.mark.parametrize("location", ["model", "adam_mean", "adam_variance"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_evidence_rejects_nonfinite_model_or_adam_tensors(location, nonfinite):
    saved = saved_checkpoint()
    tensor = saved["state_dict"]["weight"] if location == "model" else (
        saved["optimizer_states"][0]["state"][1][
            "exp_avg" if location == "adam_mean" else "exp_avg_sq"
        ]
    )
    tensor.reshape(-1)[0] = nonfinite
    with pytest.raises(ValueError, match="nonfinite|Nonfinite"):
        snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)


def test_evidence_accepts_lazy_uninitialized_parameter_and_reports_count():
    saved = saved_checkpoint()
    del saved["optimizer_states"][0]["state"][1]

    evidence = snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)

    assert evidence["optimizer_parameter_count"] == 2
    assert evidence["adam_parameter_states"] == 1
    assert evidence["uninitialized_optimizer_parameter_count"] == 1
    assert evidence["adam_parameter_step"] == STEP
    assert evidence["full_optimizer_and_scheduler_preserved"] is True
    assert saved["optimizer_states"][0]["param_groups"][0]["params"] == [0, 1]
    assert set(saved["optimizer_states"][0]["state"]) == {0}


def test_evidence_rejects_unexpected_optimizer_state_id():
    saved = saved_checkpoint()
    saved["optimizer_states"][0]["state"][2] = copy.deepcopy(
        saved["optimizer_states"][0]["state"][0]
    )
    with pytest.raises(ValueError, match="undeclared parameter"):
        snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)


@pytest.mark.parametrize("moment", ["exp_avg", "exp_avg_sq"])
def test_evidence_rejects_missing_adam_moment(moment):
    saved = saved_checkpoint()
    del saved["optimizer_states"][0]["state"][1][moment]
    with pytest.raises(ValueError):
        snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)


@pytest.mark.parametrize("location", ["optimizer", "scheduler"])
def test_evidence_rejects_nonfinite_resume_learning_rate(location):
    saved = saved_checkpoint()
    if location == "optimizer":
        saved["optimizer_states"][0]["param_groups"][0]["lr"] = float("nan")
    else:
        saved["lr_schedulers"][0]["_last_lr"][0] = float("inf")
    with pytest.raises(ValueError):
        snapshot.checkpoint_training_evidence(saved, STEP, SCHEDULE_STEPS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bound_run(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "shards").mkdir(parents=True)
    accepted = dataset / "accepted.json"
    write_json(accepted, [{"split": "train", "shard": "shards/train.npz"}])
    shard = dataset / "shards/train.npz"
    shard.write_bytes(b"synthetic bound shard")
    manifest = dataset / "native_manifest.json"
    write_json(manifest, {"artifact_sha256": {
        "accepted.json": digest(accepted), "shards/train.npz": digest(shard),
    }})

    code_root = tmp_path / "code"
    (code_root / "scripts").mkdir(parents=True)
    source = code_root / "scripts/training.py"
    source.write_bytes(b"TRAINING_STEPS = 2000000\r\n")
    monkeypatch.setattr(snapshot, "__file__", str(code_root / "scripts/export_native_mb_snapshot.py"))

    run = tmp_path / "tokenizer-2000000"
    (run / "checkpoints").mkdir(parents=True)
    config = run / "native_config.yaml"
    config.write_bytes(b"schedule_steps: 2000000\n")
    contract_path = run / "run_contract.json"
    contract = {
        "stage": "native_tokenizer", "status": "running", "dataset": str(dataset),
        "config_sha256": digest(config), "dataset_manifest_sha256": digest(manifest),
        "stop_steps": SCHEDULE_STEPS, "original_schedule_steps": SCHEDULE_STEPS,
        "resume": {"saved_step": 100}, "quality_approved": False, "ue_deployment": False,
        "source_lf_sha256": {"scripts/training.py": hashlib.sha256(
            source.read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()},
    }
    write_json(contract_path, contract)
    checkpoint = run / "checkpoints/native-step=50000.ckpt"
    torch.save(saved_checkpoint(), checkpoint)
    return {
        "run": run, "checkpoint": checkpoint, "output": tmp_path / "snapshot-50000",
        "dataset": dataset, "manifest": manifest, "source": source,
        "config": config, "contract": contract, "contract_path": contract_path,
    }


@pytest.mark.parametrize("stage", ["native_tokenizer", "native_root"])
def test_export_byte_copies_checkpoint_and_preserves_honest_bounded_provenance(bound_run, stage):
    case = bound_run
    contract = copy.deepcopy(case["contract"])
    contract["stage"] = stage
    write_json(case["contract_path"], contract)
    originals = {path: path.read_bytes() for path in (
        case["checkpoint"], case["contract_path"], case["config"], case["manifest"], case["source"],
    )}

    result = snapshot.export_snapshot(case["run"], case["checkpoint"], case["output"], STEP)

    final_checkpoint = case["output"] / "checkpoints/final.ckpt"
    assert final_checkpoint.read_bytes() == originals[case["checkpoint"]]
    assert result["checkpoint_sha256"] == digest(case["checkpoint"]) == digest(final_checkpoint)
    assert (case["output"] / "native_config.yaml").read_bytes() == originals[case["config"]]
    exported = json.loads((case["output"] / "run_contract.json").read_text(encoding="utf-8"))
    training = json.loads((case["output"] / "training_result.json").read_text(encoding="utf-8"))
    provenance = exported["snapshot_provenance"]
    assert exported["stage"] == stage
    assert exported["status"] == "completed"
    assert exported["stop_steps"] == STEP
    assert exported["original_schedule_steps"] == SCHEDULE_STEPS
    assert exported["source_lf_sha256"] == contract["source_lf_sha256"]
    assert exported["dataset_manifest_sha256"] == digest(case["manifest"])
    assert exported["quality_approved"] is exported["ue_deployment"] is False
    assert provenance["artifact_kind"] == "selected_checkpoint_export"
    assert provenance["source_original_stop_steps"] == SCHEDULE_STEPS
    assert provenance["requested_stop_steps"] == STEP
    assert provenance["source_contract_sha256"] == digest(case["contract_path"])
    assert Path(provenance["source_run"]) == case["run"]
    assert Path(provenance["source_checkpoint"]) == case["checkpoint"]
    assert provenance["original_long_run_completed"] is provenance["quality_approved"] is False
    assert training["optimizer_updates"] == STEP
    assert training["updates_this_run"] == STEP - contract["resume"]["saved_step"]
    assert training["final_checkpoint_sha256"] == digest(final_checkpoint)
    assert training["elapsed_seconds"] is training["peak_gpu_memory_mb"] is None
    assert training["skipped_batches"] is None
    assert training["evidence"] == exported["checkpoint_training_evidence"]
    for path, original in originals.items():
        assert path.read_bytes() == original
    assert not list(case["output"].rglob("*.tmp"))


@pytest.mark.parametrize("changed", ["config", "manifest", "artifact", "source"])
def test_export_rejects_changed_bindings_before_creating_output(bound_run, changed):
    case = bound_run
    path = case["dataset"] / "shards/train.npz" if changed == "artifact" else case[changed]
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="changed"):
        snapshot.export_snapshot(case["run"], case["checkpoint"], case["output"], STEP)
    assert not case["output"].exists()


@pytest.mark.parametrize("location", ["existing", "source", "nested"])
def test_export_requires_fresh_output_outside_source_run(bound_run, location):
    case = bound_run
    output = case["output"]
    if location == "existing":
        output.mkdir()
        (output / "keep.txt").write_bytes(b"preserve existing output")
    elif location == "source":
        output = case["run"]
    else:
        output = case["run"] / "nested-snapshot"
    original = case["checkpoint"].read_bytes()
    with pytest.raises(FileExistsError, match="fresh run outside"):
        snapshot.export_snapshot(case["run"], case["checkpoint"], output, STEP)
    assert case["checkpoint"].read_bytes() == original
    if location == "existing":
        assert (output / "keep.txt").read_bytes() == b"preserve existing output"
        assert not (output / "checkpoints").exists()
    elif location == "nested":
        assert not output.exists()


def test_export_rejects_checkpoint_outside_source_run(bound_run, tmp_path):
    checkpoint = tmp_path / "foreign.ckpt"
    torch.save(saved_checkpoint(), checkpoint)
    with pytest.raises(ValueError, match="belong to the source run"):
        snapshot.export_snapshot(bound_run["run"], checkpoint, bound_run["output"], STEP)
    assert not bound_run["output"].exists()


@pytest.mark.parametrize("changed", ["stage", "budget", "initialization", "checkpoint_step"])
def test_export_rejects_unsupported_stage_or_out_of_bounds_update(bound_run, changed):
    case = bound_run
    contract = copy.deepcopy(case["contract"])
    if changed == "stage":
        contract["stage"] = "native_pose"
    elif changed == "budget":
        contract["stop_steps"] = STEP - 1
    elif changed == "initialization":
        contract["resume"]["saved_step"] = STEP
    else:
        saved = saved_checkpoint()
        saved["global_step"] -= 1
        torch.save(saved, case["checkpoint"])
    write_json(case["contract_path"], contract)
    with pytest.raises(ValueError):
        snapshot.export_snapshot(case["run"], case["checkpoint"], case["output"], STEP)
    assert not case["output"].exists()
