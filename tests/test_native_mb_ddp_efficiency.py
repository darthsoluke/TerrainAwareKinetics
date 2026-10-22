import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_paper_protocol import paper_config
from scripts import train_native_mb_ddp as ddp
from scripts import prepare_native_pose_ddp as prepare
from scripts import run_native_pose_ddp_workflow as workflow


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_paper_batch_plan_preserves_256_per_gpu_and_implied_global_batch(world):
    assert ddp.ddp_batch_plan("paper", 256, world) == {
        "profile": "paper", "per_gpu": 256, "global": 256 * world,
        "gradient_accumulation": 1,
    }


@pytest.mark.parametrize(("world", "per_gpu"), [(1, 256), (2, 128), (4, 64), (8, 32)])
def test_fixed_global_batch_plan_splits_explicit_256_equally(world, per_gpu):
    assert ddp.ddp_batch_plan("fixed_global_batch", per_gpu, world, global_batch_size=256) == {
        "profile": "fixed_global_batch", "per_gpu": per_gpu, "global": 256,
        "gradient_accumulation": 1,
    }


@pytest.mark.parametrize(("profile", "batch", "world", "global_batch"), [
    ("paper", 64, 4, None), ("paper", 256, 4, 1024),
    ("fixed_global_batch", 64, 4, None), ("fixed_global_batch", 64, 4, 128),
    ("fixed_global_batch", 64, 4, 512), ("fixed_global_batch", 64, 4, 256.),
    ("fixed_global_batch", 64, 4, True), ("fixed_global_batch", 85, 3, 256),
    ("fixed_global_batch", 256, 4, 256), ("fixed_global_batch", 64, 2, 256),
    ("fixed_global_batch", 0, 4, 256), ("fixed_global_batch", -64, 4, 256),
    ("fixed_global_batch", 64., 4, 256), ("fixed_global_batch", True, 4, 256),
    ("fixed_global_batch", 64, 0, 256), ("fixed_global_batch", 64, -4, 256),
    ("fixed_global_batch", 64, 4., 256), ("fixed_global_batch", 64, True, 256),
    ("unknown", 64, 4, 256),
])
def test_batch_plan_rejects_implicit_mismatched_or_invalid_batch_contracts(
    profile, batch, world, global_batch
):
    with pytest.raises(ValueError):
        ddp.ddp_batch_plan(profile, batch, world, global_batch_size=global_batch)


@pytest.mark.parametrize("count", [2, 4, 5])
def test_paper_gpu_selection_retains_all_idle_devices_at_batch256(count):
    gpus = [{"index": index, "uuid": f"GPU-{index}"} for index in range(count)]
    original = copy.deepcopy(gpus)
    selected, batch = workflow.select_batch_gpus(gpus, "paper")
    assert selected == original
    assert batch == 256
    assert gpus == original


def test_fixed_global_gpu_selection_uses_first_four_at_batch64_and_leaves_fifth_unclaimed():
    gpus = [{"index": index, "uuid": f"GPU-{index}"} for index in range(5)]
    original = copy.deepcopy(gpus)
    selected, batch = workflow.select_batch_gpus(gpus, "fixed_global_batch", global_batch_size=256)
    assert selected == original[:4]
    assert original[4] not in selected
    assert batch == 64
    assert ddp.ddp_batch_plan("fixed_global_batch", batch, len(selected), 256)["global"] == 256
    assert gpus == original


@pytest.mark.parametrize(("profile", "count", "global_batch"), [
    ("paper", 0, None), ("paper", 1, None), ("paper", 4, 256),
    ("fixed_global_batch", 3, 256), ("fixed_global_batch", 4, None),
    ("fixed_global_batch", 4, 512), ("unknown", 4, 256),
])
def test_gpu_selection_rejects_insufficient_idle_devices_and_incompatible_profiles(profile, count, global_batch):
    gpus = [{"index": index, "uuid": f"GPU-{index}"} for index in range(count)]
    with pytest.raises(ValueError):
        workflow.select_batch_gpus(gpus, profile, global_batch_size=global_batch)


def benchmark_environment():
    return {"WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0",
            "CUDA_VISIBLE_DEVICES": "GPU-benchmark"}


def test_single_rank_environment_requires_explicit_keyword_only_benchmark_opt_in():
    parameter = inspect.signature(ddp.distributed_environment).parameters["allow_single_rank"]
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False
    with pytest.raises(ValueError):
        ddp.distributed_environment(benchmark_environment())
    assert ddp.distributed_environment(benchmark_environment(), allow_single_rank=True) == {
        "world_size": 1, "rank": 0, "local_rank": 0, "visible_devices": ["GPU-benchmark"],
    }


@pytest.mark.parametrize(("field", "value"), [
    ("WORLD_SIZE", "0"), ("RANK", "1"), ("LOCAL_RANK", "1"),
    ("CUDA_VISIBLE_DEVICES", "GPU-benchmark,GPU-extra"),
])
def test_single_rank_benchmark_opt_in_keeps_rank_and_device_identity_strict(field, value):
    env = benchmark_environment()
    env[field] = value
    with pytest.raises(ValueError):
        ddp.distributed_environment(env, allow_single_rank=True)


class FakeTorch:
    def __init__(self, *, ignore_setting=False):
        self.precision = "highest"
        self.ignore_setting = ignore_setting
        self.calls = []

    def set_float32_matmul_precision(self, value):
        self.calls.append(("set", value))
        if not self.ignore_setting:
            self.precision = value

    def get_float32_matmul_precision(self):
        self.calls.append(("get", self.precision))
        return self.precision


def test_matmul_applies_official_high_without_changing_native_config_or_training_precision():
    config = paper_config("pose")
    original = OmegaConf.to_container(config, resolve=False)
    fake_torch = FakeTorch()
    assert config.matmul_precision == "high"

    assert ddp.apply_native_matmul_precision(config, torch_module=fake_torch) == "high"

    assert fake_torch.calls == [("set", "high"), ("get", "high")]
    assert OmegaConf.to_container(config, resolve=False) == original


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
def test_matmul_applies_and_verifies_the_declared_valid_setting(precision):
    fake_torch = FakeTorch()
    assert ddp.apply_native_matmul_precision({"matmul_precision": precision}, fake_torch) == precision
    assert fake_torch.calls == [("set", precision), ("get", precision)]


@pytest.mark.parametrize("config", [{}, {"matmul_precision": None}, {"matmul_precision": ""},
                                  {"matmul_precision": "bf16"}, {"matmul_precision": "High"},
                                  {"matmul_precision": 32}])
def test_matmul_rejects_missing_or_invalid_setting_before_touching_torch(config):
    fake_torch = FakeTorch()
    with pytest.raises(ValueError):
        ddp.apply_native_matmul_precision(config, fake_torch)
    assert fake_torch.calls == []


def test_matmul_rejects_a_setter_that_does_not_apply_the_setting():
    fake_torch = FakeTorch(ignore_setting=True)
    with pytest.raises(RuntimeError, match="not applied"):
        ddp.apply_native_matmul_precision({"matmul_precision": "high"}, fake_torch)
    assert fake_torch.calls == [("set", "high"), ("get", "highest")]


def hash_lf(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def resumable_run(tmp_path):
    code_root = tmp_path / "code"
    sources = (*ddp.SOURCE_FILES, "motionbricks/data/native_mb_paper_protocol.py", ddp.DDP_SOURCE)
    for relative in sources:
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("# synthetic frozen source: " + relative + "\r\n").encode())
    run = tmp_path / "paper-pose"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/final.ckpt"
    checkpoint.write_bytes(b"synthetic checkpoint; model and optimizer bytes preserved")
    previous = {
        "stage": "native_pose", "status": "trained_not_reviewed", "seed": 7,
        "batch_size": 256, "dataset_manifest_sha256": "dataset-hash",
        "config_sha256": "unchanged-config-hash", "stop_steps": 50_000,
        "original_schedule_steps": 2_000_000,
        "tokenizer": {"checkpoint_sha256": "frozen-tokenizer-checkpoint"},
        "training_profile": {"training_profile": "paper_s7_4_optimizer_release_architecture_soma30"},
        "distributed_training": {"world_size": 4},
        "sampling_contract": {"global_input_batch_size": 1024},
        "source_lf_sha256": {relative: hash_lf(code_root / relative) for relative in sources},
    }
    write_json(run / "run_contract.json", previous)
    write_json(run / "training_result.json", {
        "optimizer_updates": 50_000,
        "final_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    })
    expected = copy.deepcopy(previous)
    expected.update(batch_size=64, stop_steps=60_000)
    expected["sampling_contract"]["global_input_batch_size"] = 256
    return {
        "run": run, "code_root": code_root, "previous": previous, "expected": expected,
        "reason": "User-approved global input batch256 across four GPUs; original model/loss/schedule unchanged.",
    }


def resume(case, **kwargs):
    return ddp.validate_ddp_resume(case["run"], case["expected"], case["code_root"], **kwargs)


def test_batch_change_opt_in_is_keyword_only_and_disabled_by_default():
    parameter = inspect.signature(ddp.validate_ddp_resume).parameters["allow_batch_change"]
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


@pytest.mark.parametrize("previous_world", [None, 2, 4])
def test_audited_batch_only_resume_records_per_gpu_and_global_transition(resumable_run, previous_world):
    case = resumable_run
    previous = case["previous"]
    if previous_world is None:
        del previous["distributed_training"]
    else:
        previous["distributed_training"]["world_size"] = previous_world
    previous["sampling_contract"]["global_input_batch_size"] = 256 * (previous_world or 1)
    contract_path = case["run"] / "run_contract.json"
    write_json(contract_path, previous)
    originals = {path: path.read_bytes() for path in (
        contract_path, case["run"] / "training_result.json", case["run"] / "checkpoints/final.ckpt",
    )}
    expected_before = copy.deepcopy(case["expected"])

    binding = resume(case, runner_change_reason=case["reason"], allow_batch_change=True)

    assert binding["batch_transition"] == {
        "previous_per_gpu": 256, "current_per_gpu": 64,
        "previous_global": 256 * (previous_world or 1), "current_global": 256,
        "reason": case["reason"],
    }
    assert binding["saved_step"] == 50_000
    assert binding["entrypoint_revision"] is None
    assert "ddp_entrypoint_revision" not in binding
    assert binding["sha256"] == hashlib.sha256(originals[case["run"] / "checkpoints/final.ckpt"]).hexdigest()
    assert case["expected"] == expected_before
    for path, original in originals.items():
        assert path.read_bytes() == original


def test_batch_change_requires_opt_in_even_with_explicit_reason(resumable_run):
    with pytest.raises(ValueError, match="batch_size"):
        resume(resumable_run, runner_change_reason=resumable_run["reason"])


@pytest.mark.parametrize("reason", [None, ""])
def test_batch_change_requires_explicit_audit_reason(resumable_run, reason):
    with pytest.raises(ValueError, match="explicit reviewed reason"):
        resume(resumable_run, runner_change_reason=reason, allow_batch_change=True)


def test_unchanged_batch_resume_needs_no_batch_exception_or_transition(resumable_run):
    case = resumable_run
    case["expected"]["batch_size"] = 256
    case["expected"]["sampling_contract"]["global_input_batch_size"] = 1024
    binding = resume(case)
    assert binding["saved_step"] == 50_000
    assert "batch_transition" not in binding


@pytest.mark.parametrize("field", ["stage", "config_sha256", "seed", "dataset_manifest_sha256",
                                   "original_schedule_steps", "tokenizer"])
def test_batch_exception_preserves_every_nonbatch_resume_binding(resumable_run, field):
    case = resumable_run
    if field == "seed":
        case["expected"][field] = 8
    elif field == "original_schedule_steps":
        case["expected"][field] = 60_000
    elif field == "tokenizer":
        case["expected"][field]["checkpoint_sha256"] = "changed-tokenizer"
    else:
        case["expected"][field] = "changed-binding"
    with pytest.raises(ValueError):
        resume(case, runner_change_reason=case["reason"], allow_batch_change=True)


@pytest.mark.parametrize("relative", (*ddp.SOURCE_FILES, "motionbricks/data/native_mb_paper_protocol.py"))
def test_batch_audit_cannot_authorize_original_training_source_changes(resumable_run, relative):
    case = resumable_run
    path = case["code_root"] / relative
    path.write_bytes(path.read_bytes() + b"# changed original source\n")
    with pytest.raises(ValueError, match="source changed"):
        resume(case, runner_change_reason=case["reason"], allow_batch_change=True)


def test_batch_transition_and_adapter_revision_are_audited_separately(resumable_run):
    case = resumable_run
    runner = case["code_root"] / ddp.DDP_SOURCE
    runner.write_bytes(b"# reviewed efficiency adapter revision\n")
    binding = resume(case, runner_change_reason=case["reason"], allow_batch_change=True)
    assert binding["ddp_entrypoint_revision"] == {
        "previous_sha256": case["previous"]["source_lf_sha256"][ddp.DDP_SOURCE],
        "current_sha256": hash_lf(runner), "reviewed_change_reason": case["reason"],
    }
    assert binding["batch_transition"]["reason"] == case["reason"]
    assert binding["batch_transition"]["current_global"] == 256


@pytest.mark.parametrize("changed", ["missing_global", "wrong_global", "per_gpu", "world"])
def test_batch_exception_still_requires_a_valid_fixed_global_plan(resumable_run, changed):
    case = resumable_run
    if changed == "missing_global":
        del case["expected"]["sampling_contract"]["global_input_batch_size"]
    elif changed == "wrong_global":
        case["expected"]["sampling_contract"]["global_input_batch_size"] = 512
    elif changed == "per_gpu":
        case["expected"]["batch_size"] = 128
    else:
        case["expected"]["distributed_training"]["world_size"] = 3
    with pytest.raises(ValueError):
        resume(case, runner_change_reason=case["reason"], allow_batch_change=True)


def test_batch_exception_still_rejects_changed_checkpoint_bytes(resumable_run):
    case = resumable_run
    (case["run"] / "checkpoints/final.ckpt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        resume(case, runner_change_reason=case["reason"], allow_batch_change=True)


@pytest.mark.parametrize("batch_transition", [False, True])
def test_production_resume_rejects_benchmark_only_checkpoint_even_with_batch_audit(resumable_run, batch_transition):
    case = resumable_run
    case["previous"]["benchmark_only"] = True
    write_json(case["run"] / "run_contract.json", case["previous"])
    case["expected"]["benchmark_only"] = False
    if not batch_transition:
        case["expected"]["batch_size"] = 256
        case["expected"]["sampling_contract"]["global_input_batch_size"] = 1024
    with pytest.raises(ValueError, match="Production cannot resume"):
        resume(case, runner_change_reason=case["reason"], allow_batch_change=batch_transition)


@pytest.fixture
def mocked_handoff(tmp_path, monkeypatch):
    source = tmp_path / "source-pose"
    old_workflow = tmp_path / "old-workflow"
    (source / "checkpoints").mkdir(parents=True)
    old_workflow.mkdir()
    write_json(source / "run_contract.json", {"stage": "native_pose", "status": "running"})
    write_json(old_workflow / "workflow_state.json", {"status": "running"})
    request = {"manager": {"pid": 100}, "pose": {"pid": 101}, "pose_run": str(source),
               "source_workflow": str(old_workflow), "target_step": 50_000}
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    output = tmp_path / "handoff"
    current = {"elapsed": 0., "manager_stopping": False, "pose_stopped": False,
               "manager_delay": .1}
    events = []

    def sleep(seconds):
        events.append(("sleep", seconds))
        current["elapsed"] += seconds

    def process_state(identity):
        if identity["pid"] == request["manager"]["pid"]:
            return "T" if current["manager_stopping"] and current["elapsed"] >= current["manager_delay"] else "R"
        assert identity == request["pose"]
        return "T" if current["pose_stopped"] else "R"

    def signal_verified(identity, sig):
        assert sig == 19
        events.append(("stop", identity["pid"]))
        if identity == request["manager"]:
            current["manager_stopping"] = True
        else:
            assert identity == request["pose"]
            current["pose_stopped"] = True

    def stable_file(path):
        assert process_state(request["manager"]) == "T"
        assert path == source / "checkpoints/native-step=50000.ckpt"
        events.append(("checkpoint_ready",))
        return True

    def export_snapshot(source_run, checkpoint, destination, step):
        assert process_state(request["manager"]) == process_state(request["pose"]) == "T"
        assert Path(source_run) == source
        assert checkpoint == source / "checkpoints/native-step=50000.ckpt"
        assert step == 50_000
        events.append(("export", current["elapsed"]))
        return {"run": str(destination), "optimizer_updates": step}

    monkeypatch.setattr(prepare, "time", SimpleNamespace(
        monotonic=lambda: current["elapsed"], sleep=sleep,
        time=lambda: 1000. + current["elapsed"],
    ))
    monkeypatch.setattr(prepare, "signal", SimpleNamespace(SIGSTOP=19, SIGCONT=18))
    monkeypatch.setattr(prepare, "process_state", process_state)
    monkeypatch.setattr(prepare, "signal_verified", signal_verified)
    monkeypatch.setattr(prepare, "stable_file", stable_file)
    monkeypatch.setattr(prepare, "export_snapshot", export_snapshot)
    monkeypatch.setattr(prepare, "terminate_verified", lambda identity: events.append(("terminate", identity["pid"])))
    monkeypatch.setattr(sys, "argv", ["prepare_native_pose_ddp.py", "--request", str(request_path),
                                     "--output", str(output), "--suspend-manager"])
    return current, events, output


def test_handoff_waits_for_asynchronous_verified_manager_stop_before_export(mocked_handoff):
    current, events, output = mocked_handoff

    prepare.main()

    state = json.loads((output / "prepare_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "ready_for_ddp"
    assert current["manager_delay"] <= current["elapsed"] < 5
    assert events[0] == ("stop", 100)
    assert any(event[0] == "sleep" for event in events)
    assert events.index(("stop", 101)) < events.index(("export", current["elapsed"]))
    assert events[-2:] == [("terminate", 101), ("terminate", 100)]


def test_handoff_rejects_manager_that_does_not_stop_within_bounded_wait(mocked_handoff):
    current, events, output = mocked_handoff
    current["manager_delay"] = 10.

    with pytest.raises(ValueError, match="manager must be suspended"):
        prepare.main()

    state = json.loads((output / "prepare_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert 5 <= current["elapsed"] <= 5.1
    assert not any(event[0] in {"export", "terminate", "checkpoint_ready"} for event in events)
    assert ("stop", 101) not in events


@pytest.fixture
def completed_fixed_global_run(tmp_path):
    run = tmp_path / "fixed-global-pose"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/final.ckpt"
    checkpoint.write_bytes(b"synthetic bound four-rank checkpoint")
    plan = {"profile": "fixed_global_batch", "per_gpu": 64, "global": 256,
            "gradient_accumulation": 1}
    numerics = {"trainer_precision": 32, "bf16": False, "effective_matmul_precision": "high"}
    contract = {
        "stage": "native_pose", "status": "trained_not_reviewed", "batch_size": 64,
        "original_schedule_steps": 2_000_000, "tokenizer_state_sha256": "frozen-tokenizer",
        "batch_profile": copy.deepcopy(plan), "runtime_numerics": copy.deepcopy(numerics),
        "distributed_training": {"world_size": 4},
    }
    receipts = [{
        "rank": rank, "optimizer_updates": 50_000, "start_step": 40_000,
        "updates_this_run": 10_000, "skipped_batches": 0, "batch_size": 64,
        "observed_input_batch_sizes": [64], "sampler_rank": rank, "sampler_replicas": 4,
        "tokenizer_state_sha256": "frozen-tokenizer", "backbone_state_sha256": "shared-backbone",
        "runtime_numerics": copy.deepcopy(numerics),
        "restored_training_state": {
            "global_step": 40_000, "max_adam_step": 40_000.,
            "scheduler_last_epoch": 40_000, "schedule_steps": 2_000_000,
        },
    } for rank in range(4)]
    result = {
        "world_size": 4, "optimizer_updates": 50_000, "updates_this_run": 10_000,
        "skipped_batches": 0, "optimizer_class": "torch.optim.adam.Adam",
        "final_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "batch_profile": copy.deepcopy(plan), "runtime_numerics": copy.deepcopy(numerics),
        "rank_receipts": receipts, "observed_input_batch_sizes": [64],
    }
    write_json(run / "run_contract.json", contract)
    write_json(run / "training_result.json", result)
    return run, contract, result


def test_workflow_verify_run_accepts_actual_four_rank_batch64_receipts(completed_fixed_global_run):
    run, _, result = completed_fixed_global_run
    assert workflow.verify_run(run, 40_000, 50_000, 4) == result


def test_workflow_verify_run_rejects_uniform_batch256_receipts_when_plan_requires64(completed_fixed_global_run):
    run, _, result = completed_fixed_global_run
    for receipt in result["rank_receipts"]:
        receipt["batch_size"] = 256
        receipt["observed_input_batch_sizes"] = [256]
    result["observed_input_batch_sizes"] = [256]
    assert ddp.verify_rank_receipts(result["rank_receipts"], 40_000, 50_000, 4, "frozen-tokenizer") is None
    write_json(run / "training_result.json", result)

    with pytest.raises(ValueError, match="batch/numerical execution receipts disagree"):
        workflow.verify_run(run, 40_000, 50_000, 4)


@pytest.mark.parametrize(("per_gpu", "global_batch"), [(32, 256), (128, 512)],
                         ids=["wrong-global-arithmetic", "fixed-global512"])
def test_workflow_verify_run_rejects_invalid_global_plan_even_when_metadata_agrees(
    completed_fixed_global_run, per_gpu, global_batch
):
    run, contract, result = completed_fixed_global_run
    plan = {"profile": "fixed_global_batch", "per_gpu": per_gpu, "global": global_batch,
            "gradient_accumulation": 1}
    contract["batch_profile"] = copy.deepcopy(plan)
    contract["batch_size"] = per_gpu
    result["batch_profile"] = copy.deepcopy(plan)
    result["observed_input_batch_sizes"] = [per_gpu]
    for receipt in result["rank_receipts"]:
        receipt["batch_size"] = per_gpu
        receipt["observed_input_batch_sizes"] = [per_gpu]
    assert ddp.verify_rank_receipts(result["rank_receipts"], 40_000, 50_000, 4, "frozen-tokenizer") is None
    write_json(run / "run_contract.json", contract)
    write_json(run / "training_result.json", result)

    with pytest.raises(ValueError, match="batch/numerical execution receipts disagree"):
        workflow.verify_run(run, 40_000, 50_000, 4)


def test_workflow_verify_run_binds_contract_batch_size_to_plan(completed_fixed_global_run):
    run, contract, _ = completed_fixed_global_run
    contract["batch_size"] = 256
    write_json(run / "run_contract.json", contract)

    with pytest.raises(ValueError, match="batch/numerical execution receipts disagree"):
        workflow.verify_run(run, 40_000, 50_000, 4)
