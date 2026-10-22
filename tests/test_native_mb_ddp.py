import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import RandomSampler


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import NativeMBMotionDataset
from scripts import train_native_mb_ddp as ddp


def torchrun_environment(rank=0, world_size=2):
    return {"WORLD_SIZE": str(world_size), "LOCAL_WORLD_SIZE": str(world_size),
            "RANK": str(rank), "LOCAL_RANK": str(rank),
            "CUDA_VISIBLE_DEVICES": ",".join(f"GPU-{index}" for index in range(world_size))}


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_environment_accepts_single_node_rank_mapping(rank):
    result = ddp.distributed_environment(torchrun_environment(rank, 3))
    assert result == {"world_size": 3, "rank": rank, "local_rank": rank,
                      "visible_devices": ["GPU-0", "GPU-1", "GPU-2"]}


@pytest.mark.parametrize(("field", "value"), [
    ("WORLD_SIZE", "1"), ("WORLD_SIZE", "0"), ("WORLD_SIZE", "not-an-integer"),
    ("LOCAL_WORLD_SIZE", "1"), ("RANK", "-1"), ("RANK", "2"),
    ("LOCAL_RANK", "-1"), ("LOCAL_RANK", "1"),
    ("CUDA_VISIBLE_DEVICES", "GPU-0"), ("CUDA_VISIBLE_DEVICES", "GPU-0,GPU-1,GPU-2"),
    ("CUDA_VISIBLE_DEVICES", "GPU-0,GPU-0"), ("CUDA_VISIBLE_DEVICES", "GPU-0,"),
    ("CUDA_VISIBLE_DEVICES", "GPU-0, "),
])
def test_environment_rejects_invalid_world_ranks_or_visible_devices(field, value):
    env = torchrun_environment()
    env[field] = value
    with pytest.raises(ValueError):
        ddp.distributed_environment(env)


@pytest.mark.parametrize("field", ["WORLD_SIZE", "RANK", "LOCAL_RANK", "CUDA_VISIBLE_DEVICES"])
def test_environment_rejects_missing_launch_identity(field):
    env = torchrun_environment()
    del env[field]
    with pytest.raises(ValueError):
        ddp.distributed_environment(env)


def test_environment_uses_explicit_visible_devices_override():
    env = torchrun_environment(1)
    env["CUDA_VISIBLE_DEVICES"] = "ignored"
    assert ddp.distributed_environment(env, visible="GPU-a,GPU-b")["visible_devices"] == [
        "GPU-a", "GPU-b",
    ]


@pytest.fixture
def native_dataset(tmp_path):
    (tmp_path / "stats").mkdir()
    (tmp_path / "shards").mkdir()
    np.save(tmp_path / "stats/mean.npy", np.zeros(370, dtype=np.float32))
    np.save(tmp_path / "stats/std.npy", np.ones(370, dtype=np.float32))
    records = []
    for index in range(2):
        features = np.full((80, 366), index + 20., dtype=np.float32)
        features[:, 0] = index * 1000. + np.arange(80, dtype=np.float32)
        shard = f"shards/train_{index}.npz"
        np.savez_compressed(tmp_path / shard, global_features=features)
        records.append({"filename": f"train_{index}", "split": "train",
                        "shard": shard, "frames_30hz": 80})
    (tmp_path / "accepted.json").write_text(json.dumps(records), encoding="utf-8")
    return NativeMBMotionDataset(tmp_path, "train", max_frames=65, min_frames=65,
                                 cache_max_bytes=2 * 80 * 366 * 4)


@pytest.mark.parametrize("rank", [0, 1])
def test_each_rank_gets_full_256_real_windows_from_two_records(native_dataset, rank):
    loader = ddp.native_ddp_loader(native_dataset, 256, 0, 7, rank=rank, world_size=2)
    batch = next(iter(loader))

    assert isinstance(loader.sampler, DistributedSamplerWrapper)
    assert loader.sampler.rank == rank
    assert loader.sampler.num_replicas == 2
    assert loader.drop_last is True
    assert len(loader) == 256
    assert batch["batch_size"] == 256
    assert batch["motion"].shape == (256, 65, 366)
    assert batch["motion_len"].tolist() == [65] * 256
    assert batch["motion_pad_mask"].all()
    raw = native_dataset.stats.unnormalize(batch["motion"])
    record_ids = raw[:, 0, 1].round() - 20
    assert set(record_ids.tolist()).issubset({0., 1.})
    starts = raw[:, 0, 0] - record_ids * 1000
    torch.testing.assert_close(starts, starts.round(), atol=2e-4, rtol=0)
    assert starts.min() >= -2e-4
    assert starts.max() <= 15.0002
    torch.testing.assert_close(raw[:, -1, 0] - raw[:, 0, 0], torch.full((256,), 64.),
                               atol=2e-4, rtol=0)


def test_seeded_replacement_draws_are_partitioned_between_ranks(native_dataset):
    loaders = [ddp.native_ddp_loader(native_dataset, 8, 0, 71, rank=rank, world_size=2)
               for rank in range(2)]
    rank_indices = [list(loader.sampler) for loader in loaders]
    for rank, loader in enumerate(loaders):
        original = loader.sampler.dataset._sampler
        assert isinstance(original, RandomSampler)
        assert original.replacement is True
        assert original.generator.initial_seed() == 71
        assert loader.generator.initial_seed() == 71 + rank
    reference = list(RandomSampler(native_dataset, replacement=True,
                                  num_samples=sum(map(len, rank_indices)),
                                  generator=torch.Generator().manual_seed(71)))
    assert rank_indices[0] == reference[0::2]
    assert rank_indices[1] == reference[1::2]
    assert rank_indices[0] != rank_indices[1]


def test_sampler_seed_is_deterministic_and_independent_of_global_rng(native_dataset):
    with torch.random.fork_rng():
        torch.manual_seed(100)
        first = ddp.native_ddp_loader(native_dataset, 8, 0, 7)
        torch.manual_seed(999)
        second = ddp.native_ddp_loader(native_dataset, 8, 0, 7)
        different = ddp.native_ddp_loader(native_dataset, 8, 0, 8)
        first_indices = list(first.sampler)
        assert first_indices == list(second.sampler)
        assert first_indices != list(different.sampler)


def hash_lf(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def resumable_run(tmp_path):
    code_root = tmp_path / "code"
    sources = (*ddp.SOURCE_FILES, "motionbricks/data/native_mb_paper_protocol.py")
    assert ddp.DDP_SOURCE not in ddp.SOURCE_FILES
    for relative in (*sources, ddp.DDP_SOURCE):
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("# unchanged synthetic source: " + relative + "\r\n").encode())
    run = tmp_path / "pose-50000"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/final.ckpt"
    checkpoint.write_bytes(b"synthetic full checkpoint identity")
    contract = {
        "stage": "native_pose", "status": "trained_not_reviewed", "seed": 7, "batch_size": 256,
        "dataset_manifest_sha256": "dataset-hash", "config_sha256": "config-hash",
        "tokenizer": {"checkpoint_sha256": "tokenizer-checkpoint-hash"},
        "stop_steps": 50_000, "original_schedule_steps": 2_000_000,
        "training_profile": {"training_profile": "paper_pose"},
        "source_lf_sha256": {relative: hash_lf(code_root / relative) for relative in sources},
    }
    expected = copy.deepcopy(contract)
    expected["stop_steps"] = 60_000
    write_json(run / "run_contract.json", contract)
    write_json(run / "training_result.json", {
        "optimizer_updates": 50_000,
        "final_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    })
    return run, expected, code_root, contract


def test_single_to_ddp_resume_allows_new_runner_without_changing_original_sources(resumable_run):
    run, expected, code_root, _ = resumable_run
    runner = code_root / ddp.DDP_SOURCE
    runner.write_bytes(b"# new external DDP runner\n")

    binding = ddp.validate_ddp_resume(run, expected, code_root)

    assert binding["saved_step"] == 50_000
    assert Path(binding["checkpoint"]) == run / "checkpoints/final.ckpt"
    assert binding["entrypoint_revision"] is None
    assert "larger global batch" in binding["sampling_note"]


@pytest.mark.parametrize("relative", ddp.SOURCE_FILES)
def test_ddp_resume_rejects_changes_to_every_original_source(resumable_run, relative):
    run, expected, code_root, _ = resumable_run
    path = code_root / relative
    path.write_bytes(path.read_bytes() + b"# changed original source\n")
    with pytest.raises(ValueError, match="source changed"):
        ddp.validate_ddp_resume(run, expected, code_root)


def test_ddp_resume_allows_only_line_ending_differences_in_original_sources(resumable_run):
    run, expected, code_root, _ = resumable_run
    for relative in (*ddp.SOURCE_FILES, "motionbricks/data/native_mb_paper_protocol.py"):
        path = code_root / relative
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    assert ddp.validate_ddp_resume(run, expected, code_root)["saved_step"] == 50_000


def test_ddp_resume_rejects_changed_paper_protocol(resumable_run):
    run, expected, code_root, _ = resumable_run
    (code_root / "motionbricks/data/native_mb_paper_protocol.py").write_bytes(b"changed protocol")
    with pytest.raises(ValueError, match="protocol source changed"):
        ddp.validate_ddp_resume(run, expected, code_root)


def test_ddp_continuation_binds_its_existing_runner(resumable_run):
    run, expected, code_root, contract = resumable_run
    contract["distributed_training"] = {"world_size": 2}
    contract["source_lf_sha256"][ddp.DDP_SOURCE] = hash_lf(code_root / ddp.DDP_SOURCE)
    write_json(run / "run_contract.json", contract)
    assert ddp.validate_ddp_resume(run, expected, code_root)["saved_step"] == 50_000
    (code_root / ddp.DDP_SOURCE).write_bytes(b"changed existing DDP runner")
    with pytest.raises(ValueError, match="DDP entrypoint changed"):
        ddp.validate_ddp_resume(run, expected, code_root)


@pytest.fixture
def resumable_ddp_run(resumable_run):
    run, _, code_root, contract = resumable_run
    contract["distributed_training"] = {"world_size": 2}
    contract["source_lf_sha256"][ddp.DDP_SOURCE] = hash_lf(code_root / ddp.DDP_SOURCE)
    write_json(run / "run_contract.json", contract)
    return resumable_run


def test_ddp_adapter_revision_records_previous_current_hashes_and_explicit_audit_reason(resumable_ddp_run):
    run, expected, code_root, contract = resumable_ddp_run
    original_contract = (run / "run_contract.json").read_bytes()
    runner = code_root / ddp.DDP_SOURCE
    runner.write_bytes(b"# reviewed external DDP adapter revision\n")
    reason = "Reviewed adapter receipt validation; original model and objective sources unchanged."

    binding = ddp.validate_ddp_resume(run, expected, code_root, runner_change_reason=reason)

    assert binding["ddp_entrypoint_revision"] == {
        "previous_sha256": contract["source_lf_sha256"][ddp.DDP_SOURCE],
        "current_sha256": hash_lf(runner), "reviewed_change_reason": reason,
    }
    assert binding["ddp_entrypoint_revision"]["previous_sha256"] != hash_lf(runner)
    assert binding["entrypoint_revision"] is None
    assert binding["saved_step"] == 50_000
    assert (run / "run_contract.json").read_bytes() == original_contract


@pytest.mark.parametrize("reason", [None, ""])
def test_ddp_adapter_revision_is_rejected_without_explicit_reason(resumable_ddp_run, reason):
    run, expected, code_root, _ = resumable_ddp_run
    (code_root / ddp.DDP_SOURCE).write_bytes(b"# changed DDP adapter\n")
    with pytest.raises(ValueError, match="DDP entrypoint changed"):
        ddp.validate_ddp_resume(run, expected, code_root, runner_change_reason=reason)


def test_ddp_adapter_revision_requires_previous_hash_even_with_audit_reason(resumable_ddp_run):
    run, expected, code_root, contract = resumable_ddp_run
    del contract["source_lf_sha256"][ddp.DDP_SOURCE]
    write_json(run / "run_contract.json", contract)
    with pytest.raises(ValueError, match="DDP entrypoint changed"):
        ddp.validate_ddp_resume(run, expected, code_root, runner_change_reason="Reviewed adapter only")


@pytest.mark.parametrize("relative", ddp.SOURCE_FILES)
def test_adapter_audit_reason_cannot_authorize_original_source_changes(resumable_ddp_run, relative):
    run, expected, code_root, _ = resumable_ddp_run
    (code_root / ddp.DDP_SOURCE).write_bytes(b"# reviewed DDP adapter revision\n")
    original_source = code_root / relative
    original_source.write_bytes(original_source.read_bytes() + b"# unauthorized original source change\n")
    with pytest.raises(ValueError, match="Control resume source changed"):
        ddp.validate_ddp_resume(run, expected, code_root, runner_change_reason="Reviewed adapter only")


def test_ddp_help_exposes_runner_audit_reason_without_launching_training(capsys):
    with pytest.raises(SystemExit) as stopped:
        ddp.main(["--help"])
    assert stopped.value.code == 0
    assert "--runner-change-reason" in capsys.readouterr().out


@pytest.mark.parametrize("changed", ["schedule", "tokenizer", "checkpoint"])
def test_ddp_resume_preserves_schedule_frozen_dependency_and_checkpoint(resumable_run, changed):
    run, expected, _, _ = resumable_run
    if changed == "schedule":
        expected["original_schedule_steps"] = 60_000
    elif changed == "tokenizer":
        expected["tokenizer"]["checkpoint_sha256"] = "changed-tokenizer"
    else:
        (run / "checkpoints/final.ckpt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError):
        ddp.validate_ddp_resume(run, expected, resumable_run[2])


def rank_receipts():
    return [{
        "rank": rank, "optimizer_updates": 60_000, "start_step": 50_000,
        "updates_this_run": 10_000, "skipped_batches": 0, "batch_size": 256,
        "observed_input_batch_sizes": [256], "sampler_rank": rank, "sampler_replicas": 2,
        "tokenizer_state_sha256": "frozen-tokenizer", "backbone_state_sha256": "shared-backbone",
        "restored_training_state": {
            "global_step": 50_000, "max_adam_step": 50_000.,
            "scheduler_last_epoch": 50_000, "schedule_steps": 2_000_000,
        },
    } for rank in range(2)]


def verify(receipts):
    return ddp.verify_rank_receipts(receipts, 50_000, 60_000, 2, "frozen-tokenizer")


def test_rank_receipts_require_each_rank_to_finish_the_same_resumed_updates():
    assert verify(list(reversed(rank_receipts()))) is None


def test_rank_receipts_accept_complete_normalized_restored_state_for_each_rank():
    receipts = json.loads(json.dumps(rank_receipts(), allow_nan=False))
    expected = {"global_step": 50_000, "max_adam_step": 50_000.,
                "scheduler_last_epoch": 50_000, "schedule_steps": 2_000_000}
    for receipt in receipts:
        assert receipt["restored_training_state"] == expected
        assert all(type(value) in (int, float) for value in receipt["restored_training_state"].values())
    assert verify(receipts) is None


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("field", [
    "global_step", "max_adam_step", "scheduler_last_epoch", "schedule_steps",
])
def test_rank_receipts_require_every_restored_state_field_on_each_rank(rank, field):
    receipts = rank_receipts()
    del receipts[rank]["restored_training_state"][field]
    with pytest.raises(ValueError):
        verify(receipts)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("missing", ["absent", "none", "empty"])
def test_rank_receipts_require_restored_training_state_on_each_rank(rank, missing):
    receipts = rank_receipts()
    if missing == "absent":
        del receipts[rank]["restored_training_state"]
    else:
        receipts[rank]["restored_training_state"] = None if missing == "none" else {}
    with pytest.raises(ValueError):
        verify(receipts)


@pytest.mark.parametrize("rank", [0, 1, "all"])
@pytest.mark.parametrize(("field", "value"), [
    ("global_step", 49_999), ("max_adam_step", 49_999.),
    ("scheduler_last_epoch", 49_999), ("schedule_steps", 60_000),
])
def test_rank_receipts_reject_restored_state_that_does_not_match_resume(rank, field, value):
    receipts = rank_receipts()
    affected = receipts if rank == "all" else [receipts[rank]]
    for receipt in affected:
        receipt["restored_training_state"][field] = value
    with pytest.raises(ValueError):
        verify(receipts)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_rank_receipts_reject_nonfinite_restored_adam_step(rank, nonfinite):
    receipts = rank_receipts()
    receipts[rank]["restored_training_state"]["max_adam_step"] = nonfinite
    with pytest.raises(ValueError):
        verify(receipts)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(("field", "value"), [
    ("optimizer_updates", 59_999), ("start_step", 49_999), ("updates_this_run", 9_999),
    ("skipped_batches", 1), ("tokenizer_state_sha256", "changed-frozen-state"),
    ("observed_input_batch_sizes", [128]), ("sampler_rank", 9), ("sampler_replicas", 3),
    ("backbone_state_sha256", "different-backbone"),
])
def test_rank_receipts_detect_update_batch_sampler_and_frozen_state_mismatches(rank, field, value):
    receipts = rank_receipts()
    receipts[rank][field] = value
    with pytest.raises(ValueError):
        verify(receipts)


@pytest.mark.parametrize("changed", ["missing", "duplicate", "unexpected"])
def test_rank_receipts_reject_missing_duplicate_or_unexpected_ranks(changed):
    receipts = rank_receipts()
    if changed == "missing":
        receipts.pop()
    else:
        receipts[1]["rank"] = 0 if changed == "duplicate" else 2
    with pytest.raises(ValueError, match="Missing or duplicate"):
        verify(receipts)


def test_rank_receipts_reject_different_self_consistent_batch_sizes_between_ranks():
    receipts = rank_receipts()
    receipts[1]["batch_size"] = 128
    receipts[1]["observed_input_batch_sizes"] = [128]
    with pytest.raises(ValueError):
        verify(receipts)
