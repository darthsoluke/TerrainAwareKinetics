from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


MOTIONBRICKS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOTIONBRICKS_ROOT))

from motionbricks.data.native_mb_online_checkpoint import (
    ONLINE_CHECKPOINT_VERSION,
    copy_online_warmstart_backbone,
    sha256_file,
    state_tree_digest,
    validate_online_config,
    validate_online_control_run,
    validate_online_resume_run,
    validate_parc_terrain_warmstart_run,
    verify_online_root_dependency,
    verify_same_tokenizer_dependency,
)
from motionbricks.data.native_mb_online_conditions import ONLINE_PROTOCOL
from motionbricks.data.native_mb_terrain_protocol import TERRAIN_PROTOCOL
from scripts import train_native_mb_control as runner


class _RootBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.original = nn.Linear(3, 4)
        self._proj_global_root_component_mask = nn.Linear(2, 4, bias=False)
        self.target_component_mask_blocks = nn.ModuleList([nn.Linear(2, 4, bias=False)])
        nn.init.zeros_(self._proj_global_root_component_mask.weight)
        nn.init.zeros_(self.target_component_mask_blocks[0].weight)


class _TerrainRootBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.original = nn.Linear(3, 4)


class _PoseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.original = nn.Linear(3, 4)


def _online_contract():
    return {
        "version": ONLINE_CHECKPOINT_VERSION,
        "protocol_version": ONLINE_PROTOCOL["version"],
        "protocol": deepcopy(ONLINE_PROTOCOL),
    }


def _valid_provenance():
    return {
        "schema": "motionbricks.parc_training_sources",
        "schema_version": 1,
        "source_row_count": 4,
        "admitted_source_count": 4,
        "excluded_source_count": 0,
        "split_counts": {"train": 2, "validation": 1, "test": 1},
        "train_origin_counts": {"parc_generated": 1, "parc_official": 1},
        "admitted_source_ids": ["a", "b", "c", "d"],
        "excluded_source_ids": [],
    }


def _valid_control_contract(stage="root", status="trained_not_reviewed"):
    return {
        "stage": f"native_{stage}",
        "status": status,
        "stop_steps": 12,
        "dataset_manifest_sha256": "d" * 64,
        "parc_only_sources": {**_valid_provenance(),
                              "actual_loader_origin_counts": {"parc_generated": 1, "parc_official": 1}},
        "terrain_conditioning": deepcopy(TERRAIN_PROTOCOL),
        "online_contract": _online_contract(),
    }


def _valid_result():
    return {"optimizer_updates": 12, "final_checkpoint_sha256": "a" * 64}


def test_root_online_warmstart_copies_originals_and_keeps_only_mask_keys_fresh():
    torch.manual_seed(1)
    source = _TerrainRootBackbone()
    target = _RootBackbone()
    with torch.no_grad():
        source.original.weight.fill_(3.0)
        source.original.bias.fill_(4.0)
        target.original.weight.zero_()
        target.original.bias.zero_()

    transfer = copy_online_warmstart_backbone(target, source, "root")

    assert torch.equal(target.original.weight, source.original.weight)
    assert torch.equal(target.original.bias, source.original.bias)
    assert torch.count_nonzero(target._proj_global_root_component_mask.weight) == 0
    assert torch.count_nonzero(target.target_component_mask_blocks[0].weight) == 0
    assert transfer["strict_key_and_shape_match"]
    assert transfer["new_parameters_kept_fresh_zero"]
    assert transfer["new_parameter_keys"] == [
        "_proj_global_root_component_mask.weight",
        "target_component_mask_blocks.0.weight",
    ]
    assert transfer["copied_parameter_count"] == 2
    assert transfer["optimizer_state"] == "not_loaded_reset"


def test_verified_online_root_warmstart_preserves_every_learned_mask_parameter():
    source, target = _RootBackbone(), _RootBackbone()
    with torch.no_grad():
        source._proj_global_root_component_mask.weight.fill_(0.375)
        source.target_component_mask_blocks[0].weight.fill_(-0.625)
    before = {key: value.detach().clone() for key, value in source.state_dict().items()}

    transfer = copy_online_warmstart_backbone(
        target, source, "root", verified_online_source=True)

    for key, value in before.items():
        assert torch.equal(target.state_dict()[key], value)
    assert transfer["transfer_scope"] == "all_backbone_parameters"
    assert transfer["component_mask_source"] == "preserved_from_verified_online_source"
    assert not transfer["new_parameters_kept_fresh_zero"]
    assert transfer["copied_parameter_count"] == len(source.state_dict())


def test_online_warmstart_rejects_non_mask_key_or_shape_drift():
    source = _TerrainRootBackbone()
    target = _RootBackbone()
    source.extra = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="original backbone keys must match exactly"):
        copy_online_warmstart_backbone(target, source, "root")

    source = _TerrainRootBackbone()
    target = _RootBackbone()
    target.original = nn.Linear(3, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        copy_online_warmstart_backbone(target, source, "root")

    with pytest.raises(ValueError, match="source must omit target component-mask"):
        copy_online_warmstart_backbone(_RootBackbone(), _RootBackbone(), "root")


def test_pose_online_warmstart_is_strict_and_adds_no_parameters():
    torch.manual_seed(2)
    source, target = _PoseBackbone(), _PoseBackbone()
    transfer = copy_online_warmstart_backbone(target, source, "pose")
    assert torch.equal(target.original.weight, source.original.weight)
    assert transfer["strict_key_and_shape_match"]
    assert transfer["new_parameter_keys"] == []
    assert not transfer["new_parameters_kept_fresh_zero"]


def test_online_run_contract_binds_stage_dataset_parc_terrain_and_protocol():
    contract = _valid_control_contract()
    result = _valid_result()
    assert validate_online_control_run(
        contract, result, stage="root", dataset_manifest_sha256="d" * 64,
        parc_only_sources=_valid_provenance(), terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
    )

    cases = [
        ({"dataset_manifest_sha256": "e" * 64}, "dataset manifest hash"),
        ({"parc_only_sources": None}, "PARC-only source provenance"),
        ({"terrain_conditioning": {"version": "wrong"}}, "terrain contract"),
        ({"online_contract": {"protocol": {}}}, "protocol contract"),
        ({"stage": "native_pose"}, "stage mismatch"),
    ]
    for changes, message in cases:
        invalid = deepcopy(contract)
        invalid.update(changes)
        with pytest.raises(ValueError, match=message):
            validate_online_control_run(
                invalid, result, stage="root", dataset_manifest_sha256="d" * 64,
                parc_only_sources=_valid_provenance(), terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
            )

    with pytest.raises(ValueError, match="fully trained state"):
        validate_online_control_run(
            contract, result, stage="root", dataset_manifest_sha256="d" * 64,
            parc_only_sources=_valid_provenance(), terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
            require_completed=True,
        )


def test_online_warmstart_accepts_same_stage_parc_terrain_baseline_without_online_protocol():
    contract = _valid_control_contract("root")
    contract.pop("online_contract")
    assert validate_parc_terrain_warmstart_run(
        contract, _valid_result(), stage="root", dataset_manifest_sha256="d" * 64,
        parc_only_sources=_valid_provenance(), terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
    )
    contract["dataset_manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="dataset manifest hash"):
        validate_parc_terrain_warmstart_run(
            contract, _valid_result(), stage="root", dataset_manifest_sha256="d" * 64,
            parc_only_sources=_valid_provenance(), terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
        )


@pytest.mark.parametrize("component_conditioning", [False, "false"])
def test_online_config_requires_versioned_root_component_support(component_conditioning):
    config = {
        "model": {
            "args": {"online_conditioning": ONLINE_PROTOCOL["version"]},
            "backbone_network": {"args": {
                "online_conditioning": ONLINE_PROTOCOL["version"],
                "root_component_conditioning": True,
            }},
        },
    }
    assert validate_online_config(config, "root") == ONLINE_PROTOCOL["version"]
    config["model"]["backbone_network"]["args"]["root_component_conditioning"] = component_conditioning
    with pytest.raises(ValueError, match="root component conditioning"):
        validate_online_config(config, "root")


def test_pose_warmstart_requires_exact_tokenizer_dependency():
    dependency = {"run": "tokenizer", "checkpoint_sha256": "b" * 64,
                  "global_step": 50_000, "dataset_manifest_sha256": "d" * 64}
    assert verify_same_tokenizer_dependency(dependency, deepcopy(dependency)) == "b" * 64
    changed = deepcopy(dependency)
    changed["run"] = "different-tokenizer"
    with pytest.raises(ValueError, match="exact tokenizer dependency"):
        verify_same_tokenizer_dependency(dependency, changed)


def test_online_root_dependency_checks_checkpoint_and_dataset_identity(tmp_path):
    root_run = tmp_path / "root-run"
    dataset = tmp_path / "dataset"
    checkpoints = root_run / "checkpoints"
    checkpoints.mkdir(parents=True)
    dataset.mkdir()
    manifest = dataset / "native_manifest.json"
    manifest.write_text('{"manifest":"same"}', encoding="utf-8")
    config_path = root_run / "native_config.yaml"
    config_path.write_text(
        "model:\n  args:\n    online_conditioning: native-mb-planar-online-v1\n"
        "  backbone_network:\n    args:\n"
        "      online_conditioning: native-mb-planar-online-v1\n"
        "      root_component_conditioning: true\n",
        encoding="utf-8",
    )
    checkpoint = checkpoints / "final.ckpt"
    checkpoint.write_bytes(b"root-checkpoint")
    dataset_hash = sha256_file(manifest)
    result = {
        "optimizer_updates": 12,
        "final_checkpoint_sha256": sha256_file(checkpoint),
    }
    contract = _valid_control_contract("root", status="trained_not_reviewed")
    contract.update({
        "dataset": str(dataset),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": dataset_hash,
        "stop_steps": 12,
    })
    (root_run / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (root_run / "training_result.json").write_text(json.dumps(result), encoding="utf-8")
    dependency = {
        "run": str(root_run),
        "checkpoint_sha256": result["final_checkpoint_sha256"],
        "state_digest": hashlib.sha256(b"frozen-state").hexdigest(),
        "dataset_manifest_sha256": dataset_hash,
        "status": "trained_not_reviewed",
        "end_state_digest": hashlib.sha256(b"frozen-state").hexdigest(),
        "frozen": True,
        "eval_mode": True,
        "registered_with_pose_model": False,
    }

    verified = verify_online_root_dependency(
        dependency, dataset_manifest_sha256=dataset_hash,
        parc_only_sources=_valid_provenance(), terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
    )
    assert verified["run"] == str(root_run.resolve())
    assert verified["status"] == "trained_not_reviewed"

    checkpoint.write_bytes(b"changed-checkpoint")
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        verify_online_root_dependency(
            dependency, dataset_manifest_sha256=dataset_hash,
            parc_only_sources=_valid_provenance(), terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
        )


def _online_resume_fixture(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = dataset / "native_manifest.json"
    manifest.write_text('{"manifest":"same"}', encoding="utf-8")
    run = tmp_path / "online-root"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    config_path = run / "native_config.yaml"
    config_path.write_text(
        "model:\n  args:\n    online_conditioning: native-mb-planar-online-v1\n"
        "  backbone_network:\n    args:\n"
        "      online_conditioning: native-mb-planar-online-v1\n"
        "      root_component_conditioning: true\n",
        encoding="utf-8",
    )
    model_state = {
        "backbone_net.original.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "backbone_net._proj_global_root_component_mask.weight": torch.full((2, 2), 0.375),
        "backbone_net.target_component_mask_blocks.0.weight": torch.full((2, 2), -0.625),
    }
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(2):
        optimizer.zero_grad()
        (parameter.square()).backward()
        optimizer.step()
        scheduler.step()
    checkpoint = checkpoints / "final.ckpt"
    torch.save({
        "global_step": 12,
        "state_dict": model_state,
        "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()],
    }, checkpoint)
    source_paths = tuple(dict.fromkeys(
        runner.SOURCE_FILES + runner.TERRAIN_SOURCE_FILES + runner.ONLINE_SOURCE_FILES
        + ("motionbricks/data/parc_training_sources.py",
           "motionbricks/data/native_mb_paper_protocol.py")))
    source_hashes = {
        relative: runner.source_lf_sha256(MOTIONBRICKS_ROOT / relative)
        for relative in source_paths
    }
    profile = {"stage": "root", "training_profile": "paper_s7_4_optimizer_release_architecture_soma30"}
    numerics = {"trainer_precision": "32", "effective_matmul_precision": "high"}
    runtime_versions = {"torch": "test", "lightning": "test", "numpy": "test"}
    mask_keys = [
        "_proj_global_root_component_mask.weight",
        "target_component_mask_blocks.0.weight",
    ]
    contract = _valid_control_contract("root", status="completed")
    contract.update({
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256_file(manifest),
        "config_sha256": sha256_file(config_path),
        "seed": 7,
        "batch_size": 256,
        "original_schedule_steps": 2_000_000,
        "training_profile": profile,
        "runtime_numerics": numerics,
        "runtime_versions": runtime_versions,
        "model_class": "test.RootModel",
        "network_class": "test.RootBackbone",
        "parameter_count": sum(value.numel() for value in model_state.values()),
        "tokenizer": None,
        "stop_steps": 12,
        "source_lf_sha256": dict(source_hashes),
    })
    contract["online_contract"]["component_mask_parameter_keys"] = mask_keys
    result = {
        "optimizer_updates": 12,
        "updates_this_run": 12,
        "final_checkpoint_sha256": sha256_file(checkpoint),
    }
    (run / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run / "training_result.json").write_text(json.dumps(result), encoding="utf-8")
    expected = {
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256_file(manifest),
        "config_sha256": sha256_file(config_path),
        "seed": 7,
        "batch_size": 256,
        "original_schedule_steps": 2_000_000,
        "stop_steps": 20,
        "training_profile": profile,
        "runtime_numerics": numerics,
        "runtime_versions": runtime_versions,
        "model_class": "test.RootModel",
        "network_class": "test.RootBackbone",
        "parameter_count": sum(value.numel() for value in model_state.values()),
        "tokenizer": None,
        "source_lf_sha256": dict(source_hashes),
    }
    return run, contract, result, expected, model_state, mask_keys, optimizer.state_dict(), scheduler.state_dict()


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_online_root_resume_preserves_full_optimizer_scheduler_and_mask_state(tmp_path):
    run, contract, result, expected, model_state, mask_keys, optimizer_state, scheduler_state = \
        _online_resume_fixture(tmp_path)
    (run / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")

    binding = validate_online_resume_run(
        run, expected, parc_only_sources=_valid_provenance(),
        terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
        expected_model_state=model_state,
        component_mask_parameter_keys=mask_keys, max_updates=2_000_000,
    )
    assert binding["saved_step"] == 12
    assert binding["optimizer_state"].startswith("restored_from_verified")
    assert binding["scheduler_state"].startswith("restored_from_verified")
    assert binding["component_mask_state_sha256"]

    saved = torch.load(binding["checkpoint"], map_location="cpu", weights_only=False)
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    optimizer.load_state_dict(saved["optimizer_states"][0])
    scheduler.load_state_dict(saved["lr_schedulers"][0])
    _assert_nested_equal(optimizer_state, optimizer.state_dict())
    _assert_nested_equal(scheduler_state, scheduler.state_dict())
    assert binding["optimizer_state_sha256"] == state_tree_digest([optimizer.state_dict()])
    assert binding["scheduler_state_sha256"] == state_tree_digest([scheduler.state_dict()])
    for key in mask_keys:
        full_key = "backbone_net." + key
        assert torch.equal(saved["state_dict"][full_key], model_state[full_key])


@pytest.mark.parametrize("mismatch", [
    "config", "numerics", "provenance", "dependency", "model_source",
    "target", "above_schedule",
])
def test_online_root_resume_rejects_mismatched_config_numerics_or_dependencies(tmp_path, mismatch):
    run, contract, _, expected, model_state, mask_keys, _, _ = _online_resume_fixture(tmp_path)
    if mismatch == "config":
        expected["config_sha256"] = "e" * 64
    elif mismatch == "numerics":
        expected["runtime_numerics"] = {"effective_matmul_precision": "highest"}
    elif mismatch == "provenance":
        contract["parc_only_sources"]["admitted_source_ids"] = ["different"]
    elif mismatch == "dependency":
        contract["online_contract"]["root_dependency"] = {"run": "unexpected"}
    elif mismatch == "model_source":
        contract["source_lf_sha256"]["motionbricks/data/native_mb_online_conditions.py"] = "f" * 64
    elif mismatch == "target":
        expected["stop_steps"] = 12
    elif mismatch == "above_schedule":
        expected["stop_steps"] = 2_000_001
    (run / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_online_resume_run(
            run, expected, parc_only_sources=_valid_provenance(),
            terrain_conditioning=deepcopy(TERRAIN_PROTOCOL),
            expected_model_state=model_state,
            component_mask_parameter_keys=mask_keys, max_updates=2_000_000,
        )


@pytest.mark.parametrize("extra, message", [
    (["--online-conditioning"], "requires --terrain-conditioning --parc-only --skip-review"),
    (["--terrain-conditioning", "--parc-only", "--online-conditioning"],
     "requires --terrain-conditioning --parc-only --skip-review"),
])
def test_runner_rejects_incomplete_online_opt_in(extra, message):
    with pytest.raises(SystemExit):
        runner.main(["--stage", "root", "--dataset", "unused", "--output", "unused",
                     "--gpu", "0", *extra])


def test_runner_requires_root_dependency_only_for_online_pose():
    with pytest.raises(SystemExit):
        runner.main(["--stage", "pose", "--dataset", "unused", "--output", "unused",
                     "--tokenizer-run", "tokenizer", "--gpu", "0", "--terrain-conditioning",
                     "--parc-only", "--skip-review", "--online-conditioning"])


def test_online_resume_is_root_only_and_generic_resume_options_remain_blocked():
    shared = ["--dataset", "unused", "--output", "unused", "--gpu", "0",
              "--terrain-conditioning", "--parc-only", "--skip-review",
              "--online-conditioning", "--training-profile", "paper",
              "--schedule-steps", "2000000", "--batch-size", "256"]
    with pytest.raises(SystemExit):
        runner.main(["--stage", "pose", "--tokenizer-run", "codec", *shared,
                     "--online-root-run", "root-run", "--online-resume-run", "prior-pose"])
    with pytest.raises(SystemExit):
        runner.main(["--stage", "root", *shared, "--resume-run", "generic-run"])
    with pytest.raises(SystemExit):
        runner.main(["--stage", "root", *shared, "--warmstart-run", "generic-run"])


def test_generic_flat_resume_remains_forbidden_for_terrain_extension():
    with pytest.raises(SystemExit):
        runner.main(["--stage", "root", "--dataset", "unused", "--output", "unused",
                     "--gpu", "0", "--terrain-conditioning", "--skip-review",
                     "--resume-run", "flat-run"])
    with pytest.raises(SystemExit):
        runner.main(["--stage", "root", "--dataset", "unused", "--output", "unused",
                     "--gpu", "0", "--online-root-run", "root-run"])


@pytest.mark.parametrize("initialization", ["--resume-run", "--warmstart-run"])
def test_runner_forbids_resume_and_crossdataset_warmstart_online(initialization):
    with pytest.raises(SystemExit):
        runner.main(["--stage", "root", "--dataset", "unused", "--output", "unused",
                     "--gpu", "0", "--terrain-conditioning", "--parc-only", "--skip-review",
                     "--online-conditioning", initialization, "prior-run"])
