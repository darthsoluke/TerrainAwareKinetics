"""Real-source CPU provider checks; the explicit CLI proof helper runs zero Adam."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from motionbricks.scene_control_v2.curriculum import BatchRequest, Curriculum
from motionbricks.scene_control_v2.provider import (
    POLICY_ID, PROVIDER_SCHEMA, PROJECT_ROOT, _json_hash, center_source_window,
    legacy_source_root_observations, load_bound_world, make_real_engineering_factory,
    prepare_real_engineering_config, resolve_project_path,
    source_root_corridor,
)
from motionbricks.scene_control_v2.quantizer import tensor_state_sha256
from motionbricks.scene_control_v2.training import TrainingBudget


REAL_SPEC = {
    "schema": PROVIDER_SCHEMA, "cpu_threads": 2,
    "artifacts": {"architecture_bundle": "out/a100_live_20261010/a100-terrain.pt",
                  "checkpoint": "out/pfnn_mixed_20261015/final_acceptance_flat_v1/weights.pt",
                  "bundle_sha256": "0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d",
                  "checkpoint_sha256": "0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646"},
    "accepted_path": "out/parc_training_20261009/native_pilot_v1/accepted.json",
    "source_id": "generated_f733c63c38e13f58",
    "motion_path": "out/parc_curated_20261009/reference_soma_v3/three_stair_flights/retarget.npz",
    "scene_path": "out/parc_training_20261009/native_pilot_v1/terrain_geometry/generated_f733c63c38e13f58-30c4d74c84_paired_heightfield.npz",
    "upstream_path": "out/parc_training_20261009/native_pilot_v1/upstream_manifests/generated_f733c63c38e13f58-30c4d74c84_upstream.json",
    "window_starts": {"8": [1], "12": [1], "16": [1]},
}


def real_curriculum():
    return {"horizon_probabilities": [1 / 3] * 3, "phases": [
        {"start_exposure": 0, "source_root_probability": 1., "source_history_probability": 1.,
         "token_probabilities": [1., 0., 0.]},
        {"start_exposure": 3, "source_root_probability": .5, "source_history_probability": .5,
         "token_probabilities": [.34, .33, .33]},
    ]}


@pytest.fixture(scope="module")
def real_source():
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    from motionbricks.motionlib.core.utils.stats import Stats

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    stats = Stats()
    stats.register_from_tensors(torch.zeros(370), torch.ones(370))
    rep = GlobalRootGlobalJoints(30, SOMASkeleton30(), "somaskel30_dual_root_global_joints", stats=stats)
    yield rep, load_bound_world(REAL_SPEC, rep)
    torch.set_num_threads(previous)


def test_paths_are_relative_to_project_not_caller_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = resolve_project_path(REAL_SPEC["motion_path"])
    assert path == (PROJECT_ROOT / REAL_SPEC["motion_path"]).resolve()
    with pytest.raises(FileNotFoundError):
        resolve_project_path("out/scene_control_v2_missing_source.npz")


def test_real_train_source_world_frame_and_continuous_endpoints(real_source):
    rep, (world, joints, scene, row, manifest) = real_source
    assert world.shape == (646, 366) and row["split"] == "train"
    assert scene["heightfield_y_m"].shape == (76, 147)
    with np.load(resolve_project_path(REAL_SPEC["motion_path"]), allow_pickle=False) as archive:
        torch.testing.assert_close(world[:, :3], torch.from_numpy(archive["root_positions"]), atol=1e-5, rtol=0)
    assert manifest["source_frames"] == 646 and manifest["fps"] == 30
    assert manifest["data_kind"] == "engineering" and not manifest["quality_accepted"]
    assert manifest["fk_roundtrip_max_m"] <= 3e-5
    assert manifest["upstream_training_approved"] is False


@pytest.mark.parametrize("horizon", [8, 12, 16])
def test_window_scene_and_ric_height_translate_together(real_source, horizon):
    from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config

    rep, (world, joints, scene, _, _) = real_source
    raw, raw_joints = world[1:4 * horizon + 2], joints[1:4 * horizon + 2]
    observation_config = terrain_observation_config("intent-horizon-range-v1")
    centered, centered_joints, shifted_scene, frame = center_source_window(
        raw, raw_joints, scene, rep, observation_config=observation_config)
    assert centered.shape == (1, 4 * horizon + 1, 366)
    torch.testing.assert_close(centered[0, :, :3] + frame.translation_world_m[0], raw[:, :3], atol=1e-6, rtol=0)
    torch.testing.assert_close(centered_joints[0] + frame.translation_world_m[0], raw_joints, atol=1e-6, rtol=0)
    torch.testing.assert_close(shifted_scene["heightfield_y_m"] + frame.translation_world_m[0, 1],
                               scene["heightfield_y_m"], atol=1e-6, rtol=0)
    ric_y = rep.indices["ric_data"][1::3]
    torch.testing.assert_close(centered[0, :, ric_y] + frame.translation_world_m[0, 1], raw[:, ric_y], atol=1e-6, rtol=0)
    observed, volume = legacy_source_root_observations(centered, shifted_scene, observation_config=observation_config)
    assert observed["points"].shape == (1, 2, 9, 9, 3)
    assert volume["features"].shape == (1, 3, 512, 15)
    assert volume["valid"].any() and volume["distance_semantics"] == "native_heightfield_prism_sdf_m"


def test_corridor_lateral_axis_uses_sideways_path_not_body_heading():
    history = torch.zeros(1, 4, 5)
    history[..., 3] = 1
    sideways = torch.stack((torch.linspace(0, 2, 9), torch.zeros(9)), -1)[None]
    _, right = source_root_corridor(sideways, 8, history)
    torch.testing.assert_close(right, torch.tensor([0., -1.]).expand(1, 8, 2), atol=0, rtol=0)
    _, backward_right = source_root_corridor(-sideways, 8, history)
    torch.testing.assert_close(backward_right, -right, atol=0, rtol=0)


def test_source_read_failure_does_not_create_retarget_or_fallback(real_source, tmp_path):
    rep, _ = real_source
    spec = deepcopy(REAL_SPEC)
    spec["motion_path"] = str(tmp_path / "missing-retarget.npz")
    with pytest.raises(FileNotFoundError):
        load_bound_world(spec, rep)
    assert not Path(spec["motion_path"]).exists()


def execute_real_provider_forward(output):
    """Explicit manual 3-batch CLI execution; not collected by pytest."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _execute_real_provider_forward(output)
    except BaseException as error:
        with (output / "failure.json").open("x", encoding="utf-8") as handle:
            json.dump({"status": "failed", "error": f"{type(error).__name__}: {error}",
                       "actual_adam_updates": 0, "formal_training_exposures": 0,
                       "provider_spec": REAL_SPEC}, handle, indent=2, sort_keys=True)
        raise


def _execute_real_provider_forward(output):
    import motionbricks.scene_control_v2.provider as provider_module
    from scripts.train_scene_control_v2 import main

    budget = TrainingBudget("diagnostic", 3, 1, 3, 0)
    config = prepare_real_engineering_config(REAL_SPEC, budget=asdict(budget), curriculum=real_curriculum())
    with (output / "resolved_config.json").open("x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    captured = {}
    original_factory = provider_module.make_real_engineering_factory

    def factory(value):
        trainer, provider = original_factory(value)
        captured.update(trainer=trainer, provider=provider,
                        model_before=tensor_state_sha256(trainer.state_dict()),
                        codec_before=tensor_state_sha256(provider.codec.state_dict()), batches=[])

        def traced(request):
            batch = provider(request)
            captured["batches"].append((request, batch))
            return batch

        return trainer, traced

    def forbidden(*args, **kwargs):
        raise AssertionError("This real-source provider execution must not construct Adam")

    with patch.object(provider_module, "make_real_engineering_factory", factory), patch.object(torch.optim, "Adam", forbidden):
        receipt = main(["--config", str(output / "resolved_config.json"), "--output", str(output / "forward"),
                        "--mode", "forward", "--factory", "motionbricks.scene_control_v2.provider:make_real_engineering_factory",
                        "--seed", "2338"])
    trainer, provider = captured["trainer"], captured["provider"]
    assert len(captured["batches"]) == 3
    assert {r.horizon_tokens for r, _ in captured["batches"]} == {8, 12, 16}
    assert receipt["actual_adam_updates"] == receipt["diagnostic_exposures"] == receipt["formal_exposures"] == 0
    assert tensor_state_sha256(trainer.state_dict()) == captured["model_before"]
    assert tensor_state_sha256(provider.codec.state_dict()) == captured["codec_before"]
    assert provider.lineage["label_manifest_sha256"] == _json_hash(provider.manifest)
    windows = []
    for request, batch in captured["batches"]:
        assert batch["masks"]["tokens"].all() and not batch["masks"]["motion_loss_frames"][:, :4].any()
        assert not batch["contact_valid"].any() and torch.isnan(batch["contact_targets"]).all()
        assert not any(batch["contacts"][key].any() for key in ("valid", "actual_valid", "anchor_valid"))
        assert batch["recovery"] is None and batch["split"] == ["train"]
        assert batch["source_frame_indices"][0].tolist() == list(range(1, request.horizon_tokens * 4 + 2))
        assert batch["pose_tokens"].shape == (1, request.horizon_tokens, 8)
        windows.append(dict(horizon_tokens=request.horizon_tokens, world_shape=list(batch["world"].shape),
                            token_shape=list(batch["pose_tokens"].shape), ce_includes_token_zero=True,
                            source_frame_start=1, source_endpoint_frame=request.horizon_tokens * 4 + 1,
                            token_sha256=tensor_state_sha256({"tokens": batch["pose_tokens"]}),
                            height_valid=int(batch["observation"]["valid"].sum()),
                            volume_valid=int(batch["volume"]["valid"].sum())))
    unsupported = BatchRequest(8, ("predicted",), ("source",), ("all_mask",), (0,), POLICY_ID)
    with pytest.raises(ValueError, match="no predicted Root"):
        provider(unsupported)
    report = {"schema": "motionbricks.scene_control_v2.real_provider_forward.v1", "device": "cpu",
              "cpu_threads": torch.get_num_threads(), "seed": 2338, "data_kind": "engineering",
              "actual_adam_updates": 0, "diagnostic_training_exposures": 0, "formal_training_exposures": 0,
              "policy_id": POLICY_ID, "quality_accepted": False, "original_split": "train",
              "model_parameters_and_buffers_unchanged": True, "codec_and_ema_buffers_unchanged": True,
              "model_tensor_sha256": captured["model_before"], "codec_tensor_sha256": captured["codec_before"],
              "label_lineage": provider.lineage, "windows": windows,
              "assembly": "actual_final_native_pose_and_frozen_shared_encoder_with_final_adapter_output_projection",
              "source_manifest": provider.manifest}
    with (output / "provider_checks.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report
