"""CPU integration; synthetic tests are not accepted training/quality evidence."""

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "out" / "vendor" / "kimodo"))

from motionbricks.scene_control_v2.contracts import (
    ContactKind, NativeWindowFrame, TorchSceneQuery, load_run_config,
    numpy_sample_to_torch_batch, paired_batch_in_window,
)
from motionbricks.scene_control_v2.data import make_synthetic_fixture, make_synthetic_scene
from motionbricks.scene_control_v2.geometry import MeshField, SceneObject, SceneQuery
from motionbricks.scene_control_v2.losses import relative_contact_losses
from motionbricks.scene_control_v2.quantizer import (
    ExplicitHeadQuantizer, FrozenCodebookIdentity, QuantizerConfig, make_token_cache,
    read_token_cache, tensor_state_sha256,
)
from motionbricks.scene_control_v2.sampling import (
    QueryBudget, QueryGuidance, offline_teacher_guidance, sample_scene_queries,
)
from scripts.preflight_scene_control_v2 import fixture_contact_preflight, run_preflight


def profile(name="engineering_8x10"):
    return ROOT / "configs" / f"scene_control_v2_{name}.json"


@pytest.fixture(scope="module")
def paired_fixture():
    sample = make_synthetic_fixture(frames=8)
    return sample, make_synthetic_scene(sample)


def test_profiles_resolve_without_training_authorization():
    for name, entries in (("engineering_8x10", 10), ("research_128", 128)):
        config = load_run_config(profile(name))
        assert config["codec"]["entries_per_head"] == entries
        assert config["execution"]["optimizer"] is None
        assert config["model"]["global_features"] == 366
        assert config["model"]["dual_features"] == config["model"]["statistics_features"] == 370
        assert config["training"]["current_runner_objective"] == "focused_valid_codec_code_ce_including_latent0"
        assert config["training"]["motion_loss_mask"] == "independent_decoded_motion_prefix_frame_mask"
        assert config["budget"]["legacy_completed_exposures"] == 38400000
        assert config["quality_gates"]["fullskin"]["vertices"] == 18056
        assert all(value == 0 for account in config["budget"]["accounts"].values() for value in account.values())


def test_object_ids_and_numpy_masks_are_preserved(paired_fixture):
    original, _ = paired_fixture
    sample = deepcopy(original)
    sample["scene"]["object_ids"] = np.array([11, 29, 35, 72], np.int32)
    sample["contacts"]["object_id"] = sample["scene"]["object_ids"][sample["contacts"]["object_id"]]
    batch = numpy_sample_to_torch_batch(sample)
    assert batch["source"] == "synthetic"
    assert batch["contacts"]["actual_valid"].dtype == torch.bool
    assert batch["contacts"]["object_table_index"][0, 0].tolist() == [0, 3]
    assert batch["contacts"]["object_id"][0, 0].tolist() == [11, 72]
    assert sample["contacts"]["object_id"][0].tolist() == [11, 72]
    batch["body_positions_world_m"].zero_()
    assert np.any(sample["body_positions_world_m"] != 0)
    assert {kind.name.lower(): int(kind) for kind in ContactKind} == {
        "support": 0, "brace": 1, "grasp": 2, "touch": 3, "slide": 4}


def test_fixture_queries_reach_real_contact_and_distance_losses():
    result = fixture_contact_preflight(load_run_config(profile()))
    assert result["source"] == "synthetic"
    assert result["guidance_role"] == "offline_teacher"
    assert result["query_count"] == 512
    assert result["point_gradient_l2"] > 0
    assert result["contact_loss_terms"]["contact_anchor_m"] > 0
    assert not result["quality_approved"]


def test_actor_rejects_future_labels_and_teacher_geometry(paired_fixture):
    sample, scene = paired_fixture
    oracle = offline_teacher_guidance(sample)
    with pytest.raises(ValueError, match="Actor guidance"):
        sample_scene_queries(scene, sample, guidance=oracle, mode="actor", as_of_s=0)
    with pytest.raises(ValueError, match="actor query time|future teacher"):
        TorchSceneQuery(scene, scene_metadata=sample["scene"], mode="actor", as_of_s=1)
    factual = QueryGuidance("factual_history", 0, sample["body_positions_world_m"],
                            sample["body_valid"], sample["contacts"])
    with pytest.raises(ValueError, match="future ground-truth"):
        sample_scene_queries(scene, sample, guidance=factual, mode="actor", as_of_s=0)


def box_bridge(*, translated=False):
    matrix = np.eye(4, dtype=np.float32)[None]
    if translated:
        angle = .7
        matrix[0, :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                              [-np.sin(angle), 0, np.cos(angle)]]
        matrix[0, :3, 3] = [.3, .2, -.4]
    obj = SceneObject(17, "synthetic_box", MeshField(trimesh.creation.box(extents=[2, 2, 2])),
                      matrix, np.ones(1, bool))
    scene = SceneQuery((obj,), [[-4, -4, -4], [4, 4, 4]])
    metadata = {"object_ids": np.array([17]), "object_to_world": matrix[:, None],
                "object_valid": np.ones((1, 1), bool),
                "object_track_provenance": [{"role": "known_schedule", "available_at_s": 0,
                                              "source": "synthetic_explicit_schedule"}]}
    return TorchSceneQuery(scene, scene_metadata=metadata, mode="actor", as_of_s=0), matrix[0]


@pytest.mark.parametrize("translated", [False, True])
def test_mesh_gradient_matches_finite_difference_at_face_edge_corner(translated):
    bridge, transform = box_bridge(translated=translated)
    local = np.array([[1.4, .2, .1], [1.4, 1.3, .2], [1.4, 1.3, 1.2], [.7, .1, .2]], np.float32)
    points = torch.tensor(local @ transform[:3, :3].T + transform[:3, 3], requires_grad=True)
    times = torch.zeros(len(points), dtype=torch.long)
    distance = bridge(points, times, timestamps_s=np.array([0.]))["distance_m"]
    gradient, = torch.autograd.grad(distance.sum(), points)
    epsilon = 1e-3
    finite_difference = torch.zeros_like(points)
    for axis in range(3):
        delta = torch.zeros_like(points)
        delta[:, axis] = epsilon
        plus = bridge(points.detach() + delta, times, timestamps_s=np.array([0.]))["distance_m"]
        minus = bridge(points.detach() - delta, times, timestamps_s=np.array([0.]))["distance_m"]
        finite_difference[:, axis] = (plus - minus) / (2 * epsilon)
    torch.testing.assert_close(gradient, finite_difference, atol=3e-4, rtol=3e-4)
    assert (gradient[2].abs() > .1).sum() == 3


@pytest.mark.skipif(os.environ.get("SCENE_CONTROL_V2_CUDA_TESTS") != "1",
                    reason="Requires a separate explicit GPU allocation")
@pytest.mark.parametrize("assigned", [False, True])
def test_optional_cuda_query_preserves_device_and_matches_cpu_vjp(assigned):
    bridge, _ = box_bridge(translated=True)
    cpu_points = torch.tensor([[1.7, 1.5, .8], [.4, .2, -.3]], requires_grad=True)
    cpu_times = torch.zeros(2, dtype=torch.long)
    ids = torch.full((2,), 17, dtype=torch.long) if assigned else None
    expected = bridge(cpu_points, cpu_times, timestamps_s=np.array([0.]), assigned_object_id=ids)
    expected_gradient, = torch.autograd.grad(expected["distance_m"].sum(), cpu_points)
    points = cpu_points.detach().to("cuda:0").requires_grad_()
    times = cpu_times.to(points.device)
    actual = bridge(points, times, timestamps_s=np.array([0.]),
                    assigned_object_id=ids.to(points.device) if assigned else None)
    gradient, = torch.autograd.grad(actual["distance_m"].sum(), points)
    for key in ("distance_m", "valid", "distance_gradient_valid", "object_id"):
        assert actual[key].device == points.device
        torch.testing.assert_close(actual[key].cpu(), expected[key], atol=1e-6, rtol=0)
    torch.testing.assert_close(gradient.cpu(), expected_gradient, atol=1e-6, rtol=0)


def test_contact_and_body_sdf_backpropagate_to_predicted_points(paired_fixture):
    sample, scene = paired_fixture
    batch = numpy_sample_to_torch_batch(sample)
    predicted = (batch["body_positions_world_m"] + torch.tensor([.01, .02, -.01])).requires_grad_()
    times = torch.arange(8)[None, :, None].expand(predicted.shape[:-1])
    bridge = TorchSceneQuery(scene, scene_metadata=sample["scene"], mode="offline", as_of_s=1)
    indices = batch["contacts"]["body_part_id"][0]
    contact = bridge(predicted.index_select(2, indices), times.index_select(2, indices),
        timestamps_s=sample["timestamps_s"], assigned_object_id=batch["contacts"]["object_id"])
    gradient, = torch.autograd.grad(contact["distance_m"].abs().sum(), predicted, retain_graph=True)
    assert gradient.abs().sum() > 0
    body = bridge(predicted, times, timestamps_s=sample["timestamps_s"])
    body_gradient, = torch.autograd.grad(body["distance_m"].sum(), predicted)
    assert torch.isfinite(body_gradient).all() and body_gradient.abs().sum() > 0


def test_unknown_scene_component_and_assigned_object_have_distinct_validity(paired_fixture):
    original, original_scene = paired_fixture
    sample = deepcopy(original)
    sample["scene"]["object_valid"][-1, -1] = False
    objects = tuple(SceneObject(obj.object_id, obj.name, obj.field, obj.object_to_world,
        sample["scene"]["object_valid"][:, i]) for i, obj in enumerate(original_scene.objects))
    query = SceneQuery(objects, sample["scene"]["valid_bounds_world_m"])
    bridge = TorchSceneQuery(query, scene_metadata=sample["scene"], mode="offline", as_of_s=1)
    point = torch.tensor([[0., .5, 0.]], requires_grad=True)
    time = torch.tensor([7])
    whole = bridge(point, time, timestamps_s=sample["timestamps_s"])
    assert not whole["valid"].any() and not whole["distance_gradient_valid"].any()
    assert whole["valid_semantics"] == "all_declared_components_known"
    gradient, = torch.autograd.grad(whole["distance_m"].sum(), point)
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assigned = bridge(point, time, timestamps_s=sample["timestamps_s"], assigned_object_id=torch.tensor([0]))
    assert assigned["valid"].all()
    assert assigned["valid_semantics"] == "assigned_object_component_known"


def test_mixed_object_track_provenance_controls_actor_admission(paired_fixture):
    sample, scene = paired_fixture
    metadata = deepcopy(sample["scene"])
    metadata["transform_role"] = "known_schedule"  # Descriptive field cannot authorize a teacher track.
    with pytest.raises(ValueError, match="future teacher"):
        TorchSceneQuery(scene, scene_metadata=metadata, mode="actor", as_of_s=1)
    metadata["object_track_provenance"][-1] = {
        "role": "model_prediction", "available_at_s": 0, "source": "synthetic_test_prediction"}
    actor = TorchSceneQuery(scene, scene_metadata=metadata, mode="actor", as_of_s=0)
    result = actor(torch.tensor([[0., .4, 0.]]), torch.tensor([7]), timestamps_s=sample["timestamps_s"])
    assert result["valid"].all()
    metadata["object_track_provenance"][-1]["role"] = "static"
    with pytest.raises(ValueError, match="contains motion"):
        TorchSceneQuery(scene, scene_metadata=metadata, mode="actor", as_of_s=0)
    metadata["object_track_provenance"][-1]["role"] = "factual_history"
    factual = TorchSceneQuery(scene, scene_metadata=metadata, mode="actor", as_of_s=0)
    with pytest.raises(ValueError, match="Future scene transforms"):
        factual(torch.tensor([[0., .4, 0.]]), torch.tensor([7]), timestamps_s=sample["timestamps_s"])


def test_sparse_object_ids_and_swapped_table_preserve_contact_losses(paired_fixture):
    original, original_scene = paired_fixture
    sample = deepcopy(original)
    scene = sample["scene"]
    scene["object_ids"] = np.array([7, 42], np.int32)
    scene["object_names"] = ("floor", "moving_brace")
    scene["object_to_world"] = scene["object_to_world"][:, [0, 3]]
    scene["object_valid"] = scene["object_valid"][:, [0, 3]]
    scene["object_track_provenance"] = [scene["object_track_provenance"][i] for i in [0, 3]]
    sample["contacts"]["object_id"] = np.tile(np.array([7, 42], np.int32), (8, 1))
    objects = tuple(SceneObject(new_id, obj.name, obj.field, obj.object_to_world, obj.valid)
                    for new_id, obj in zip([7, 42], [original_scene.objects[0], original_scene.objects[3]]))
    query = SceneQuery(objects, scene["valid_bounds_world_m"])
    batch = numpy_sample_to_torch_batch(sample)
    indices = batch["contacts"]["body_part_id"][0]
    points = batch["body_positions_world_m"].clone()
    points[..., 0] += torch.linspace(.01, .05, 8)[None, :, None]
    points[..., 1] += .02
    bridge = TorchSceneQuery(query, scene_metadata=scene, mode="offline", as_of_s=1)
    times = torch.arange(8)[None, :, None].expand(1, 8, 2)
    contact_query = bridge(points.index_select(2, indices), times, timestamps_s=sample["timestamps_s"],
                           assigned_object_id=batch["contacts"]["object_id"])

    def losses(value):
        return relative_contact_losses(points, contact_query["distance_m"], value["contacts"], value["scene"],
            indices, contact_query["valid"], torch.ones((1, 7, 2), dtype=torch.bool),
            torch.ones((1, 8), dtype=torch.bool), fps=30)

    expected = losses(batch)
    scene["object_ids"] = scene["object_ids"][[1, 0]]
    scene["object_names"] = tuple(reversed(scene["object_names"]))
    scene["object_to_world"] = scene["object_to_world"][:, [1, 0]]
    scene["object_valid"] = scene["object_valid"][:, [1, 0]]
    scene["object_track_provenance"] = list(reversed(scene["object_track_provenance"]))
    swapped = losses(numpy_sample_to_torch_batch(sample))
    for key in expected:
        torch.testing.assert_close(swapped[key], expected[key], atol=1e-7, rtol=0)
    assert expected["contact_gap_m"] > 0 and expected["relative_slip_mps"] > 0


def test_world_to_window_translates_scene_and_targets_together(paired_fixture):
    sample, scene = paired_fixture
    queries = sample_scene_queries(scene, sample, guidance=offline_teacher_guidance(sample),
        mode="offline", as_of_s=1, budget=QueryBudget(8, 8, 8))
    batch = numpy_sample_to_torch_batch(sample, queries=queries)
    frame = NativeWindowFrame(torch.tensor([[2., .6, -3.]]), torch.tensor([[1.2]]))
    window = paired_batch_in_window(batch, frame)
    torch.testing.assert_close(frame.points_to_world(window["body_positions_world_m"]), batch["body_positions_world_m"])
    before = batch["scene"]["object_to_world"][..., :3, 3]
    after = window["scene"]["object_to_world"][..., :3, 3]
    torch.testing.assert_close(frame.points_to_world(after), before)
    torch.testing.assert_close(window["contacts"]["anchor_local_m"], batch["contacts"]["anchor_local_m"])
    assert window["window_frame"]["geometry_rotation"] == "heading_retained_no_extra_yaw"
    torch.testing.assert_close(frame.points_to_world(window["queries"]["closest_point_world_m"]),
                               batch["queries"]["closest_point_world_m"])
    centered_scene = SceneQuery(tuple(SceneObject(obj.object_id, obj.name, obj.field,
        window["scene"]["object_to_world"][0, :, i].numpy(), obj.valid)
        for i, obj in enumerate(scene.objects)), window["scene"]["valid_bounds_world_m"][0].numpy())
    actual = centered_scene.query_world_torch(window["queries"]["world_points_m"][0],
                                              window["queries"]["time_index"][0])
    torch.testing.assert_close(actual["distance_m"], batch["queries"]["distance_m"][0], atol=2e-6, rtol=0)


def test_native_window_center_shifts_root_and_ric_height():
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30

    rep = GlobalRootGlobalJoints(30, SOMASkeleton30(), "somaskel30_dual_root_global_joints")
    # Synthetic feature fixture exercises only the actual representation's translation channels.
    world = torch.zeros(1, 4, 366)
    world[..., :3] = torch.tensor([2., 1.8, -3.])
    world[..., rep.indices["ric_data"][1::3]] = 2.1
    original = world.clone()
    frame = NativeWindowFrame(torch.tensor([[2., .6, -3.]]), torch.tensor([[0.]]))
    centered = frame.center_global_motion(world, rep)
    torch.testing.assert_close(centered[..., 1], torch.full((1, 4), 1.2))
    torch.testing.assert_close(centered[..., rep.indices["ric_data"][1::3]], torch.full((1, 4, 29), 1.5))
    torch.testing.assert_close(world, original)


@pytest.mark.parametrize("entries", [10, 128, 256])
def test_explicit_heads_do_not_pack_overall_indices(entries):
    quantizer = ExplicitHeadQuantizer(QuantizerConfig(entries_per_head=entries)).eval()
    tokens = torch.full((1, 8, 8), entries - 1, dtype=torch.long)
    decoded = quantizer.dequantize(tokens)
    assert decoded.shape == (1, 8, 256)
    output = quantizer(torch.randn(1, 256, 8))
    assert output.indices.shape == (1, 8, 8)


def test_engineering_codec_cache_lineage_cannot_authorize_formal_training(monkeypatch):
    from motionbricks.scene_control_v2.training import (
        CONFIG_SCHEMA, attach_final_codec_labels, make_training_label_lineage,
        preflight_training_config, validate_label_lineage,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Lineage admission must not construct Adam")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    cfg = QuantizerConfig(entries_per_head=10)
    quantizer = ExplicitHeadQuantizer(cfg).eval()
    encoder = torch.nn.Linear(256, 256)
    normalization_sha256 = tensor_state_sha256({"mean": torch.zeros(370), "std": torch.ones(370)})
    identity = FrozenCodebookIdentity(quantizer_config=cfg, codebook_sha256=quantizer.freeze(),
        encoder_sha256=tensor_state_sha256(encoder.state_dict()),
        representation_id="engineering_fixture_only", normalization_sha256=normalization_sha256)
    indices = torch.zeros(1, 8, 8, dtype=torch.long)
    cache = make_token_cache(indices, identity)
    # Engineering identities are genuine tensor hashes, never real-data receipts.
    lineage = make_training_label_lineage(identity, data_kind="engineering",
        codec_sha256=tensor_state_sha256(quantizer.state_dict()),
        scene_encoder_sha256=tensor_state_sha256({"engineering_scene": torch.ones(2, 2)}),
        normalization_sha256=normalization_sha256,
        label_manifest_sha256=tensor_state_sha256({"engineering_indices": indices}))
    labeled = attach_final_codec_labels({"data_kind": "engineering"}, cache, identity, lineage=lineage)
    assert labeled["label_lineage"]["data_kind"] == "engineering"
    torch.testing.assert_close(labeled["pose_tokens"], indices)
    missing = dict(lineage)
    del missing["data_kind"]
    with pytest.raises(KeyError, match="data_kind"):
        validate_label_lineage(missing)
    config = {"schema": CONFIG_SCHEMA, "label_lineage": lineage, "policy_id": "engineering-only",
        "curriculum": {"horizon_probabilities": [.34, .33, .33], "phases": [{"start_exposure": 0,
            "source_root_probability": .5, "source_history_probability": .5,
            "token_probabilities": [.3, .4, .3]}]},
        "budget": {"kind": "formal", "max_batches": 1, "batch_size": 1,
            "max_exposures": 1, "max_adam_updates": 1, "approval_id": "test-rejection-only",
            "learning_rate": 1e-4, "clip_norm": .5}}
    with pytest.raises(ValueError, match="Engineering label identities"):
        preflight_training_config(config, mode="train")


def test_normalization_identity_change_rejects_existing_cache():
    cfg = QuantizerConfig(entries_per_head=10)
    quantizer = ExplicitHeadQuantizer(cfg).eval()
    original_norm = tensor_state_sha256({"mean": torch.zeros(370), "std": torch.ones(370)})
    changed_norm = tensor_state_sha256({"mean": torch.ones(370), "std": torch.ones(370)})
    identity = FrozenCodebookIdentity(quantizer_config=cfg, codebook_sha256=quantizer.freeze(),
        encoder_sha256=tensor_state_sha256({"encoder_fixture": torch.zeros(256, 268)}),
        representation_id="unchanged_engineering_representation", normalization_sha256=original_norm)
    labels = torch.zeros((1, 8, 8), dtype=torch.long)
    cache = make_token_cache(labels, identity)
    assert cache["normalization_sha256"] == cache["vocabulary"]["normalization_sha256"] == original_norm
    torch.testing.assert_close(read_token_cache(cache, identity), labels)
    changed = replace(identity, normalization_sha256=changed_norm)
    assert changed.identity_id != identity.identity_id
    with pytest.raises(ValueError, match="different frozen normalization"):
        read_token_cache(cache, changed)


def test_real_provider_cli_config_matches_executed_zero_adam_proof():
    from motionbricks.scene_control_v2.curriculum import Curriculum
    from motionbricks.scene_control_v2.provider import POLICY_ID, make_real_engineering_factory, resolve_project_path
    from motionbricks.scene_control_v2.training import preflight_training_config

    config = json.loads((ROOT / "configs" / "scene_control_v2_real_provider_cpu.json").read_text())
    evidence = ROOT / "out" / "scene_control_v2_20261018" / "real_provider_cpu_v1"
    executed = json.loads((evidence / "resolved_config.json").read_text())
    proof = json.loads((evidence / "provider_checks.json").read_text())
    assert config == executed
    assert callable(make_real_engineering_factory)
    summary = preflight_training_config(config)
    assert summary["actual_adam_updates"] == summary["formal_exposures"] == 0
    assert config["policy_id"] == POLICY_ID
    assert config["label_lineage"] == proof["label_lineage"]
    assert config["label_lineage"]["data_kind"] == "engineering"
    spec = config["provider"]
    assert spec["cpu_threads"] == 2
    for path in [spec[key] for key in ("accepted_path", "motion_path", "scene_path", "upstream_path")] + [
            spec["artifacts"][key] for key in ("architecture_bundle", "checkpoint")]:
        assert not Path(path).is_absolute() and ".." not in Path(path).parts
        assert resolve_project_path(path) == (ROOT / path).resolve()
    curriculum = Curriculum.from_dict(config["curriculum"], policy_id=POLICY_ID)
    requests = [curriculum.draw(1, exposure=i, seed=2338 + i) for i in range(3)]
    assert [request.horizon_tokens for request in requests] == [12, 16, 8]
    assert all(request.root_source == request.history_source == ("source",) for request in requests)
    assert all(request.token_mode == ("all_mask",) for request in requests)
    assert all(window["world_shape"] == [1, 4 * window["horizon_tokens"] + 1, 366]
               and window["ce_includes_token_zero"] for window in proof["windows"])
    assert proof["source_manifest"]["root_roundtrip_max_abs_m"] <= 1e-5
    assert proof["source_manifest"]["fk_roundtrip_max_m"] <= 3e-5
    assert proof["source_manifest"]["volume_distance_semantics"] == "native_heightfield_prism_sdf_m"
    assert proof["source_manifest"]["height_observation_config"]["forward_spacing_m"] == .4
    assert proof["source_manifest"]["corridor_lateral_axis"] == "path_tangent_native_curve_rule"


def test_actual_cli_preflight_uses_checkpoint_and_never_constructs_adam(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No optimizer is authorized")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    report = run_preflight(profile())
    assert report["native"]["indices_shape"] == [1, 1, 8]
    assert report["native"]["seed_fk_world_max_error_m"] < 2e-5
    assert report["native"]["quantizer_state_unchanged"]
    assert report["native"]["source"] == "existing_bundle_initial_history_real_seed"
    assert report["actual_adam_updates"] == report["formal_training_exposures"] == 0
    assert not report["native"]["real_paired_scene_loaded"]
    assert not report["quality_approved"]
    json.dumps(report)
