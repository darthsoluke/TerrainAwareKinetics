import copy

import numpy as np
import pytest

from motionbricks.scene_control_v2.data import (
    PAIRED_SCHEMA, adapt_public_window, assert_family_disjoint, empty_contacts,
    inspect_parc15_sidecar, inspect_public_metadata, make_synthetic_fixture,
    make_synthetic_scene, validate_paired_sample,
)
from motionbricks.scene_control_v2.sampling import (
    QueryBudget, QueryGuidance, offline_teacher_guidance, sample_scene_queries,
)


@pytest.fixture(scope="module")
def paired():
    sample = make_synthetic_fixture()
    return sample, make_synthetic_scene(sample)


def test_fixture_contract_is_explicitly_synthetic(paired):
    sample, _ = paired
    validate_paired_sample(sample)
    assert sample["schema"] == PAIRED_SCHEMA
    assert sample["source"] == "synthetic"
    assert sample["provenance"]["purpose"] == "offline_oracle_diagnostic"
    assert not np.array_equal(sample["scene"]["object_to_world"][0, 3],
                              sample["scene"]["object_to_world"][-1, 3])


def test_fixed_budget_covers_every_body_time_and_contacts(paired):
    sample, scene = paired
    out = sample_scene_queries(scene, sample, guidance=offline_teacher_guidance(sample),
                               mode="offline", as_of_s=1, seed=11)
    assert len(out["valid"]) == 512
    assert np.bincount(out["region_id"]).tolist() == [128, 256, 128]
    body = out["region_id"] == 1
    assert len(set(zip(out["time_index"][body], out["body_part_id"][body]))) == 8 * 13
    contacts = out["region_id"] == 2
    assert set(out["target_object_id"][contacts]) == {0, 3}
    assert out["sample_valid"].all()
    repeated = sample_scene_queries(scene, sample, guidance=offline_teacher_guidance(sample),
                                    mode="offline", as_of_s=1, seed=11)
    np.testing.assert_array_equal(out["world_points_m"], repeated["world_points_m"])


def test_missing_guidance_does_not_borrow_labels_or_budget(paired):
    sample, scene = paired
    guidance = QueryGuidance("offline_teacher", 1, np.zeros_like(sample["body_positions_world_m"]),
                             np.zeros_like(sample["body_valid"]), empty_contacts(8))
    out = sample_scene_queries(scene, sample, guidance=guidance, mode="offline", as_of_s=1)
    assert len(out["valid"]) == 512
    assert out["sample_valid"].sum() == 128
    assert not out["valid"][128:].any()
    assert np.all(out["object_id"][128:] == -1)


def test_actor_rejects_oracle_and_unavailable_guidance(paired):
    sample, scene = paired
    with pytest.raises(ValueError, match="Actor guidance"):
        sample_scene_queries(scene, sample, guidance=offline_teacher_guidance(sample), mode="actor", as_of_s=0)
    prediction = QueryGuidance("model_prediction", 1, sample["body_positions_world_m"],
                               sample["body_valid"], sample["contacts"])
    with pytest.raises(ValueError, match="available"):
        sample_scene_queries(scene, sample, guidance=prediction, mode="actor", as_of_s=0)


def test_factual_future_body_and_contact_leakage_rejected(paired):
    sample, scene = paired
    guidance = QueryGuidance("factual_history", 0, sample["body_positions_world_m"],
                             sample["body_valid"], sample["contacts"])
    with pytest.raises(ValueError, match="future ground-truth"):
        sample_scene_queries(scene, sample, guidance=guidance, mode="actor", as_of_s=0)
    body_valid = np.zeros_like(sample["body_valid"])
    guidance = QueryGuidance("factual_history", 0, sample["body_positions_world_m"],
                             body_valid, sample["contacts"])
    with pytest.raises(ValueError, match="future ground-truth"):
        sample_scene_queries(scene, sample, guidance=guidance, mode="actor", as_of_s=0)


def test_actor_prediction_uses_own_positions_and_known_schedule(paired):
    source, _ = paired
    sample = copy.deepcopy(source)
    sample["scene"]["transform_role"] = "known_schedule"
    sample["scene"]["object_track_provenance"][3] = {
        "role": "known_schedule", "available_at_s": 0, "source": "synthetic_test_declared_schedule"}
    scene = make_synthetic_scene(sample)
    predicted = np.full_like(sample["body_positions_world_m"], 0.2)
    guidance = QueryGuidance("model_prediction", 0, predicted,
                             sample["body_valid"], empty_contacts(8))
    out = sample_scene_queries(scene, sample, guidance=guidance, mode="actor", as_of_s=0,
                               budget=QueryBudget(0, 208, 0, body_radius_m=0))
    np.testing.assert_allclose(out["world_points_m"], 0.2)
    assert (out["time_s"] > 0).any()
    sample["scene"]["transform_role"] = "factual_history"
    sample["scene"]["object_track_provenance"][3]["role"] = "factual_history"
    with pytest.raises(ValueError, match="Future queries"):
        sample_scene_queries(scene, sample, guidance=guidance, mode="actor", as_of_s=0)


def test_scene_binding_and_teacher_transform_rejected(paired):
    sample, scene = paired
    guidance = QueryGuidance("model_prediction", 0, sample["body_positions_world_m"],
                             sample["body_valid"], empty_contacts(8))
    with pytest.raises(ValueError, match="offline teacher tracks|available at actor"):
        sample_scene_queries(scene, sample, guidance=guidance, mode="actor", as_of_s=0)
    different = copy.deepcopy(sample)
    different["scene"]["object_to_world"][0, 0, 0, 3] += 0.1
    with pytest.raises(ValueError, match="binding|transforms"):
        sample_scene_queries(scene, different, guidance=offline_teacher_guidance(sample),
                             mode="offline", as_of_s=1)


@pytest.mark.parametrize("leakage", ["family", "recording", "parent", "scene", "cross_source"])
def test_family_split_leakage(leakage, paired):
    first = copy.deepcopy(paired[0])
    second = copy.deepcopy(first)
    second.update(split="test", family="other", recording_id="other")
    second["provenance"]["scene_family"] = "other_scene"
    if leakage == "family":
        second["family"] = first["family"]
    elif leakage == "recording":
        second["recording_id"] = first["recording_id"]
    elif leakage == "parent":
        second["provenance"]["parent_recording_ids"] = [first["recording_id"]]
    elif leakage == "scene":
        second["provenance"]["scene_family"] = first["provenance"]["scene_family"]
    else:
        second["source"] = "OTHER"
        first["provenance"]["lineage_ids"] = ["shared-global-recording"]
        second["provenance"]["lineage_ids"] = ["shared-global-recording"]
    with pytest.raises(ValueError, match="Split leakage"):
        assert_family_disjoint([first, second])


def test_disjoint_and_same_split_variants_allowed(paired):
    first = copy.deepcopy(paired[0])
    second = copy.deepcopy(first)
    second["recording_id"] += "_mirror"
    assert_family_disjoint([first, second])
    second.update(split="val", family="other", recording_id="independent")
    second["provenance"]["scene_family"] = "unseen"
    assert_family_disjoint([first, second])


@pytest.mark.parametrize("dataset", ["LINGO", "TRUMANS", "CIRCLE"])
def test_public_missing_data_is_explicit(dataset, tmp_path):
    report = inspect_public_metadata(dataset, tmp_path)
    assert report["entity_samples_loaded"] == 0
    assert not report["paired_training_ready"]
    assert report["missing"]
    assert "license_evidence" in report["unresolved"]


def test_lingo_small_metadata_headers(tmp_path):
    np.save(tmp_path / "human_pose.npy", np.zeros((3, 63), np.float32))
    np.save(tmp_path / "human_joints_aligned.npy", np.zeros((3, 28, 3), np.float32))
    report = inspect_public_metadata("LINGO", tmp_path)
    assert report["fields"]["human_pose.npy"]["shape"] == [3, 63]
    assert report["fields"]["human_joints_aligned.npy"]["shape_valid"]
    assert "left_hand_inter_frame.npy" in report["missing"]


def test_trumans_object_list_metadata_does_not_unpickle(tmp_path):
    np.save(tmp_path / "object_list.npy", np.array(["chair", "table"], dtype=object))
    report = inspect_public_metadata("TRUMANS", tmp_path)
    field = report["fields"]["object_list.npy"]
    assert field["shape"] == [2] and field["dtype"] == "object"
    assert field["shape_valid"] and not field["payload_loaded"]


def test_factual_history_never_samples_future_times(paired):
    sample = copy.deepcopy(paired[0])
    sample["scene"]["object_track_provenance"][3] = {
        "role": "factual_history", "available_at_s": 0, "source": "synthetic_observed_t0"}
    scene = make_synthetic_scene(sample)
    body_valid = np.zeros_like(sample["body_valid"])
    body_valid[0] = True
    contacts = copy.deepcopy(sample["contacts"])
    contacts["valid"][1:] = contacts["anchor_valid"][1:] = False
    guidance = QueryGuidance("factual_history", 0, sample["body_positions_world_m"], body_valid, contacts)
    out = sample_scene_queries(scene, sample, guidance=guidance, mode="actor", as_of_s=0)
    assert (out["time_s"] == 0).all()


def test_parc15_source_labels_not_certified(tmp_path):
    path = tmp_path / "parc15.npz"
    np.savez(path, source_contacts=np.zeros((3, 15)), source_contact_names=np.array([f"p{i}" for i in range(15)]),
             source_frame_indices=np.arange(3), fps=np.array(30))
    report = inspect_parc15_sidecar(path)
    assert report["channels"] == 15 and not report["certified_contact"]
    assert not report["anchor_valid"]


def test_adapter_preserves_native_fields_and_requires_bindings(paired):
    sample = paired[0]
    kwargs = {k: sample[k] for k in (
        "body_positions_world_m", "body_valid", "body_part_names", "timestamps_s", "fps",
        "skeleton", "recording_id", "family", "split", "scene_id", "scene", "provenance", "contacts")}
    with pytest.raises(ValueError, match="native fields"):
        adapt_public_window("LINGO", native_fields={}, **kwargs)
    fields = {"human_pose": np.zeros((8, 63)), "human_orient": np.zeros((8, 3)),
              "transl_aligned": np.zeros((8, 3)), "left_hand_inter_frame": -1}
    out = adapt_public_window("LINGO", native_fields=fields, **kwargs)
    assert out["native_fields"]["left_hand_inter_frame"] == -1
    assert out["source"] == "LINGO"


def test_invalid_contact_kind_rejected(paired):
    sample = copy.deepcopy(paired[0])
    sample["contacts"]["kind"][0, 0] = 5
    with pytest.raises(ValueError, match="canonical"):
        validate_paired_sample(sample)


def test_static_track_cannot_hide_future_object_truth(paired):
    original, _ = paired
    sample = copy.deepcopy(original)
    sample["scene"]["object_track_provenance"][3] = {
        "role": "static", "available_at_s": 0, "source": "incorrect_test_static_claim"}
    scene = make_synthetic_scene(sample)
    guidance = QueryGuidance("model_prediction", 0, sample["body_positions_world_m"],
                             sample["body_valid"], empty_contacts(8))
    with pytest.raises(ValueError, match="Static object track contains motion"):
        sample_scene_queries(scene, sample, guidance=guidance, mode="actor", as_of_s=0)


def test_body_and_contact_channel_quotas_are_balanced(paired):
    sample, scene = paired
    out = sample_scene_queries(scene, sample, guidance=offline_teacher_guidance(sample),
                               mode="offline", as_of_s=1)
    mask = out["region_id"] == 1
    counts = np.bincount(out["body_part_id"][mask])
    assert counts.max() - counts.min() <= 1
    mask = out["region_id"] == 2
    assert np.bincount(out["contact_index"][mask]).tolist() == [64, 64]
