import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "out" / "vendor" / "kimodo"))

from scripts import relabel_native_mb_stair_plant as relabel


def analytic_motion():
    skeleton = SimpleNamespace(
        left_foot_joint_idx=[1, 3], right_foot_joint_idx=[2, 4],
        foot_joint_idx=[4, 1, 2, 3],
    )
    positions = torch.zeros(5, 5, 3)
    positions[:, [1, 3], 1] = 0.55
    positions[:, [2, 4], 1] = 1.05
    contacts = np.ones((5, 4), dtype=np.float32)
    ids = {"left": np.full(5, 10), "right": np.full(5, 20)}
    support = {"patches": [
        {"id": 10, "height_m": 0.5, "status": "weak_support"},
        {"id": 20, "height_m": 1.0, "status": "weak_support"},
    ]}
    return skeleton, positions, contacts, ids, support


def test_elevated_stationary_feet_use_their_own_support_heights():
    skeleton, positions, contacts, ids, support = analytic_motion()
    before = positions.clone()
    labels, report = relabel.stationary_plant(positions, contacts, ids, support, skeleton)
    assert labels.all()
    assert torch.equal(positions, before)
    assert report["positive_joint_frames_before"] == report["positive_joint_frames_after"] == 20
    assert report["support_heights"]["left"]["min_m"] == 0.5
    assert report["support_heights"]["right"]["max_m"] == 1.0


def test_native_joint_id_mapping_intersection_and_rolling_joint_speed():
    skeleton, positions, contacts, ids, support = analytic_motion()
    # Material-point evidence may remain positive while the right toe joint rolls.
    positions[:, 4, 0] = torch.arange(5) * 0.01  # 0.30 m/s at 30 Hz
    contacts[:, 1] = 0.5  # Detector eligibility cannot create source evidence.
    labels, report = relabel.stationary_plant(positions, contacts, ids, support, skeleton)
    np.testing.assert_array_equal(labels, np.tile([False, False, True, True], (5, 1)))
    assert report["source_labeled_joint_speed_exceed_0_15_mps_count"] == 5
    assert report["source_labeled_joint_speed_exceed_0_15_mps_fraction"] == pytest.approx(1 / 3)
    assert report["positive_joint_frames_before"] == 15
    assert report["positive_joint_frames_after"] == 10


def test_stationary_material_pivot_does_not_plant_a_fast_rolling_joint():
    skeleton, positions, contacts, ids, support = analytic_motion()
    pivot = torch.tensor([0.0, 1.0, 0.0]).expand(5, 3)
    angles = torch.arange(5) * 0.2
    positions[:, 4] = pivot + torch.stack(
        [0.08 * torch.sin(angles), 0.08 * torch.cos(angles), torch.zeros(5)], dim=-1
    )
    assert torch.equal(torch.diff(pivot, dim=0), torch.zeros(4, 3))
    labels, report = relabel.stationary_plant(positions, contacts, ids, support, skeleton)
    assert not labels[:, 0].any()  # Native channel zero is the moving right toe.
    assert labels[:, 1:].all()
    assert report["source_labeled_joint_speed_exceed_0_15_mps_count"] == 5


def test_labels_are_invariant_to_translation_and_yaw():
    skeleton, positions, contacts, ids, support = analytic_motion()
    positions[:, 1, 2] = torch.arange(5) * 0.002
    positions[:, 4, 0] = torch.arange(5) * 0.01
    expected, _ = relabel.stationary_plant(positions, contacts, ids, support, skeleton)
    angle = torch.tensor(0.73)
    c, s = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    shifted = positions @ rotation.T + torch.tensor([3.0, 2.0, -4.0])
    shifted_support = copy.deepcopy(support)
    for patch in shifted_support["patches"]:
        patch["height_m"] += 2.0
    actual, report = relabel.stationary_plant(
        shifted, contacts, ids, shifted_support, skeleton
    )
    np.testing.assert_array_equal(actual, expected)
    assert report["support_heights"]["left"]["min_m"] == 2.5


@pytest.mark.parametrize("unknown", ["negative_id", "missing_id", "rejected", "nan_height"])
def test_unknown_same_foot_support_never_produces_positive(unknown):
    skeleton, positions, contacts, ids, support = analytic_motion()
    if unknown == "negative_id":
        ids["left"][:] = -1
    elif unknown == "missing_id":
        ids["left"][:] = 999
    elif unknown == "rejected":
        support["patches"][0]["status"] = "rejected"
    else:
        support["patches"][0]["height_m"] = float("nan")
    labels, report = relabel.stationary_plant(positions, contacts, ids, support, skeleton)
    np.testing.assert_array_equal(labels, np.tile([True, False, True, False], (5, 1)))
    assert report["support_heights"]["left"]["unknown_frames"] == 5
    assert report["support_heights"]["left"]["min_m"] is None


def test_last_native_derivative_repeats_and_support_changes_are_not_velocity():
    skeleton, positions, contacts, ids, support = analytic_motion()
    support["patches"].append({"id": 30, "height_m": 0.53, "status": "weak_support"})
    ids["left"][2:] = 30
    labels, _ = relabel.stationary_plant(positions, contacts, ids, support, skeleton)
    assert labels.all()  # Support-height reassignment must not affect joint velocity.
    positions[-1, 1, 0] = 0.01
    labels, report = relabel.stationary_plant(positions, contacts, ids, support, skeleton)
    assert labels[:, 1].tolist() == [True, True, True, False, False]
    assert report["source_labeled_joint_speed_exceed_0_15_mps_count"] == 2


def test_original_height_threshold_and_joint_id_validation():
    skeleton, positions, contacts, ids, support = analytic_motion()
    positions[:, 3, 1] = 0.62
    labels, _ = relabel.stationary_plant(positions, contacts, ids, support, skeleton)
    assert not labels[:, 3].any()
    skeleton.foot_joint_idx = [4, 1, 2, 2]
    with pytest.raises(ValueError, match="permutation"):
        relabel.stationary_plant(positions, contacts, ids, support, skeleton)


@pytest.fixture
def source_dataset(tmp_path):
    representation = relabel.build_native_representation()
    skeleton = representation.skeleton
    frames = 6
    rotations = torch.eye(3).expand(frames, skeleton.nbjoints, 3, 3).clone()
    root = torch.zeros(frames, 3)
    root[:, 1] = 1.2
    features = representation({
        "local_joint_rots": rotations, "translation": root,
        "foot_contacts": torch.ones(frames, 4),
    }, to_normalize=False, lengths=torch.tensor(frames)).numpy()
    contact_columns = representation.indices["foot_contacts"]
    features[:, contact_columns[0]] = 0.0
    joints = representation.inverse(torch.from_numpy(features), False)["posed_joints"]
    patches = []
    for index, side in enumerate(("left", "right")):
        foot = getattr(skeleton, side + "_foot_joint_idx")
        patches.append({"id": index, "status": "weak_support",
                        "height_m": float(joints[:, foot, 1].max()) - 0.05,
                        "polygon_xz_m": [[-1, -1], [1, -1], [1, 1], [-1, 1]]})
    source = tmp_path / "source"
    (source / "shards").mkdir(parents=True)
    (source / "reports").mkdir()
    (source / "stats").mkdir()
    np.save(source / "stats/mean.npy", np.arange(370, dtype=np.float32))
    np.save(source / "stats/std.npy", np.ones(370, dtype=np.float32))
    np.savez_compressed(source / "shards/flat.npz", global_features=features)
    arrays = {
        "global_features": features,
        "dual_features": np.arange(frames * 370, dtype=np.float32).reshape(frames, 370),
        "fps": np.asarray(30.0, dtype=np.float32),
        "patch_id_left": np.array([0, -1, 0, 0, 0, 0]),
        "patch_id_right": np.ones(frames, dtype=np.int64),
        "stair_contact_labels": features[:, contact_columns].copy(),
        "stair_contact_label_mask": np.ones((frames, 4), dtype=bool),
        "vertex_measured_left": np.ones((frames - 1, 98), dtype=bool),
        "sole_vertices_left": np.zeros((frames, 98, 3), dtype=np.float32),
        # Deliberately unusable cached FK: the converter must reconstruct from features.
        "posed_joints": np.full((frames, skeleton.nbjoints, 3), np.nan, dtype=np.float32),
    }
    records = [{"shard": "shards/flat.npz", "behavior": "walk", "split": "test"}]
    for direction, split in (("up", "train"), ("down", "validation")):
        shard, report = f"shards/{direction}.npz", f"reports/{direction}.json"
        np.savez_compressed(source / shard, **arrays)
        (source / report).write_text(json.dumps({"patches": patches}), encoding="utf-8")
        records.append({"shard": shard, "behavior": f"stairs_climbing_{direction}",
                        "split": split, "support_report": report})
    (source / "accepted.json").write_text(json.dumps(records, indent=3) + "\n", encoding="utf-8")
    (source / "unbound_notes.txt").write_bytes(b"copy the full source dataset\r\n")
    artifacts = {path.relative_to(source).as_posix(): relabel.sha256_file(path)
                 for path in source.rglob("*") if path.is_file() and path.name != "unbound_notes.txt"}
    manifest = {
        "contract": {"version": "native-mb-soma30-v1", "target_fps": 30.0,
                     "representation": relabel.REPRESENTATION_NAME},
        "artifact_sha256": artifacts, "accepted_json_sha256": artifacts["accepted.json"],
        "source_dataset": "original-flat-parent", "source_manifest_sha256": "earlier-lineage",
        "split_counts": {"train": 1, "validation": 1, "test": 1},
    }
    (source / "native_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source, representation


def test_fresh_clone_preserves_source_flat_stats_evidence_and_binds_lineage(source_dataset, tmp_path):
    source, representation = source_dataset
    output = tmp_path / "plant-v1"
    source_hashes = {p.relative_to(source).as_posix(): relabel.sha256_file(p)
                     for p in source.rglob("*") if p.is_file()}
    relabel.main(["--dataset", str(source), "--output", str(output)])
    parent = json.loads((source / "native_manifest.json").read_text())
    child = json.loads((output / "native_manifest.json").read_text())
    report = json.loads((output / relabel.REPORT_FILE).read_text())
    assert child["parent_native_manifest_sha256"] == source_hashes["native_manifest.json"]
    assert child["source_manifest_sha256"] == parent["source_manifest_sha256"]
    assert child["source_dataset"] == parent["source_dataset"]
    assert child["split_counts"] == parent["split_counts"]
    assert (output / relabel.PARENT_FILE).read_bytes() == (source / "native_manifest.json").read_bytes()
    assert len(report["shards"]) == 2
    assert "final backward interval" in report["protocol"]["derivative"]
    assert "dual_features" in report["protocol"]["original_evidence"]
    for relative, digest in source_hashes.items():
        assert relabel.sha256_file(source / relative) == digest
        if relative not in {"native_manifest.json", "shards/up.npz", "shards/down.npz"}:
            assert relabel.sha256_file(output / relative) == digest
    for relative, digest in child["artifact_sha256"].items():
        assert relabel.sha256_file(output / relative) == digest
    assert child["stationary_plant_relabel"]["report_sha256"] == relabel.sha256_file(output / relabel.REPORT_FILE)
    contact_columns = representation.indices["foot_contacts"]
    noncontact = np.setdiff1d(np.arange(representation.motion_rep_dim), contact_columns)
    with np.load(source / "shards/up.npz", allow_pickle=False) as before, np.load(
            output / "shards/up.npz", allow_pickle=False) as after:
        assert set(after.files) == set(before.files) | set(relabel.NEW_ARRAYS)
        for key in before.files:
            if key != "global_features":
                assert after[key].dtype == before[key].dtype
                assert after[key].shape == before[key].shape
                assert after[key].tobytes() == before[key].tobytes()
        assert after["global_features"][:, noncontact].tobytes() == before["global_features"][:, noncontact].tobytes()
        plant = after["native_stationary_plant"]
        assert plant.dtype == after["native_stationary_plant_mask"].dtype == np.bool_
        np.testing.assert_array_equal(plant, after["native_stationary_plant_mask"])
        np.testing.assert_array_equal(after["global_features"][:, contact_columns], plant.astype(np.float32))
        assert not (plant & ~(before["global_features"][:, contact_columns] > 0.5)).any()
        assert plant.sum() < (before["global_features"][:, contact_columns] > 0.5).sum()
        provenance = json.loads(after["native_stationary_plant_provenance"].item())
        assert provenance["arrays"] == {name: relabel.VERSION for name in relabel.NEW_ARRAYS[:2]}
        assert provenance["parent_native_manifest_sha256"] == child["parent_native_manifest_sha256"]
        assert provenance["source_shard_sha256"] == source_hashes["shards/up.npz"]


@pytest.mark.parametrize("relative", ["shards/flat.npz", "shards/up.npz", "stats/mean.npy", "reports/up.json"])
def test_corrupted_source_hash_fails_before_destination_is_created(source_dataset, tmp_path, relative):
    source, _ = source_dataset
    path = source / relative
    path.write_bytes(path.read_bytes() + b"corruption")
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="Source artifact hash mismatch"):
        relabel.relabel_dataset(source, output)
    assert not output.exists()


def test_unbound_support_report_is_rejected_before_cloning(source_dataset, tmp_path):
    source, _ = source_dataset
    path = source / "native_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["artifact_sha256"]["reports/up.json"]
    path.write_text(json.dumps(manifest))
    output = tmp_path / "unbound"
    with pytest.raises(ValueError, match="must bind"):
        relabel.relabel_dataset(source, output)
    assert not output.exists()


def test_existing_destination_and_nested_destination_are_rejected(source_dataset, tmp_path):
    source, _ = source_dataset
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel").write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="must not exist"):
        relabel.relabel_dataset(source, output)
    assert (output / "sentinel").read_bytes() == b"keep"
    with pytest.raises(ValueError, match="separate"):
        relabel.relabel_dataset(source, source / "child")
    assert not (source / "child").exists()
