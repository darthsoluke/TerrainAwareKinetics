import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "out/vendor/kimodo"))

from motionbricks.data.native_mb_data import (
    NativeMBMotionDataset,
    build_native_representation,
    collate_native_batch,
)
from motionbricks.data.native_mb_terrain_protocol import validate_terrain_dataset
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


_SCRIPT = Path(__file__).parents[1] / "scripts" / "build_parc_native_dataset.py"
_SPEC = importlib.util.spec_from_file_location("build_parc_native_dataset", _SCRIPT)
_BUILDER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BUILDER)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture(tmp_path, *, frames=65, source_fps=30.0, split_rows=None, admission_status="passed"):
    inputs = tmp_path / "input"
    inputs.mkdir(parents=True)
    skeleton = SOMASkeleton30()
    source_names = np.asarray([
        "pelvis", "torso", "head", "right_upper_arm", "right_lower_arm", "right_hand",
        "left_upper_arm", "left_lower_arm", "left_hand", "right_thigh", "right_shin",
        "right_foot", "left_thigh", "left_shin", "left_foot",
    ])
    items = []
    for index, source_id in enumerate(("course-a", "course-b", "course-c")):
        folder = inputs / source_id
        folder.mkdir()
        raw_recording = folder / "recorded_raw.pkl"
        raw_recording.write_bytes(f"raw-physics-recording-{source_id}".encode())
        export_root = folder / "export"
        converted_dir = export_root / "recordings"
        converted_dir.mkdir(parents=True)
        source = converted_dir / "converted.pkl"
        source.write_bytes(f"official-converted-recording-{source_id}".encode())
        t = np.arange(frames, dtype=np.float32)
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 0] = 0.008 * t
        root[:, 1] = 1.0 + 0.015 * np.sin(t * 0.08)
        local = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 30, 3, 3)).copy()
        yaw = np.zeros(frames, dtype=np.float32)
        if index == 1:
            yaw[:] = np.pi / 2
        cosine, sine = np.cos(yaw), np.sin(yaw)
        local[:, 0, 0, 0] = cosine
        local[:, 0, 0, 2] = sine
        local[:, 0, 2, 0] = -sine
        local[:, 0, 2, 2] = cosine
        with torch.no_grad():
            global_rot, _, _ = skeleton.fk(torch.from_numpy(local), torch.from_numpy(root))
        global_rot = global_rot.numpy()
        contacts = (np.arange(frames * 15, dtype=np.float32).reshape(frames, 15) % 2)
        # PARC grid is [source-X, source-Y] and intentionally asymmetric.
        hf = np.arange(100 * 90, dtype=np.float32).reshape(100, 90) / 10000.0
        retarget = folder / "retarget.npz"
        np.savez_compressed(
            retarget,
            local_rot_mats=local,
            root_positions=root,
            target_global_rot_mats=global_rot,
            target_names=np.asarray(skeleton.bone_order_names),
            target_parents=skeleton.joint_parents.numpy(),
            hf=hf,
            min_point=np.asarray([-4.95, -4.45], dtype=np.float32),
            dx=np.asarray(0.1, dtype=np.float32),
            source_contacts=contacts,
            source_names=source_names,
            source_fps=np.asarray(source_fps, dtype=np.float32),
            scale_scene=np.asarray(1.0, dtype=np.float32),
        )
        intent = folder / "intent.npz"
        source_commands = np.zeros((frames + 1, 2), dtype=np.float32)
        source_commands[:, 0] = 1.0
        source_frame_indices = np.delete(np.arange(frames + 1, dtype=np.int64), 3)
        aligned_commands = source_commands[source_frame_indices]
        np.savez_compressed(intent, command_world_xy=aligned_commands,
                            source_frame_indices=source_frame_indices, fps=np.asarray(30.0))
        report = folder / "report.json"
        report.write_text(json.dumps({
            "execution_status": "processed", "screen_status": "pass",
            "training_approved": False, "fps": source_fps,
            "frames": frames, "source_frames": frames,
            "shard_sha256": _hash(retarget),
            "source": {"sha256": _hash(source)},
        }), encoding="utf-8")
        physics = folder / "physics_report.json"
        origins = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        offsets = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        aligned = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        physics.write_text(json.dumps({
            "status": "official_recording_finished",
            "physics_recording_completed": True,
            "inputs_unchanged_after_run": True,
            "engine_origin_alignment": {
                "source": "gym.get_env_origin", "applied_before_first_reset": True,
                "shared_buffer_preserved": True,
                "before": {"engine_origins": origins, "observation_offsets": offsets,
                           "max_error_m": 4.0},
                "after": {"engine_origins": origins, "observation_offsets": aligned,
                          "max_error_m": 0.0},
            },
            "engine_origin_contract": {
                "engine_origins": origins, "observation_offsets": aligned,
                "max_error_m": 0.0,
            },
            "packed_terrain_validation": {
                "grid_spacing_matches": True, "all_tile_cells_checked": 100,
                "maximum_height_error_m": 0.0,
            },
            "inputs": [{"candidate_key": source_id, "sha256": _hash(raw_recording),
                        "source_frames": frames}],
            "saved": [{"candidate_key": source_id, "source_sha256": _hash(raw_recording),
                       "sha256": _hash(raw_recording), "recorded_state": "simulated",
                       "whole_motion": True, "start_fraction": 0.0,
                       "frames": frames, "source_frames": frames, "fps": 30.0,
                       "recorded_terrain_error_m": 0.0}],
        }), encoding="utf-8")
        parity = {key: True for key in _BUILDER._EXPORT_PARITY_FIELDS}
        export_manifest = export_root / "manifest.json"
        export_manifest.write_text(json.dumps({
            "schema": "parc.physics_recording_export.v1",
            "status": "exported_all_reported_physics_recordings",
            "run_report": "../physics_report.json",
            "run_report_sha256": _hash(physics),
            "run_status": "official_recording_finished",
            "physics_recording_completed": True,
            "official_converter_sha256": "a" * 64,
            "conversion": "PARC scripts/convert_legacy_motions.py:motion_to_ms_file",
            "recordings": [{
                "candidate_key": source_id,
                "source_sha256": _hash(raw_recording),
                "filename": raw_recording.name,
                "sha256": _hash(raw_recording),
                "frames": frames, "fps": 30.0, "source_frames": frames,
                "start_fraction": 0.0, "whole_motion": True,
                "recorded_state": "simulated", "recorded_terrain_error_m": 0.0,
                "converted_path": "recordings/converted.pkl",
                "converted_sha256": _hash(source), "conversion_parity": parity,
            }],
        }), encoding="utf-8")
        items.append({
            "source_id": source_id,
            "status": admission_status,
            "source_path": str(source), "source_sha256": _hash(source),
            "retarget_path": str(retarget), "retarget_sha256": _hash(retarget),
            "intent_path": str(intent), "intent_sha256": _hash(intent),
            "physics_report_path": str(physics), "physics_report_sha256": _hash(physics),
            "export_manifest_path": str(export_manifest),
            "export_manifest_sha256": _hash(export_manifest),
            "physics_candidate_key": source_id,
            "physics_recording_sha256": _hash(raw_recording),
            "behavior": "mixed_terrain",
        })

    admissions = inputs / "admissions.json"
    admissions.write_text(json.dumps({
        "status": "passed", "retarget_status": "passed",
        "motion_terrain_pairing_status": "passed", "physics_retest_status": "passed",
        "engine_origin_measurement_status": "passed", "items": items,
    }), encoding="utf-8")
    if split_rows is None:
        split_rows = [
            {"source_id": "course-a", "source_group": "group-a",
             "terrain_family": "stairs", "terrain_seed": "seed-a", "split": "train"},
            {"source_id": "course-b", "source_group": "group-b",
             "terrain_family": "rocks", "terrain_seed": "seed-b", "split": "validation"},
            {"source_id": "course-c", "source_group": "group-c",
             "terrain_family": "platforms", "terrain_seed": "seed-c", "split": "test"},
        ]
    splits = inputs / "splits.json"
    splits.write_text(json.dumps({"items": split_rows}), encoding="utf-8")
    return admissions, splits, items


def _flat_native_dataset(tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "build_native_mb_dataset.py"
    spec = importlib.util.spec_from_file_location("build_native_mb_dataset_for_reuse", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, output = tmp_path / "flat-source", tmp_path / "flat-native"
    (source / "shards").mkdir(parents=True)
    rows = []
    frames = 141
    time = np.arange(frames, dtype=np.float32) / 60.0
    for split in ("train", "validation"):
        local = np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 30, 3, 3)).copy()
        root = np.zeros((frames, 3), dtype=np.float32)
        root[:, 0] = 0.2 * time + 0.01 * np.sin(5 * time)
        root[:, 1] = 1.0
        root[:, 2] = 0.05 * np.sin(2 * time)
        relative = f"shards/{split}.npz"
        np.savez_compressed(
            source / relative, local_rot_mats=local, root_positions=root,
            global_features=np.zeros((frames, 366), dtype=np.float32),
        )
        rows.append({"filename": split, "recording_group": split, "behavior": "walk",
                     "split": split, "shard": relative, "source_fps": 60.0,
                     "frames_60hz": frames})
    (source / "accepted.json").write_text(json.dumps(rows), encoding="utf-8")
    module.build_dataset(source, output)
    return output


def test_parc_heightfield_and_command_axes_use_shared_y_to_z():
    source_hf = np.asarray([[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    hf, origin, dx = _BUILDER.parc_heightfield_to_native(source_hf, [5.0, 7.0], 0.5)
    np.testing.assert_array_equal(hf, [[2.0, 12.0], [1.0, 11.0], [0.0, 10.0]])
    np.testing.assert_allclose(origin, [5.0, 0.0, -8.0])
    assert dx == 0.5

    commands = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    local = _BUILDER.parc_world_command_to_native_local_xz(
        commands, np.asarray([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(local, [[1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], atol=1e-6)


def test_official_mb_foot_contacts_are_terrain_relative_and_height_translation_invariant():
    skeleton = SOMASkeleton30()
    positions = torch.zeros((4, skeleton.nbjoints, 3), dtype=torch.float32)
    foot_indices = skeleton.left_foot_joint_idx + skeleton.right_foot_joint_idx
    positions[:, foot_indices, 1] = 1.05
    heightfield = torch.ones((11, 11), dtype=torch.float32)
    origin = torch.tensor([-0.5, 1.0, -0.5], dtype=torch.float32)

    upstairs = _BUILDER._terrain_relative_foot_contacts(
        positions, skeleton, clip_heightfield_y_m=heightfield,
        clip_origin_xyz_m=origin, dx_m=0.1)
    shifted = _BUILDER._terrain_relative_foot_contacts(
        positions + torch.tensor([0.0, 1.0, 0.0]), skeleton,
        clip_heightfield_y_m=heightfield + 1.0,
        clip_origin_xyz_m=origin + torch.tensor([0.0, 1.0, 0.0]), dx_m=0.1)
    assert upstairs.shape == (4, 4)
    assert torch.equal(upstairs, torch.ones_like(upstairs))
    assert torch.equal(shifted, upstairs)


def test_official_mb_foot_contacts_reject_unknown_terrain():
    skeleton = SOMASkeleton30()
    positions = torch.zeros((3, skeleton.nbjoints, 3), dtype=torch.float32)
    positions[:, skeleton.left_foot_joint_idx + skeleton.right_foot_joint_idx, 0] = 20.0
    with pytest.raises(ValueError, match="Unknown terrain under"):
        _BUILDER._terrain_relative_foot_contacts(
            positions, skeleton, clip_heightfield_y_m=torch.zeros((5, 5)),
            clip_origin_xyz_m=[-0.2, 0.0, -0.2], dx_m=0.1)


def test_explicit_admissions_are_mandatory(tmp_path):
    admissions, splits, _ = _fixture(tmp_path)
    original = json.loads(admissions.read_text())
    original["items"][0]["retarget_sha256"] = "0" * 64
    admissions.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="retarget SHA256 mismatch"):
        _BUILDER.build_dataset(admissions, splits, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_old_physics_admissions_without_measured_origin_retest_are_rejected(tmp_path):
    admissions, splits, _ = _fixture(tmp_path)
    original = json.loads(admissions.read_text())
    original.pop("physics_retest_status")
    original.pop("engine_origin_measurement_status")
    original["items"][0].pop("physics_report_path")
    original["items"][0].pop("physics_report_sha256")
    original["items"][0].pop("export_manifest_path")
    original["items"][0].pop("export_manifest_sha256")
    original["items"][0].pop("physics_candidate_key")
    admissions.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="physics_retest_status='passed'"):
        _BUILDER.build_dataset(admissions, splits, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("failure", ["origin", "partial", "stale_source"])
def test_physics_report_must_prove_corrected_full_recording(tmp_path, failure):
    admissions, splits, items = _fixture(tmp_path)
    physics_path = Path(items[0]["physics_report_path"])
    physics = json.loads(physics_path.read_text())
    if failure == "origin":
        physics["engine_origin_alignment"]["applied_before_first_reset"] = False
    elif failure == "partial":
        physics["saved"][0]["whole_motion"] = False
        physics["saved"][0]["start_fraction"] = 0.25
    else:
        physics["saved"][0]["sha256"] = "0" * 64
    physics_path.write_text(json.dumps(physics))
    admission_doc = json.loads(admissions.read_text())
    admission_doc["items"][0]["physics_report_sha256"] = _hash(physics_path)
    export_path = Path(items[0]["export_manifest_path"])
    export_manifest = json.loads(export_path.read_text())
    export_manifest["run_report_sha256"] = _hash(physics_path)
    export_path.write_text(json.dumps(export_manifest))
    admission_doc["items"][0]["export_manifest_sha256"] = _hash(export_path)
    admissions.write_text(json.dumps(admission_doc))
    expected = {
        "origin": "measured, applied, pre-reset engine-origin alignment",
        "partial": "partial, stale, or not hash-linked",
        "stale_source": "partial, stale, or not hash-linked",
    }[failure]
    with pytest.raises(ValueError, match=expected):
        _BUILDER.build_dataset(admissions, splits, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("failure", ["fps", "nonmonotonic", "length"])
def test_online_intent_matches_correction_writer_schema(tmp_path, failure):
    admissions, splits, items = _fixture(tmp_path)
    intent_path = Path(items[0]["intent_path"])
    with np.load(intent_path, allow_pickle=False) as data:
        commands = data["command_world_xy"].copy()
        frame_ids = data["source_frame_indices"].copy()
        fps = data["fps"].copy()
    if failure == "fps":
        fps = np.asarray(60.0)
    elif failure == "nonmonotonic":
        frame_ids[[0, 1]] = frame_ids[[1, 0]]
    else:
        commands = commands[:-1]
    np.savez_compressed(intent_path, command_world_xy=commands,
                        source_frame_indices=frame_ids, fps=fps)
    admissions_doc = json.loads(admissions.read_text())
    admissions_doc["items"][0]["intent_sha256"] = _hash(intent_path)
    admissions.write_text(json.dumps(admissions_doc))
    with pytest.raises(ValueError, match="Invalid online intent contract|source_frame_indices"):
        _BUILDER.build_dataset(admissions, splits, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("identity", ["source_group", "terrain_family", "terrain_seed"])
def test_source_and_terrain_lineages_cannot_cross_splits(tmp_path, identity):
    split_rows = [
        {"source_id": "course-a", "source_group": "g1", "terrain_family": "stairs",
         "terrain_seed": "s1", "split": "train"},
        {"source_id": "course-b", "source_group": "g2", "terrain_family": "rocks",
         "terrain_seed": "s2", "split": "validation"},
        {"source_id": "course-c", "source_group": "g3", "terrain_family": "platforms",
         "terrain_seed": "s3", "split": "test"},
    ]
    split_rows[1][identity] = split_rows[0][identity]
    case = tmp_path / identity
    admissions, splits, _ = _fixture(case, split_rows=split_rows)
    with pytest.raises(ValueError, match=f"{identity} .* leaks across splits"):
        _BUILDER.build_dataset(admissions, splits, case / "out")


@pytest.mark.parametrize("frames,source_fps,admission_status,match", [
    (64, 30.0, "passed", "too short"),
    (65, 60.0, "passed", "full 30 FPS"),
    (65, 30.0, "screened", "status='passed'"),
])
def test_invalid_rate_short_or_unapproved_inputs_fail(tmp_path, frames, source_fps,
                                                       admission_status, match):
    admissions, splits, _ = _fixture(tmp_path, frames=frames, source_fps=source_fps,
                                      admission_status=admission_status)
    with pytest.raises(ValueError, match=match):
        _BUILDER.build_dataset(admissions, splits, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_builds_trainer_compatible_native_terrain_dataset_and_preserves_contacts(tmp_path):
    admissions, splits, inputs = _fixture(tmp_path)
    output = tmp_path / "native"
    manifest = _BUILDER.build_dataset(admissions, splits, output)
    rows = json.loads((output / "accepted.json").read_text())
    assert len(rows) == 3
    assert all(row["source_shard_fps"] == 30.0 and row["frames_30hz"] == 65 for row in rows)
    train = next(row for row in rows if row["split"] == "train")
    acceptance = json.loads((output / "acceptance_report.json").read_text())
    assert acceptance["retarget_screen_is_not_automatically_promoted"] is True
    assert acceptance["physics_retest_status"] == "passed"
    assert acceptance["engine_origin_measurement_status"] == "passed"
    assert acceptance["items"][0]["engine_origin_alignment_max_error_m"] <= 1e-5
    assert manifest["terrain_conditioning"]["acceptance"]["physics_retest_status"] == "passed"
    source_item = next(row for row in inputs if row["source_id"] == train["source_id"])
    assert (output / train["physics_report_snapshot"]).is_file()
    assert (output / train["physics_export_manifest_snapshot"]).is_file()
    assert train["physics_report_sha256"] == source_item["physics_report_sha256"]
    assert train["physics_export_manifest_sha256"] == source_item["export_manifest_sha256"]
    assert train["source_original_sha256"] != source_item["physics_recording_sha256"]
    source_retarget = Path(source_item["retarget_path"])
    with np.load(source_retarget, allow_pickle=False) as original, np.load(
            output / train["source_contact_sidecar"], allow_pickle=False) as contacts:
        np.testing.assert_array_equal(contacts["source_contacts"], original["source_contacts"])
        assert contacts["source_contacts"].shape == (65, 15)
        expected_source_ids = np.delete(np.arange(66, dtype=np.int64), 3)
        np.testing.assert_array_equal(contacts["source_frame_indices"], expected_source_ids)
    with np.load(output / train["source_intent_sidecar"], allow_pickle=False) as intent:
        original_intent_path = Path(source_item["intent_path"])
        with np.load(original_intent_path, allow_pickle=False) as original_intent:
            np.testing.assert_array_equal(intent["command_world_xy"],
                                          original_intent["command_world_xy"])
            np.testing.assert_array_equal(intent["source_frame_indices"],
                                          original_intent["source_frame_indices"])
        assert intent["fps"].item() == 30.0
        assert np.any(np.diff(intent["source_frame_indices"]) > 1)

    with np.load(output / train["shard"], allow_pickle=False) as shard:
        assert shard["global_features"].shape == (65, 366)
        assert shard["dual_features"].shape == (65, 370)
        assert shard["terrain_points"].shape == (65, 2, 9, 9, 3)
        assert shard["terrain_valid"].shape == (65, 2, 9, 9)
        assert shard["fps"].item() == 30.0
        # The aligned +PARC-X command faces native +X. The shared v2 query grid
        # spans 0.2..3.4 m forward, whose mean is 1.8 m.
        np.testing.assert_allclose(shard["terrain_points"][0, 1, :, :, 0].mean(),
                                   1.8, atol=1e-6)

    dataset = NativeMBMotionDataset(output, "train", max_frames=65, min_frames=65,
                                    require_terrain=True)
    item = dataset[0]
    assert item["terrain_points"].shape == (65, 2, 9, 9, 3)
    batch = collate_native_batch([item])
    assert batch["terrain_points"].shape == (1, 65, 2, 9, 9, 3)
    assert batch["terrain_valid"].dtype == torch.bool

    native_rep = build_native_representation(output / "stats")
    assert item["motion"].shape[-1] == native_rep.motion_rep_dim == 366
    with np.load(output / train["shard"], allow_pickle=False) as shard:
        assert shard["mb_foot_contacts"].shape == (65, 4)
        np.testing.assert_array_equal(
            shard["global_features"][:, native_rep.indices["foot_contacts"]],
            shard["mb_foot_contacts"],
        )
    assert manifest["terrain_conditioning"]["acceptance"]["retarget_status"] == "passed"
    validate_terrain_dataset(manifest, rows)
    architecture = manifest["architecture_status"]
    assert architecture["stage"] == "pilot"
    assert architecture["final_architecture"] is False
    assert architecture["active_protocol"] == "native-mb-local-terrain-v2"
    assert architecture["tokenizer"] == "official native MotionBricks tokenizer and motion representation are unchanged"
    assert architecture["terrain_conditioned_decoder"] == "not implemented"
    assert architecture["training_performed_by_dataset_builder"] is False
    assert manifest["terrain_conditioning"]["architecture_status"] == architecture
    assert acceptance["architecture_status"] == architecture
    assert "local_terrain_encoder_pilot" in manifest["contract"]["code_sha256"]
    assert "root_backbone_terrain_pilot" in manifest["contract"]["code_sha256"]
    assert "pose_backbone_terrain_pilot" in manifest["contract"]["code_sha256"]

    artifacts = manifest["artifact_sha256"]
    for relative, expected in artifacts.items():
        assert _hash(output / relative) == expected
    assert (output / "acceptance_report.json").is_file()


def test_verified_flat_stats_can_be_reused_byte_for_byte_for_warmstart(tmp_path):
    admissions, splits, _ = _fixture(tmp_path)
    flat = _flat_native_dataset(tmp_path)
    output = tmp_path / "native-reuse-stats"
    manifest = _BUILDER.build_dataset(admissions, splits, output, reuse_stats_from=flat)
    for name in ("mean.npy", "std.npy"):
        assert (output / "stats" / name).read_bytes() == (flat / "stats" / name).read_bytes()
    assert manifest["normalization_reuse"]["normalization_byte_identical"] is True
    assert manifest["stats_contract"]["split"] == "train"
    assert manifest["terrain_conditioning"]["acceptance"]["report"] == "acceptance_report.json"


def test_terrain_observation_is_sampled_from_current_pose_and_command():
    hf = np.zeros((80, 80), dtype=np.float32)
    hf[:, :39] = 0.2
    native_hf, origin, dx = _BUILDER.parc_heightfield_to_native(hf, [-3.95, -3.95], 0.1)
    clip = {
        "frames": 1,
        "root_positions": np.asarray([[0.0, 0.8, 0.0]], dtype=np.float32),
        "target_global_rot_mats": np.broadcast_to(np.eye(3, dtype=np.float32), (1, 30, 3, 3)).copy(),
        "command_local_xz": np.asarray([[0.0, 1.0]], dtype=np.float32),
        "heightfield_y_m": native_hf,
        "min_point_world_xyz_m": origin,
        "dx_m": dx,
    }
    points, valid = _BUILDER._build_terrain_observations(clip)
    assert points.shape == (1, 2, 9, 9, 3)
    assert valid.all()
    assert points[0, 1, 1, 4, 1] == pytest.approx(0.2 - 0.8)
    assert points[0, 1, -1, 4, 2] == pytest.approx(3.4)
