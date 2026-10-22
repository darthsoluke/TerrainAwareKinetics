"""Exercise actual official skinning, source identity, builder and output contracts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime():
    builder = script("build_lafan_weak_support_pairs")
    builder._init_runtime()
    torch.set_num_threads(2)
    representation = builder.build_representation()
    surfaces = builder.official_sole_surfaces(representation.skeleton)[0]
    skinner = builder.load_official_skinner(representation.skeleton)
    return builder, representation, surfaces, skinner


def build_item(tmp_path, runtime, name="walk_test"):
    builder, representation, surfaces, skinner = runtime
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "shards").mkdir(exist_ok=True)
    output = tmp_path / "output"
    output.mkdir(exist_ok=True)
    local = np.broadcast_to(np.eye(3, dtype=np.float32), (36, 30, 3, 3)).copy()
    root = np.zeros((36, 3), dtype=np.float32)
    root[:, 1] = 1.
    motion = builder.complete_motion_dict(torch.from_numpy(local), torch.from_numpy(root), representation.skeleton, 60.)
    shard = source / "shards" / f"{name}.npz"
    np.savez_compressed(shard, local_rot_mats=local, root_positions=root,
                        source_positions=motion["posed_joints"].numpy(), target_fps=60.)
    record = {"filename": name, "shard": f"shards/{name}.npz", "frames_60hz": 36,
              "shard_sha256": builder._sha256_file(shard), "recording_group": name,
              "subject_id": "subject4", "split": "validation", "behavior": "walk"}
    original_mesh_vertices = skinner.bind_vertices.shape[0]
    row = builder.process_record(record, source, output, representation, surfaces, skinner, 16,
                                 "accepted-hash", "builder-hash", "core-hash", "mesh-hash")
    assert skinner.bind_vertices.shape[0] == original_mesh_vertices
    item = output / row["item_relative_path"]
    report = json.loads((item / "report.json").read_text())
    with np.load(item / "pair.npz", allow_pickle=False) as data:
        data = {key: data[key] for key in data.files}
    return builder, row, data, report


def test_real_builder_retains_source_and_inherited_split(tmp_path, runtime):
    builder, row, data, report = build_item(tmp_path, runtime)
    assert row["split"] == "validation" and row["recording_group"] == "walk_test"
    assert report["source_arrays_unchanged"]
    assert report["official_selected_lbs_parity_max_error_m"] <= 1e-6
    assert report["additional_floor_translation_m"] == 0
    assert len(data["sole_vertex_indices_left"]) == len(data["sole_vertex_indices_right"]) == 98
    assert not set(data["sole_vertex_indices_left"]) & set(data["sole_vertex_indices_right"])
    assert builder._sha256_array(data["root_positions"]) == report["source_root_positions_sha256"]
    assert builder._sha256_array(data["local_rot_mats"]) == report["source_rotations_sha256"]
    review = script("check_weak_support_pairs")
    stats = review.validate_pair(data, report)
    assert stats["left"]["weak_support_frames"] > 0
    assert stats["right"]["measured_vertex_intervals"] > 0
    assert not report["training_eligible"]  # Not force/terrain GT approval.


def test_selected_skinner_is_safe_to_reuse_for_another_record(tmp_path, runtime):
    build_item(tmp_path, runtime, "first")
    _, _, _, report = build_item(tmp_path, runtime, "second")
    assert report["official_selected_lbs_parity_max_error_m"] <= 1e-6


@pytest.mark.parametrize("corruption", ["motion", "interval", "unknown_zero", "slip_minimum"])
def test_artifact_contract_rejects_corruption(tmp_path, runtime, corruption):
    _, _, data, report = build_item(tmp_path, runtime)
    if corruption == "motion":
        data["root_positions"][0, 1] += .01
    elif corruption == "interval":
        data["vertex_measured_left"] = data["vertex_measured_left"][:-1]
    elif corruption == "unknown_zero":
        data["patch_id_left"][0] = -1
        data["support_confidence_left"][0] = 0
        data["support_gap_m_left"][0] = 0  # Unknown is NaN, never ground height zero.
    else:
        data["slip_mps_left"][data["vertex_measured_left"]] += .5
    with pytest.raises((ValueError, AssertionError)):
        script("check_weak_support_pairs").validate_pair(data, report)


def test_same_recording_cannot_leak_across_splits(tmp_path):
    rows = [{"recording_group": "same", "split": "train"},
            {"recording_group": "same", "split": "test"}]
    (tmp_path / "accepted.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="multiple inherited splits"):
        script("build_lafan_weak_support_pairs")._read_manifest(tmp_path)


def test_render_sampling_has_frames_and_does_not_score_select_confidence():
    renderer = script("render_stair_weak_pairs")
    root = np.zeros((600, 3))
    root[:, 2] = np.arange(600) / 60
    first, last = renderer._segment(root, 60, len(root))
    frames = renderer._sample_frames(first, last, 60)
    assert len(frames) >= 1 and len(frames) <= renderer.MAX_GIF_FRAMES
    assert frames[0] >= first and frames[-1] < last
