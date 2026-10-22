import hashlib
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_native_mb_lafan as evaluator
from scripts.evaluate_native_mb_lafan import select_validation_windows


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation_selector_uses_only_real_subject4_walk_run_shards(tmp_path):
    source, dataset = tmp_path / "source", tmp_path / "native"
    (source / "shards").mkdir(parents=True)
    (dataset / "shards").mkdir(parents=True)
    source_rows, native_rows = [], []
    cases = [
        ("s4-walk", "subject4", "validation", "walk"),
        ("s4-run", "subject4", "validation", "run"),
        ("s5-run", "subject5", "test", "run"),
        ("s4-idle", "subject4", "validation", "idle"),
        ("s4-train", "subject4", "train", "walk"),
    ]
    for name, subject, split, behavior in cases:
        source_shard = f"shards/{name}.npz"
        native_shard = f"shards/{name}_30hz.npz"
        (source / source_shard).write_bytes(name.encode())
        np.savez(dataset / native_shard, global_features=np.zeros((80, 7), dtype=np.float32))
        source_rows.append({"filename": name, "shard": source_shard, "subject_id": subject,
                            "split": split, "behavior": behavior})
        native_rows.append({"filename": name, "source_shard": source_shard, "shard": native_shard,
                            "split": split, "behavior": behavior})
    (source / "accepted.json").write_text(json.dumps(source_rows))
    (dataset / "accepted.json").write_text(json.dumps(native_rows))

    windows, excluded = select_validation_windows(source, dataset)
    assert excluded == []
    assert {row["behavior"] for row in windows} == {"walk", "run"}
    assert {row["subject_id"] for row in windows} == {"subject4"}
    assert {row["split"] for row in windows} == {"validation"}
    assert len(windows) == 6
    assert all(row["source_shard_sha256"] == _digest(source / row["source_shard"])
               for row in windows)


def test_validation_selector_reports_short_heldout_windows_without_fake_behaviors(tmp_path):
    source, dataset = tmp_path / "source", tmp_path / "native"
    (source / "shards").mkdir(parents=True)
    (dataset / "shards").mkdir(parents=True)
    records = []
    for name, behavior, frames in (("walk", "walk", 40), ("run", "run", 70)):
        shard = f"shards/{name}.npz"
        native_shard = f"shards/{name}_30hz.npz"
        (source / shard).write_bytes(name.encode())
        np.savez(dataset / native_shard, global_features=np.zeros((frames, 5), dtype=np.float32))
        records.append(({"filename": name, "shard": shard, "subject_id": "subject4",
                         "split": "validation", "behavior": behavior},
                        {"filename": name, "source_shard": shard, "shard": native_shard,
                         "split": "validation", "behavior": behavior}))
    (source / "accepted.json").write_text(json.dumps([pair[0] for pair in records]))
    (dataset / "accepted.json").write_text(json.dumps([pair[1] for pair in records]))

    windows, excluded = select_validation_windows(source, dataset)
    assert {row["behavior"] for row in windows} == {"run"}
    assert len(excluded) == 1 and excluded[0]["behavior"] == "walk"


def test_feature_window_contract_is_a_65_frame_native_global_feature_array(tmp_path):
    path = tmp_path / "clip.npz"
    np.savez(path, global_features=np.arange(70 * 9, dtype=np.float32).reshape(70, 9))
    row = {"shard": "clip.npz", "start": 3}
    result = evaluator._read_features(tmp_path, row)
    assert result.shape == (65, 9)
    np.testing.assert_array_equal(result[0], np.arange(27, 36, dtype=np.float32))


@pytest.mark.parametrize("stage, expected_caption", [
    ("tokenizer", "source-encoded tokenizer reconstruction\nno pose keyframes | source root decoder condition"),
    ("pose", "generated tokens + predicted root\nposes 3/63 only | no runtime/post-hoc repair"),
])
def test_render_set_uses_stage_specific_low_level_helper_caption(tmp_path, monkeypatch, stage, expected_caption):
    calls = []
    helper = types.ModuleType("scripts.render_native_mb_review")

    def render_motion_set(path, motions, fps, output_dir, title, **kwargs):
        calls.append((path, motions, fps, title, kwargs))
        return path.with_suffix(".gif").name

    helper._render_motion_set = render_motion_set
    monkeypatch.setitem(sys.modules, "scripts.render_native_mb_review", helper)
    samples = {"walk": (np.zeros((64, 30, 3)), np.ones((64, 30, 3)), 30.0)}
    report = {"stage": stage, "task": "smoke", "windows": [{"behavior": "walk"}], "examples": {}}
    result = evaluator._render_set(tmp_path / stage, samples, report)
    assert result["gifs"] == ["walk_reconstruction.gif"]
    assert expected_caption in calls[0][3]
    assert calls[0][4]["panel_labels"] == (
        ("Source motion", "Tokenizer reconstruction") if stage == "tokenizer"
        else ("Source motion", "Generated pose"))


def test_actual_native_review_renderer_api_and_gif_smoke(tmp_path, monkeypatch):
    import importlib

    skeleton_module = types.ModuleType("motionbricks.motionlib.core.skeletons.soma")

    class Skeleton:
        joint_parents = np.asarray([-1, *range(29)], dtype=np.int64)

    skeleton_module.SOMASkeleton30 = Skeleton
    monkeypatch.setitem(sys.modules, "motionbricks.motionlib.core.skeletons.soma", skeleton_module)
    vendor = ROOT / "out" / "vendor" / "kimodo"
    if vendor.is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    renderer = importlib.import_module("scripts.render_native_mb_review")
    _render_motion_set = renderer._render_motion_set

    signature = inspect.signature(_render_motion_set)
    assert {"path", "motions", "fps", "output_dir", "title", "panel_labels"}.issubset(signature.parameters)
    folder = tmp_path / "visuals"
    folder.mkdir()
    source = np.zeros((3, 30, 3), dtype=np.float32)
    source[:, :, 1] = np.linspace(0.0, 1.7, 30)
    source[:, :, 0] = np.arange(3, dtype=np.float32)[:, None] * 0.1
    generated = source.copy()
    generated[:, 12, 0] = 0.05
    output = _render_motion_set(folder / "walk_reconstruction.npz", (source, generated), 30.0,
                                folder, "offline LAFAN evaluator smoke", panel_labels=("Source", "Generated"))
    assert output == "walk_reconstruction.gif"
    from PIL import Image
    with Image.open(folder / output) as image:
        assert image.n_frames == 3 and image.width > 500 and image.height > 250
