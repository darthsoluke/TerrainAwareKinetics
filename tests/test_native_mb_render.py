import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "out" / "vendor" / "kimodo"))
from scripts import render_native_mb_review as renderer
from scripts.render_native_mb_review import render


def _review(root, identity=None, source=None, prediction=None, fps=30.0):
    root.mkdir()
    (root / "metrics.json").write_text(json.dumps({
        "task": "source-encoded reconstruction; not WASD generation",
        "examples": {"walk": identity or {
            "shard": "shards/walk.npz", "start": 12, "conditioning": "context4"
        }},
    }))
    source = np.zeros((2, 30, 3), dtype=np.float32) if source is None else source
    prediction = np.ones((2, 30, 3), dtype=np.float32) * 0.1 if prediction is None else prediction
    np.savez(root / "walk_reconstruction.npz", source=source,
             prediction=prediction, fps=np.asarray(fps))
    return root


def _bytes(path):
    return hashlib.sha256(path.read_bytes()).digest()


def test_reference_review_encodes_three_panel_gif_without_changing_inputs(tmp_path):
    pilot, expanded = tmp_path / "pilot", tmp_path / "expanded"
    src = np.arange(2 * 30 * 3, dtype=np.float32).reshape(2, 30, 3) / 100
    _review(pilot, source=src, prediction=src + 0.2)
    _review(expanded, source=src + np.float32(5e-6), prediction=src + 0.3)
    pilot_npz, expanded_npz = pilot / "walk_reconstruction.npz", expanded / "walk_reconstruction.npz"
    original_bytes = (_bytes(pilot_npz), _bytes(expanded_npz))

    output = tmp_path / "three_panel_review"
    result = render(expanded, pilot, output, "Pilot 12k", "Expanded 45k")

    assert result["gifs"] == ["walk_reconstruction.gif"]
    assert (output / "walk_reconstruction.gif").is_file()
    assert (output / "walk_reconstruction.png").is_file()
    assert (_bytes(pilot_npz), _bytes(expanded_npz)) == original_bytes
    from PIL import Image
    with Image.open(output / "walk_reconstruction.gif") as gif:
        assert gif.n_frames == 2
        assert gif.size[0] > 0 and gif.size[1] > 0
    with Image.open(output / "walk_reconstruction.png") as png:
        assert png.width > 900


@pytest.mark.parametrize("case", ["source", "shard", "start", "conditioning", "fps", "shape"])
def test_reference_review_rejects_incompatible_pair(tmp_path, case):
    pilot, expanded = tmp_path / "pilot", tmp_path / "expanded"
    _review(pilot)
    options = {}
    if case == "source":
        options["source"] = np.ones((2, 30, 3), dtype=np.float32)
    elif case == "fps":
        options["fps"] = 24.0
    elif case == "shape":
        options["source"] = np.zeros((3, 30, 3), dtype=np.float32)
        options["prediction"] = np.ones((3, 30, 3), dtype=np.float32)
    identity = {"shard": "shards/walk.npz", "start": 12, "conditioning": "context4"}
    if case == "shard":
        identity["shard"] = "shards/other.npz"
    elif case == "start":
        identity["start"] = 13
    elif case == "conditioning":
        identity["conditioning"] = "none"
    _review(expanded, identity=identity, **options)

    with pytest.raises(ValueError):
        render(expanded, pilot, tmp_path / "three_panel_review")
    assert not (tmp_path / "three_panel_review").exists()


def test_reference_review_rejects_empty_frames(tmp_path):
    pilot, expanded = tmp_path / "pilot", tmp_path / "expanded"
    empty = np.empty((0, 30, 3), dtype=np.float32)
    _review(pilot, source=empty, prediction=empty)
    _review(expanded, source=empty, prediction=empty)

    with pytest.raises(ValueError, match="Invalid reconstruction arrays"):
        render(expanded, pilot, tmp_path / "three_panel_review")


def test_default_mode_keeps_two_panel_output_location(tmp_path):
    review = tmp_path / "review"
    _review(review)

    result = render(review)

    assert result["gifs"] == ["walk_reconstruction.gif"]
    assert (review / "walk_reconstruction.gif").is_file()
    assert not (tmp_path / "review_comparison").exists()


@pytest.mark.parametrize("mode,description", [
    ("context4_stress_probe", "Context-4 stress probe"),
    ("inbetween_endpoint_pose", "Endpoint-pose in-between"),
    ("native_keyframe1_inbetween", "Native keyframe-1 in-between"),
])
def test_pose_render_uses_generated_token_and_condition_captions(tmp_path, monkeypatch, mode, description):
    review = _review(tmp_path / mode)
    metrics_path = review / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update(stage="pose", render_mode=mode,
                   task="native MotionBricks Pose conditional held-out generation")
    metrics["examples"]["walk"]["conditioning_mode"] = "stale_example_mode"
    metrics_path.write_text(json.dumps(metrics))
    captions = []
    render_motion_set = renderer._render_motion_set

    def capture(path, motions, fps, output_dir, title, **kwargs):
        assert output_dir == review
        assert len(motions) == 2
        captions.append((title, kwargs["panel_labels"]))
        return render_motion_set(path, motions, fps, output_dir, title, **kwargs)

    monkeypatch.setattr(renderer, "_render_motion_set", capture)
    result = render(review)

    assert result["gifs"] == ["walk_reconstruction.gif"]
    assert (review / "walk_reconstruction.png").is_file()
    assert (review / "walk_reconstruction.gif").is_file()
    title, labels = captions[0]
    assert "generated tokens" in title
    assert "known source root" in title
    assert mode in title and description in title
    assert "no WASD" in title
    assert "source-encoded" not in title
    assert "stale_example_mode" not in title
    assert labels == ("Source", "Generated pose")


def test_pose_caption_falls_back_to_example_conditioning_mode():
    title, labels = renderer._two_panel_caption("walk", {
        "stage": "pose",
        "examples": {"walk": {"conditioning_mode": "context4_stress_probe"}},
    })
    assert "context4_stress_probe" in title
    assert "generated tokens + known source root" in title
    assert "source-encoded" not in title
    assert labels == ("Source", "Generated pose")


def test_pose_caption_requires_an_identified_mode():
    with pytest.raises(ValueError, match="render_mode or example conditioning_mode"):
        renderer._two_panel_caption("walk", {"stage": "pose"})


def test_pipeline_caption_does_not_claim_a_known_source_trajectory():
    title, _ = renderer._two_panel_caption("walk", {
        "stage": "pose", "render_mode": "native_keyframe1_inbetween",
        "root_condition_source": "root_model",
    })
    assert "model-predicted root" in title
    assert "known source root" not in title
    assert "source-encoded" not in title


def test_pose_caption_rejects_unknown_root_condition():
    with pytest.raises(ValueError, match="Unknown root condition"):
        renderer._two_panel_caption("walk", {
            "stage": "pose", "render_mode": "native_keyframe1_inbetween",
            "root_condition_source": "unidentified",
        })


def test_tokenizer_caption_preserves_existing_title_and_labels():
    assert renderer._two_panel_caption("walk", {}) == (
        "walk | source-encoded tokens, context4 | no postprocessing",
        ("Source motion", "Native MB reconstruction"),
    )
