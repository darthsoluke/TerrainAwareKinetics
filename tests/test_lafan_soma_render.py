import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import render_lafan_soma_retarget as renderer


def fixture(path, frames=18, *, soles=True):
    source = np.zeros((frames, 75, 3), dtype=np.float32)
    target = np.zeros((frames, 30, 3), dtype=np.float32)
    for frame in range(frames):
        source[frame, :, 0] = frame * 0.025
        source[frame, :, 1] = np.linspace(0.05, 1.75, 75)
        source[frame, :, 2] = np.linspace(-0.2, 0.2, 75)
        target[frame, :, 0] = frame * 0.018
        target[frame, :, 1] = np.linspace(0.04, 1.7, 30)
        target[frame, :, 2] = np.linspace(-0.18, 0.18, 30)
    source[:, 0, :] = np.stack((np.arange(frames) * 0.025,
                                np.full(frames, 0.9), np.zeros(frames)), axis=-1)
    target[:, 0, :] = np.stack((np.arange(frames) * 0.018,
                                np.full(frames, 0.88), np.zeros(frames)), axis=-1)
    values = {
        "source_positions": source,
        "source_global_rot_mats": np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 75, 3, 3)).copy(),
        "source_names": np.asarray([f"GenoJoint{i}" for i in range(75)]),
        "source_parents": np.asarray([-1, *range(74)], dtype=np.int64),
        "local_rot_mats": np.broadcast_to(np.eye(3, dtype=np.float32), (frames, 30, 3, 3)).copy(),
        "root_positions": target[:, 0].copy(), "target_positions": target,
        "target_fps": np.asarray(60.0), "uniform_scale": np.asarray(0.8),
    }
    if soles:
        values["sole_heights"] = np.full((frames, 4), 0.025, dtype=np.float32)
    np.savez_compressed(path, **values)
    return values


def test_renders_real_gif_frames_and_fixed_camera_covers_full_segment(tmp_path):
    source = tmp_path / "retarget.npz"
    arrays = fixture(source)
    output = tmp_path / "review.gif"
    result = renderer.render_retarget(source, output, behavior="run", start=2,
                                      duration_frames=12, frame_stride=3, gif_fps=8)

    assert result["rendered_frame_count"] == 4
    with Image.open(output) as image:
        assert image.n_frames == 4
        assert image.size[0] > 500 and image.size[1] > 300
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["floor_plane"].endswith("not asserted physical ground truth")
    assert metadata["retarget_method"] == "offline GMR retarget"
    assert metadata["runtime_posthoc_repair"] is False
    assert metadata["sole_heights_included"] is True
    assert "display only" in metadata["display_transform"]
    assert "not scored" in metadata["display_transform"]
    tracking = metadata["root_tracking_original"]
    assert len(tracking["source_root_xz_m"]) == 12
    assert len(tracking["target_minus_source_root_xz_m"]) == 12
    assert tracking["max_error_m"] > 0
    assert renderer.SOLE_NAMES == ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")

    source_view, target_view = renderer._display_positions(arrays, 2, 14)
    np.testing.assert_allclose(source_view[:, 0, (0, 2)], 0.0, atol=1e-8)
    expected_root_error_x = (arrays["target_positions"][2:14, 0, 0]
                             - arrays["source_positions"][2:14, 0, 0] * arrays["uniform_scale"])
    np.testing.assert_allclose(target_view[:, 0, 0], expected_root_error_x, atol=1e-8)
    assert np.max(np.abs(target_view[:, 0, 0])) < 0.05
    domain = result["camera_domain_xyz"]
    assert domain["x"][1] - domain["x"][0] < 1.0
    assert np.min(source_view[..., 0]) >= domain["x"][0]
    assert np.max(source_view[..., 0]) <= domain["x"][1]
    assert np.min(target_view[..., 2]) >= domain["z"][0]
    assert np.max(target_view[..., 1]) <= domain["y"][1]


def test_optional_sole_heights_and_output_collision(tmp_path):
    source = tmp_path / "retarget.npz"
    fixture(source, soles=False)
    output = tmp_path / "review.gif"
    result = renderer.render_retarget(source, output, duration_frames=6, frame_stride=2, gif_fps=6)
    assert result["rendered_frame_count"] == 3
    assert json.loads(output.with_suffix(".json").read_text())["sole_heights_included"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        renderer.render_retarget(source, output, duration_frames=6, frame_stride=2, gif_fps=6)


def test_rejects_incomplete_six_second_segment_and_bad_parents(tmp_path):
    source = tmp_path / "retarget.npz"
    arrays = fixture(source)
    with pytest.raises(ValueError, match="exceeds"):
        renderer.render_retarget(source, tmp_path / "too_long.gif", start=1, duration_frames=18)

    arrays["source_parents"][12] = 74
    np.savez_compressed(source, **arrays)
    with pytest.raises(ValueError, match="earlier joint"):
        renderer.render_retarget(source, tmp_path / "invalid.gif", duration_frames=6)


def test_camera_domain_is_fixed_and_includes_floor_and_all_frames():
    source = np.zeros((3, 75, 3), dtype=np.float32)
    target = np.zeros((3, 30, 3), dtype=np.float32)
    source[-1, 20] = [1.4, -0.03, -0.7]
    target[1, 12] = [-0.9, 1.8, 0.5]
    domain = renderer.camera_domain(source, target)
    assert domain["x"][0] < -1.4 and domain["x"][1] > 1.4
    assert domain["z"][0] < -0.7 and domain["z"][1] > 0.5
    assert domain["y"][0] < -0.03 and domain["y"][1] > 1.8
