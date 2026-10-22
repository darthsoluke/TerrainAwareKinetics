"""Offline review clips retain their source geometry and cannot masquerade as live."""

import hashlib
import json

import numpy as np
import pytest

from motionbricks.data.soma_lab_review import (
    REVIEW_SCHEMA, load_review_clips, native_terrain_to_parc,
)


def make_review(root):
    roots = np.array([[1, 2, 3], [1.1, 2.1, 3.1]], dtype=np.float32)
    local = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 30, 3, 3)).copy()
    data = dict(roots_source=roots, roots_codec=roots + [.1, .2, .3],
                local_source=local, local_codec=local.copy(),
                hf=np.array([[0, .5], [1, 2]], dtype=np.float32),
                min_point=np.array([2, .3, -5], dtype=np.float32), dx=np.float32(.25), fps=30)
    path = root / "example.npz"
    np.savez_compressed(path, **data)
    manifest = dict(schema=REVIEW_SCHEMA, coordinate_frame="soma_y_up_world", fps=30,
                    examples=[dict(id="first", label="Stairs", file=path.name,
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    (root / "review_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, data


def test_source_and_codec_world_poses_remain_unchanged(tmp_path):
    _, data = make_review(tmp_path)
    calls = []

    def mesh(hf, origin, dx):
        calls.append((hf.copy(), origin.copy(), dx))
        return {"vertices": [0, 0, 0], "faces": [0, 0, 0]}

    clips = load_review_clips(tmp_path, 18056, mesh)
    assert len(calls) == 1
    for branch in ("source", "codec"):
        clip = clips[f"review_first_{branch}"]
        np.testing.assert_array_equal(clip["roots"], data[f"roots_{branch}"])
        np.testing.assert_array_equal(clip["local"], data[f"local_{branch}"])
        assert clip["role"] == branch
        assert "not live" in clip["task"]
        assert clip["frame_stride_floats"] == 3 + 18056 * 3
    np.testing.assert_array_equal(calls[0][0], data["hf"])
    np.testing.assert_array_equal(calls[0][1], data["min_point"])


@pytest.mark.parametrize("defect", ["hash", "path", "coordinate_frame", "duplicate", "rotation"])
def test_invalid_review_is_rejected(tmp_path, defect):
    manifest, data = make_review(tmp_path)
    if defect == "hash":
        manifest["examples"][0]["sha256"] = "0" * 64
    elif defect == "path":
        manifest["examples"][0]["file"] = "../example.npz"
    elif defect == "coordinate_frame":
        manifest["coordinate_frame"] = "canonical"
    elif defect == "duplicate":
        manifest["examples"] *= 2
    else:
        data["local_codec"][0, 1, 0, 0] = 2
        path = tmp_path / "example.npz"
        np.savez_compressed(path, **data)
        manifest["examples"][0]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "review_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_review_clips(tmp_path, 9, lambda *args: {})


def test_heightfield_inverse_preserves_axis_origin_and_vertical_offset():
    from scripts.build_parc_native_dataset import parc_heightfield_to_native

    hf = np.arange(12, dtype=np.float32).reshape(3, 4) / 10
    origin = np.array([4.2, -3.7])
    native_hf, native_origin, dx = parc_heightfield_to_native(hf, origin, .2)
    recovered, recovered_origin = native_terrain_to_parc(native_hf, native_origin, dx)
    np.testing.assert_allclose(recovered, hf, atol=1e-6)
    np.testing.assert_allclose(recovered_origin, origin, atol=1e-6)
    native_origin[1] = .7
    elevated, _ = native_terrain_to_parc(native_hf, native_origin, dx)
    np.testing.assert_allclose(elevated, hf + .7, atol=1e-6)
