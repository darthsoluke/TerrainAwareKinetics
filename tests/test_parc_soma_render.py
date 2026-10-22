from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import render_parc_soma_retarget as renderer


def synthetic_clip(frames=4):
    source = np.zeros((frames, 15, 3), dtype=np.float32)
    source[:, :, 2] = np.linspace(0.0, 1.6, 15)
    source[:, 0, 0] = np.arange(frames, dtype=np.float32) * 0.1
    source[:, 0, 2] = 0.9
    target = np.zeros((frames, 30, 3), dtype=np.float32)
    target[:, :, 1] = np.linspace(0.0, 1.7, 30)
    target[:, 0, 0] = np.arange(frames, dtype=np.float32) * 0.12
    target[:, 0, 1] = 0.88
    soles = np.zeros((frames, 4, 3), dtype=np.float32)
    soles[:, :, 1] = 0.02
    soles[:, :, 0] = np.asarray((-0.1, 0.1, -0.1, 0.1), dtype=np.float32)
    rotations = np.zeros((frames, 15, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    return {
        "source_positions_zup": source,
        "source_rotations_xyzw": rotations,
        "source_names": np.asarray(["pelvis", *[f"body_{i}" for i in range(1, 15)]]),
        "source_contacts": np.zeros((frames, 15), dtype=np.uint8),
        "target_positions_yup": target,
        "target_names": np.asarray(["Hips", *[f"joint_{i}" for i in range(1, 30)]]),
        "target_parents": np.asarray([-1, *range(29)], dtype=np.int64),
        "target_sole_vertices_yup": soles,
        "hf": np.asarray(((0.1, 0.2, 0.25), (0.15, 0.3, 0.4)), dtype=np.float32),
        "min_point": np.asarray((2.0, -1.0), dtype=np.float32),
        "dx": np.asarray(0.25),
        "uniform_scale": np.asarray(0.8),
        "source_fps": np.asarray(30.0),
    }


def write_clip(path, **overrides):
    values = synthetic_clip()
    values.update(overrides)
    np.savez_compressed(path, **values)
    return values


def test_loads_renderer_contract_and_converts_target_coordinates(tmp_path):
    path = tmp_path / "synthetic.npz"
    original = write_clip(path)
    loaded = renderer.load_clip(path)

    assert loaded["source_positions_zup"].shape == (4, 15, 3)
    assert loaded["target_positions_yup"].shape == (4, 30, 3)
    assert loaded["target_parents"].dtype == np.int64
    np.testing.assert_array_equal(loaded["source_positions_zup"], original["source_positions_zup"])
    np.testing.assert_allclose(
        renderer.target_yup_to_zup(np.asarray(((1.0, 2.0, 3.0),))),
        ((1.0, -3.0, 2.0),), atol=0, rtol=0)


def test_rejects_invalid_tree_and_nonunit_source_quaternions(tmp_path):
    path = tmp_path / "invalid.npz"
    values = synthetic_clip()
    values["target_parents"][4] = 4
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="cycle"):
        renderer.load_clip(path)

    values = synthetic_clip()
    values["source_rotations_xyzw"][0, 0] = 0.0
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="unit xyzw"):
        renderer.load_clip(path)


def test_scales_geom_local_dimensions_without_scaling_xml_body_poses():
    root = ET.fromstring(
        '<mujoco><worldbody><body name="pelvis" pos="1 2 3">'
        '<geom type="capsule" pos=".1 -.2 .3" fromto="0 0 0 0 0 1" '
        'size=".05 .4"/></body></worldbody></mujoco>')
    scaled = renderer.scale_primitive_xml(root, 2.0)
    body = scaled.find(".//body")
    geom = scaled.find(".//geom")

    assert body.get("pos") == "1 2 3"
    assert geom.get("pos") == "0.2 -0.4 0.6"
    assert geom.get("fromto") == "0 0 0 0 0 2"
    assert geom.get("size") == "0.1 0.8"
    assert root.find(".//geom").get("size") == ".05 .4"


def test_shared_camera_tracks_both_roots_and_terrain_uses_cell_centers():
    focus = renderer.shared_camera_lookat((2.0, 1.0, 0.9), (4.0, 3.0, 1.1))
    np.testing.assert_allclose(focus, (3.0, 2.0, 1.18), atol=1e-12)
    moved_focus = renderer.shared_camera_lookat((3.0, 1.0, 0.9), (5.0, 3.0, 1.1))
    np.testing.assert_allclose(moved_focus - focus, (1.0, 0.0, 0.0), atol=1e-12)

    values = synthetic_clip()
    source = values["source_positions_zup"] * float(values["uniform_scale"])
    target = renderer.target_yup_to_zup(values["target_positions_yup"])
    soles = renderer.target_yup_to_zup(values["target_sole_vertices_yup"])
    distance = renderer.shared_camera_distance(source, target, soles,
                                               np.zeros(len(source), dtype=np.int64), 0)
    assert distance >= renderer.MIN_CAMERA_DISTANCE

    cells = list(renderer.terrain_cells(values["hf"], values["min_point"], values["dx"]))
    assert len(cells) == values["hf"].size
    x, y, center, halfsize = cells[1]
    assert (x, y) == (0, 1)
    assert center[0] == pytest.approx(2.0)
    assert center[1] == pytest.approx(-0.75)
    assert center[2] + halfsize[2] == pytest.approx(float(values["hf"][x, y]))


def test_streamed_gif_keeps_every_source_frame_and_real_time_schedule(tmp_path):
    from PIL import Image

    durations = renderer._gif_durations(3, 30.0)
    path = tmp_path / "streamed.gif"
    frames = [Image.new("RGB", (24, 16), color) for color in ("red", "green", "blue")]
    with path.open("wb") as stream:
        palette = None
        for frame, duration in zip(frames, durations):
            palette = renderer._write_gif_frame(stream, palette, frame, duration)
        stream.write(b";")

    with Image.open(path) as animation:
        assert animation.n_frames == 3
        observed = []
        for index in range(animation.n_frames):
            animation.seek(index)
            observed.append(animation.info["duration"])
    assert observed == durations == [30, 40, 30]
    assert renderer._frame_tags(1) == {0: ["first", "mid", "last"]}


def test_soma_capsules_follow_joint_distances_and_sole_points_exactly():
    mujoco = pytest.importorskip("mujoco")
    values = synthetic_clip(frames=1)
    positions = renderer.target_yup_to_zup(values["target_positions_yup"][0])
    soles = renderer.target_yup_to_zup(values["target_sole_vertices_yup"][0])
    scene = renderer.make_target_scene(
        values["target_names"].tolist(), values["target_parents"], len(soles),
        values["hf"], values["min_point"], float(values["dx"]), 480, 400)
    (model, state, edges, joint_ids, joint_addresses, link_ids, link_addresses,
     link_geom_ids, sole_ids, sole_addresses) = scene

    errors = renderer.set_target_frame(
        mujoco, model, state, edges, joint_ids, joint_addresses, link_ids,
        link_addresses, link_geom_ids, sole_ids, sole_addresses, positions, soles)
    assert max(errors) < 1e-6
    parent, child = edges[0]
    expected_length = np.linalg.norm(positions[child] - positions[parent]) / 2.0
    assert model.geom_size[link_geom_ids[0], 1] == pytest.approx(expected_length)
    np.testing.assert_allclose(state.xpos[joint_ids], positions, atol=1e-6)
    np.testing.assert_allclose(state.xpos[sole_ids], soles, atol=1e-6)
