"""Focused tests for the PARC long-course render helpers."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_parc_long_course.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("parc_long_course_render", SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def _assert_exact_rectangle_coverage(heightfield, rectangles):
    coverage = np.zeros(heightfield.shape, dtype=np.int32)
    for rectangle in rectangles:
        cells = np.s_[rectangle.i0:rectangle.i1, rectangle.j0:rectangle.j1]
        assert rectangle.i0 < rectangle.i1
        assert rectangle.j0 < rectangle.j1
        assert np.all(heightfield[cells] == rectangle.height)
        coverage[cells] += 1
    np.testing.assert_array_equal(coverage, np.ones(heightfield.shape, dtype=np.int32))


def test_rectangles_cover_every_cell_once_at_the_original_height():
    heightfield = np.asarray([
        [0.0, 0.0, 0.0, 0.2, 0.2],
        [0.0, 0.0, 0.0, 0.2, 0.2],
        [0.0, 0.1, 0.1, 0.2, 0.2],
        [0.0, 0.1, 0.1, 0.0, 0.0],
    ], dtype=np.float32)

    rectangles = renderer.merge_heightfield_rectangles(heightfield)

    assert len(rectangles) < heightfield.size
    _assert_exact_rectangle_coverage(heightfield, rectangles)


def test_rectangles_preserve_cell_center_footprint_bounds():
    heightfield = np.zeros((3, 4), dtype=np.float32)
    rectangle = renderer.merge_heightfield_rectangles(heightfield)[0]
    origin = np.asarray((2.5, -7.0))
    dx = 0.2

    center, half_size = renderer.terrain_rectangle_xy_geometry(rectangle, origin, dx)

    np.testing.assert_allclose(center - half_size, origin - dx / 2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        center + half_size,
        origin + (np.asarray(heightfield.shape) - 0.5) * dx,
        atol=1e-12,
        rtol=0,
    )


def test_random_heightfields_have_exact_nonoverlapping_rectangular_coverage():
    random = np.random.default_rng(12)
    heightfield = random.choice(np.asarray((0.0, 0.12, 0.24)), size=(31, 27))

    rectangles = renderer.merge_heightfield_rectangles(heightfield)

    _assert_exact_rectangle_coverage(heightfield, rectangles)


def test_video_duration_uses_all_frames_at_source_fps():
    assert renderer.video_duration_seconds(300, 30) == pytest.approx(10.0)
    assert renderer.video_duration_seconds(1, 30) == pytest.approx(1 / 30)


@pytest.mark.parametrize(("frame_count", "fps"), [(0, 30), (10, 0), (4, float("nan"))])
def test_video_duration_rejects_invalid_stream_timing(frame_count, fps):
    with pytest.raises(ValueError):
        renderer.video_duration_seconds(frame_count, fps)


def test_gif_preview_is_bounded_and_keeps_cumulative_source_timing():
    assert renderer.gif_preview_frame_count(1200, 30) == 300
    assert renderer.gif_preview_frame_count(100, 15) == 100
    durations = renderer.gif_durations_ms(300, 30)
    assert len(durations) == 300
    assert set(durations) == {30, 40}
    assert sum(durations) == 10_000


def test_imageio_ffmpeg_receives_fps_by_keyword_after_pixel_format(monkeypatch, tmp_path):
    captured = {}

    def write_frames(path, size, pix_fmt_in="rgb24", pix_fmt_out="yuv420p", fps=16, **kwargs):
        captured.update(path=path, size=size, pix_fmt_in=pix_fmt_in,
                        pix_fmt_out=pix_fmt_out, fps=fps, kwargs=kwargs)

        def stream():
            yield None

        return stream()

    fake_ffmpeg = types.ModuleType("imageio_ffmpeg")
    fake_ffmpeg.get_ffmpeg_exe = lambda: "test-ffmpeg"
    fake_ffmpeg.write_frames = write_frames
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

    with renderer.open_mp4_writer(tmp_path / "test.mp4", 640, 352, 30):
        pass

    assert captured["size"] == (640, 352)
    assert captured["fps"] == 30
    assert captured["pix_fmt_in"] == "rgb24"
    assert captured["pix_fmt_out"] == "yuv420p"


def _free_body_case():
    model = renderer.mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='probe'><freejoint/>"
        "<geom type='sphere' size='0.1'/></body></worldbody></mujoco>"
    )
    data = renderer.mujoco.MjData(model)
    body_ids = np.asarray([
        renderer.mujoco.mj_name2id(model, renderer.mujoco.mjtObj.mjOBJ_BODY, "probe")
    ], dtype=np.int32)
    addresses = model.jnt_qposadr[model.body_jntadr[body_ids]]
    return model, data, body_ids, addresses


def test_small_quaternion_norm_drift_passes_both_orientation_parity_checks():
    model, data, body_ids, addresses = _free_body_case()
    norm = 1.0000005066632998
    rotations = np.asarray([[norm, 0.0, 0.0, 0.0]], dtype=np.float64)
    original_rotations = rotations.copy()

    raw_matrix = np.empty(9, dtype=np.float64)
    renderer.mujoco.mju_quat2Mat(raw_matrix, rotations[0, [3, 0, 1, 2]])
    unit_matrix = np.empty(9, dtype=np.float64)
    renderer.mujoco.mju_quat2Mat(
        unit_matrix, renderer.normalized_mujoco_quaternion_wxyz(rotations[0])
    )
    assert np.max(np.abs(raw_matrix - unit_matrix)) > 1e-6

    error = renderer.set_frame(
        model, data, body_ids, addresses, np.zeros((1, 3)), rotations,
        np.zeros(1), model.geom_rgba.copy(),
    )
    assert error <= 1e-6
    assert renderer._rotation_parity_error(data, body_ids, rotations) <= 1e-6
    np.testing.assert_array_equal(rotations, original_rotations)


@pytest.mark.parametrize("rotation", [
    np.zeros(4),
    np.asarray([1.00011, 0.0, 0.0, 0.0]),
    np.asarray([np.nan, 0.0, 0.0, 0.0]),
    np.asarray([np.inf, 0.0, 0.0, 0.0]),
])
def test_both_orientation_parity_checks_reject_invalid_quaternion_norms(rotation):
    rotations = np.asarray([rotation], dtype=np.float64)
    with pytest.raises(ValueError, match="quaternion"):
        renderer.set_frame(None, None, (), (), None, rotations, None, None)
    with pytest.raises(ValueError, match="quaternion"):
        renderer._rotation_parity_error(None, np.asarray([1]), rotations)


def test_both_orientation_parity_checks_reject_a_real_orientation_mismatch(monkeypatch):
    model, data, body_ids, addresses = _free_body_case()
    rotations = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    forward = renderer.mujoco.mj_forward

    def corrupt_orientation(model_arg, data_arg):
        forward(model_arg, data_arg)
        data_arg.xmat[body_ids[0]] = np.eye(3).reshape(9)

    monkeypatch.setattr(renderer.mujoco, "mj_forward", corrupt_orientation)
    with pytest.raises(AssertionError, match="orientation"):
        renderer.set_frame(
            model, data, body_ids, addresses, np.zeros((1, 3)), rotations,
            np.zeros(1), model.geom_rgba.copy(),
        )
    with pytest.raises(AssertionError, match="orientation"):
        renderer._rotation_parity_error(data, body_ids, rotations)


@pytest.mark.parametrize("panel_width", [320, 480])
def test_composed_header_and_panel_titles_fit_at_supported_widths(panel_width):
    panel_height = 240
    rgb = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    frame_index, frame_count, fps = 380, 10_000, 30
    frame = renderer.compose_frame(rgb, rgb, frame_index, frame_count, fps)
    draw = ImageDraw.Draw(frame)
    total_width = panel_width * 2

    title = draw.textbbox((18, 8), renderer.HEADER_TITLE, font=renderer.font(22, True))
    status_font = renderer.font(12, True)
    status_1 = draw.textbbox((19, 34), renderer.STATUS_LINE_1, font=status_font)
    status_2_text = renderer.STATUS_LINE_2.format(fps=fps)
    status_2 = draw.textbbox((19, 52), status_2_text, font=status_font)
    clock = f"{frame_index / fps:.2f} / {renderer.video_duration_seconds(frame_count, fps):.2f} s"
    clock_font = renderer.font(13, True)
    clock_bounds = draw.textbbox((0, 0), clock, font=clock_font)
    clock_x = total_width - (clock_bounds[2] - clock_bounds[0]) - 18
    clock_box = draw.textbbox((clock_x, 53), clock, font=clock_font)

    assert frame.size == (total_width, renderer.HEADER_HEIGHT + renderer.PANEL_HEADER_HEIGHT + panel_height)
    assert title[2] <= total_width - 12
    assert title[3] < status_1[1] and status_1[2] <= total_width - 12
    assert status_1[3] < status_2[1] < renderer.HEADER_HEIGHT
    assert status_2[2] + 8 <= clock_box[0]
    assert clock_box[2] <= total_width - 18

    overview_box = draw.textbbox(
        (12, renderer.HEADER_HEIGHT + 7), renderer.OVERVIEW_PANEL_TITLE,
        font=renderer.font(15, True),
    )
    follow_box = draw.textbbox(
        (panel_width + 12, renderer.HEADER_HEIGHT + 8), renderer.FOLLOW_PANEL_TITLE,
        font=renderer.font(13, True),
    )
    assert overview_box[2] <= panel_width - 12
    assert follow_box[2] <= total_width - 12
    assert max(overview_box[3], follow_box[3]) <= renderer.HEADER_HEIGHT + renderer.PANEL_HEADER_HEIGHT
