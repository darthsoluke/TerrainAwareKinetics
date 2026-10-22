"""Regression checks for long-course inference orchestration, not motion quality."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_parc_long_stairs import (chunked_motion_score, passage_evidence,
                                       route_evidence, official_generation_settings, frame_intent_from_calls)
from build_parc_stair_course import build_course


class Frames:
    def __init__(self, values):
        self.root_pos = values

    def get_slice(self, selection):
        return Frames(self.root_pos[:, selection])

    def get_idx(self, selection):
        return Frames(self.root_pos[selection])


def test_pre_generation_commands_follow_candidate_and_retained_frame_axes():
    import numpy as np
    calls = [np.asarray([[i, -i], [100 + i, 200 + i]], dtype=float) for i in range(4)]
    commands = frame_intent_from_calls(calls, 1, 25, 15, 2, 5)
    np.testing.assert_array_equal(commands[:9], np.tile([100, 200], (9, 1)))
    np.testing.assert_array_equal(commands[9:16], np.tile([101, 201], (7, 1)))
    np.testing.assert_array_equal(commands[16:23], np.tile([102, 202], (7, 1)))
    np.testing.assert_array_equal(commands[23:], np.tile([103, 203], (2, 1)))
    # A fully completed batch discards its last diffusion call entirely.
    np.testing.assert_array_equal(frame_intent_from_calls(calls, 1, 23, 15, 2, 5), commands[:23])
    with pytest.raises(ValueError, match='shorter'):
        frame_intent_from_calls(calls, 1, 37, 15, 2, 5)


def test_chunked_score_preserves_all_frames_batch_rows_and_partial_tail():
    frames = Frames(torch.arange(2 * 19 * 3, dtype=torch.float32).reshape(2, 19, 3))

    def score(motion, *args, **kwargs):
        result = motion.root_pos.sum((1, 2))
        return {"total_loss": result, "contact_loss": result * 2, "pen_loss": result * 3}

    actual = chunked_motion_score(score, frames, None, None, None, None, 1, 1, 1, frame_chunk=4)
    for key, expected in score(frames).items():
        torch.testing.assert_close(actual[key], expected)
    with pytest.raises(ValueError):
        chunked_motion_score(score, frames, None, None, None, None, 1, 1, 1, frame_chunk=0)


def test_batch_chunking_preserves_candidate_identity_and_tail():
    frames = Frames(torch.arange(19 * 17 * 3, dtype=torch.float32).reshape(19, 17, 3))
    def score(motion, *args, **kwargs):
        assert motion.root_pos.shape[0] <= 8
        return {'total_loss': motion.root_pos.sum((1, 2))}
    result = chunked_motion_score(score, frames, None, None, None, None, 1, 1, 1,
                                  frame_chunk=4, batch_chunk=8)
    torch.testing.assert_close(result['total_loss'], frames.root_pos.sum((1, 2)))


def test_course_passage_rejects_endpoint_shortcut_and_requires_order():
    terrain, nodes, course = build_course()
    endpoints = nodes[[0, -1]].clone()
    endpoints[:, 2] += .08
    assert not passage_evidence(endpoints, terrain, course)["all_summits_visited_in_order"]
    route = nodes.clone()
    route[:, 2] += .08
    assert passage_evidence(route, terrain, course)["all_summits_visited_in_order"]
    assert not passage_evidence(route.flip(0), terrain, course)["all_summits_visited_in_order"]


def test_official_score_chunk_parity():
    terrain, _, _ = build_course()
    from parc.anim.kin_char_model import KinCharModel
    from parc.motion_synthesis.procgen.mdm_path import compute_motion_loss
    from parc.util import geom_util, motion_util, terrain_util

    root = Path(__file__).resolve().parents[1]
    char = KinCharModel(torch.device("cpu"))
    char.load_char_file(str(root / "out/vendor/PARC/data/assets/humanoid.xml"))
    terrain = terrain_util.SubTerrain("score_test", x_dim=8, y_dim=8, dx=.4, dy=.4,
                                      min_x=-1.6, min_y=-1.6, device="cpu")
    frames = motion_util.MotionFrames()
    frames.init_blank_frames(char, 7, batch_size=2)
    frames.root_pos[..., 2] = .8
    frames.root_pos[1, :, 0] = .2
    points = geom_util.get_char_point_samples(char)
    args = (frames, None, terrain, char, points, .1, .1, 1.)
    with torch.no_grad():
        expected = compute_motion_loss(*args, verbose=False)
        actual = chunked_motion_score(compute_motion_loss, *args, frame_chunk=3)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)


def test_route_milestones_reject_isolated_summits_and_skipped_descent():
    _, nodes, course = build_course()
    dense = torch.repeat_interleave(nodes, 2, dim=0)
    dense[:, 2] += .08
    assert route_evidence(dense, nodes, course)["all_segments_covered_in_order"]
    summits = [0] + [f["path_segments"]["ascent"]["path_node_end"] for f in course["stair_flights"]] + [-1]
    assert not route_evidence(nodes[summits], nodes, course)["all_segments_covered_in_order"]
    assert not route_evidence(dense.flip(0), nodes, course)["all_segments_covered_in_order"]
    assert not route_evidence(dense + torch.tensor([0., 0., 1.]), nodes, course)["all_segments_covered_in_order"]


def test_sampling_uses_released_generation_settings_not_training_test_stride():
    class Settings:
        pass
    repo = Path(__file__).resolve().parents[1] / "out/vendor/PARC"
    path, gen, _ = official_generation_settings(repo, Settings, Settings, 8, 45.)
    assert gen.ddim_stride == 5
    assert gen.use_cfg and gen.use_ddim and gen.use_prev_state
    assert gen.cfg_scale == .65
    assert path.mdm_batch_size == 8 and path.max_motion_length == 45.
    assert path.rewind_num_frames == 5 and path.next_node_lookahead == 7


def test_loader_selects_ema_weights_and_never_silently_falls_back(tmp_path, monkeypatch):
    from run_parc_local_terrain_pilot import load_official_model
    build_course()
    from parc.motion_generator import mdm

    class TinyMDM:
        def __init__(self, cfg):
            self._denoise_model = torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(mdm, "MDM", TinyMDM)
    file = tmp_path / "weights.ckpt"
    torch.save({"cfg": {}, "model": {"weight": torch.tensor([[1.]])},
                "ema_model": {"weight": torch.tensor([[7.]])}}, file)
    assert load_official_model(tmp_path, file, "cpu", "ema_model")._denoise_model.weight.item() == 7.
    assert load_official_model(tmp_path, file, "cpu")._denoise_model.weight.item() == 1.
    torch.save({"cfg": {}, "model": {"weight": torch.tensor([[1.]])}}, file)
    with pytest.raises(ValueError, match="unavailable"):
        load_official_model(tmp_path, file, "cpu", "ema_model")
