from types import SimpleNamespace

import pytest
import torch
from PIL import Image, ImageChops

from scripts import evaluate_sole_region_ab as evaluator


def render_fixture(monkeypatch, frames=4):
    names = ("source", "A0", "B", "A")
    points = torch.tensor([[0.0, 0.9, 0.0], [0.0, 1.4, 0.0], [0.1, 0.2, 0.0]])
    points = points[None].repeat(frames, 1, 1)
    if frames:
        points[:, :, 0] += torch.arange(frames)[:, None] * 0.01
    motions = {name: (torch.zeros(frames, 30, 3, 3), torch.zeros(frames, 3), points)
               for name in names}
    vertices = {name: {side: points[:, :2] for side in ("Left", "Right")} for name in names}
    monkeypatch.setattr(evaluator, "RENDER_VERTICES", vertices)
    monkeypatch.setattr(evaluator, "RENDER_LIMITS", ((-0.2, 0.3), (-0.1, 1.5)))
    monkeypatch.setattr(evaluator, "region_geometry_from_motion",
                        lambda rots, *args: (torch.zeros(rots.shape[0], 2),
                                            torch.zeros(rots.shape[0], 2, 2)))
    monkeypatch.setattr(evaluator.shutil, "which", lambda name: None)
    skeleton = SimpleNamespace(joint_parents=torch.tensor([-1, 0, 1]))
    return motions, {"Left": [0, 1], "Right": [0, 1]}, {"skeleton": skeleton}


def test_four_arm_animation_has_all_frames_and_changes(monkeypatch, tmp_path):
    motions, indices, joints = render_fixture(monkeypatch)
    result = evaluator.render(tmp_path, "walk", motions, indices, {}, 60.0, joints)
    assert result["gif"] == "walk_source_A0_B_A.gif"
    with Image.open(tmp_path / result["gif"]) as animation:
        assert animation.n_frames == 4
        first = animation.convert("RGB")
        animation.seek(3)
        assert ImageChops.difference(first, animation.convert("RGB")).getbbox() is not None


def test_empty_animation_is_rejected_before_writer(monkeypatch, tmp_path):
    motions, indices, joints = render_fixture(monkeypatch, frames=0)
    with pytest.raises(ValueError, match="nonempty frame axis"):
        evaluator.render(tmp_path, "walk", motions, indices, {}, 60.0, joints)
    assert not list(tmp_path.glob("*.gif"))


def test_draw_failure_preserves_original_error(monkeypatch, tmp_path):
    motions, indices, joints = render_fixture(monkeypatch)
    evaluator.RENDER_VERTICES.pop("A0")
    with pytest.raises(KeyError, match="A0"):
        evaluator.render(tmp_path, "walk", motions, indices, {}, 60.0, joints)
    assert not list(tmp_path.glob("*.gif"))


def test_all_window_plan_has_no_training_or_test_leakage():
    records = {
        "train": {"split": "train", "frames_60hz": 400, "behavior": "walk"},
        "test": {"split": "test", "frames_60hz": 400, "behavior": "run"},
        "val": {"split": "validation", "frames_60hz": 400, "behavior": "turn"},
    }
    planned = evaluator.evaluation_windows(records, True)
    assert len(planned) == 3
    assert all(name == "val" for _, name, _ in planned)
    assert [start for _, _, start in planned] == [0, 135, 271]


def test_slip_aggregate_is_count_weighted_not_a_claim_of_pooled_p95():
    def row(speed, count):
        return {"mpjpe_m": 0.01, "region_penetration_m": {"mean": 0.001, "peak": 0.004},
                "measured_slip_mps": {"mean": speed, "p95": speed + 1, "peak": speed + 2},
                "valid_samples": {"measured_stable_intervals": count}}
    result = evaluator.aggregate_region_results({"a": {"arms": {"B": row(1.0, 1)}},
                                                 "b": {"arms": {"B": row(3.0, 3)}}})["B"]
    assert result["measured_slip_count_weighted_mean_mps"] == 2.5
    assert result["measured_slip_peak_mps"] == 5.0
    assert result["mean_window_slip_p95_mps"] == 3.0
