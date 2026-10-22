from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.soma_primitive_scheduler import NativePrimitiveScheduler


class Backend:
    commit_frames = 8
    horizon_frames = 64

    def __init__(self, planned_frames=60):
        self.history = torch.zeros(1, 4, 5)
        self.representation = SimpleNamespace(fps=60)
        self.published_frames = 0
        self.calls = []
        self.planned_frames = planned_frames

    def predict_plan(self, velocity, heading):
        self.calls.append(self.history.clone())
        start = self.history[0, -1, 0].item()
        motion = torch.arange(start + 1, start + 1 + self.planned_frames)[None, :, None].expand(1, -1, 5).clone()
        return {"planned_motion": motion, "generation_seconds": 0.01}


def test_contiguous_material_frames_and_last_four_history():
    backend = Backend()
    scheduler = NativePrimitiveScheduler(backend, 56)
    chunks = [scheduler.step([[0., 1.]], [0.]) for _ in range(8)]
    assert [result["replanned"] for result in chunks] == [True] + [False] * 6 + [True]
    assert [result["primitive_frame_offset"] for result in chunks] == [0, 8, 16, 24, 32, 40, 48, 0]
    assert torch.equal(torch.cat([result["motion"] for result in chunks], 1)[0, :, 0], torch.arange(1., 65.))
    assert torch.equal(backend.calls[1][0, :, 0], torch.arange(53., 57.))
    assert scheduler.model_calls == 2
    assert chunks[-1]["replan_reason"] == "buffer_low"
    assert chunks[0]["motion"][0, 0, 0] == 1
    assert backend.published_frames == 64


def test_command_change_interrupts_before_unpublished_block():
    backend = Backend()
    scheduler = NativePrimitiveScheduler(backend, 56)
    first = scheduler.step([[0., 1.]], [0.])
    scheduler.step([[0., 1.]], [0.])
    changed = scheduler.step([[0., 0.]], [0.])
    assert changed["replan_reason"] == "command_changed"
    assert changed["start_frame"] == 16
    assert torch.equal(backend.calls[1][0, :, 0], torch.arange(13., 17.))
    assert torch.equal(first["motion"][0, :, 0], torch.arange(1., 9.))
    assert scheduler.step([[0., 0.]], [0.1])["replan_reason"] == "command_changed"


def test_interval_limit_does_not_wait_until_whole_primitive_exhausted():
    scheduler = NativePrimitiveScheduler(Backend(), 16)
    assert scheduler.step([[0., 1.]], [0.])["replan_reason"] == "initial"
    assert scheduler.step([[0., 1.]], [0.])["generation_seconds"] == 0
    assert scheduler.step([[0., 1.]], [0.])["replan_reason"] == "interval_limit"


def test_short_native_primitive_triggers_buffer_low():
    scheduler = NativePrimitiveScheduler(Backend(planned_frames=20), 56)
    scheduler.step([[0., 1.]], [0.])
    scheduler.step([[0., 1.]], [0.])
    assert scheduler.step([[0., 1.]], [0.])["replan_reason"] == "buffer_low"


@pytest.mark.parametrize("interval", [0, 4, 9, 64])
def test_invalid_interval(interval):
    with pytest.raises(ValueError, match="whole published blocks"):
        NativePrimitiveScheduler(Backend(), interval)


def test_invalid_command_rejected_before_generation():
    scheduler = NativePrimitiveScheduler(Backend(), 8)
    with pytest.raises(ValueError, match="finite"):
        scheduler.step([[float("nan"), 0.]], [0.])
    assert scheduler.model_calls == 0


def test_eight_frame_policy_matches_existing_backend_exactly():
    from pathlib import Path

    from motionbricks.data.bones_rollout import SomaLocomotionBackend
    from scripts.portable_bones_control import load_bundle

    path = Path(__file__).parents[1] / "out/soma_control_v3/baseline_backend.pt"
    if not path.exists():
        pytest.skip("Local frozen SOMA bundle is unavailable")
    inference, bundle = load_bundle(path, "cpu")
    native = SomaLocomotionBackend(inference, bundle["initial_history"], argmax=True, commit_frames=8)
    scheduled = NativePrimitiveScheduler(
        SomaLocomotionBackend(inference, bundle["initial_history"], argmax=True, commit_frames=8), 8)
    for velocity, heading in (([0., 0.], 0.), ([0., 1.2], 0.), ([0., 1.2], 0.), ([1., 0.], 1.)):
        expected = native.step([velocity], [heading])
        actual = scheduled.step([velocity], [heading])
        assert torch.equal(actual["motion"], expected["motion"])
        assert torch.equal(scheduled.backend.history, native.history)
        assert actual["start_frame"] == expected["start_frame"]
