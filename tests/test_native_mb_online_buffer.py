from types import SimpleNamespace

import torch

from motionbricks.data import native_mb_online_live as live


def test_generated_buffer_is_reused_only_after_actual_playback(monkeypatch):
    inference = SimpleNamespace(device='cpu', global_motion_rep=SimpleNamespace(indices={'root': [0, 1, 2, 3, 4]}))
    history = torch.zeros(1, 4, 5)
    history[..., 3] = 1
    calls = []

    def generate(_inference, observed, _goal, _scene, **kwargs):
        calls.append(observed.clone())
        result = torch.arange(32 * 5, dtype=torch.float32).reshape(1, 32, 5) + len(calls) * 1000
        return result, torch.tensor([[8]])

    monkeypatch.setattr(live, 'predict_planar_terrain', generate)
    backend = live.NativeTerrainOnlineBackend(inference, history, {})
    first = backend.predict_plan([[0, 1]], [0])['planned_motion']
    torch.testing.assert_close(backend.history, history)
    # Publishing is not observing: without a played-history commit, regenerate.
    backend.predict_plan([[0, 1]], [0])
    assert len(calls) == 2
    backend = live.NativeTerrainOnlineBackend(inference, history, {})
    first = backend.predict_plan([[0, 1]], [0])['planned_motion']
    backend.history = first[:, -4:].clone()
    second = backend.predict_plan([[0, 1]], [0])['planned_motion']
    assert len(calls) == 3 and not backend.last_generation['neural_generation']
    expected = torch.arange(32 * 5, dtype=torch.float32).reshape(1, 32, 5) + 3000
    torch.testing.assert_close(first, expected[:, 4:12])
    torch.testing.assert_close(second, expected[:, 12:20])
    backend.history = second[:, -4:].clone()
    changed = backend.predict_plan([[1, 0]], [1])['planned_motion']
    assert len(calls) == 4 and backend.last_generation['neural_generation']
    torch.testing.assert_close(calls[-1], second[:, -4:])
    assert not torch.equal(changed, expected[:, 20:28])
