from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.data import native_mb_online_live as live
from motionbricks.data.soma_lab import SomaLabSession, pack_geometry_frames


class _Inference:
    device = "cpu"
    global_motion_rep = SimpleNamespace(indices={"all": list(range(5)), "root": list(range(5))})


def _history(offset=0):
    history = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5) + offset
    history[..., 3] = 1
    history[..., 4] = 0
    return history


def _runtime(monkeypatch, *, fail_generation=None):
    inference = _Inference()
    calls = []
    predictions = []

    def generate(_inference, observed, _goal, _scene, **_kwargs):
        calls.append(observed.detach().clone())
        if fail_generation == len(calls):
            raise RuntimeError("fake generation failure")
        prediction = torch.arange(32 * 5, dtype=torch.float32).reshape(1, 32, 5)
        prediction = prediction + len(calls) * 1000
        predictions.append(prediction)
        _kwargs["diagnostics"].update({
            "root_stage_world_xyz_m": [[float(i), 0., 0.] for i in range(24)],
            "decoded_world_xyz_m": [[float(i), 1., 0.] for i in range(24)],
            "observation_root_world_xyz_m": observed[0, -1, :3].tolist(),
            "observation_source": "current_played_history",
            "observation_snapshot_note": "generation-time snapshot",
            "terrain_patches": [{"name": "immutable snapshot"}],
        })
        return prediction, torch.tensor([[8]])

    monkeypatch.setattr(live, "predict_planar_terrain", generate)
    initial = _history(-10)
    backends = []

    def create_backend(mode):
        assert mode == "terrain"
        backend = live.NativeTerrainOnlineBackend(inference, initial, {})
        backends.append(backend)
        return backend

    skin_calls = []

    def skin_features_with_joints(motion, received_inference):
        assert received_inference is inference
        skin_calls.append(motion.detach().clone())
        roots = motion[0, :, :3].numpy()
        vertices = (motion[0, :, :3] + motion.new_tensor([10., 20., 30.])).numpy()[:, None]
        joints = vertices.copy()
        return roots, vertices, joints

    runtime = SimpleNamespace(
        fps=30, create_backend=create_backend,
        skin_features_with_joints=skin_features_with_joints,
    )
    return runtime, calls, predictions, backends, skin_calls


def _plan(session, request_id=1, velocity=(0., 1.)):
    return session.plan({
        "type": "plan", "mode": "terrain", "request_id": request_id,
        "velocity_xz": list(velocity), "heading": 0.,
    })


def _prefetch(session, request_id=2, source_request_id=1, played_frame_count=4,
              velocity=(0., 1.)):
    return session.prefetch({
        "type": "prefetch", "mode": "terrain", "request_id": request_id,
        "source_request_id": source_request_id,
        "played_frame_count": played_frame_count,
        "velocity_xz": list(velocity), "heading": 0.,
    })


def test_zero_played_prefetch_does_not_reencode_or_advance_observed_features(monkeypatch):
    runtime,_,_,_,_=_runtime(monkeypatch)
    session=SomaLabSession(runtime,mode='terrain')
    _plan(session)
    original=session.backend.history.clone()
    session.backend.committed_history_prefetch=True
    session.backend.observe_history=lambda *args:pytest.fail('Zero new frames must preserve existing observed features')
    session.backend.prepare_next_plan=lambda *args,**kwargs:{'deferred':True}
    metadata,payload=_prefetch(session,played_frame_count=0)
    assert metadata['type']=='prefetch_deferred' and payload is None
    torch.testing.assert_close(session.backend.history,original,atol=0,rtol=0)


def test_combined_played_and_plan_uses_only_acknowledged_history(monkeypatch, tmp_path):
    import asyncio
    from aiohttp import WSMsgType
    from aiohttp.test_utils import TestClient, TestServer
    from scripts.soma_lab_server import create_app

    runtime, calls, predictions, backends, _ = _runtime(monkeypatch)
    create = runtime.create_backend
    runtime.create_backend = lambda mode: create('terrain')
    runtime.clips = {}
    runtime.config = lambda: {}
    runtime.ready = lambda mode='live': {'type': 'ready', 'mode': mode}
    (tmp_path / 'vendor').mkdir()

    async def run():
        async with TestClient(TestServer(create_app(runtime, tmp_path))) as client:
            async with client.ws_connect('/ws') as ws:
                await ws.receive_json()
                await ws.send_json({'type': 'reset', 'mode': 'terrain'})
                await ws.receive_json()
                await ws.send_json({'type': 'plan', 'mode': 'terrain', 'request_id': 1,
                                    'velocity_xz': [0., 1.], 'heading': 0.})
                await ws.receive_json()
                assert (await ws.receive()).type == WSMsgType.BINARY
                follow = {'type': 'plan', 'mode': 'terrain', 'request_id': 2,
                          'velocity_xz': [1., 0.], 'heading': 0.}
                await ws.send_json({'type': 'played', 'request_id': 1, 'frame_count': 4, 'next_plan': follow})
                assert (await ws.receive_json())['type'] == 'error'
                assert len(calls) == 1
                await ws.send_json({'type': 'played', 'request_id': 1, 'frame_count': 8, 'next_plan': follow})
                assert (await ws.receive_json())['type'] == 'played'
                assert (await ws.receive_json())['request_id'] == 2
                assert (await ws.receive()).type == WSMsgType.BINARY
                assert len(calls) == 2
                torch.testing.assert_close(calls[1], predictions[0][:, 8:12])

    asyncio.run(run())


@pytest.mark.parametrize('played_frames', [2, 4])
def test_changed_command_defers_fresh_inference_until_actual_played_boundary(monkeypatch, played_frames):
    runtime, calls, predictions, backends, skin_calls = _runtime(monkeypatch)
    session = SomaLabSession(runtime, mode="terrain")
    original_metadata, _ = _plan(session)
    backend = backends[0]
    current = session.pending_playback["motion"].clone()
    old_buffer = backend._future_buffer.clone()
    old_published_history = backend._last_published_history.clone()

    metadata, payload = _prefetch(session, velocity=(1., 0.), played_frame_count=played_frames)

    assert len(calls) == 1 and len(skin_calls) == 1
    assert payload is None and metadata['type'] == 'prefetch_deferred'
    actual_prefix = torch.cat((session.pending_playback['prior_history'], current[:, :played_frames]), 1)[:, -4:]
    torch.testing.assert_close(backend.history, actual_prefix)
    torch.testing.assert_close(session.pending_playback["motion"], current)
    torch.testing.assert_close(backend._future_buffer, old_buffer)
    torch.testing.assert_close(backend._last_published_history, old_published_history)
    assert session.request_id == 2
    assert metadata["request_id"] == 2 and metadata["source_request_id"] == 1
    assert session.pending_playback.get('prepared') is None
    assert original_metadata["request_id"] == 1
    with pytest.raises(ValueError, match='fully played'):
        _plan(session, request_id=3, velocity=(1., 0.))
    with pytest.raises(ValueError, match="entire published block"):
        session.acknowledge_played({"request_id": 1, "frame_count": 4})
    torch.testing.assert_close(backend._future_buffer, old_buffer)
    assert session.pending_playback["request_id"] == 1

    session.acknowledge_played({"request_id": 1, "frame_count": 8})
    assert session.pending_playback is None
    torch.testing.assert_close(backend.history, current[:, -4:])
    next_metadata, _ = _plan(session, request_id=3, velocity=(1., 0.))
    assert next_metadata["request_id"] == 3
    assert len(calls) == 2
    torch.testing.assert_close(calls[-1], current[:, -4:])
    torch.testing.assert_close(session.pending_playback['motion'], predictions[1][:, 4:12])


def test_same_command_prefetch_uses_cached_next_eight_without_neural_generation(monkeypatch):
    runtime, calls, predictions, backends, _ = _runtime(monkeypatch)
    session = SomaLabSession(runtime, mode="terrain")
    _plan(session)
    backend = backends[0]
    current = session.pending_playback["motion"].clone()
    original_snapshot = backend._debug_future["observation_root_world_xyz_m"]
    original_buffer = backend._future_buffer.clone()

    metadata, payload = _prefetch(session)

    expected = predictions[0][:, 12:20]
    assert len(calls) == 1
    assert payload == pack_geometry_frames(
        expected[0, :, :3].numpy(),
        (expected[0, :, :3] + expected.new_tensor([10., 20., 30.])).numpy()[:, None],
    )
    torch.testing.assert_close(session.pending_playback["motion"], current)
    torch.testing.assert_close(backend.history, current[:, :4])
    torch.testing.assert_close(backend._future_buffer, original_buffer)
    assert metadata["generation"]["neural_generation"] is False
    assert "forecast_alignment_lag_frames" not in metadata
    assert metadata["debug"]["observation_root_world_xyz_m"] == original_snapshot
    assert metadata["debug"]["observation_snapshot_note"] == "generation-time snapshot"
    assert metadata["debug"]["root_stage_world_xyz_m"][0] == [8., 0., 0.]

    session.acknowledge_played({"request_id": 1, "frame_count": 8})
    torch.testing.assert_close(backend._future_buffer, predictions[0][:, 20:28])
    torch.testing.assert_close(backend.history, current[:, -4:])
    assert session.pending_playback["request_id"] == 2
    session.acknowledge_played({"request_id": 2, "frame_count": 8})
    torch.testing.assert_close(backend.history, expected[:, -4:])
    assert session.pending_playback is None


def test_two_played_frames_keep_previous_two_observations_not_unplayed_future(monkeypatch):
    runtime, calls, predictions, backends, _ = _runtime(monkeypatch)
    session = SomaLabSession(runtime, mode='terrain')
    initial = session.backend.history.clone()
    _plan(session)
    motion = session.pending_playback['motion'].clone()
    metadata, _ = _prefetch(session, played_frame_count=2, velocity=(1., 0.))
    expected_observed = torch.cat((initial[:, -2:], motion[:, :2]), dim=1)
    assert len(calls) == 1
    torch.testing.assert_close(backends[-1].history, expected_observed)
    assert metadata['type'] == 'prefetch_deferred'
    session.acknowledge_played({'request_id': 1, 'frame_count': 8})
    assert session.pending_playback is None
    torch.testing.assert_close(session.backend.history, motion[:, -4:])


def test_invalid_midpoint_receipts_do_not_advance_history_or_request_counter(monkeypatch):
    runtime, calls, _, backends, _ = _runtime(monkeypatch)
    session = SomaLabSession(runtime, mode="terrain")
    _plan(session)
    backend = backends[0]
    initial_history = backend.history.clone()
    original_buffer = backend._future_buffer.clone()

    invalid = [
        {"type": "prefetch", "mode": "terrain", "request_id": 2,
         "source_request_id": 99, "played_frame_count": 4,
         "velocity_xz": [0., 1.], "heading": 0.},
        {"type": "prefetch", "mode": "terrain", "request_id": 2,
         "source_request_id": 1, "played_frame_count": 3,
         "velocity_xz": [0., 1.], "heading": 0.},
        {"type": "prefetch", "mode": "terrain", "request_id": 3,
         "source_request_id": 1, "played_frame_count": 4,
         "velocity_xz": [0., 1.], "heading": 0.},
    ]
    for message in invalid:
        with pytest.raises(ValueError):
            session.prefetch(message)
        torch.testing.assert_close(backend.history, initial_history)
        torch.testing.assert_close(backend._future_buffer, original_buffer)
        assert session.request_id == 1
        assert session.pending_playback.get("prepared") is None
    assert len(calls) == 1

    _prefetch(session)
    prepared_motion = session.pending_playback["prepared"]["motion"].clone()
    with pytest.raises(ValueError, match="already awaiting"):
        _prefetch(session)
    with pytest.raises(ValueError, match="does not match"):
        session.acknowledge_played({"request_id": 2, "frame_count": 8})
    with pytest.raises(ValueError, match="entire published block"):
        session.acknowledge_played({"request_id": 1, "frame_count": 7})
    torch.testing.assert_close(session.pending_playback["prepared"]["motion"], prepared_motion)
    assert session.request_id == 2


def test_failed_boundary_generation_keeps_truthful_fully_played_history(monkeypatch):
    runtime, calls, _, backends, _ = _runtime(monkeypatch, fail_generation=2)
    session = SomaLabSession(runtime, mode="terrain")
    _plan(session)
    backend = backends[0]
    current = session.pending_playback["motion"].clone()
    original_buffer = backend._future_buffer.clone()
    original_published_history = backend._last_published_history.clone()

    metadata, payload = _prefetch(session, velocity=(1., 0.))
    assert metadata['type'] == 'prefetch_deferred' and payload is None
    assert len(calls) == 1
    torch.testing.assert_close(backend.history, current[:, :4])
    torch.testing.assert_close(backend._future_buffer, original_buffer)
    torch.testing.assert_close(backend._last_published_history, original_published_history)
    assert session.request_id == 2
    assert session.pending_playback.get("prepared") is None
    session.acknowledge_played({'request_id': 1, 'frame_count': 8})
    with pytest.raises(RuntimeError, match="fake generation failure"):
        _plan(session, request_id=3, velocity=(1., 0.))
    assert len(calls) == 2
    torch.testing.assert_close(calls[-1], current[:, -4:])
    torch.testing.assert_close(backend.history, current[:, -4:])
    assert session.pending_playback is None and session.request_id == 2


def test_exhausted_buffer_defers_even_when_command_is_unchanged(monkeypatch):
    runtime, calls, _, backends, _ = _runtime(monkeypatch)
    session = SomaLabSession(runtime, mode='terrain')
    _plan(session)
    backends[-1]._future_buffer = backends[-1]._future_buffer[:, :0]
    metadata, payload = _prefetch(session)
    assert metadata['type'] == 'prefetch_deferred' and payload is None
    assert len(calls) == 1 and session.pending_playback.get('prepared') is None


def test_pause_without_midpoint_receipt_and_new_session_reset_discard_prefetch_state(monkeypatch):
    runtime, calls, _, backends, _ = _runtime(monkeypatch)
    session = SomaLabSession(runtime, mode="terrain")
    _plan(session)
    old_backend = backends[0]
    old_history = old_backend.history.clone()
    old_motion = session.pending_playback["motion"].clone()

    # Pausing before the midpoint sends no receipt, so no generated motion becomes observed.
    torch.testing.assert_close(old_backend.history, old_history)
    assert session.pending_playback.get("prepared") is None
    assert session.request_id == 1 and len(calls) == 1

    reset_session = SomaLabSession(runtime, mode="terrain")
    assert reset_session.backend is not old_backend
    assert reset_session.request_id == 0
    assert reset_session.pending_playback is None
    torch.testing.assert_close(reset_session.backend.history, _history(-10))
    torch.testing.assert_close(session.pending_playback["motion"], old_motion)


def test_websocket_prefetch_promotes_only_after_full_ack_and_reset_discards_it(monkeypatch, tmp_path):
    import asyncio
    from aiohttp import WSMsgType
    from aiohttp.test_utils import TestClient, TestServer
    from scripts.soma_lab_server import create_app

    runtime, calls, _, backends, _ = _runtime(monkeypatch)
    original_create = runtime.create_backend
    runtime.create_backend = lambda mode: original_create('terrain')
    runtime.clips = {}
    runtime.config = lambda: {}
    runtime.ready = lambda mode='live': {'type': 'ready', 'mode': mode}
    (tmp_path / 'vendor').mkdir()

    async def run():
        async with TestClient(TestServer(create_app(runtime, tmp_path))) as client:
            async with client.ws_connect('/ws') as ws:
                await ws.receive_json()
                await ws.send_json({'type': 'reset', 'mode': 'terrain'})
                await ws.receive_json()
                await ws.send_json({'type': 'plan', 'mode': 'terrain', 'request_id': 1,
                                    'velocity_xz': [0., 1.], 'heading': 0.})
                await ws.receive_json()
                assert (await ws.receive()).type == WSMsgType.BINARY
                original = backends[-1]._last_published_history.clone()
                await ws.send_json({'type': 'prefetch', 'mode': 'terrain', 'request_id': 2,
                    'source_request_id': 1, 'played_frame_count': 4, 'velocity_xz': [0., 1.], 'heading': 0.})
                metadata = await ws.receive_json()
                assert metadata['prefetched'] and metadata['request_id'] == 2
                assert (await ws.receive()).type == WSMsgType.BINARY
                assert not torch.equal(backends[-1].history, original)
                await ws.send_json({'type': 'played', 'request_id': 1, 'frame_count': 8})
                assert (await ws.receive_json())['type'] == 'played'
                torch.testing.assert_close(backends[-1].history, original)
                await ws.send_json({'type': 'prefetch', 'mode': 'terrain', 'request_id': 3,
                    'source_request_id': 2, 'played_frame_count': 2, 'velocity_xz': [1., 0.], 'heading': 0.})
                deferred = await ws.receive_json()
                assert deferred['type'] == 'prefetch_deferred' and deferred['request_id'] == 3
                await ws.send_json({'type': 'played', 'request_id': 2, 'frame_count': 8})
                # A deferral sends no binary packet; the next response is the ACK.
                assert (await ws.receive_json())['type'] == 'played'
                actual = backends[-1].history.clone()
                await ws.send_json({'type': 'plan', 'mode': 'terrain', 'request_id': 4,
                    'velocity_xz': [1., 0.], 'heading': 0.})
                assert (await ws.receive_json())['type'] == 'plan'
                assert (await ws.receive()).type == WSMsgType.BINARY
                torch.testing.assert_close(calls[-1], actual)
                await ws.send_json({'type': 'reset', 'mode': 'terrain'})
                assert (await ws.receive_json())['type'] == 'ready'
                assert backends[-1]._future_buffer is None
                torch.testing.assert_close(backends[-1].history, _history(-10))

    asyncio.run(run())
