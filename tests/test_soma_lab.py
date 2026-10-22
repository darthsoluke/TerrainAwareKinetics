"""Browser transport keeps native motion and every official SOMA influence."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.data.soma_lab import SomaLabSession, pack_geometry_frames, validate_command


def test_binary_transport_preserves_roots_and_all_vertices():
    roots = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    vertices = np.arange(2 * 9 * 3, dtype=np.float32).reshape(2, 9, 3) / 10
    payload = pack_geometry_frames(roots, vertices)
    restored = np.frombuffer(payload, dtype="<f4").reshape(2, 30)
    np.testing.assert_array_equal(restored[:, :3], roots)
    np.testing.assert_array_equal(restored[:, 3:].reshape(2, 9, 3), vertices)


@pytest.mark.parametrize("velocity,heading,request_id", [
    ([4, 0], 0, 1), ([np.nan, 0], 0, 1), ([0, 0], np.inf, 1),
    ([0, 0], 0, True), ([0, 0], 0, 2), ([0], 0, 1),
])
def test_invalid_intent_or_request_order_is_rejected(velocity, heading, request_id):
    with pytest.raises(ValueError):
        validate_command({"velocity_xz": velocity, "heading": heading, "request_id": request_id}, 0)


def test_stationary_and_backward_commands_are_not_reinterpreted():
    assert validate_command({"velocity_xz": [0, -1.2], "heading": .5, "request_id": 1}, 0) == (1, [0, -1.2], .5)
    assert validate_command({"velocity_xz": [0, 0], "heading": .5, "request_id": 2}, 1) == (2, [0, 0], .5)


def test_next_plan_uses_the_last_four_committed_frames():
    future = torch.arange(8 * 6).reshape(1, 8, 6).float()
    previous = torch.zeros(1, 4, 6)
    calls = []
    backend = SimpleNamespace(PUBLISH_FRAMES=8, history=previous, last_style="walk", last_generation={})

    def predict(velocity, heading):
        calls.append((velocity, heading, backend.history.clone()))
        return {"planned_motion": future, "generation_seconds": .02}

    backend.predict_plan = predict
    runtime = SimpleNamespace(
        fps=30, create_backend=lambda mode: backend,
        skin_features=lambda motion: (motion[0, :, :3].numpy(), motion[0, :, 3:].numpy().reshape(8, 1, 3)),
    )
    session = SomaLabSession(runtime)
    metadata, payload = session.plan({"request_id": 1, "velocity_xz": [0, 1.2], "heading": 0})
    assert calls[0][0:2] == ([[0, 1.2]], [0.0])
    torch.testing.assert_close(calls[0][2], previous)
    torch.testing.assert_close(backend.history, future[:, -4:])
    assert metadata["frame_count"] == 8 and metadata["fps"] == 30
    np.testing.assert_array_equal(np.frombuffer(payload, dtype="<f4").reshape(8, 6), future[0].numpy())
    session.plan({"request_id": 2, "velocity_xz": [0, 0], "heading": 0})
    torch.testing.assert_close(calls[1][2], future[:, -4:])


def test_terrain_mode_forwards_intent_and_commits_only_played_motion():
    future = torch.arange(48).reshape(1, 8, 6).float()
    backend = SimpleNamespace(PUBLISH_FRAMES=8, history=torch.zeros(1, 4, 6), last_generation={})
    received = []

    def predict(velocity, heading):
        received.append((velocity, heading))
        return {"planned_motion": future, "generation_seconds": .03}

    backend.predict_plan = predict
    runtime = SimpleNamespace(
        fps=30,
        create_backend=lambda mode: backend if mode == "terrain" else pytest.fail("wrong backend selected"),
        skin_features=lambda motion: (
            motion[0, :, :3].numpy(), motion[0, :, 3:].numpy().reshape(8, 1, 3)),
    )
    session = SomaLabSession(runtime, mode="terrain")
    metadata, payload = session.plan({"mode": "terrain", "request_id": 1,
                                      "velocity_xz": [-.6, 1.1], "heading": .4})
    assert received == [([[-.6, 1.1]], [.4])]
    assert metadata["quality_status"] == "experimental_not_approved"
    assert metadata["display_only"] is True and metadata["actuating"] is False
    packed = np.frombuffer(payload, dtype="<f4").reshape(8, 6)
    np.testing.assert_array_equal(packed[:, :3], future[0, :, :3].numpy())
    assert torch.count_nonzero(backend.history) == 0
    with pytest.raises(ValueError, match="entire published block"):
        session.acknowledge_played({"request_id": 1, "frame_count": 7})
    torch.testing.assert_close(backend.history, torch.zeros_like(backend.history))
    session.acknowledge_played({"request_id": 1, "frame_count": 8})
    torch.testing.assert_close(backend.history, future[:, -4:])


def test_terrain_mode_fails_closed_without_a_loaded_builder():
    from scripts.soma_lab_server import SomaLabRuntime

    runtime = object.__new__(SomaLabRuntime)
    runtime.terrain_backend_builder = None
    with pytest.raises(ValueError, match="matching online assets were not loaded"):
        runtime.create_backend("terrain")

    runtime.fps = 30
    runtime.vertex_count = 1
    runtime.initial_vertices = np.zeros((1, 1, 3), dtype=np.float32)
    runtime.initial_root = np.zeros((1, 3), dtype=np.float32)
    runtime.faces = np.array([[0, 0, 0]], dtype=np.uint32)
    runtime.clips = {}
    config = runtime.config()
    assert config["terrain_mode"]["available"] is False
    assert all(mode["id"] != "terrain" for mode in config["modes"])


def test_diagnostic_metadata_does_not_change_geometry_or_commit_history():
    future = torch.arange(48).reshape(1, 8, 6).float()
    joints = np.arange(8 * 30 * 3).reshape(8, 30, 3).astype(np.float32)
    diagnostic = {"root_stage_world_xyz_m": [[1., 2., 3.]],
                  "decoded_world_xyz_m": [[4., 5., 6.]],
                  "observation_source": "current_played_history"}
    backend = SimpleNamespace(PUBLISH_FRAMES=8, history=torch.zeros(1, 4, 6), last_generation={})
    backend.predict_plan = lambda *_: {"planned_motion": future, "generation_seconds": 0.,
                                      "debug": diagnostic}
    roots = future[0, :, :3].numpy()
    vertices = future[0, :, 3:].numpy().reshape(8, 1, 3)
    runtime = SimpleNamespace(fps=30, create_backend=lambda _: backend,
        skin_features_with_joints=lambda motion, inference: (roots, vertices, joints))
    session = SomaLabSession(runtime, mode="terrain")
    meta, payload = session.plan({"request_id": 1, "velocity_xz": [0., 1.], "heading": 0.})
    assert payload == pack_geometry_frames(roots, vertices)
    assert meta["debug"]["joints_world_xyz_m"] == joints.tolist()
    assert meta["debug"]["root_stage_world_xyz_m"] == diagnostic["root_stage_world_xyz_m"]
    assert meta["debug"]["decoded_world_xyz_m"] == diagnostic["decoded_world_xyz_m"]
    assert "joints_world_xyz_m" not in diagnostic
    assert torch.count_nonzero(backend.history) == 0
    session.acknowledge_played({"request_id": 1, "frame_count": 8})
    torch.testing.assert_close(backend.history, future[:, -4:])


def test_runtime_skeleton_diagnostics_reuse_the_same_inverse_and_skin():
    from scripts.soma_lab_server import SomaLabRuntime

    runtime = object.__new__(SomaLabRuntime)
    features = torch.arange(16).reshape(1, 2, 8).float()
    original = features.clone()
    joints = torch.arange(180).reshape(1, 2, 30, 3).float()
    rotations = torch.eye(3).expand(1, 2, 30, 3, 3)
    calls = []

    def inverse(motion, normalized, return_all):
        calls.append("inverse")
        assert motion is features and not normalized and not return_all
        return {"global_joint_rots": rotations, "posed_joints": joints}

    def skin(rots, positions, rot_is_global):
        calls.append("skin")
        assert rot_is_global
        torch.testing.assert_close(positions, joints[0])
        return positions[:, :3]

    runtime.inference = SimpleNamespace(motion_rep=SimpleNamespace(inverse=inverse))
    runtime.skinner = SimpleNamespace(skin=skin)
    roots, vertices, debug_joints = runtime.skin_features_with_joints(features)
    assert calls == ["inverse", "skin"]
    np.testing.assert_array_equal(debug_joints, joints[0].numpy())
    np.testing.assert_array_equal(roots, features[0, :, :3].numpy())
    np.testing.assert_array_equal(vertices, joints[0, :, :3].numpy())
    torch.testing.assert_close(features, original)


def test_terrain_bundle_rejects_future_or_reference_inputs(monkeypatch):
    import sys

    from scripts.soma_lab_server import load_terrain_live_bundle

    bundle = {
        "initial_history": torch.zeros(1, 4, 5),
        "scene": {},
        "future_pose_reference": torch.zeros(1, 8, 5),
    }
    loader = SimpleNamespace(load_terrain_bundle=lambda *_: (object(), bundle))
    monkeypatch.setitem(sys.modules, "scripts.portable_parc_online_soma", loader)
    with pytest.raises(ValueError, match="reference, future, or clip inputs"):
        load_terrain_live_bundle("unused", "cpu")


def test_live_scene_override_is_paired_and_contains_observed_history_only(monkeypatch, tmp_path):
    import sys
    import json
    from scripts.soma_lab_server import load_terrain_live_bundle
    from motionbricks.data.bones_contract import file_sha256

    scene = {'heightfield_y_m': np.zeros((5, 5), dtype=np.float32),
             'min_point_world_xyz_m': np.zeros(3, dtype=np.float32), 'dx_m': np.asarray(.2)}
    bundle = {'initial_history': torch.zeros(1, 4, 5), 'scene': scene,
              'metadata': {'dataset_manifest_sha256': 'dataset'}}
    monkeypatch.setitem(sys.modules, 'scripts.portable_parc_online_soma',
                       SimpleNamespace(load_terrain_bundle=lambda *_: (object(), bundle)))
    path = tmp_path / 'scene.npz'
    meta = {'schema': 'parc.live_scene.v1', 'dataset_manifest_sha256': 'dataset',
            'history_frames': 4, 'future_motion_included': False}
    def write(history, **extra):
        np.savez(path, **scene, initial_history=history, metadata_json=np.asarray(json.dumps(meta)), **extra)
        path.with_suffix('.json').write_text(json.dumps({'scene_sha256': file_sha256(path)}))
    write(np.ones((1, 4, 5), dtype=np.float32))
    _, loaded, history = load_terrain_live_bundle('unused', 'cpu', path)
    assert torch.equal(history, torch.ones(1, 4, 5)) and set(loaded) == set(scene)
    write(np.zeros((1, 8, 5), dtype=np.float32))
    with pytest.raises(ValueError, match='exactly four'):
        load_terrain_live_bundle('unused', 'cpu', path)
    write(np.zeros((1, 4, 5), dtype=np.float32), future_pose=np.zeros(1))
    with pytest.raises(ValueError, match='four-frame history only'):
        load_terrain_live_bundle('unused', 'cpu', path)
    meta['dataset_manifest_sha256'] = 'different-dataset'
    write(np.zeros((1, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match='observation contracts differ'):
        load_terrain_live_bundle('unused', 'cpu', path)


def test_real_official_skinning_survives_transport_without_losing_influences():
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    from motionbricks.data.soma_skin_reference import load_official_skinner

    skeleton = SOMASkeleton30()
    skinner = load_official_skinner(skeleton)
    local = torch.eye(3).expand(2, 30, 3, 3).clone()
    angle = torch.tensor(.35)
    local[1, skeleton.bone_order_names.index("LeftToeBase")] = torch.tensor([
        [1, 0, 0], [0, torch.cos(angle), -torch.sin(angle)], [0, torch.sin(angle), torch.cos(angle)]])
    roots = torch.tensor([[0, 1, 0], [.1, 1, .2]])
    positions = roots[:, None].expand(-1, 30, -1)
    vertices = skinner.skin(local, positions).numpy()
    recovered = np.frombuffer(pack_geometry_frames(roots.numpy(), vertices), dtype="<f4").reshape(2, -1)
    np.testing.assert_array_equal(recovered[:, 3:].reshape(vertices.shape), vertices)
    assert skinner.lbs_weights.shape[1] > 4


def test_websocket_terrain_history_waits_for_playback_ack_and_reset(tmp_path):
    import asyncio
    from aiohttp import WSMsgType
    from aiohttp.test_utils import TestClient, TestServer
    from scripts.soma_lab_server import create_app

    (tmp_path / "vendor").mkdir()
    future = torch.arange(48).reshape(1, 8, 6).float()
    backends = []

    def create_backend(mode):
        backend = SimpleNamespace(PUBLISH_FRAMES=8, history=torch.zeros(1, 4, 6),
                                  last_style="walk", last_generation={}, mode=mode)
        backend.predict_plan = lambda *args: {"planned_motion": future, "generation_seconds": .01}
        backends.append(backend)
        return backend

    runtime = SimpleNamespace(
        fps=30, clips={}, create_backend=create_backend, config=lambda: {"fps": 30},
        ready=lambda mode="live": {"type": "ready", "fps": 30, "mode": mode},
        skin_features=lambda motion: (motion[0, :, :3].numpy(), motion[0, :, 3:].numpy().reshape(8, 1, 3)),
    )

    async def run():
        async with TestClient(TestServer(create_app(runtime, tmp_path))) as client:
            async with client.ws_connect("/ws") as socket:
                assert (await socket.receive_json())["mode"] == "live"
                await socket.send_json({"type": "reset", "mode": "terrain"})
                assert (await socket.receive_json())["mode"] == "terrain"
                assert backends[1].mode == "terrain"
                await socket.send_json({"type": "plan", "mode": "terrain", "request_id": 2,
                                        "velocity_xz": [0, 0], "heading": 0})
                assert (await socket.receive_json())["type"] == "error"
                assert torch.count_nonzero(backends[1].history) == 0
                await socket.send_json({"type": "plan", "mode": "terrain", "request_id": 1,
                                        "velocity_xz": [0, 1], "heading": 0})
                metadata = await socket.receive_json()
                binary = await socket.receive()
                assert metadata["encoding"] == "float32-le" and metadata["frame_count"] == 8
                assert metadata["mode"] == "terrain"
                assert binary.type == WSMsgType.BINARY
                np.testing.assert_array_equal(np.frombuffer(binary.data, "<f4").reshape(8, 6), future[0].numpy())
                assert torch.count_nonzero(backends[1].history) == 0
                await socket.send_json({"type": "plan", "request_id": 2,
                                        "velocity_xz": [0, 0], "heading": 0, "mode": "terrain"})
                assert "fully played" in (await socket.receive_json())["message"]
                await socket.send_json({"type": "played", "request_id": 1, "frame_count": 8})
                assert (await socket.receive_json())["type"] == "played"
                torch.testing.assert_close(backends[1].history, future[:, -4:])
                await socket.send_json({"type": "plan", "mode": "terrain", "request_id": 2,
                                        "velocity_xz": [0, 0], "heading": 0})
                assert (await socket.receive_json())["type"] == "plan"
                await socket.receive()
                await socket.send_json({"type": "reset", "mode": "terrain"})
                assert (await socket.receive_json())["mode"] == "terrain"
                assert len(backends) == 3 and torch.count_nonzero(backends[2].history) == 0

    asyncio.run(run())


def test_world_reset_selects_session_history_and_preserves_other_sessions(tmp_path):
    import asyncio
    from aiohttp.test_utils import TestClient, TestServer
    from scripts.soma_lab_server import create_app
    (tmp_path/'vendor').mkdir()
    created=[]
    def create_backend(mode,spawn_id=None):
        if spawn_id not in (None,'rocks','stairs'):
            raise ValueError('Unknown terrain entry point')
        backend=SimpleNamespace(history=torch.full((1,4,6),1. if spawn_id=='rocks' else 2.))
        created.append(backend)
        return backend
    runtime=SimpleNamespace(default_mode='terrain',fps=30,clips={},config=lambda:{},
        create_backend=create_backend,
        ready=lambda mode='terrain',spawn_id=None:{'type':'ready','mode':mode,'spawn_id':spawn_id})
    async def run():
        async with TestClient(TestServer(create_app(runtime,tmp_path))) as client:
            async with client.ws_connect('/ws') as socket:
                assert (await socket.receive_json())['mode']=='terrain'
                original=created[0].history.clone()
                await socket.send_json({'type':'reset','mode':'terrain','spawn_id':'rocks'})
                assert (await socket.receive_json())['spawn_id']=='rocks'
                assert torch.all(created[1].history==1)
                torch.testing.assert_close(created[0].history,original)
                await socket.send_json({'type':'reset','mode':'terrain','spawn_id':'missing'})
                assert (await socket.receive_json())['type']=='error'
    asyncio.run(run())
