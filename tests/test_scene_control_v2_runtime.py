import asyncio
from contextlib import asynccontextmanager, nullcontext
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.scene_control_v2.runtime import (
    Command, EncodedMotion, RuntimeSettings, RuntimeState, SceneControlDispatcher,
    SerializedWorker, SOMADualBackend, StepResult,
)


def command(number=1, *, velocity=1., scene=0, settings=0):
    return dict(type="command", command_id=number, velocity_xz=[velocity, 0.], heading=0.,
                scene_revision=scene, settings_revision=settings)


def ack(packet, **changes):
    value = dict(type="played", packet_id=packet.packet_id,
                 frame_count=packet.end_frame - packet.start_frame,
                 scene_revision=packet.scene_revision, settings_revision=packet.settings_revision)
    return dict(value, **changes)


async def until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 3.)


class CPUBackend:
    def __init__(self):
        self.calls, self.commits = [], []
        self.entered, self.release = threading.Event(), threading.Event()
        self.block_call = None
        self.delay = 0.
        self.fail = False

    def initial_state(self):
        history = torch.zeros(1, 4, 6)
        history[..., 3] = 1
        return RuntimeState(history, dict(frames=0))

    def step(self, state, intent, settings):
        self.calls.append((state.clone(), intent, settings))
        if len(self.calls) == self.block_call:
            self.entered.set()
            if not self.release.wait(3):
                raise TimeoutError("Test inference release was not signalled")
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise ValueError("contact gate failure retained")
        motion = state.history.clone()
        motion[0, :, 0] = state.history[0, -1, 0] + torch.arange(1, 5)
        motion[..., 1] = intent.command_id
        following = RuntimeState(motion.clone(), dict(frames=state.model_state["frames"] + 4,
                                                     command_id=intent.command_id))
        return StepResult(motion, following, dict(dense_points=[1, 2, 3]))

    def encode(self, motion, *, dense_debug):
        return EncodedMotion(motion.numpy().astype("<f4").tobytes(),
                             dict(frame_stride_floats=6, vertex_count=1))

    def commit(self, state):
        self.commits.append(state.clone())


@asynccontextmanager
async def harness(*, frames=4, backend=None):
    backend = backend or CPUBackend()
    sent, clock = [], [0.]
    worker = SerializedWorker()

    async def send(meta, payload):
        value = json.loads(meta)
        # These assertions track plans/failures; receipt ordering is tested below.
        if value["type"] != "played":
            sent.append((value, payload))

    dispatch = SceneControlDispatcher(backend, backend.initial_state(), send, worker,
                                     settings=RuntimeSettings(frames), clock=lambda: clock[0])
    dispatch.start()
    try:
        yield dispatch, backend, sent, clock
    finally:
        backend.release.set()
        await dispatch.close()
        worker.close()


@pytest.mark.parametrize("frames", [4, 8])
def test_frames_and_candidate_state_advance_only_after_complete_ack(frames):
    async def run():
        async with harness(frames=frames) as (dispatch, backend, sent, clock):
            dispatch.receive_command(command())
            await until(lambda: dispatch.pending is not None and dispatch.candidate is not None)
            first = dispatch.pending
            assert not backend.commits
            assert dispatch.committed_frame == 0
            assert dispatch.committed_state.model_state == dict(frames=0)
            block = np.frombuffer(first.payload, dtype="<f4").reshape(frames, 6)
            np.testing.assert_equal(block[:, 0], np.arange(1, frames + 1))
            assert "debug" not in json.loads(first.metadata_json)
            clock[0] = 1.
            await dispatch.acknowledge(ack(first))
            assert dispatch.committed_frame == frames
            assert backend.commits[0].model_state["frames"] == frames
            await until(lambda: len(sent) >= 2)
            second = dispatch.pending
            assert (second.start_frame, second.end_frame) == (frames, 2 * frames)
            np.testing.assert_equal(np.frombuffer(second.payload, dtype="<f4").reshape(frames, 6)[:, 0],
                                    np.arange(frames + 1, 2 * frames + 1))
            clock[0] = 2.
            await dispatch.acknowledge(ack(second))
            assert backend.commits[1].model_state["frames"] == 2 * frames
            assert first.payload == sent[0][1]
    asyncio.run(run())


def test_latest_command_received_during_slow_step_and_used_by_second_subplan():
    async def run():
        backend = CPUBackend()
        backend.block_call = 1
        async with harness(frames=8, backend=backend) as (dispatch, _, sent, _clock):
            dispatch.receive_command(command(1))
            await until(backend.entered.is_set)
            assert not sent
            receipt = dispatch.receive_command(command(2, velocity=2.))
            assert receipt["command_id"] == 2
            assert dispatch.latest.command_id == 2
            backend.release.set()
            await until(lambda: bool(sent))
            metadata = sent[0][0]
            assert metadata["command_ids"] == [1, 2]
            assert backend.calls[1][0].model_state["frames"] == 4
            assert not backend.commits
    asyncio.run(run())


def test_duplicate_and_rapidly_changing_intent_cannot_starve_publication():
    async def run():
        backend = CPUBackend()
        backend.delay = .008
        async with harness(frames=8, backend=backend) as (dispatch, _, sent, _clock):
            dispatch.receive_command(command(1))
            for _ in range(30):
                assert dispatch.receive_command(command(1))["duplicate"]
                await asyncio.sleep(.001)
            await until(lambda: bool(sent))
            published = dispatch.pending
            await until(lambda: dispatch.candidate is not None)
            ready = dispatch.candidate
            dispatch.receive_command(command(2))
            assert dispatch.candidate is ready
            for number in range(3, 45):
                dispatch.receive_command(command(number, velocity=float(number % 3)))
                await asyncio.sleep(.001)
            assert dispatch.pending is published
            assert published.payload == sent[0][1]
            assert dispatch.discarded_candidates > 0
        async with harness(frames=8, backend=backend) as (dispatch, _, sent, _clock):
            dispatch.receive_command(command(1))
            for number in range(2, 40):
                dispatch.receive_command(command(number, velocity=float(number % 3)))
                await asyncio.sleep(.001)
            assert sent and sent[0][0]["frame_count"] == 8
    asyncio.run(run())


@pytest.mark.parametrize("change", [{"frame_count": 2}, {"packet_id": 2},
                                    {"scene_revision": 1}, {"settings_revision": 1},
                                    {"frame_count": True}])
def test_rejects_partial_wrong_and_stale_ack_without_commit(change):
    async def run():
        async with harness() as (dispatch, backend, _sent, clock):
            dispatch.receive_command(command())
            await until(lambda: dispatch.sent_complete_at is not None)
            pending = dispatch.pending
            clock[0] = 1.
            with pytest.raises(ValueError, match="entire published packet"):
                await dispatch.acknowledge(ack(pending, **change))
            assert dispatch.pending is pending and not backend.commits
            with pytest.raises(ValueError, match="Drain"):
                dispatch.replace_scene(backend, backend.initial_state(), 1)
    asyncio.run(run())


def test_early_ack_rejected_and_valid_last_presentation_time_accepted():
    async def run():
        async with harness() as (dispatch, backend, _sent, clock):
            dispatch.receive_command(command())
            await until(lambda: dispatch.sent_complete_at is not None)
            pending = dispatch.pending
            with pytest.raises(ValueError, match="Early"):
                await dispatch.acknowledge(ack(pending))
            clock[0] = 3 / 30
            await dispatch.acknowledge(ack(pending))
            assert len(backend.commits) == 1
            with pytest.raises(ValueError):
                await dispatch.acknowledge(ack(pending))
    asyncio.run(run())


def test_unpublished_candidate_replacement_preserves_all_published_frames_and_state():
    async def run():
        async with harness() as (dispatch, backend, sent, clock):
            dispatch.receive_command(command(1))
            await until(lambda: dispatch.candidate is not None)
            published = dispatch.pending
            candidate = dispatch.candidate
            dispatch.receive_command(command(2, velocity=2.))
            await until(lambda: dispatch.candidate is not None and dispatch.candidate is not candidate)
            assert dispatch.pending is published
            assert dispatch.candidate.state.model_state == dict(frames=8, command_id=2)
            assert dispatch.candidate.base_frame == 4
            clock[0] = 1.
            await dispatch.acknowledge(ack(published))
            await until(lambda: len(sent) == 2)
            second = dispatch.pending
            assert json.loads(second.metadata_json)["command_ids"] == [2]
            clock[0] = 2.
            await dispatch.acknowledge(ack(second))
            assert backend.commits[-1].model_state == dict(frames=8, command_id=2)
    asyncio.run(run())


def test_settings_change_during_inference_isolates_candidate_but_not_published_ack():
    async def run():
        async with harness() as (dispatch, backend, sent, clock):
            dispatch.receive_command(command())
            await until(lambda: dispatch.candidate is not None)
            published = dispatch.pending
            dispatch.update_settings(RuntimeSettings(8, 12), 1)
            with pytest.raises(ValueError, match="Stale"):
                dispatch.receive_command(command(2))
            dispatch.receive_command(command(2, settings=1))
            await until(lambda: dispatch.candidate is not None)
            assert dispatch.pending is published
            assert dispatch.candidate.frame_count == 8
            clock[0] = 1.
            await dispatch.acknowledge(ack(published))
            await until(lambda: len(sent) >= 2)
            assert sent[1][0]["settings_revision"] == 1
            assert (dispatch.pending.start_frame, dispatch.pending.end_frame) == (4, 12)
            assert backend.commits[0].model_state["frames"] == 4
    asyncio.run(run())


def test_scene_change_rejects_old_computation_and_old_command():
    async def run():
        backend = CPUBackend()
        backend.block_call = 1
        async with harness(backend=backend) as (dispatch, _, sent, _clock):
            dispatch.receive_command(command())
            await until(backend.entered.is_set)
            replacement = CPUBackend()
            initial = replacement.initial_state()
            initial.history[..., 0] = 100
            dispatch.replace_scene(replacement, initial, 1)
            with pytest.raises(ValueError, match="Stale"):
                dispatch.receive_command(command(2))
            dispatch.receive_command(command(2, scene=1))
            backend.release.set()
            await until(lambda: bool(sent))
            assert sent[0][0]["scene_revision"] == 1
            assert dispatch.pending.state.history[0, -1, 0] == 104
            assert dispatch.discarded_candidates == 1
            assert not backend.commits
    asyncio.run(run())


def test_pause_retains_pending_and_discards_only_unpublished_work():
    async def run():
        async with harness() as (dispatch, backend, sent, clock):
            dispatch.receive_command(command())
            await until(lambda: dispatch.candidate is not None)
            published = dispatch.pending
            dispatch.set_paused(True)
            assert dispatch.pending is published and dispatch.candidate is None
            clock[0] = 1.
            await dispatch.acknowledge(ack(published))
            await asyncio.sleep(.02)
            assert len(sent) == 1 and backend.commits[0].model_state["frames"] == 4
            dispatch.set_paused(False)
            await until(lambda: len(sent) >= 2)
            assert sent[1][0]["start_frame"] == 4
    asyncio.run(run())


def test_failures_are_retained_without_retry_loop_or_history_advance():
    async def run():
        backend = CPUBackend()
        backend.fail = True
        async with harness(backend=backend) as (dispatch, _, sent, _clock):
            dispatch.receive_command(command())
            await until(lambda: bool(sent))
            await asyncio.sleep(.02)
            assert len(backend.calls) == 1
            assert dispatch.failures == [sent[0][0]]
            assert "contact gate failure" in sent[0][0]["message"]
            assert not backend.commits and dispatch.committed_frame == 0
    asyncio.run(run())


def test_shared_model_worker_serializes_sessions_and_ack_commits():
    async def run():
        worker = SerializedWorker()
        active = [0, 0]
        backend = CPUBackend()
        original_step, original_commit = backend.step, backend.commit

        def tracked(function, *args):
            active[0] += 1
            active[1] = max(active)
            try:
                time.sleep(.004)
                return function(*args)
            finally:
                active[0] -= 1

        backend.step = lambda *args: tracked(original_step, *args)
        backend.commit = lambda *args: tracked(original_commit, *args)
        async def send(_metadata, _payload):
            pass
        dispatchers = [SceneControlDispatcher(backend, backend.initial_state(), send, worker,
                                              clock=lambda: time.monotonic()) for _ in range(2)]
        try:
            for dispatch in dispatchers:
                dispatch.start()
                dispatch.receive_command(command())
            await until(lambda: all(d.sent_complete_at is not None for d in dispatchers))
            await asyncio.sleep(.11)
            await asyncio.gather(*(d.acknowledge(ack(d.pending)) for d in dispatchers))
            assert active[1] == 1 and len(backend.commits) == 2
        finally:
            for dispatch in dispatchers:
                await dispatch.close()
            worker.close()
    asyncio.run(run())


def test_native_adapter_step_uses_factual_observer_and_no_dense_debug_by_default():
    from motionbricks.data.native_mb_online_conditions import VELOCITY_PROTOCOL

    adapter = SOMADualBackend.__new__(SOMADualBackend)
    adapter.runtime = SimpleNamespace(scene_revision=4)
    adapter.scene_revision = 4
    observed = []
    full = torch.zeros(1, 32, 6)
    full[0, :, 0] = torch.arange(32)
    engine = SimpleNamespace(protocol_version=VELOCITY_PROTOCOL["version"], scene={},
                             controller=SimpleNamespace(context=lambda *_args: nullcontext()))
    engine._prediction = lambda *_args: dict(motion=full, count=torch.tensor([8]))
    def observe(motion):
        observed.append(motion.clone())
        return motion + 10
    engine.observe = observe
    engine._describe_prediction = lambda *_args: pytest.fail("Dense debug entered the hot path")
    adapter.engine = engine
    result = adapter.step(CPUBackend().initial_state(), Command(7, (1., 0.), 0., 0.), RuntimeSettings())
    torch.testing.assert_close(result.motion, full[:, 4:8], atol=0, rtol=0)
    torch.testing.assert_close(observed[0], result.motion, atol=0, rtol=0)
    torch.testing.assert_close(result.state.history, result.motion + 10, atol=0, rtol=0)
    assert result.diagnostics == {}
    full.fill_(123)
    assert result.motion[0, 0, 0] == 4


@pytest.mark.parametrize("frames", [4, 8])
def test_native_adapter_encodes_all_18056_vertices_as_exact_float32(frames):
    adapter = SOMADualBackend.__new__(SOMADualBackend)
    vertices = np.arange(frames * 18056 * 3, dtype=np.float32).reshape(frames, 18056, 3) / 1000
    roots = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    adapter.runtime = SimpleNamespace(skin_features=lambda *_args: (roots, vertices))
    adapter.inference = None
    encoded = adapter.encode(torch.zeros(1, frames, 6), dense_debug=False)
    block = np.frombuffer(encoded.payload, dtype="<f4").reshape(frames, 54171)
    np.testing.assert_equal(block[:, :3], roots)
    np.testing.assert_equal(block[:, 3:].reshape(frames, 18056, 3), vertices)
    assert len(encoded.payload) == frames * 54171 * 4
    assert encoded.metadata["geometry_format"] == "soma-full-vertices-v1"


def test_cli_defaults_and_reserved_ports_and_memory_limit():
    from scripts.soma_scene_control_v2_server import parse_args
    args = parse_args(["--bundle", "a.pt", "--catalog", "b.json"])
    assert (args.port, args.packet_frames, args.lookahead_tokens) == (28945, 4, 8)
    for port in (28943, 28944):
        with pytest.raises(SystemExit):
            parse_args(["--bundle", "a.pt", "--catalog", "b.json", "--port", str(port)])
    for fraction in ("0", "-1", "nan", "1.1"):
        with pytest.raises(SystemExit):
            parse_args(["--bundle", "a.pt", "--catalog", "b.json", "--cuda-memory-fraction", fraction])
    args = parse_args(["--bundle", "a.pt", "--catalog", "b.json", "--cuda-memory-fraction", ".35",
                       "--native-proof-report", "proof.json", "--native-proof-packets", "2"])
    assert args.cuda_memory_fraction == .35 and args.native_proof_packets == 2


def test_ack_timing_uses_send_start_without_rejecting_backpressured_normal_playback():
    async def run():
        backend, worker, clock = CPUBackend(), SerializedWorker(), [0.]
        entered, release = asyncio.Event(), asyncio.Event()
        async def send(_metadata, payload):
            if payload is not None:
                entered.set()
                await release.wait()
        dispatch = SceneControlDispatcher(backend, backend.initial_state(), send, worker,
                                          clock=lambda: clock[0])
        dispatch.start()
        try:
            dispatch.receive_command(command())
            await entered.wait()
            packet = dispatch.pending
            clock[0] = .1
            ack_task = asyncio.create_task(dispatch.acknowledge(ack(packet)))
            await asyncio.sleep(.01)
            assert not ack_task.done() and not backend.commits
            clock[0] = .2
            release.set()
            response = await ack_task
            assert response["send_complete_at"] == .2
            assert response["ack_received_at"] == .1
            assert len(backend.commits) == 1
        finally:
            release.set()
            await dispatch.close()
            worker.close()
    asyncio.run(run())


def test_dense_debug_is_explicit_and_cannot_change_native_binary():
    async def run():
        async with harness() as (dispatch, _backend, sent, _clock):
            dispatch.update_settings(RuntimeSettings(dense_debug=True), 1)
            dispatch.receive_command(command(settings=1))
            await until(lambda: bool(sent))
            assert sent[0][0]["debug"] == [dict(dense_points=[1, 2, 3])]
            actual = sent[0][1]
        async with harness() as (dispatch, _backend, sent, _clock):
            dispatch.receive_command(command())
            await until(lambda: bool(sent))
            assert "debug" not in sent[0][0] and sent[0][1] == actual
    asyncio.run(run())


def test_server_static_config_command_and_ack_order(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from scripts.soma_scene_control_v2_server import API, create_app

    (tmp_path / "vendor").mkdir()
    for name in ("index.html", "app.js", "styles.css"):
        (tmp_path / name).write_text(name)
    backend = CPUBackend()
    runtime = SimpleNamespace(scene_revision=3, vertex_count=18056,
                              terrain_initial_vertices=np.zeros((1, 18056, 3)),
                              terrain_initial_root=np.array([[4., 5., 6.]]),
                              config=lambda: dict(faces=[0, 1, 2], terrain_scene=dict(hf=[[0.]])),
                              close=lambda: None)

    async def run():
        app = create_app(runtime, backend_factory=lambda _runtime: backend,
                         web_root=tmp_path, vendor_root=tmp_path / "vendor")
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/scene-control-v2/")).status == 200
            assert (await client.get("/scene-control-v2/app.js")).status == 200
            config = await (await client.get(API + "/config")).json()
            assert config["initial_root"] == [4., 5., 6.]
            assert len(config["initial_vertices"]) == 18056 * 3
            socket = await client.ws_connect(API + "/ws")
            ready = await socket.receive_json()
            assert ready["type"] == "ready" and ready["heading"] == 0.
            await socket.send_json(command(scene=3))
            receipts = []
            while True:
                meta = await socket.receive_json()
                receipts.append(meta["type"])
                if meta["type"] == "plan":
                    binary = await socket.receive_bytes()
                    break
            assert receipts[0] == "command_received" and len(binary) == 4 * 6 * 4
            await socket.send_json(dict(type="pause", paused=True, scene_revision=3, settings_revision=0))
            assert (await socket.receive_json())["type"] == "pause_applied"
            await asyncio.sleep(.11)
            await socket.send_json(dict(type="played", packet_id=meta["packet_id"], frame_count=4,
                                        scene_revision=3, settings_revision=0))
            assert (await socket.receive_json())["type"] == "played"
            assert len(backend.commits) == 1
            await socket.close()
    asyncio.run(run())


def test_socket_can_receive_new_intent_while_ack_waits_for_slow_speculative_inference(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from scripts.soma_scene_control_v2_server import API, create_app

    backend = CPUBackend()
    backend.block_call = 2
    runtime = SimpleNamespace(scene_revision=0, vertex_count=18056, close=lambda: None)

    async def run():
        app = create_app(runtime, backend_factory=lambda _runtime: backend,
                         web_root=tmp_path, vendor_root=tmp_path)
        async with TestClient(TestServer(app)) as client:
            socket = await client.ws_connect(API + "/ws")
            await socket.receive_json()
            await socket.send_json(command())
            while True:
                meta = await socket.receive_json()
                if meta["type"] == "plan":
                    await socket.receive_bytes()
                    break
            await until(backend.entered.is_set)
            await socket.send_json(dict(type="pause", scene_revision=0, settings_revision=0, paused=True))
            assert (await socket.receive_json())["type"] == "pause_applied"
            await asyncio.sleep(.11)
            await socket.send_json(dict(type="played", packet_id=meta["packet_id"], frame_count=4,
                                        scene_revision=0, settings_revision=0))
            await socket.send_json(command(2, velocity=2.))
            receipt = await asyncio.wait_for(socket.receive_json(), .5)
            assert receipt["type"] == "command_received" and receipt["command_id"] == 2
            assert not backend.commits
            backend.release.set()
            assert (await socket.receive_json())["type"] == "played"
            await socket.close()
    try:
        asyncio.run(run())
    finally:
        backend.release.set()


def test_played_receipt_always_precedes_successor_plan_and_map_selection_is_drain_only(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from scripts.soma_scene_control_v2_server import API, create_app

    runtime = SimpleNamespace(scene_revision=0, vertex_count=18056, close=lambda: None,
                              entries={"a": {}, "b": {}}, terrain_initial_root=np.zeros((1, 3)),
                              terrain_initial_vertices=np.zeros((1, 18056, 3)),
                              config=lambda: dict(faces=[0, 1, 2]))
    def select_map(_map_id):
        runtime.scene_revision += 1
        runtime.terrain_initial_root = np.ones((1, 3))
    runtime.select_map = select_map

    async def run():
        app = create_app(runtime, backend_factory=lambda _runtime: CPUBackend(),
                         web_root=tmp_path, vendor_root=tmp_path)
        async with TestClient(TestServer(app)) as client:
            socket = await client.ws_connect(API + "/ws")
            await socket.receive_json()
            await socket.send_json(command())
            assert (await socket.receive_json())["type"] == "command_received"
            plan = await socket.receive_json()
            await socket.receive_bytes()
            assert (await client.post(API + "/maps/select", json=dict(map_id="b"))).status == 409
            for index in range(3):
                await asyncio.sleep(.11)
                await socket.send_json(dict(type="played", packet_id=plan["packet_id"], frame_count=4,
                                            scene_revision=0, settings_revision=0))
                receipt = await socket.receive_json()
                assert receipt["type"] == "played" and receipt["committed_frame"] == (index + 1) * 4
                plan = await socket.receive_json()
                assert plan["type"] == "plan" and plan["start_frame"] == receipt["committed_frame"]
                await socket.receive_bytes()
            await socket.send_json(dict(type="pause", scene_revision=0, settings_revision=0, paused=True))
            assert (await socket.receive_json())["type"] == "pause_applied"
            await asyncio.sleep(.11)
            await socket.send_json(dict(type="played", packet_id=plan["packet_id"], frame_count=4,
                                        scene_revision=0, settings_revision=0))
            assert (await socket.receive_json())["type"] == "played"
            request = asyncio.create_task(client.post(API + "/maps/select", json=dict(map_id="b")))
            assert (await socket.receive_json())["type"] == "scene_changed"
            await socket.receive()
            response = await request
            assert response.status == 200
            config = await response.json()
            assert config["scene_control_v2"]["scene_revision"] == 1
            assert config["initial_root"] == [1., 1., 1.]
    asyncio.run(run())


def test_memory_fraction_is_applied_before_constructing_any_model(tmp_path, monkeypatch):
    import sys
    from scripts import soma_scene_control_v2_server as server

    order = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *_args: order.append("cap"))
    def build(*_args, **_kwargs):
        order.append("model")
        return SimpleNamespace(close=lambda: order.append("close"))
    monkeypatch.setitem(sys.modules, "scripts.soma_pfnn_maps_server", SimpleNamespace(PFNNMapRuntime=build))
    def proof(*_args):
        order.append("proof")
        return dict(passed=True)
    monkeypatch.setattr(server, "native_proof", proof)
    server.main(["--bundle", "a.pt", "--catalog", "b.json", "--cuda-memory-fraction", ".35",
                 "--native-proof-report", str(tmp_path / "proof.json")])
    assert order == ["cap", "model", "proof", "close"]


class FailingFutureBackend(CPUBackend):
    def __init__(self, fail_call):
        super().__init__()
        self.fail_call = fail_call

    def step(self, *args):
        self.fail = len(self.calls) + 1 == self.fail_call
        return super().step(*args)


@pytest.mark.parametrize("frames,failed_substep", [(4, 0), (8, 1)])
def test_failed_future_waits_for_full_ack_and_played_send_then_reports_once(frames, failed_substep):
    async def run():
        backend = FailingFutureBackend(frames // 4 + failed_substep + 1)
        async with harness(frames=frames, backend=backend) as (dispatch, _, sent, clock):
            wire = []
            receipt_started, receipt_release = asyncio.Event(), asyncio.Event()
            async def send(metadata, payload):
                value = json.loads(metadata)
                if value["type"] == "played":
                    receipt_started.set()
                    await receipt_release.wait()
                wire.append((value, payload))
            dispatch.send = send
            dispatch.receive_command(command())
            await until(lambda: len(dispatch.failures) == 1 and "roots" in dispatch.failures[0])
            packet = dispatch.pending
            original = packet.payload
            failure = dispatch.failures[0]
            assert [m[0]["type"] for m in wire] == ["plan"]
            assert failure["dependency_frame"] == frames
            assert failure["predecessor_packet_id"] == packet.packet_id
            assert failure["committed_frame"] == 0
            assert failure["pending_packet"]["end_frame"] == frames
            assert failure["phase"] == "backend.step"
            assert "contact gate failure retained" in failure["traceback"]
            assert failure["build_history_role"] == "published_not_acknowledged_at_build_start"
            assert failure["roots"]["factual_root5"][-1][0] == 0
            assert failure["roots"]["build_input_root5"][-1][0] == frames
            assert failure["roots"]["failed_replan_input_root5"][-1][0] == frames + failed_substep * 4
            assert failure["roots"]["pending_end_root5"][-1][0] == frames
            assert not dispatch.failure_reported and not backend.commits
            with pytest.raises(ValueError, match="Early"):
                await dispatch.acknowledge(ack(packet))
            calls = len(backend.calls)
            dispatch.receive_command(command(2, velocity=2.))
            await asyncio.sleep(.01)
            assert len(backend.calls) == calls and dispatch.pending is packet
            clock[0] = 1.
            task = asyncio.create_task(dispatch.acknowledge(ack(packet)))
            await receipt_started.wait()
            assert dispatch.committed_frame == frames and dispatch.pending is None
            assert len(backend.commits) == 1
            dispatch.receive_command(command(3, velocity=3.))
            await asyncio.sleep(.01)
            assert [m[0]["type"] for m in wire] == ["plan"]
            assert not dispatch.failure_reported and len(backend.calls) == calls
            receipt_release.set()
            receipt = await task
            await until(lambda: len(wire) == 3)
            assert receipt["committed_frame"] == frames
            assert [m[0]["type"] for m in wire] == ["plan", "played", "runtime_failure"]
            assert wire[2][0] == failure and dispatch.failure_reported
            assert packet.payload == original
            dispatch.receive_command(command(4, velocity=4.))
            await asyncio.sleep(.01)
            assert len(wire) == 3 and len(backend.calls) == calls and len(dispatch.failures) == 1
            assert backend.commits[0].model_state["frames"] == frames
    asyncio.run(run())


@pytest.mark.parametrize("ack_started", [False, True])
def test_close_preserves_deferred_failure_without_deadlock_or_fabricated_ack(ack_started):
    async def run():
        backend = FailingFutureBackend(2)
        async with harness(backend=backend) as (dispatch, _, _sent, clock):
            wire = []
            receipt_started, receipt_release = asyncio.Event(), asyncio.Event()
            async def send(metadata, payload):
                value = json.loads(metadata)
                if value["type"] == "played":
                    receipt_started.set()
                    await receipt_release.wait()
                wire.append(value)
            dispatch.send = send
            dispatch.receive_command(command())
            await until(lambda: len(dispatch.failures) == 1 and "roots" in dispatch.failures[0])
            packet = dispatch.pending
            task = None
            if ack_started:
                clock[0] = 1.
                task = asyncio.create_task(dispatch.acknowledge(ack(packet)))
                await receipt_started.wait()
            await asyncio.wait_for(dispatch.close(), .5)
            assert len(dispatch.failures) == 1 and not dispatch.failure_reported
            assert dispatch.committed_frame == (4 if ack_started else 0)
            assert len(backend.commits) == int(ack_started)
            assert [m["type"] for m in wire] == ["plan"]
            if task is not None:
                receipt_release.set()
                await asyncio.wait_for(task, .5)
                assert [m["type"] for m in wire] == ["plan", "played"]
            else:
                assert dispatch.pending is packet
    asyncio.run(run())


def test_failed_plan_send_unblocks_waiting_ack_and_close_without_committing():
    async def run():
        async with harness() as (dispatch, backend, _sent, clock):
            send_started, fail_send = asyncio.Event(), asyncio.Event()
            sends = []
            async def send(metadata, payload):
                sends.append(json.loads(metadata)["type"])
                send_started.set()
                await fail_send.wait()
                raise ConnectionResetError("plan binary transport failed")
            dispatch.send = send
            dispatch.receive_command(command())
            await send_started.wait()
            packet = dispatch.pending
            with pytest.raises(ValueError, match="Early"):
                await dispatch.acknowledge(ack(packet))
            clock[0] = .1
            task = asyncio.create_task(dispatch.acknowledge(ack(packet)))
            await asyncio.sleep(.01)
            assert not task.done()
            fail_send.set()
            with pytest.raises(ValueError, match="transport failed"):
                await asyncio.wait_for(task, .5)
            await asyncio.wait_for(dispatch.close(), .5)
            assert dispatch.transport_failed and dispatch.pending is None
            assert dispatch.aborted_publication is packet
            assert dispatch.committed_frame == 0 and not backend.commits
            assert sends == ["plan"]
            assert len(dispatch.failures) == 1 and not dispatch.failure_reported
            failure = dispatch.failures[0]
            assert failure["phase"] == "transport.send_plan"
            assert failure["aborted_packet"]["publication_status"] == "transport_send_failed"
            assert "plan binary transport failed" in failure["traceback"]
    asyncio.run(run())


def test_failed_played_send_terminates_without_publishing_successor():
    async def run():
        async with harness() as (dispatch, backend, _sent, clock):
            wire = []
            async def send(metadata, payload):
                value = json.loads(metadata)
                wire.append(value["type"])
                if value["type"] == "played":
                    raise ConnectionResetError("played receipt transport failed")
            dispatch.send = send
            dispatch.receive_command(command())
            await until(lambda: dispatch.candidate is not None)
            packet = dispatch.pending
            clock[0] = 1.
            with pytest.raises(ConnectionResetError):
                await dispatch.acknowledge(ack(packet))
            await asyncio.wait_for(dispatch.close(), .5)
            assert wire == ["plan", "played"]
            assert dispatch.transport_failed and dispatch.committed_frame == 4
            assert len(backend.commits) == 1
            assert dispatch.failures[-1]["phase"] == "transport.send_played"
    asyncio.run(run())


def test_socket_reports_failed_future_only_after_published_packet_played_receipt(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from scripts.soma_scene_control_v2_server import API, create_app

    backend = FailingFutureBackend(2)
    runtime = SimpleNamespace(scene_revision=0, vertex_count=18056, close=lambda: None)
    async def run():
        app = create_app(runtime, backend_factory=lambda _runtime: backend,
                         web_root=tmp_path, vendor_root=tmp_path)
        async with TestClient(TestServer(app)) as client:
            socket = await client.ws_connect(API + "/ws")
            await socket.receive_json()
            await socket.send_json(command())
            assert (await socket.receive_json())["type"] == "command_received"
            plan = await socket.receive_json()
            binary = await socket.receive_bytes()
            await until(lambda: len(backend.calls) == 2)
            await asyncio.sleep(.02)
            await socket.send_json(command(2, velocity=2.))
            assert (await socket.receive_json())["type"] == "command_received"
            assert not backend.commits
            await asyncio.sleep(.11)
            await socket.send_json(dict(type="played", packet_id=plan["packet_id"], frame_count=4,
                                        scene_revision=0, settings_revision=0))
            receipt = await socket.receive_json()
            failure = await socket.receive_json()
            assert receipt["type"] == "played" and receipt["committed_frame"] == 4
            assert failure["type"] == "runtime_failure" and failure["dependency_frame"] == 4
            assert failure["predecessor_packet_id"] == plan["packet_id"]
            assert len(backend.calls) == 2 and len(backend.commits) == 1
            np.testing.assert_equal(np.frombuffer(binary, dtype="<f4").reshape(4, 6)[:, 0], [1, 2, 3, 4])
            await socket.close()
    asyncio.run(run())
