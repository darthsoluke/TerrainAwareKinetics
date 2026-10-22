"""CPU doubles and ephemeral-loopback protocol tests; no GPU or legacy service."""

import asyncio
import json
import math
import os
from pathlib import Path
import socket
import struct
from types import SimpleNamespace

from aiohttp import ClientSession, WSMsgType, web
from aiohttp.test_utils import make_mocked_request
import pytest

from scripts.soma_scene_control_v3_server import (
    API, FRAME_BYTES, ClosedSessions, CapacityRuntime, ReplaySession, PreparedFrame, create_app, parse_args, status_contract,
)


class ReferenceDouble:
    frame_count = 646
    mode = "source"

    def __init__(self):
        self.calls = []

    def new_session(self):
        return None

    def commit(self, _session, _frame):
        pass

    def frame(self, _session, index, command):
        self.calls.append((index, command))
        return PreparedFrame(bytes(FRAME_BYTES), dict(native_rig_ms=1., necessary_inference_ms=None))

    def config(self):
        return dict(**status_contract(), source_frame_count=646)

    def reference(self):
        return dict(vertex_count=18056, frame_index=0)


class PoseDouble:
    def __init__(self, frame_id):
        self.frame_id = frame_id

    def __getitem__(self, _index):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [[self.frame_id / 30, 0, 0, 0, 0, 0, 1]] * 3


class PipelineDouble:
    def __init__(self, adapter, actor, scene, initial_history, *, object_velocity):
        self.adapter, self.actor, self.scene = adapter, actor, scene
        self.state = SimpleNamespace(history=initial_history, hidden=0, frame_id=0)
        self.pending, self.commands, self.known = None, [], True

    def propose(self, command):
        if self.pending:
            raise RuntimeError("One pending capacity frame")
        if not self.known:
            raise RuntimeError("Unknown scene: no capacity frame published")
        self.commands.append(tuple(command))
        next_id = self.state.frame_id + 1
        state = SimpleNamespace(history=self.state.history + command[0] + command[1],
                                hidden=self.state.hidden + 1, frame_id=next_id, object_pose=PoseDouble(next_id))
        self.pending = SimpleNamespace(frame_id=next_id, next_state=state,
            payload=struct.pack("<f", state.history) + bytes(FRAME_BYTES - 4),
            query_metadata=dict(total=512, source_counts=dict(current_body=256,
                committed_root_plan=128, fresh_intent=128), plan_origin_frame_id=self.state.frame_id))
        return self.pending

    def commit(self, frame_id):
        if not self.pending or self.pending.frame_id != frame_id:
            raise ValueError("Wrong capacity ACK")
        self.state, self.pending = self.pending.next_state, None


class CapacityDouble(ReferenceDouble):
    mode = "capacity"

    def __init__(self):
        super().__init__()
        self.adapter, self.actor, self.scene = object(), object(), object()
        self.initial_history, self.object_velocity = 5., None

    def new_session(self):
        return PipelineDouble(self.adapter, self.actor, self.scene, self.initial_history,
                              object_velocity=self.object_velocity)

    def frame(self, session, index, command):
        self.calls.append((index, command))
        return CapacityRuntime.frame(self, session, index, command)

    def commit(self, session, frame):
        CapacityRuntime.commit(session, frame)


def fixture(runtime=None):
    runtime, sent = ReferenceDouble() if runtime is None else runtime, []

    async def send(value, payload):
        sent.append((value, payload))

    async def run(function, *args):
        return function(*args)

    return runtime, sent, ReplaySession(runtime, send, run, runtime.new_session())


def command(session, number=1, **changes):
    return dict(type="command", session_id=session.session_id, command_id=number,
                command=[0., 1., 0.], client_input_ms=number * 10., **changes)


def ack(session, **changes):
    keys = ("session_id", "packet_id", "frame_index", "command_id", "client_input_ms")
    return dict(type="rendered", **{key: session.pending[key] for key in keys},
                client_render_started_ms=100., client_render_completed_ms=102., **changes)


def test_single_pending_frame_commits_only_after_matching_render_ack():
    async def run():
        runtime, sent, session = fixture()
        session.receive_command(command(session))
        assert not await session.publish_once()
        session.set_paused(False)
        assert await session.publish_once()
        assert session.committed == 0 and len(sent) == 1 and len(sent[0][1]) == 4940
        assert not await session.publish_once()
        for key, value in (("command_id", 8), ("packet_id", 2), ("frame_index", 1), ("session_id", "other")):
            wrong = ack(session); wrong[key] = value
            with pytest.raises(ValueError):
                session.acknowledge(wrong)
        wrong = ack(session); wrong["client_render_completed_ms"] = 99.
        with pytest.raises(ValueError):
            session.acknowledge(wrong)
        value = ack(session)
        receipt = session.acknowledge(value)
        assert receipt["committed_frame_count"] == 1
        assert session.pending is None
        with pytest.raises(ValueError):
            session.acknowledge(value)
        assert runtime.calls[0][0] == 0
    asyncio.run(run())


def test_latest_command_is_sampled_for_next_frame_not_relabelled_onto_pending_frame():
    async def run():
        runtime, sent, session = fixture()
        session.receive_command(command(session)); session.set_paused(False)
        await session.publish_once()
        session.receive_command(command(session, 2)); session.receive_command(command(session, 3))
        assert session.pending["command_id"] == 1
        assert session.superseded == 1
        session.acknowledge(ack(session))
        await session.publish_once()
        assert sent[1][0]["command_id"] == 3 and sent[1][0]["frame_index"] == 1
        assert runtime.calls[1][1]["command_id"] == 3
        assert session.pending["command_used_for_motion"] is False
        assert session.pending["timings"]["necessary_inference_ms"] is None
        for number in range(4, 1004):
            session.receive_command(command(session, number))
        snapshot = session.diagnostics()
        assert len(snapshot["events"]) == 256
        assert snapshot["frame_queue_capacity"] == snapshot["command_queue_capacity"] == 1
    asyncio.run(run())


def test_pause_eof_and_late_clock_never_skip_repeat_or_burst_source_frames():
    async def run():
        runtime, sent, session = fixture(); runtime.frame_count = 2
        session.receive_command(command(session)); session.set_paused(False)
        await session.publish_once(); session.set_paused(True)
        assert not await session.publish_once()
        session.acknowledge(ack(session)); assert not await session.publish_once()
        session.set_paused(False); await session.publish_once()
        receipt = session.acknowledge(ack(session)); assert receipt["completed"]
        assert not await session.publish_once()
        assert [value[0]["frame_index"] for value in sent] == [0, 1]
        assert [row[0] for row in runtime.calls] == [0, 1]
    asyncio.run(run())


def test_command_validation_and_zero_training_contract():
    _, _, session = fixture()
    for change in ({"command_id": True}, {"command_id": 2}, {"client_input_ms": math.nan},
                   {"command": [4., 0., 0.]}, {"command": [0., 0., math.inf]}, {"command": [True, 0, 0]}):
        value = command(session); value.update(change)
        with pytest.raises(ValueError):
            session.receive_command(value)
    assert status_contract()["optimizer_updates"] == 0
    assert status_contract()["actor_status"] == "UNTRAINED"
    assert status_contract()["quality_response"] == "unmeasured"
    assert not status_contract()["learned_wasd"]
    with pytest.raises(SystemExit):
        parse_args(["--bundle", "unused", "--source-config", "unused", "--port", "28945"])
    assert parse_args([]).mode == "source" and parse_args([]).device == "cpu"
    with pytest.raises(SystemExit):
        parse_args(["--mode", "capacity", "--device", "cuda:0"])
    capacity = parse_args(["--mode", "capacity", "--device", "cuda:0", "--scene", "true3d_clutter",
                           "--hidden-dim", "1024", "--allocation-id", "parent-test-identity"])
    assert capacity.hidden_dim == 1024


def test_capacity_command_reaches_pipeline_and_history_hidden_commit_only_on_matching_ack():
    async def run():
        runtime, sent, session = fixture(CapacityDouble())
        initial = session.state.state
        session.receive_command(command(session)); session.set_paused(False)
        await session.publish_once()
        candidate = session.pending_frame.candidate
        assert session.state.commands == [(0., 1., 0.)]
        assert session.state.state is initial and initial.hidden == 0 and initial.history == 5.
        assert session.pending["benchmark_role"] == "architecture_capacity_only"
        assert session.pending["command_used_for_motion"] is True
        assert session.pending["timings"]["necessary_inference_ms"] >= 0
        assert session.pending["object_pose"] == candidate.next_state.object_pose.tolist()
        assert session.pending["query_metadata"] == candidate.query_metadata
        assert session.pending["query_metadata"]["plan_origin_frame_id"] == 0
        session.receive_command(command(session, 2))
        session.command["command"] = [1., 0., .2]
        assert not await session.publish_once()
        wrong = ack(session); wrong["command_id"] = 2
        with pytest.raises(ValueError):
            session.acknowledge(wrong)
        assert session.state.state is initial
        session.acknowledge(ack(session))
        assert session.state.state is candidate.next_state and session.state.state.hidden == 1
        assert session.state.state.history == 6.
        await session.publish_once()
        assert session.state.commands[-1] == (1., 0., .2)
        assert session.pending["query_metadata"]["plan_origin_frame_id"] == 1
        assert sent[0][0]["command_id"] == 1 and sent[1][0]["command_id"] == 2
        before = session.state.state
        session.closed = True
        _, _, reset = fixture(runtime)
        assert reset.state.state.hidden == 0 and reset.state.state.history == 5.
        assert reset.state.actor is session.state.actor and reset.state.scene is session.state.scene
        assert session.state.state is before
    asyncio.run(run())


def test_capacity_unknown_scene_publishes_nothing_and_preserves_history_hidden():
    async def run():
        _, sent, session = fixture(CapacityDouble())
        initial = session.state.state; session.state.known = False
        session.receive_command(command(session)); session.set_paused(False)
        with pytest.raises(RuntimeError, match="no capacity frame published"):
            await session.publish_once()
        assert not sent and session.pending is None and session.state.pending is None
        assert session.state.state is initial and session.committed == 0
    asyncio.run(run())


def test_pause_during_capacity_preparation_keeps_candidate_pending_until_rendered():
    async def run():
        _, sent, session = fixture(CapacityDouble())
        original_run = session.run

        async def pause_after_preparation(function, *args):
            result = await original_run(function, *args)
            session.set_paused(True)
            return result

        session.run = pause_after_preparation
        session.receive_command(command(session)); session.set_paused(False)
        initial = session.state.state
        await session.publish_once()
        assert len(sent) == 1 and session.paused and session.pending_frame is not None
        assert session.state.state is initial
        session.set_paused(False)
        assert not await session.publish_once()
        session.acknowledge(ack(session))
        assert session.state.state.hidden == 1 and session.state.pending is None
    asyncio.run(run())


def test_capacity_new_connection_uses_shared_pipeline_without_rebuilding_static_resources(monkeypatch):
    from motionbricks.scene_control_v3 import pipeline

    monkeypatch.setattr(pipeline, "CapacityPipeline", PipelineDouble)
    runtime = CapacityDouble()
    first = CapacityRuntime.new_session(runtime)
    first.commit(first.propose([.2, .4, .3]).frame_id)
    second = CapacityRuntime.new_session(runtime)
    assert second is not first and second.state.frame_id == second.state.hidden == 0
    assert first.actor is second.actor is runtime.actor and first.scene is second.scene is runtime.scene
    assert first.adapter is second.adapter is runtime.adapter
    assert second.state.history == runtime.initial_history


def test_close_during_capacity_preparation_does_not_send_or_commit_abandoned_candidate():
    async def run():
        _, sent, session = fixture(CapacityDouble())
        entered, release = asyncio.Event(), asyncio.Event()
        original_run = session.run

        async def delayed(function, *args):
            result = await original_run(function, *args)
            entered.set(); await release.wait()
            return result

        session.run = delayed
        initial = session.state.state
        session.receive_command(command(session)); session.set_paused(False)
        task = asyncio.create_task(session.publish_once())
        await entered.wait()
        session.closed = True; release.set()
        assert not await asyncio.wait_for(task, 1)
        assert not sent and session.pending is None and session.state.state is initial
    asyncio.run(run())


def test_closed_session_archive_is_bounded_and_survives_reload(tmp_path):
    path = tmp_path / "closed.json"
    archive = ClosedSessions(path)
    for number in range(20):
        archive.append(dict(session_id=str(number), failure=dict(message="actual failure")))
    reloaded = ClosedSessions(path)
    assert len(reloaded.rows) == 16 and reloaded.rows[0]["session_id"] == "4"
    assert reloaded.rows[-1]["session_id"] == "19"


def test_actual_async_pump_failure_is_logged_archived_and_sent_as_close_reason(tmp_path, caplog):
    class FailingReference(ReferenceDouble):
        def frame(self, session, index, command):
            if index == 1:
                raise RuntimeError("Unknown scene: no capacity frame published")
            return super().frame(session, index, command)

    async def run():
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0)); port = listener.getsockname()[1]
        path = tmp_path / "closed.json"
        runner = web.AppRunner(create_app(FailingReference(), port=port, closed_diagnostics=path))
        await runner.setup(); await web.SockSite(runner, listener).start()
        base = f"http://127.0.0.1:{port}"
        try:
            async with ClientSession() as client:
                async with client.ws_connect(base + API + "/ws") as ws:
                    ready = await ws.receive_json()
                    session_id = ready["session_id"]
                    await ws.send_json(dict(type="command", session_id=session_id, command_id=1,
                                            command=[0., 1., 0.], client_input_ms=10.))
                    await ws.send_json(dict(type="pause", session_id=session_id, paused=False))
                    while True:
                        value = await ws.receive_json()
                        if value["type"] == "frame":
                            break
                    assert (await ws.receive()).type == WSMsgType.BINARY
                    identity = ("session_id", "packet_id", "frame_index", "command_id", "client_input_ms")
                    await ws.send_json(dict(type="rendered", **{key: value[key] for key in identity},
                                            client_render_started_ms=20., client_render_completed_ms=21.))
                    while True:
                        message = await asyncio.wait_for(ws.receive(), 3)
                        if message.type == WSMsgType.CLOSE:
                            assert message.data == 1011
                            assert "RuntimeError: Unknown scene" in message.extra
                            break
                        assert message.type == WSMsgType.TEXT
                async with client.get(base + API + "/diagnostics") as response:
                    diagnostics = await response.json()
                assert diagnostics["sessions"] == []
                closed = diagnostics["closed_sessions"][-1]
                assert closed["committed_frame_count"] == 1
                assert closed["failure"]["error_type"] == "RuntimeError"
                assert "no capacity frame published" in closed["failure"]["message"]
                assert "raise RuntimeError" in closed["failure"]["traceback"]
                assert json.loads(path.read_text())["closed_sessions"][-1]["failure"] == closed["failure"]
        finally:
            await runner.cleanup()
    asyncio.run(run())
    assert "failed after 1 committed frames" in caplog.text
    assert "RuntimeError: Unknown scene" in caplog.text


def test_read_only_routes_and_same_origin_loopback_admission_without_binding():
    async def run():
        app = create_app(ReferenceDouble())
        paths = {route.resource.canonical for route in app.router.routes()}
        assert API + "/diagnostics" in paths and API + "/reference" in paths
        assert not any("train" in path or "v2" in path for path in paths)
        middleware = app.middlewares[0]

        async def handler(_request):
            return web.json_response(dict(ok=True))

        transport = SimpleNamespace(get_extra_info=lambda key, default=None:
                                    ("127.0.0.1", 12345) if key == "peername" else default)
        for host, origin, accepted in (("127.0.0.1:28946", "http://127.0.0.1:28946", True),
                                       ("external.example", "http://external.example", False),
                                       ("127.0.0.1:28946", "https://external.example", False)):
            request = make_mocked_request("GET", "/", headers={"Host": host, "Origin": origin}, transport=transport)
            if accepted:
                response = await middleware(request, handler)
                assert json.loads(response.text)["ok"] and response.headers["Cache-Control"] == "no-store"
            else:
                with pytest.raises(web.HTTPForbidden):
                    await middleware(request, handler)
        await app.cleanup()
    asyncio.run(run())


@pytest.mark.parametrize("capacity", [False, True])
def test_real_websocket_with_actual_frontend_fakeclock_on_temporary_cpu_test_port(capacity):
    async def run():
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        runtime = CapacityDouble() if capacity else ReferenceDouble()
        app = create_app(runtime, port=port)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.SockSite(runner, listener)
        await site.start()
        try:
            env = dict(os.environ, SCENE_V3_WS_TEST_URL=f"http://127.0.0.1:{port}")
            process = await asyncio.create_subprocess_exec(
                "node", "--test", "web/scene_control_v3/protocol.test.mjs",
                cwd=Path(__file__).resolve().parents[1], env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            communication = asyncio.create_task(process.communicate())
            try:
                output, _ = await asyncio.wait_for(asyncio.shield(communication), 15)
            finally:
                if process.returncode is None:
                    process.kill()
                    await communication
            assert process.returncode == 0, output.decode()
            assert [row[0] for row in runtime.calls] == [0, 1, 2]
        finally:
            await runner.cleanup()
    asyncio.run(run())
