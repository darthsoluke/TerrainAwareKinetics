"""Isolated CPU protocol fixtures; no service, renderer, model, or physics launch."""

import asyncio
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from aiohttp import WSMsgType, web
import pytest

from scripts.mb_sonic_lab_server import (API, DISPLAY_HZ, PORT, WASD_PORT, SCHEMA,
    HighResolutionProactorEventLoop, LabSession, create_app, parse_args)


def configuration(scene_id="pfnn_test"):
    return dict(schema=SCHEMA, optimizer_updates=0, physics_dt_s=.005, controller_hz=50, episode_id=1,
        intent_enabled=True, control_mode="trained_mb_reference", start_required=True, start_action="begin_episode",
        world_up="z", scene_catalog=[dict(id=name, label=name, available=True, unavailable_reason=None)
                                    for name in ("pfnn_test", "true3d_test")],
        scene=dict(manifest_id=scene_id, manifest_sha256=scene_id * 4, geometries=[]),
        robot=dict(name="isolated_protocol_fixture", source="unit_fixture", geometries=[]),
        reference=dict(version="fixture-v1", source="unit_fixture", coverage_start_s=0.,
                       coverage_end_s=1., parents=[-1, 0]),
        initial_camera=dict(position=[3., -3., 2.], target=[0., 0., 1.]))


class RuntimeFixture:
    """Test-only independent native clocks and a one-value intent mailbox."""

    def __init__(self):
        self.current_config = configuration()
        self.physics_ticks = self.controller_ticks = 0
        self.latest_intent = self.used_intent = None
        self.submissions = []
        self.wall_s = 0.
        self.begin_count = 0
        self.snapshot_subscribers = set()

    def subscribe_snapshots(self, callback):
        self.snapshot_subscribers.add(callback)
        callback()
        return lambda: self.snapshot_subscribers.discard(callback)

    def notify_snapshot(self):
        for callback in tuple(self.snapshot_subscribers):
            callback()

    def config(self):
        return deepcopy(self.current_config)

    def advance_wall(self, seconds):
        self.wall_s += seconds
        self.physics_ticks = round(self.wall_s / .005)
        self.controller_ticks = round(self.wall_s / .02)
        if self.controller_ticks > 0:
            self.used_intent = deepcopy(self.latest_intent)
        self.notify_snapshot()

    def submit_intent(self, command):
        self.latest_intent = command
        self.submissions.append(command)

    def begin_episode(self):
        if self.begin_count:
            raise ValueError("Episode already started")
        self.begin_count += 1
        return dict(started=True, command_consumed=False, control_mode="official_reference_read_only")

    def select_scene(self, scene_id):
        episode_id = self.current_config["episode_id"] + 1
        self.current_config = configuration(scene_id)
        self.current_config["episode_id"] = episode_id
        self.notify_snapshot()
        return self.config()

    def snapshot(self):
        scene = self.current_config["scene"]
        return dict(frame_id=self.physics_ticks, sim_time_s=self.physics_ticks * .005,
            episode_id=self.current_config["episode_id"],
            server_monotonic_ns=1000, scene_manifest_id=scene["manifest_id"],
            scene_manifest_sha256=scene["manifest_sha256"], status="unit_fixture",
            command=self.used_intent, actual=dict(root_position=[0., 0., 1.], geom_poses=[],
                body_onset_command_id=None, body_onset_sim_time_s=None), reference=None,
            contacts=[], timing=dict(chain_ms=None, physics_steps=self.physics_ticks,
                controller_steps=self.controller_ticks, realtime_factor=1., tracking_error_m=None,
                heading_error_rad=None), errors=[])

    def step(self):
        raise AssertionError("Viewer must never step physics")

    def tick(self):
        raise AssertionError("Viewer must never tick the controller")

    def start(self):
        raise AssertionError("Tests must not launch runtime")

    def stop(self):
        raise AssertionError("Tests must not launch runtime")

    def shutdown(self):
        raise AssertionError("Tests must not shut down an actual runtime")

    def asset(self, _asset_id):
        return b""


def intent(number=1, **changes):
    return dict(command_id=number, command_seq=number, client_input_ms=number * 10.,
                scene_id="pfnn_test", scene_manifest_sha256="pfnn_test" * 4, episode_id=1,
                velocity_xy_mps=[.8, 0.], yaw_rate_radps=0., **changes)


def acknowledgement(packet):
    command = packet["state"]["command"]
    return dict(type="rendered", packet_id=packet["packet_id"],
        frame_id=packet["state"]["frame_id"], command_id=None if command is None else command["command_id"],
        client_render_started_ms=20., client_render_completed_ms=21.)


def test_newest_intent_has_no_backlog_and_is_not_associated_before_consumption():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    for number in range(1, 301):
        session.intent(intent(number))
    assert runtime.latest_intent["command_id"] == 300
    packet = session.offer(runtime.snapshot())
    assert packet["state"]["command"] is None
    assert len(session.commands) == 256
    session.rendered(acknowledgement(packet))
    runtime.advance_wall(.02)
    packet = session.offer(runtime.snapshot())
    assert packet["state"]["command"]["command_id"] == 300
    assert packet["state"]["command"]["session_id"] == session.id


def test_explicit_start_and_read_only_mode_never_fabricate_intent_consumption():
    runtime = RuntimeFixture()
    runtime.current_config.update(intent_enabled=False, control_mode="official_reference_read_only")
    session = LabSession(runtime, runtime.config())
    assert runtime.begin_count == 0 and session.offer(runtime.snapshot())["state"]["command"] is None
    with pytest.raises(ValueError, match="read-only"):
        session.intent(intent())
    reply = session.begin_episode()
    assert reply["type"] == "episode_started" and reply["started"]
    assert reply["command_consumed"] is False and runtime.begin_count == 1
    with pytest.raises(ValueError, match="already started"):
        session.begin_episode()
    session.disconnect()
    assert not runtime.submissions and not session.commands


def test_strafe_and_pure_turn_forward_raw_yaw_without_fabricating_effective_intent():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    session.intent(dict(intent(), velocity_xy_mps=[0., -1.2], yaw_rate_radps=.7))
    raw = runtime.latest_intent
    assert raw["velocity_xy_mps"] == [0., -1.2] and raw["yaw_rate_radps"] == .7
    assert raw["command_seq"] == raw["command_id"] == 1
    assert raw["input_frame"] == "world_z_up" and raw["effective_intent"] is None
    assert "heading_rad" not in raw
    assert runtime.snapshot()["command"] is None
    runtime.advance_wall(.02)
    packet = session.offer(runtime.snapshot())
    session.intent(dict(intent(2), velocity_xy_mps=[0., 0.], yaw_rate_radps=-.5))
    assert runtime.latest_intent["velocity_xy_mps"] == [0., 0.]
    assert runtime.latest_intent["yaw_rate_radps"] == -.5
    assert packet["state"]["command"]["command_seq"] == 1
    assert packet["state"]["actual"]["body_onset_command_id"] is None


@pytest.mark.parametrize("number", [1, 2, 30, 120])
def test_physics_200_and_controller_50_are_independent_of_display_and_missing_acks(number):
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    first = session.offer(runtime.snapshot())
    for _ in range(number):
        runtime.advance_wall(1 / number)
        assert session.offer(runtime.snapshot()) is None
    assert runtime.physics_ticks == 200
    assert runtime.controller_ticks == 50
    assert session.pending is first and session.sent_count == 1
    session.rendered(acknowledgement(first))
    assert runtime.physics_ticks == 200 and runtime.controller_ticks == 50
    latest = session.offer(runtime.snapshot())
    assert latest["state"]["frame_id"] == 200
    assert session.sent_count == 2


def test_real_30hz_display_sampling_does_not_advance_200hz_native_time():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    for _ in range(DISPLAY_HZ):
        runtime.advance_wall(1 / DISPLAY_HZ)
        packet = session.offer(runtime.snapshot())
        session.rendered(acknowledgement(packet))
    assert session.rendered_count == 30
    assert runtime.physics_ticks == 200 and runtime.controller_ticks == 50
    assert runtime.snapshot()["sim_time_s"] == 1.


def test_render_identity_is_exact_and_duplicate_physics_state_is_not_resent():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    packet = session.offer(runtime.snapshot())
    receipt = acknowledgement(packet)
    with pytest.raises(ValueError, match="does not match"):
        session.rendered(dict(receipt, frame_id=50))
    assert session.pending is packet
    acknowledgement_result = session.rendered(receipt)
    assert acknowledgement_result["server_publish_to_ack_ms"] >= 0
    assert session.receipts[-1]["server_ack_received_ns"] == acknowledgement_result["server_ack_received_ns"]
    assert session.offer(runtime.snapshot()) is None
    with pytest.raises(ValueError, match="No outstanding"):
        session.rendered(receipt)


def test_scene_transition_invalidates_old_slot_without_reset_or_failure():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    runtime.advance_wall(.4)
    old = session.offer(runtime.snapshot())
    response = session.select_scene("true3d_test")
    assert response["scene_revision"] == 1
    assert session.pending is None
    assert runtime.physics_ticks == 80 and runtime.controller_ticks == 20
    assert session.rendered(acknowledgement(old))["invalidated_by_scene_change"] is True
    packet = session.offer(runtime.snapshot())
    assert packet["scene_revision"] == 1
    assert packet["state"]["scene_manifest_id"] == "true3d_test"
    assert not session.failures


def test_reset_current_scene_requires_explicit_action_and_preserves_prior_evidence():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    packet = session.offer(runtime.snapshot())
    session.rendered(acknowledgement(packet))
    session.fail(RuntimeError("retained prior failure"))
    assert not session.lifecycle and runtime.begin_count == 0
    result = session.reset_episode()
    assert result["scene_revision"] == 1 and result["config"]["scene"]["manifest_id"] == "pfnn_test"
    assert runtime.begin_count == 0
    assert session.rendered_count == 1 and len(session.receipts) == 1
    assert session.failures[0]["message"] == "retained prior failure"
    assert session.lifecycle[-1]["action"] == "reset_episode"
    session.begin_episode()
    assert runtime.begin_count == 1 and session.lifecycle[-1]["action"] == "begin_episode"


def test_unknown_scene_or_manifest_is_not_replaced():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    unknown = runtime.snapshot()
    unknown["scene_manifest_sha256"] = "different"
    with pytest.raises(ValueError, match="canonical"):
        session.offer(unknown)
    runtime.current_config["scene_catalog"][1].update(available=False, unavailable_reason="asset missing")
    session.config = runtime.config()
    with pytest.raises(ValueError, match="asset missing"):
        session.select_scene("true3d_test")
    assert runtime.config()["scene"]["manifest_id"] == "pfnn_test"


@pytest.mark.parametrize("change", [dict(command_id=0), dict(command_id=True),
    dict(velocity_xy_mps=[float("nan"), 0]), dict(velocity_xy_mps=[1]),
    dict(yaw_rate_radps=float("inf")), dict(client_input_ms=None)])
def test_invalid_inputs_do_not_reach_runtime(change):
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    with pytest.raises(ValueError):
        session.intent(dict(intent(), **change))
    assert not runtime.submissions


def test_disconnect_sends_zero_intent_and_retains_failures_without_reset():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    runtime.advance_wall(.5)
    session.intent(intent())
    session.fail(RuntimeError("measured fall"))
    session.disconnect()
    assert runtime.latest_intent["velocity_xy_mps"] == [0., 0.]
    assert runtime.latest_intent["source"] == "disconnect_stop"
    assert runtime.physics_ticks == 100
    assert session.diagnostics()["failures"][0]["message"] == "measured fall"
    assert runtime.latest_intent["scene_id"] == "pfnn_test"
    assert runtime.latest_intent["episode_id"] == 1


@pytest.mark.parametrize('scene_id',[None,'true3d_test'])
def test_disconnect_after_reset_does_not_claim_the_new_episode(scene_id):
    runtime = RuntimeFixture()
    session = LabSession(runtime,runtime.config())
    session.intent(intent())
    if scene_id is None:
        session.reset_episode()
    else:
        session.select_scene(scene_id)
    session.disconnect()
    assert len(runtime.submissions) == 1
    assert runtime.latest_intent['episode_id'] == 1
    assert session.diagnostics()['commands'][0]['episode_id'] == 1
    assert session.config['episode_id'] == 2
    new = LabSession(runtime,runtime.config())
    scene = new.config['scene']
    new.intent(dict(intent(),episode_id=2,scene_id=scene['manifest_id'],
        scene_manifest_sha256=scene['manifest_sha256']))
    assert runtime.latest_intent['session_id'] == new.id


def test_disconnect_after_new_episode_input_still_sends_stop():
    runtime = RuntimeFixture()
    session = LabSession(runtime,runtime.config())
    session.intent(intent())
    session.reset_episode()
    session.intent(dict(intent(2),episode_id=2))
    session.disconnect()
    assert len(runtime.submissions) == 3
    assert runtime.latest_intent['episode_id'] == 2
    assert runtime.latest_intent['command_seq'] == 3
    assert runtime.latest_intent['source'] == 'disconnect_stop'
    assert runtime.latest_intent['velocity_xy_mps'] == [0.,0.]
    assert len(session.diagnostics()['commands']) == 2


def test_old_scene_and_same_scene_old_episode_inputs_never_reach_new_mailbox():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    session.intent(intent())
    session.select_scene("true3d_test")
    with pytest.raises(ValueError, match="current scene and episode"):
        session.intent(intent(2))
    assert len(runtime.submissions) == 1
    current = dict(intent(2), scene_id="true3d_test", scene_manifest_sha256="true3d_test" * 4, episode_id=2)
    session.intent(current)
    session.reset_episode()
    with pytest.raises(ValueError, match="current scene and episode"):
        session.intent(dict(current, command_id=3, command_seq=3))
    assert len(runtime.submissions) == 2
    session.intent(dict(current, command_id=3, command_seq=3, episode_id=3))
    assert runtime.latest_intent["episode_id"] == 3 and not session.failures


def test_same_origin_is_independent_for_the_two_allocated_ports_without_a_listener():
    async def run():
        async def handler(_request):
            return web.Response(text="isolated_fixture")
        app = create_app(RuntimeFixture(), port=WASD_PORT)
        same = SimpleNamespace(remote="127.0.0.1", host="127.0.0.1:28948",
                               headers={"Origin": "http://127.0.0.1:28948"})
        assert (await app.middlewares[0](same, handler)).text == "isolated_fixture"
        for origin in ("http://127.0.0.1:28947", "http://localhost:28948", "https://example.invalid"):
            with pytest.raises(web.HTTPForbidden):
                await app.middlewares[0](SimpleNamespace(remote=same.remote, host=same.host,
                    headers={"Origin": origin}), handler)
    asyncio.run(run())


def test_reserved_port_explicit_allocation_and_binary_route_without_startup():
    args = parse_args(["--config", "explicit-config.json", "--allocation-id", "parent-e3"])
    assert args.port == PORT == 28947
    assert parse_args(["--config", "wasd.json", "--allocation-id", "parent-e3",
                       "--port", "28948"]).port == WASD_PORT
    assert create_app(RuntimeFixture(), port=WASD_PORT)
    with pytest.raises(ValueError, match="allocated lab ports"):
        create_app(RuntimeFixture(), port=28946)
    with pytest.raises(SystemExit):
        parse_args(["--config", "explicit-config.json", "--allocation-id", "parent-e3", "--port", "28946"])
    with pytest.raises(SystemExit):
        parse_args(["--config", "explicit-config.json"])
    app = create_app(RuntimeFixture())
    assert app.on_startup and app.on_cleanup
    routes = [str(route.resource) for route in app.router.routes()]
    assert any(API + "/assets/" in row for row in routes)


def test_unused_ready_client_disconnect_does_not_bind_a_runtime_controller_session():
    runtime = RuntimeFixture()
    session = LabSession(runtime, runtime.config())
    session.disconnect()
    assert not runtime.submissions and session.last_command_id == 0


class MemorySocket:
    def __init__(self, **_):
        self.closed = False
        self.sent, self.incoming = asyncio.Queue(), asyncio.Queue()

    async def prepare(self, _request):
        pass

    async def send_str(self, value):
        await self.sent.put(json.loads(value))

    async def close(self, **_):
        self.closed = True
        await self.incoming.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return SimpleNamespace(type=WSMsgType.TEXT, data=json.dumps(item))


@pytest.mark.parametrize("closed_before_result", [True, False])
def test_cold_scene_heartbeat_budget_and_transport_close_never_double_send(monkeypatch, closed_before_result):
    class ColdRuntime(RuntimeFixture):
        def __init__(self):
            super().__init__()
            self.selecting, self.release = threading.Event(), threading.Event()

        def select_scene(self, scene_id):
            self.selecting.set()
            assert self.release.wait(2)
            return super().select_scene(scene_id)

    class ClosingSocket(MemorySocket):
        def __init__(self):
            super().__init__()
            self.attempts = []

        async def send_str(self, value):
            kind = json.loads(value)["type"]
            self.attempts.append(kind)
            if kind == "config":
                raise ConnectionResetError("cold scene transport closed during write")
            await super().send_str(value)

    async def run():
        runtime, socket = ColdRuntime(), ClosingSocket()

        def make_socket(**options):
            assert options["heartbeat"] == 180
            return socket

        monkeypatch.setattr("scripts.mb_sonic_lab_server.web.WebSocketResponse", make_socket)
        app = create_app(runtime, port=WASD_PORT)
        handler = next(route.handler for route in app.router.routes()
                       if route.method == "GET" and route.resource.canonical == API + "/ws")
        task = asyncio.create_task(handler(None))
        ready = await asyncio.wait_for(socket.sent.get(), 2)
        assert (await asyncio.wait_for(socket.sent.get(), 2))["type"] == "frame"
        await socket.incoming.put(dict(type="scene", scene_id="true3d_test", session_id=ready["session_id"]))
        assert await asyncio.to_thread(runtime.selecting.wait, 1)
        if closed_before_result:
            await socket.close()
        runtime.release.set()
        await asyncio.wait_for(task, 2)
        expected = ["ready", "frame"] + ([] if closed_before_result else ["config"])
        assert socket.attempts == expected
        assert runtime.current_config["scene"]["manifest_id"] == "true3d_test"
        assert runtime.begin_count == runtime.physics_ticks == 0
        assert not runtime.snapshot_subscribers
        diagnostics = next(route.handler for route in app.router.routes()
                           if route.method == "GET" and route.resource.canonical == API + "/diagnostics")
        receipt = json.loads((await diagnostics(None)).text)["closed_sessions"][0]
        assert any(row["error_type"] == "ConnectionResetError" for row in receipt["failures"])
        assert not any(kind in socket.attempts for kind in ("failure", "error"))

    asyncio.run(run())


def test_snapshot_config_and_explicit_scene_change_serialize_without_a_listening_server(monkeypatch):
    class BlockingRuntime(RuntimeFixture):
        def __init__(self):
            super().__init__()
            self.reading, self.release = threading.Event(), threading.Event()
            self.selected = False

        def snapshot(self):
            self.reading.set()
            assert self.release.wait(2), "test failed to release snapshot"
            return super().snapshot()

        def select_scene(self, scene_id):
            self.selected = True
            return super().select_scene(scene_id)

    async def run():
        runtime, socket = BlockingRuntime(), MemorySocket()
        monkeypatch.setattr("scripts.mb_sonic_lab_server.web.WebSocketResponse", lambda **_: socket)
        app = create_app(runtime)
        handler = next(route.handler for route in app.router.routes()
                       if route.method == "GET" and route.resource.canonical == API + "/ws")
        task = asyncio.create_task(handler(None))
        ready = await asyncio.wait_for(socket.sent.get(), 2)
        assert await asyncio.to_thread(runtime.reading.wait, 1)
        await socket.incoming.put(dict(type="scene", session_id=ready["session_id"], scene_id="true3d_test"))
        await asyncio.sleep(.01)
        assert not runtime.selected
        runtime.release.set()
        old = await asyncio.wait_for(socket.sent.get(), 2)
        changed = await asyncio.wait_for(socket.sent.get(), 2)
        assert old["type"] == "frame" and old["scene_revision"] == 0
        assert changed["type"] == "config" and changed["scene_revision"] == 1
        await socket.incoming.put(dict(acknowledgement(old), session_id=ready["session_id"]))
        replies = [await asyncio.wait_for(socket.sent.get(), 2) for _ in range(2)]
        frame = next(row for row in replies if row["type"] == "frame")
        receipt = next(row for row in replies if row["type"] == "render_receipt")
        assert frame["scene_revision"] == 1
        assert frame["state"]["scene_manifest_id"] == "true3d_test"
        assert receipt["invalidated_by_scene_change"]
        await socket.close()
        await asyncio.wait_for(task, 2)

    asyncio.run(run())


def test_ack_prefetches_native_latest_without_a_30hz_producer_sleep(monkeypatch):
    delays = []
    sleep = asyncio.sleep

    async def observed_sleep(delay):
        delays.append(delay)
        await sleep(delay)

    async def run():
        runtime, socket = RuntimeFixture(), MemorySocket()
        monkeypatch.setattr("scripts.mb_sonic_lab_server.web.WebSocketResponse", lambda **_: socket)
        monkeypatch.setattr("scripts.mb_sonic_lab_server.asyncio.sleep", observed_sleep)
        app = create_app(runtime)
        handler = next(route.handler for route in app.router.routes()
                       if route.method == "GET" and route.resource.canonical == API + "/ws")
        task = asyncio.create_task(handler(None))
        ready = await asyncio.wait_for(socket.sent.get(), 2)
        first = await asyncio.wait_for(socket.sent.get(), 2)
        runtime.advance_wall(.005)
        await socket.incoming.put(dict(acknowledgement(first), session_id=ready["session_id"]))
        replies = [await asyncio.wait_for(socket.sent.get(), 2) for _ in range(2)]
        frame = next(row for row in replies if row["type"] == "frame")
        assert frame["state"]["frame_id"] == 1
        assert runtime.physics_ticks == 1 and runtime.controller_ticks == 0
        assert not delays
        await socket.close()
        await asyncio.wait_for(task, 2)

    asyncio.run(run())


def test_runtime_failure_retains_connection_for_explicit_reset_without_auto_retry(monkeypatch):
    class FailedRuntime(RuntimeFixture):
        def __init__(self):
            super().__init__()
            self.failed, self.snapshot_calls = True, 0

        def snapshot(self):
            self.snapshot_calls += 1
            if self.failed:
                raise RuntimeError("actual terminal failure fixture")
            return super().snapshot()

        def select_scene(self, scene_id):
            self.failed = False
            return super().select_scene(scene_id)

    async def run():
        runtime, socket = FailedRuntime(), MemorySocket()
        monkeypatch.setattr("scripts.mb_sonic_lab_server.web.WebSocketResponse", lambda **_: socket)
        app = create_app(runtime, port=WASD_PORT)
        handler = next(route.handler for route in app.router.routes()
                       if route.method == "GET" and route.resource.canonical == API + "/ws")
        task = asyncio.create_task(handler(None))
        ready = await asyncio.wait_for(socket.sent.get(), 2)
        failure = await asyncio.wait_for(socket.sent.get(), 2)
        assert failure["type"] == "failure" and "actual terminal" in failure["message"]
        await asyncio.to_thread(runtime.notify_snapshot)
        await asyncio.sleep(.01)
        assert runtime.snapshot_calls == 1 and not socket.closed and runtime.begin_count == 0
        assert len(runtime.snapshot_subscribers) == 1
        await socket.incoming.put(dict(type="reset_episode", session_id=ready["session_id"]))
        replies = [await asyncio.wait_for(socket.sent.get(), 2) for _ in range(2)]
        changed = next(row for row in replies if row["type"] == "config")
        frame = next(row for row in replies if row["type"] == "frame")
        assert changed["config"]["episode_id"] == frame["state"]["episode_id"] == 2
        assert runtime.begin_count == 0 and runtime.physics_ticks == 0
        await socket.close()
        await asyncio.wait_for(task, 2)
        assert not runtime.snapshot_subscribers
    asyncio.run(run())


def test_publication_pump_has_no_idle_poll_or_pending_ack_wake_and_unsubscribes_on_close(monkeypatch):
    class CountingRuntime(RuntimeFixture):
        def __init__(self):
            super().__init__()
            self.snapshot_calls = 0

        def snapshot(self):
            self.snapshot_calls += 1
            return super().snapshot()

    events = []

    class ObservedEvent(asyncio.Event):
        def __init__(self):
            super().__init__()
            self.set_calls = 0
            events.append(self)

        def set(self):
            self.set_calls += 1
            super().set()

    async def run():
        runtime, socket = CountingRuntime(), MemorySocket()
        loop = asyncio.get_running_loop()
        cross_thread_notices = []
        call_soon_threadsafe = loop.call_soon_threadsafe

        def counted_schedule(callback, *args, **kwargs):
            if getattr(callback, "__name__", "") == "publication_ready":
                cross_thread_notices.append(callback)
            return call_soon_threadsafe(callback, *args, **kwargs)

        monkeypatch.setattr(loop, "call_soon_threadsafe", counted_schedule)
        monkeypatch.setattr("scripts.mb_sonic_lab_server.web.WebSocketResponse", lambda **_: socket)
        monkeypatch.setattr("scripts.mb_sonic_lab_server.asyncio.Event", ObservedEvent)
        app = create_app(runtime)
        handler = next(route.handler for route in app.router.routes()
                       if route.method == "GET" and route.resource.canonical == API + "/ws")
        task = asyncio.create_task(handler(None))
        ready = await asyncio.wait_for(socket.sent.get(), 2)
        first = await asyncio.wait_for(socket.sent.get(), 2)
        assert first["type"] == "frame" and runtime.snapshot_calls == 1
        assert len(events) == 1 and len(runtime.snapshot_subscribers) == 1
        wake_count = events[0].set_calls
        notice_count = len(cross_thread_notices)

        def publications():
            for _ in range(40):
                runtime.advance_wall(.005)

        await asyncio.to_thread(publications)
        await asyncio.sleep(.01)
        assert runtime.snapshot_calls == 1 and events[0].set_calls == wake_count
        assert len(cross_thread_notices) == notice_count
        await socket.incoming.put(dict(acknowledgement(first), session_id=ready["session_id"]))
        replies = [await asyncio.wait_for(socket.sent.get(), 2) for _ in range(2)]
        latest = next(row for row in replies if row["type"] == "frame")
        assert latest["state"]["frame_id"] == 40 and runtime.snapshot_calls == 2

        await socket.incoming.put(dict(acknowledgement(latest), session_id=ready["session_id"]))
        ack = await asyncio.wait_for(socket.sent.get(), 2)
        assert ack["type"] == "render_receipt"
        await asyncio.sleep(.03)
        assert runtime.snapshot_calls == 3 and socket.sent.empty()

        def final_publication_burst():
            for _ in range(200):
                runtime.notify_snapshot()
            runtime.advance_wall(.005)

        notice_count = len(cross_thread_notices)
        publisher = threading.Thread(target=final_publication_burst)
        publisher.start()
        publisher.join(timeout=1)
        assert not publisher.is_alive()
        assert len(cross_thread_notices) == notice_count + 1
        frame = await asyncio.wait_for(socket.sent.get(), 2)
        assert frame["type"] == "frame" and frame["state"]["frame_id"] == 41
        assert runtime.snapshot_calls == 4
        notice_count = len(cross_thread_notices)
        in_flight_callback = next(iter(runtime.snapshot_subscribers))
        await socket.close()
        await asyncio.wait_for(task, 2)
        assert not runtime.snapshot_subscribers
        await asyncio.to_thread(in_flight_callback)
        await asyncio.sleep(.01)
        assert runtime.snapshot_calls == 4
        assert len(cross_thread_notices) == notice_count

    asyncio.run(run())


def test_last_failure_notification_during_snapshot_copy_survives_coalescing_and_manual_reset(monkeypatch):
    class RacingRuntime(RuntimeFixture):
        def __init__(self):
            super().__init__()
            self.snapshot_calls, self.failed = 0, False
            self.reading, self.release = threading.Event(), threading.Event()

        def snapshot(self):
            self.snapshot_calls += 1
            if self.failed:
                raise RuntimeError("last publication was terminal failure")
            value = super().snapshot()
            if self.snapshot_calls == 2:
                self.reading.set()
                assert self.release.wait(2)
            return value

        def select_scene(self, scene_id):
            self.failed = False
            return super().select_scene(scene_id)

    async def run():
        runtime, socket = RacingRuntime(), MemorySocket()
        monkeypatch.setattr("scripts.mb_sonic_lab_server.web.WebSocketResponse", lambda **_: socket)
        app = create_app(runtime)
        handler = next(route.handler for route in app.router.routes()
                       if route.method == "GET" and route.resource.canonical == API + "/ws")
        task = asyncio.create_task(handler(None))
        ready = await asyncio.wait_for(socket.sent.get(), 2)
        first = await asyncio.wait_for(socket.sent.get(), 2)
        await socket.incoming.put(dict(acknowledgement(first), session_id=ready["session_id"]))
        assert (await asyncio.wait_for(socket.sent.get(), 2))["type"] == "render_receipt"
        assert await asyncio.to_thread(runtime.reading.wait, 1)
        runtime.failed = True
        await asyncio.to_thread(runtime.notify_snapshot)
        runtime.release.set()
        failure = await asyncio.wait_for(socket.sent.get(), 2)
        assert failure["type"] == "failure" and "last publication" in failure["message"]
        assert runtime.snapshot_calls == 3
        await asyncio.to_thread(runtime.notify_snapshot)
        await asyncio.sleep(.01)
        assert runtime.snapshot_calls == 3
        await socket.incoming.put(dict(type="reset_episode", session_id=ready["session_id"]))
        replies = [await asyncio.wait_for(socket.sent.get(), 2) for _ in range(2)]
        assert {row["type"] for row in replies} == {"config", "frame"}
        assert next(row for row in replies if row["type"] == "frame")["state"]["episode_id"] == 2
        assert runtime.begin_count == 0
        await socket.close()
        await asyncio.wait_for(task, 2)
        assert not runtime.snapshot_subscribers

    asyncio.run(run())


def test_subscription_registration_error_releases_websocket_ownership_without_starting_pump(monkeypatch):
    class RejectingRuntime(RuntimeFixture):
        def subscribe_snapshots(self, callback):
            callback()
            raise RuntimeError("subscription registration fixture failure")

        def snapshot(self):
            raise AssertionError("A failed registration must not start the pump")

    async def run():
        runtime, socket = RejectingRuntime(), MemorySocket()
        monkeypatch.setattr("scripts.mb_sonic_lab_server.web.WebSocketResponse", lambda **_: socket)
        app = create_app(runtime)
        handler = next(route.handler for route in app.router.routes()
                       if route.method == "GET" and route.resource.canonical == API + "/ws")
        with pytest.raises(RuntimeError, match="registration fixture"):
            await handler(None)
        await asyncio.sleep(0)
        diagnostics = next(route.handler for route in app.router.routes()
                           if route.method == "GET" and route.resource.canonical == API + "/diagnostics")
        body = json.loads((await diagnostics(None)).text)
        assert body["sessions"] == [] and len(body["closed_sessions"]) == 1
        assert socket.sent.empty() and not runtime.submissions

    asyncio.run(run())


def test_only_final_application_cleanup_shuts_down_runtime_once_after_scene_reset_and_disconnect(monkeypatch):
    class ShutdownRuntime(RuntimeFixture):
        def __init__(self):
            super().__init__()
            self.shutdown_count = 0

        def shutdown(self):
            self.shutdown_count += 1

    async def run():
        runtime, socket = ShutdownRuntime(), MemorySocket()
        monkeypatch.setattr("scripts.mb_sonic_lab_server.web.WebSocketResponse", lambda **_: socket)
        app = create_app(runtime, port=WASD_PORT)
        handler = next(route.handler for route in app.router.routes()
                       if route.method == "GET" and route.resource.canonical == API + "/ws")
        task = asyncio.create_task(handler(None))
        ready = await asyncio.wait_for(socket.sent.get(), 2)
        assert (await asyncio.wait_for(socket.sent.get(), 2))["type"] == "frame"
        for action in (dict(type="scene", scene_id="true3d_test"), dict(type="reset_episode")):
            await socket.incoming.put(dict(action, session_id=ready["session_id"]))
            replies = [await asyncio.wait_for(socket.sent.get(), 2) for _ in range(2)]
            assert {row["type"] for row in replies} == {"config", "frame"}
            assert runtime.shutdown_count == 0
        await socket.close()
        await asyncio.wait_for(task, 2)
        assert runtime.shutdown_count == 0 and not runtime.snapshot_subscribers
        app.freeze()
        await asyncio.gather(app.cleanup(), app.cleanup())
        await app.cleanup()
        assert runtime.shutdown_count == 1

    asyncio.run(run())


def test_static_packed_mesh_layout_roundtrip_is_not_json_geometry():
    vertices = [0., 0., 0., 1., 0., 0., 0., 1., 0.]
    indices = [0, 1, 2]
    buffer = struct.pack("<9f3I", *vertices, *indices)
    description = dict(asset_id=hashlib.sha256(buffer).hexdigest(), vertex_count=3, index_count=3)
    assert len(buffer) == (description["vertex_count"] * 3 + description["index_count"]) * 4
    assert list(struct.unpack_from("<3I", buffer, description["vertex_count"] * 12)) == indices


def test_follow_is_default_on_in_the_width_constrained_wrapping_layer_bar():
    class LayerControls(HTMLParser):
        def __init__(self):
            super().__init__()
            self.depth, self.layer_depth, self.follow = 0, 0, None

        def handle_starttag(self, tag, attributes):
            attrs = dict(attributes)
            if tag == "div":
                self.depth += 1
                if "layers" in attrs.get("class", "").split():
                    self.layer_depth = self.depth
            if tag == "input" and attrs.get("id") == "follow":
                self.follow = dict(attrs, in_layer_bar=self.layer_depth > 0)

        def handle_endtag(self, tag):
            if tag == "div":
                if self.depth == self.layer_depth:
                    self.layer_depth = 0
                self.depth -= 1

    root = Path(__file__).parents[1] / "web/mb_sonic_lab"
    controls = LayerControls()
    controls.feed((root / "index.html").read_text())
    assert controls.follow["type"] == "checkbox" and "checked" in controls.follow
    assert controls.follow["in_layer_bar"]
    css = (root / "styles.css").read_text()
    start = css.index(".layers {")
    declaration = css[start:css.index("}", start)]
    assert "display: flex;" in declaration and "flex-wrap: wrap;" in declaration
    assert "max-width: calc(100% - 32px);" in declaration


def test_browser_protocol_on_cpu_without_browser_or_service():
    path = Path(__file__).resolve().parents[1] / "web/mb_sonic_lab/protocol.test.mjs"
    result = subprocess.run(["node", "--experimental-vm-modules", "--test", str(path),
                             str(path.with_name("mesh_worker.test.mjs"))],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_high_resolution_proactor_timeout_never_fires_early_with_thread_wakeups():
    async def run():
        durations, completions = [], 0
        stopped = asyncio.Event()
        waiting = asyncio.Event()

        async def interleave_thread_completions():
            nonlocal completions
            while not stopped.is_set():
                await asyncio.to_thread(lambda: None)
                completions += 1

        worker = asyncio.create_task(interleave_thread_completions())
        try:
            for _ in range(100):
                await asyncio.to_thread(lambda: None)
                started = time.perf_counter()
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(waiting.wait(), timeout=.005)
                durations.append(time.perf_counter() - started)
        finally:
            stopped.set()
            await worker
        tolerance = time.get_clock_info("perf_counter").resolution * 2
        assert completions >= 100
        assert min(durations) >= .005 - tolerance
        milliseconds = sorted(value * 1000 for value in durations)
        print(f"High-resolution100x5ms: min={milliseconds[0]:.4f}ms "
              f"P50={milliseconds[49]:.4f}ms mean={sum(milliseconds)/100:.4f}ms "
              f"thread_completions={completions}")

    loop = HighResolutionProactorEventLoop()
    try:
        assert isinstance(loop, asyncio.ProactorEventLoop)
        assert loop._clock_resolution == time.get_clock_info("perf_counter").resolution
        before = time.perf_counter()
        assert before <= loop.time() <= time.perf_counter()
        loop.run_until_complete(run())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        loop.close()


def test_cli_passes_explicit_high_resolution_loop_without_policy_or_service_start(monkeypatch):
    from scripts import mb_sonic_lab_server as server

    args = parse_args(["--config", "fixture.json", "--allocation-id", "fixture", "--port", "28948"])
    calls = {}
    policy = asyncio.get_event_loop_policy()
    monkeypatch.setattr(server, "parse_args", lambda: args)
    monkeypatch.setitem(sys.modules, "motionbricks.mb_sonic.runtime", SimpleNamespace(
        create_lab_runtime=lambda config_path, allocation_id: RuntimeFixture()))

    def capture_run_app(app, **kwargs):
        calls.update(kwargs)
        assert app.on_startup and app.on_cleanup

    monkeypatch.setattr(server.web, "run_app", capture_run_app)
    try:
        server.main()
        assert isinstance(calls["loop"], HighResolutionProactorEventLoop)
        assert calls["host"] == "127.0.0.1" and calls["port"] == 28948
        assert asyncio.get_event_loop_policy() is policy
    finally:
        if "loop" in calls:
            calls["loop"].close()
