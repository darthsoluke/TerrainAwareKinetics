"""Loopback MB + SONIC G1 lab. Runtime owns all physics/control clocks."""

import argparse
import asyncio
from collections import deque
import json
import logging
import math
from pathlib import Path
import threading
import time
import uuid

from aiohttp import WSMsgType, web


ROOT = Path(__file__).resolve().parents[1]
API = "/api/mb-sonic"
SCHEMA = "motionbricks.mb_sonic.lab.v1"
PORT = 28947
WASD_PORT = 28948
LAB_PORTS = (PORT, WASD_PORT, 28949)
DISPLAY_HZ = 30
LOGGER = logging.getLogger("mb_sonic_lab")


class HighResolutionProactorEventLoop(asyncio.ProactorEventLoop):
    """Use one high-resolution clock for deadlines and early-timer tolerance."""

    def __init__(self):
        super().__init__()
        self._clock_resolution = time.get_clock_info("perf_counter").resolution

    def time(self):
        return time.perf_counter()


def json_text(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class LabSession:
    """A single transport slot; render receipts never commit simulation state."""

    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        self.id = uuid.uuid4().hex
        self.last_command_id = 0
        self.packet_id = 0
        self.last_frame_id = -1
        self.pending = None
        self.commands = deque(maxlen=256)
        self.receipts = deque(maxlen=256)
        self.failures = []
        self.lifecycle = deque(maxlen=64)
        self.sent_count = self.rendered_count = 0
        self.closed = False
        self.scene_revision = 0
        self.invalidated = deque(maxlen=32)
        self.last_ack_ns = None

    def intent(self, value):
        if not self.config["intent_enabled"]:
            raise ValueError("Official reference is read-only; no input command was consumed")
        scope = dict(scene_id=self.config["scene"]["manifest_id"],
                     scene_manifest_sha256=self.config["scene"]["manifest_sha256"],
                     episode_id=self.config["episode_id"])
        if any(value.get(key) != expected for key, expected in scope.items()):
            raise ValueError("Intent does not belong to the current scene and episode")
        command_id, velocity = value["command_id"], value["velocity_xy_mps"]
        if (type(command_id) is not int or command_id <= self.last_command_id
                or type(value["command_seq"]) is not int or value["command_seq"] != command_id
                or not isinstance(velocity, list) or len(velocity) != 2
                or not all(finite_number(x) for x in velocity)
                or not finite_number(value["yaw_rate_radps"])
                or not finite_number(value["client_input_ms"])):
            raise ValueError("Expected increasing intent ID, finite world XY velocity, yaw rate and input time")
        command = dict(**scope, session_id=self.id, command_id=command_id, command_seq=command_id,
                       client_input_ms=value["client_input_ms"],
                       velocity_xy_mps=velocity, yaw_rate_radps=value["yaw_rate_radps"],
                       input_frame="world_z_up", effective_intent=None,
                       server_received_ns=time.perf_counter_ns())
        self.runtime.submit_intent(command)
        self.last_command_id = command_id
        self.commands.append(command)
        return dict(type="intent_received", command_id=command_id,
                    server_received_ns=command["server_received_ns"])

    def begin_episode(self):
        if not self.config["start_required"] or self.config["start_action"] != "begin_episode":
            raise ValueError("This runtime does not request an explicit episode start")
        response = self.runtime.begin_episode()
        self.lifecycle.append(dict(action="begin_episode", scene_revision=self.scene_revision,
                                   server_monotonic_ns=time.perf_counter_ns()))
        return dict(type="episode_started", **response)

    def offer(self, snapshot):
        if self.pending is not None or snapshot is None:
            return None
        if snapshot["frame_id"] <= self.last_frame_id:
            return None
        scene = self.config["scene"]
        if (snapshot["scene_manifest_id"] != scene["manifest_id"]
                or snapshot["scene_manifest_sha256"] != scene["manifest_sha256"]):
            raise ValueError("Snapshot does not match the canonical displayed SceneManifest")
        if self.config["intent_enabled"] and snapshot["episode_id"] != self.config["episode_id"]:
            raise ValueError("Snapshot does not match the current input episode")
        self.packet_id += 1
        self.last_frame_id = snapshot["frame_id"]
        published_ns = time.perf_counter_ns()
        self.pending = dict(type="frame", schema=SCHEMA, session_id=self.id,
                            packet_id=self.packet_id, scene_revision=self.scene_revision,
                            published_server_ns=published_ns,
                            server_ack_to_publish_ms=None if self.last_ack_ns is None else
                            (published_ns - self.last_ack_ns) / 1e6,
                            state=snapshot)
        self.sent_count += 1
        return self.pending

    def rendered(self, value):
        if value["packet_id"] in self.invalidated:
            return dict(type="render_receipt", packet_id=value["packet_id"], invalidated_by_scene_change=True)
        pending = self.pending
        if pending is None:
            raise ValueError("No outstanding display sample")
        state = pending["state"]
        command = state["command"]
        command_id = command["command_id"] if command is not None else None
        start, end = value["client_render_started_ms"], value["client_render_completed_ms"]
        if (value["packet_id"] != pending["packet_id"] or value["frame_id"] != state["frame_id"]
                or value["command_id"] != command_id
                or not finite_number(start) or not finite_number(end) or end < start):
            raise ValueError("Render receipt does not match the published physics sample")
        self.last_ack_ns = time.perf_counter_ns()
        publish_to_ack_ms = (self.last_ack_ns - pending["published_server_ns"]) / 1e6
        self.receipts.append(dict(value, server_ack_received_ns=self.last_ack_ns,
                                 server_publish_to_ack_ms=publish_to_ack_ms))
        self.rendered_count += 1
        self.pending = None
        return dict(type="render_receipt", packet_id=value["packet_id"], frame_id=value["frame_id"],
                    server_ack_received_ns=self.last_ack_ns, server_publish_to_ack_ms=publish_to_ack_ms)

    def select_scene(self, scene_id, *, action="scene"):
        entry = next(row for row in self.config["scene_catalog"] if row["id"] == scene_id)
        if not entry["available"]:
            raise ValueError(entry["unavailable_reason"])
        self.config = self.runtime.select_scene(scene_id)
        if self.pending is not None:
            self.invalidated.append(self.pending["packet_id"])
        self.pending = None
        self.scene_revision += 1
        self.last_frame_id = -1
        self.last_ack_ns = None
        self.lifecycle.append(dict(action=action, scene_id=scene_id, scene_revision=self.scene_revision,
                                   server_monotonic_ns=time.perf_counter_ns()))
        return dict(type="config", config=self.config, scene_revision=self.scene_revision)

    def reset_episode(self):
        return self.select_scene(self.config["scene"]["manifest_id"], action="reset_episode")

    def fail(self, error):
        row = dict(error_type=type(error).__name__, message=str(error),
                   server_monotonic_ns=time.perf_counter_ns())
        self.failures.append(row)
        return dict(type="failure", **row)

    def disconnect(self):
        self.closed = True
        if not self.config["intent_enabled"] or not self.commands:
            return
        scope = dict(scene_id=self.config["scene"]["manifest_id"],
                     scene_manifest_sha256=self.config["scene"]["manifest_sha256"],
                     episode_id=self.config["episode_id"])
        # Reset retains diagnostics, but old commands do not own the new episode.
        if any(self.commands[-1][key] != value for key, value in scope.items()):
            return
        self.runtime.submit_intent(dict(session_id=self.id, command_id=self.last_command_id + 1,
            command_seq=self.last_command_id + 1,
            client_input_ms=None, velocity_xy_mps=[0., 0.], yaw_rate_radps=0.,
            input_frame="world_z_up", effective_intent=None,
            server_received_ns=time.perf_counter_ns(), source="disconnect_stop",
            **scope))

    def diagnostics(self):
        return dict(schema=SCHEMA, session_id=self.id, closed=self.closed,
                    display_hz=DISPLAY_HZ, physics_clock_owner="runtime", controller_clock_owner="runtime",
                    presentation_clock_owner="browser", display_stream_mode="runtime_publication_and_ack",
                    command_queue_capacity=1, transport_slot_capacity=1,
                    sent_count=self.sent_count, rendered_count=self.rendered_count,
                    outstanding_packet_id=None if self.pending is None else self.pending["packet_id"],
                    commands=list(self.commands), render_receipts=list(self.receipts),
                    lifecycle=list(self.lifecycle), failures=self.failures)


def create_app(runtime, *, port=PORT, evidence_path=None):
    if port not in LAB_PORTS:
        raise ValueError("Only independently allocated lab ports28947/28948 are supported")
    sessions, history = {}, []
    active_lock = asyncio.Lock()
    state_lock = asyncio.Lock()
    cleanup_lock = asyncio.Lock()
    runtime_closed = False

    @web.middleware
    async def loopback(request, handler):
        if (request.remote != "127.0.0.1" or request.host != f"127.0.0.1:{port}"
                or request.headers.get("Origin", f"http://127.0.0.1:{port}") != f"http://127.0.0.1:{port}"):
            raise web.HTTPForbidden(text="Loopback same-origin lab only")
        response = await handler(request)
        if "Cache-Control" not in response.headers:
            response.headers["Cache-Control"] = "no-store"
        return response

    app = web.Application(client_max_size=16384, middlewares=[loopback])

    async def config(_request):
        async with state_lock:
            return web.json_response(await asyncio.to_thread(runtime.config), dumps=json_text)

    async def binary_asset(request):
        asset_id = request.match_info["asset_id"]
        data = await asyncio.to_thread(runtime.asset, asset_id)
        return web.Response(body=data, content_type="application/octet-stream",
                            headers={"Cache-Control": "private, max-age=31536000, immutable"})

    async def diagnostics(_request):
        return web.json_response(dict(schema=SCHEMA, sessions=[s.diagnostics() for s in sessions.values()],
                                      closed_sessions=history), dumps=json_text)

    async def websocket(request):
        # Cold scene RPCs can occupy this reader for180s; motion dead-man is separate.
        socket = web.WebSocketResponse(max_msg_size=16384, heartbeat=180)
        send_lock = asyncio.Lock()
        transport_failed = False

        async def send(value):
            nonlocal transport_failed
            async with send_lock:
                if socket.closed or transport_failed:
                    if not transport_failed:
                        session.fail(ConnectionResetError(f"WebSocket closed before {value['type']} publish"))
                        transport_failed = True
                    return False
                try:
                    await socket.send_str(json_text(value))
                except ConnectionError as error:
                    transport_failed = True
                    session.fail(error)
                    return False
                return True

        async with active_lock:
            if sessions:
                raise web.HTTPConflict(text="One G1 controlling session is already connected")
            async with state_lock:
                session = LabSession(runtime, await asyncio.to_thread(runtime.config))
            await socket.prepare(request)
            sessions[socket] = session
        wake = asyncio.Event()
        loop = asyncio.get_running_loop()
        notification_lock = threading.Lock()
        subscribed = True
        slot_free, notice_queued, dirty = True, False, False

        def publication_ready():
            with notification_lock:
                if subscribed and slot_free and dirty and not socket.closed:
                    wake.set()

        def snapshot_published():
            nonlocal notice_queued, dirty
            with notification_lock:
                if not subscribed:
                    return
                dirty = True
                if not slot_free or notice_queued:
                    return
                notice_queued = True
            loop.call_soon_threadsafe(publication_ready)

        async def pump():
            nonlocal slot_free, notice_queued, dirty
            while not socket.closed and not transport_failed:
                wake.clear()
                if session.pending is None:
                    with notification_lock:
                        slot_free, notice_queued, dirty = False, False, False
                    async with state_lock:
                        snapshot_started_ns = time.perf_counter_ns()
                        snapshot = await asyncio.to_thread(runtime.snapshot)
                        snapshot_ms = (time.perf_counter_ns() - snapshot_started_ns) / 1e6
                        frame = session.offer(snapshot)
                        if frame is not None:
                            frame["server_snapshot_wall_ms"] = snapshot_ms
                            if not await send(frame):
                                return
                    if session.pending is None:
                        with notification_lock:
                            slot_free = True
                            if dirty:
                                notice_queued = True
                                wake.set()
                await wake.wait()

        async def guarded_pump():
            while not socket.closed and not transport_failed:
                try:
                    await pump()
                except Exception as error:
                    failed_revision = session.scene_revision
                    LOGGER.exception("G1 display stream failed; waiting for explicit scene/reset")
                    if not await send(session.fail(error)):
                        return
                    while not socket.closed and session.scene_revision == failed_revision:
                        wake.clear()
                        await wake.wait()

        task, unsubscribe = None, None
        try:
            unsubscribe = runtime.subscribe_snapshots(snapshot_published)
            if not await send(dict(type="ready", schema=SCHEMA, session_id=session.id,
                                   scene_revision=session.scene_revision, config=session.config)):
                return socket
            task = asyncio.create_task(guarded_pump())
            async for incoming in socket:
                if incoming.type != WSMsgType.TEXT:
                    if incoming.type == WSMsgType.BINARY:
                        await socket.close(code=1003, message=b"JSON messages required")
                    continue
                try:
                    value = json.loads(incoming.data)
                    kind = value["type"]
                    if value["session_id"] != session.id:
                        raise ValueError("Wrong controlling session")
                    if kind == "intent":
                        response = session.intent(value)
                    elif kind == "begin_episode":
                        async with state_lock:
                            response = await asyncio.to_thread(session.begin_episode)
                        wake.set()
                    elif kind == "rendered":
                        response = session.rendered(value)
                        wake.set()
                    elif kind in ("scene", "reset_episode"):
                        async with state_lock:
                            if kind == "reset_episode":
                                response = await asyncio.to_thread(session.reset_episode)
                            else:
                                response = await asyncio.to_thread(session.select_scene, value["scene_id"])
                            if not await send(dict(schema=SCHEMA, session_id=session.id, **response)):
                                break
                        wake.set()
                        continue
                    elif kind == "receipt":
                        response = dict(type="diagnostics", diagnostics=session.diagnostics())
                    else:
                        raise ValueError("Unknown lab message")
                    if not await send(dict(schema=SCHEMA, session_id=session.id, **response)):
                        break
                except (KeyError, TypeError, ValueError, StopIteration) as error:
                    if not await send(dict(type="error", schema=SCHEMA, message=str(error),
                                           scene_revision=session.scene_revision)):
                        break
                except Exception as error:
                    LOGGER.exception("G1 runtime operation failed")
                    if not await send(session.fail(error)):
                        break
        finally:
            with notification_lock:
                subscribed = False
                slot_free, notice_queued, dirty = False, False, False
            if unsubscribe is not None:
                unsubscribe()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            try:
                session.disconnect()
            except Exception as error:
                session.fail(error)
            history.append(session.diagnostics())
            if evidence_path is not None:
                path = Path(evidence_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(schema=SCHEMA, closed_sessions=history),
                                           allow_nan=False, indent=2), encoding="utf-8")
            sessions.pop(socket, None)
        return socket

    async def index(_request):
        return web.FileResponse(ROOT / "web/mb_sonic_lab/index.html")

    async def asset(request):
        return web.FileResponse(ROOT / "web/mb_sonic_lab" / request.match_info["name"])

    async def startup(_app):
        await asyncio.to_thread(runtime.start)

    async def shutdown(_app):
        for socket in list(sessions):
            await socket.close(code=1001, message=b"Parent stopped lab")

    async def cleanup(_app):
        nonlocal runtime_closed
        async with cleanup_lock:
            if runtime_closed:
                return
            runtime_closed = True
            await asyncio.to_thread(runtime.shutdown)

    app.router.add_get("/", index)
    app.router.add_get("/{name:app\\.js|protocol\\.mjs|styles\\.css}", asset)
    app.router.add_static("/vendor/", ROOT / "web/full_corpus_lab/vendor", show_index=False)
    app.router.add_get(API + "/config", config)
    app.router.add_get(API + "/assets/{asset_id:[0-9a-f]{64}}", binary_asset)
    app.router.add_get(API + "/diagnostics", diagnostics)
    app.router.add_get(API + "/ws", websocket)
    app.on_startup.append(startup)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(cleanup)
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--port", type=int, choices=LAB_PORTS, default=PORT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument('--stable-gpu-profile',action='store_true',
        help='Opt-in development-only stable GPU power scope; restores on server exit')
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    from motionbricks.mb_sonic.runtime import create_lab_runtime

    from contextlib import nullcontext
    from scripts.windows_gpu_profile import StableGpuProfile

    scope = StableGpuProfile() if args.stable_gpu_profile else nullcontext()
    with scope:
        runtime = create_lab_runtime(args.config, allocation_id=args.allocation_id)
        runtime.hardware_execution_profile = (
            'development_stable_gpu_power' if args.stable_gpu_profile else 'normal_dynamic_gpu_power')
        web.run_app(create_app(runtime, port=args.port, evidence_path=args.evidence),
                    host="127.0.0.1", port=args.port, loop=HighResolutionProactorEventLoop())


if __name__ == "__main__":
    main()
