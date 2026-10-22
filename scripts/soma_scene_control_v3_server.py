"""Loopback-only SOMA v3 source-reference and untrained-capacity workbench."""

import argparse
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import time
import traceback
import uuid

from aiohttp import WSMsgType, web


ROOT = Path(__file__).resolve().parents[1]
API = "/api/scene-control-v3"
SCHEMA = "motionbricks.scene_control_v3.viewer.v1"
FRAME_BYTES = (3 + 77 * 16) * 4
LOGGER = logging.getLogger("scene_control_v3")


class ClosedSessions:
    limit = 16

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self.rows = deque(maxlen=self.limit)
        if self.path is not None and self.path.exists():
            self.rows.extend(json.loads(self.path.read_text(encoding="utf-8"))["closed_sessions"])

    def append(self, value):
        self.rows.append(dict(value, closed_utc=datetime.now(timezone.utc).isoformat()))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(schema=SCHEMA, closed_sessions=list(self.rows)),
                                             allow_nan=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)


def status_contract(mode="source"):
    return dict(mode=mode, actor_status="UNTRAINED", optimizer_updates=0, quality_response="unmeasured",
                learned_wasd=False, benchmark_role=("source_replay_reference" if mode == "source"
                                                    else "architecture_capacity_only"),
                command_used_for_motion=mode == "capacity",
                display_measurement="client_render_submission_not_scanout")


@dataclass(frozen=True)
class PreparedFrame:
    payload: bytes
    timings: dict
    candidate: object = None
    object_pose: object = None
    query_metadata: object = None


class SourceReplayRuntime:
    """Read the record's original world trajectory and its bound scene, without a model."""

    mode = "source"

    def __init__(self, bundle_path, source_config):
        import torch
        from motionbricks.scene_control_v3.native import NativeSOMAAdapter
        from motionbricks.scene_control_v2.provider import load_bound_world

        self.adapter = NativeSOMAAdapter(bundle_path, device="cpu")
        spec = json.loads(Path(source_config).read_text(encoding="utf-8"))["provider"]
        self.world, _, self.scene, _, self.lineage = load_bound_world(spec, self.adapter.representation)
        self.frame_count = len(self.world)
        with torch.inference_mode():
            self.initial = self.adapter.source_frame(self.world[None, :1])
        self._reference = None

    def config(self):
        heights = self.scene["heightfield_y_m"].tolist()
        heights = [[value if math.isfinite(value) else None for value in row] for row in heights]
        return dict(schema=SCHEMA, **status_contract(), fps=30, vertex_count=18056,
                    source_frame_count=self.frame_count, source=self.lineage,
                    soma_skin=self.adapter.skin_asset(),
                    initial_root=self.initial["root"].reshape(-1).tolist(),
                    initial_rig=self.initial["rig"].float().reshape(-1).tolist(),
                    scene=dict(kind="bound_source_heightfield", heightfield_y_m=heights,
                               min_point_world_xyz_m=self.scene["min_point_world_xyz_m"].tolist(),
                               dx_m=self.scene["dx_m"]), websocket_path=API + "/ws",
                    proposed_budgets=dict(inference_p99_ms=20, command_associated_render_p95_ms=100,
                                          target_unique_fps=30, measured=False))

    def new_session(self):
        return None

    def commit(self, _session, _frame):
        pass

    def frame(self, _session, index, command):
        import torch
        from motionbricks.data.soma_lab import pack_rig_frames

        started = time.perf_counter()
        with torch.inference_mode():
            value = self.adapter.source_frame(self.world[None, index:index + 1])
        rig_finished = time.perf_counter()
        payload = pack_rig_frames(value["root"].reshape(1, 3).cpu().numpy(),
                                  value["rig"].reshape(1, 77, 4, 4).cpu().numpy())
        return PreparedFrame(payload, dict(native_rig_ms=(rig_finished - started) * 1000,
                                         transport_encode_ms=(time.perf_counter() - rig_finished) * 1000,
                                         necessary_inference_ms=None))

    def reference(self):
        import torch

        if self._reference is None:
            with torch.inference_mode():
                vertices = self.adapter.fullskin(self.initial["rig"])
            self._reference = dict(schema=SCHEMA, frame_index=0, vertex_count=18056,
                                   vertices=vertices.reshape(-1).cpu().tolist(),
                                   provenance="official_fullskin_of_source_frame0_rig_outside_timed_loop",
                                   source=self.lineage)
        return self._reference


class CapacityRuntime:
    """Shared static resources; a new transactional pipeline for each connection."""

    mode = "capacity"

    def __init__(self, bundle_path, config_path, *, device, scene_id, hidden_dim,
                 max_frames, allocation_id):
        import torch
        from motionbricks.scene_control_v3 import CausalSceneActor, SceneActorConfig
        from motionbricks.scene_control_v3.geometry import build_true3d_scene
        from motionbricks.scene_control_v3.native import NativeSOMAAdapter

        if scene_id != "true3d_clutter":
            raise ValueError("Capacity viewer requires the explicit true3d_clutter scene")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if device == "cuda:0":
            torch.cuda.set_per_process_memory_fraction(config["benchmark"]["torch_memory_fraction"], device)
        self.seed, self.device, self.scene_id = config["seed"], device, scene_id
        self.device_name = torch.cuda.get_device_name(device) if device == "cuda:0" else "cpu"
        self.frame_count, self.allocation_id = max_frames, allocation_id
        torch.manual_seed(self.seed)
        self.adapter = NativeSOMAAdapter(bundle_path, device=device)
        self.actor = CausalSceneActor(SceneActorConfig(hidden_dim=hidden_dim,
            preview_frames=config["benchmark"]["preview_frames"])).to(device).eval().requires_grad_(False)
        self.scene = build_true3d_scene(device=device)
        self.initial_history = self.adapter.initial_history.clone()
        self.object_velocity = self.initial_history.new_zeros(1, self.scene.object_count, 6)
        self.object_velocity[:, 1, 0] = .06
        with torch.inference_mode():
            self.initial = self.adapter.source_frame(self.initial_history[:, -1:])
        template = self.new_session()
        self.scene_asset, self.pipeline_contract = template.scene_asset(), template.contract()
        self._reference = None

    def new_session(self):
        from motionbricks.scene_control_v3.pipeline import CapacityPipeline

        return CapacityPipeline(self.adapter, self.actor, self.scene, self.initial_history,
                                object_velocity=self.object_velocity)

    @staticmethod
    def commit(session, frame):
        session.commit(frame.candidate.frame_id)

    def frame(self, session, _index, command):
        started = time.perf_counter()
        candidate = session.propose(command["command"])
        pose = candidate.next_state.object_pose[0].detach().cpu().tolist()
        duration = (time.perf_counter() - started) * 1000
        return PreparedFrame(candidate.payload, dict(necessary_inference_ms=duration,
            native_rig_ms=None, measured_scope="contact_query_encode_actor_native_rig_validity_payload_and_object_pose_D2H"),
            candidate=candidate, object_pose=pose, query_metadata=candidate.query_metadata)

    def config(self):
        return dict(schema=SCHEMA, **status_contract(self.mode), fps=30, vertex_count=18056,
                    source_frame_count=None, generated_frame_limit=self.frame_count,
                    source=None, soma_skin=self.adapter.skin_asset(),
                    initial_root=self.initial["root"].reshape(-1).cpu().tolist(),
                    initial_rig=self.initial["rig"].float().reshape(-1).cpu().tolist(),
                    scene=self.scene_asset, websocket_path=API + "/ws",
                    capacity=dict(scene_id=self.scene_id, device=self.device, seed=self.seed,
                                  hidden_dim=self.actor.config.hidden_dim,
                                  preview_frames=self.actor.config.preview_frames,
                                  parameter_count=sum(p.numel() for p in self.actor.parameters()),
                                  actor_dtype=str(next(self.actor.parameters()).dtype),
                                  process_id=os.getpid(),
                                  device_name=self.device_name,
                                  allocation_id=self.allocation_id,
                                  contact_observations="synthetic_capacity_fixture_not_measured_contacts",
                                  execution="eager", scene_feature_dim=self.actor.config.scene_dim,
                                  pipeline=self.pipeline_contract),
                    proposed_budgets=dict(inference_p99_ms=20, command_associated_render_p95_ms=100,
                                          target_unique_fps=30, measured=False))

    def reference(self):
        import torch

        if self._reference is None:
            with torch.inference_mode():
                vertices = self.adapter.fullskin(self.initial["rig"])
            self._reference = dict(schema=SCHEMA, frame_index=None, vertex_count=18056,
                vertices=vertices.reshape(-1).cpu().tolist(), source=None,
                provenance="official_fullskin_of_native_initial_seed_outside_timed_loop")
        return self._reference


class ReplaySession:
    """One pending frame, latest command only; ACK is the sole committed frontier."""

    def __init__(self, runtime, send, run, state):
        self.runtime, self.send, self.run = runtime, send, run
        self.state, self.pending_frame = state, None
        self.session_id = uuid.uuid4().hex
        self.paused, self.closed = True, False
        self.command, self.pending = None, None
        self.committed, self.packet_id, self.superseded = 0, 0, 0
        self.last_sampled = 0
        self.failure, self.last_query_metadata = None, None
        self.events = deque(maxlen=256)
        self.wake = asyncio.Event()

    def record(self, kind, **fields):
        self.events.append(dict(type=kind, server_monotonic_ns=time.perf_counter_ns(), **fields))

    def receive_command(self, value):
        number, vector = value["command_id"], value["command"]
        if (value["session_id"] != self.session_id or type(number) is not int
                or number != (self.command["command_id"] + 1 if self.command else 1)):
            raise ValueError("Expected the next command_id in this session")
        if (not isinstance(vector, list) or len(vector) != 3
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)
                or math.hypot(*vector[:2]) > 3.000001 or abs(vector[2]) > math.pi):
            raise ValueError("Expected finite vx,vz,heading within 3 m/s and +/-pi")
        stamp = value["client_input_ms"]
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp < 0:
            raise ValueError("Expected a finite client input timestamp")
        if self.command and self.command["command_id"] > self.last_sampled:
            self.superseded += 1
        self.command = dict(command_id=number, command=list(vector), client_input_ms=stamp,
                            command_received_ns=time.perf_counter_ns())
        self.record("command_received", **self.command)
        self.wake.set()
        return dict(type="command_received", **self.command)

    def set_paused(self, value):
        if type(value) is not bool:
            raise ValueError("paused must be boolean")
        self.paused = value
        self.wake.set()
        return dict(type="pause_applied", paused=value)

    async def publish_once(self):
        if self.paused or self.pending or not self.command or self.committed == self.runtime.frame_count:
            return False
        command = dict(self.command)
        started = time.perf_counter_ns()
        frame = await self.run(self.runtime.frame, self.state, self.committed, command)
        if self.closed:
            return False
        if len(frame.payload) != FRAME_BYTES:
            raise ValueError("Expected one complete official SOMA77 rig frame")
        self.packet_id += 1
        self.last_sampled = command["command_id"]
        meta = dict(type="frame", schema=SCHEMA, session_id=self.session_id,
                    packet_id=self.packet_id, frame_index=self.committed, frame_count=1, fps=30,
                    vertex_count=18056, rig_count=77, geometry_format="soma77-rig-v1",
                    encoding="float32-le", frame_stride_floats=3 + 77 * 16,
                    **command, **status_contract(self.runtime.mode), timings=frame.timings,
                    server_prepare_started_ns=started, server_published_ns=time.perf_counter_ns())
        if frame.object_pose is not None:
            meta["object_pose"] = frame.object_pose
        if frame.query_metadata is not None:
            meta["query_metadata"] = frame.query_metadata
        self.pending = meta
        self.pending_frame = frame
        self.last_query_metadata = frame.query_metadata
        self.record("frame_published", packet_id=self.packet_id, frame_index=self.committed,
                    command_id=command["command_id"])
        await self.send(meta, frame.payload)
        return True

    def acknowledge(self, value):
        pending = self.pending
        identity = ("session_id", "packet_id", "frame_index", "command_id", "client_input_ms")
        if not pending or any(value[key] != pending[key] for key in identity):
            raise ValueError("ACK does not identify the one pending rendered frame")
        start, end = value["client_render_started_ms"], value["client_render_completed_ms"]
        if (any(type(v) not in (float, int) or not math.isfinite(v) for v in (start, end))
                or start < pending["client_input_ms"] or end < start):
            raise ValueError("ACK requires ordered client render timestamps")
        self.runtime.commit(self.state, self.pending_frame)
        self.committed += 1
        self.pending = None
        self.pending_frame = None
        self.record("frame_acknowledged", **{key: value[key] for key in identity},
                    client_render_started_ms=start, client_render_completed_ms=end)
        self.wake.set()
        return dict(type="committed", **{key: value[key] for key in identity},
                    committed_frame_count=self.committed, completed=self.committed == self.runtime.frame_count)

    async def pump(self):
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()
            await self.publish_once()

    def diagnostics(self):
        return dict(session_id=self.session_id, **status_contract(self.runtime.mode),
                    paused=self.paused, committed_frame_count=self.committed,
                    pending_packet_id=self.pending["packet_id"] if self.pending else None,
                    latest_command=self.command, command_queue_capacity=1, frame_queue_capacity=1,
                    superseded_commands=self.superseded, events=list(self.events),
                    failure=self.failure, last_query_metadata=self.last_query_metadata)


def create_app(runtime, *, port=28946, web_root=ROOT / "web/scene_control_v3",
               vendor_root=ROOT / "web/full_corpus_lab/vendor", closed_diagnostics=None):
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-v3-cpu")
    sessions = {}
    session_lock = asyncio.Lock()
    closed = ClosedSessions(closed_diagnostics)

    async def run(function, *args):
        return await asyncio.get_running_loop().run_in_executor(executor, function, *args)

    @web.middleware
    async def loopback(request, handler):
        origin = f"http://127.0.0.1:{port}"
        if request.remote != "127.0.0.1" or request.host != f"127.0.0.1:{port}" or (
                "Origin" in request.headers and request.headers["Origin"] != origin):
            raise web.HTTPForbidden(text="127.0.0.1 same-origin workbench only")
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    app = web.Application(client_max_size=4096, middlewares=[loopback])

    async def config(_request):
        return web.json_response(await run(runtime.config), dumps=lambda v: json.dumps(v, allow_nan=False))

    async def reference(_request):
        return web.json_response(await run(runtime.reference))

    async def diagnostics(_request):
        return web.json_response(dict(schema=SCHEMA, **status_contract(runtime.mode),
                                      captured_server_monotonic_ns=time.perf_counter_ns(),
                                      sessions=[value.diagnostics() for value in sessions.values()],
                                      closed_session_limit=closed.limit, closed_sessions=list(closed.rows)))

    async def websocket(request):
        socket = web.WebSocketResponse(max_msg_size=4096, heartbeat=20)
        send_lock = asyncio.Lock()

        async def send(value, payload=None):
            async with send_lock:
                await socket.send_str(json.dumps(value, allow_nan=False))
                if payload is not None:
                    await socket.send_bytes(payload)

        async with session_lock:
            if sessions:
                raise web.HTTPConflict(text="One performance session at a time")
            state = await run(runtime.new_session)
            session = ReplaySession(runtime, send, run, state)
            await socket.prepare(request)
            sessions[socket] = session
        await send(dict(type="ready", schema=SCHEMA, session_id=session.session_id, paused=True,
                        frame_limit=runtime.frame_count, **status_contract(runtime.mode)))
        task = asyncio.create_task(session.pump())

        def pump_done(done):
            if done.cancelled():
                return
            try:
                done.result()
            except Exception as error:
                session.failure = dict(error_type=type(error).__name__, message=str(error),
                    traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)))
                session.record("stream_failed", error_type=type(error).__name__, message=str(error))
                LOGGER.exception("Session %s failed after %d committed frames", session.session_id, session.committed)
                reason = f"{type(error).__name__}: {error}".encode("utf-8")[:123].decode("utf-8", errors="ignore")
                asyncio.create_task(socket.close(code=1011, message=reason.encode("utf-8")))

        task.add_done_callback(pump_done)
        try:
            async for incoming in socket:
                if incoming.type != WSMsgType.TEXT:
                    if incoming.type == WSMsgType.BINARY:
                        await socket.close(code=1003, message=b"Expected JSON commands")
                    continue
                try:
                    value = json.loads(incoming.data)
                    if not isinstance(value, dict) or value.get("session_id") != session.session_id:
                        raise ValueError("Expected a message for this session")
                    kind = value["type"]
                    if kind == "command":
                        response = session.receive_command(value)
                    elif kind == "rendered":
                        response = session.acknowledge(value)
                    elif kind == "pause":
                        response = session.set_paused(value["paused"])
                    else:
                        raise ValueError("Unknown v3 message")
                    await send({"schema": SCHEMA, "session_id": session.session_id, **response})
                except (ValueError, KeyError, TypeError) as error:
                    await send(dict(type="error", schema=SCHEMA, message=str(error)))
        finally:
            session.closed = True
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            closed.append(dict(session.diagnostics(), close_code=socket.close_code))
            sessions.pop(socket, None)
        return socket

    async def index(_request):
        return web.FileResponse(web_root / "index.html")

    async def asset(request):
        return web.FileResponse(web_root / request.match_info["name"])

    async def shutdown(_app):
        for socket in list(sessions):
            await socket.close()

    async def cleanup(_app):
        executor.shutdown(wait=True, cancel_futures=False)

    app.router.add_get("/", index)
    app.router.add_get("/scene-control-v3/", index)
    app.router.add_get("/scene-control-v3/{name:app\\.js|styles\\.css}", asset)
    app.router.add_static("/vendor/", vendor_root, show_index=False)
    app.router.add_get(API + "/config", config)
    app.router.add_get(API + "/reference", reference)
    app.router.add_get(API + "/diagnostics", diagnostics)
    app.router.add_get(API + "/ws", websocket)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(cleanup)
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("source", "capacity"), default="source")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--scene", choices=("true3d_clutter",), default="true3d_clutter")
    parser.add_argument("--hidden-dim", choices=(256, 512, 1024), type=int, default=256)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--allocation-id")
    parser.add_argument("--closed-diagnostics", type=Path,
                        default=ROOT / "out/scene_control_v3_20261018/closed_sessions.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scene_control_v3_performance.json")
    parser.add_argument("--bundle", type=Path, default=ROOT / "out/a100_live_20261010/a100-terrain.pt")
    parser.add_argument("--source-config", type=Path, default=ROOT / "configs/scene_control_v2_real_provider_cpu.json")
    parser.add_argument("--port", default=28946, type=int, choices=(28946,))
    args = parser.parse_args(argv)
    if args.mode == "source" and args.device != "cpu":
        parser.error("Source replay uses CPU; CUDA requires explicit --mode capacity")
    if args.device == "cuda:0" and not args.allocation_id:
        parser.error("CUDA requires --allocation-id from the parent")
    if not 1 <= args.max_frames <= 3000:
        parser.error("--max-frames must be in 1..3000")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "2"
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    runtime = (SourceReplayRuntime(args.bundle, args.source_config) if args.mode == "source"
               else CapacityRuntime(args.bundle, args.config, device=args.device, scene_id=args.scene,
                                    hidden_dim=args.hidden_dim, max_frames=args.max_frames,
                                    allocation_id=args.allocation_id))
    web.run_app(create_app(runtime, port=args.port, closed_diagnostics=args.closed_diagnostics),
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
