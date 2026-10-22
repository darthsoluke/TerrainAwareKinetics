"""Independent SOMA scene-control-v2 viewer/server; never reuses an old port."""

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path

from aiohttp import WSMsgType, web

from motionbricks.scene_control_v2.runtime import (
    RUNTIME_SCHEMA, RuntimeSettings, SceneControlDispatcher, SerializedWorker, SOMADualBackend,
)


ROOT = Path(__file__).resolve().parents[1]
API = "/api/scene-control-v2"


def create_app(runtime, *, settings=RuntimeSettings(), backend_factory=SOMADualBackend,
               web_root=ROOT / "web/scene_control_v2",
               vendor_root=ROOT / "web/full_corpus_lab/vendor"):
    worker = SerializedWorker()
    sessions = {}
    scene_lock = asyncio.Lock()
    selecting = False
    app = web.Application(client_max_size=4096)

    def ready(dispatch):
        history = dispatch.committed_state.history
        return dict(type="ready", schema=RUNTIME_SCHEMA, scene_revision=dispatch.scene_revision,
                    settings_revision=dispatch.settings_revision, settings=dispatch.settings.contract(),
                    vertex_count=runtime.vertex_count, paused=dispatch.paused,
                    heading=math.atan2(float(history[0, -1, 4]), float(history[0, -1, 3])))

    async def configuration(_request):
        return web.json_response(await worker.run(config_value))

    def config_value():
        value = runtime.config()
        value["initial_vertices"] = runtime.terrain_initial_vertices[0].reshape(-1).tolist()
        value["initial_root"] = runtime.terrain_initial_root[0].tolist()
        value["scene_control_v2"] = dict(schema=RUNTIME_SCHEMA, settings=settings.contract(),
                                         scene_revision=runtime.scene_revision, settings_revision=0,
                                         websocket_path=API + "/ws",
                                         quality_gates=dict(vertices=18056, substeps=8,
                                                            peak_penetration_m=.02, required=True))
        return value

    async def select_map(request):
        nonlocal selecting
        value = await request.json()
        if not isinstance(value, dict) or set(value) != {"map_id"}:
            raise web.HTTPBadRequest(text="Expected one map_id")
        async with scene_lock:
            if selecting or any(dispatch.pending is not None for dispatch in sessions.values()):
                raise web.HTTPConflict(text="Drain all published packets before changing maps")
            if not isinstance(value["map_id"], str) or value["map_id"] not in runtime.entries:
                raise web.HTTPBadRequest(text="Unknown map_id")
            selecting = True
            try:
                active = list(sessions.items())
                for _socket, dispatch in active:
                    dispatch.stop()
                for _socket, dispatch in active:
                    await dispatch.close()
                await worker.run(runtime.select_map, value["map_id"])
                result = await worker.run(config_value)
                for socket, _dispatch in active:
                    await socket.send_json(dict(type="scene_changed", schema=RUNTIME_SCHEMA,
                                                scene_revision=runtime.scene_revision))
                    await socket.close(code=1012, message=b"Scene changed; reconnect with fresh config")
                return web.json_response(result)
            finally:
                selecting = False

    async def websocket(request):
        if selecting:
            raise web.HTTPConflict(text="Scene selection is in progress")
        socket = web.WebSocketResponse(max_msg_size=4096, heartbeat=20)
        await socket.prepare(request)
        send_lock = asyncio.Lock()

        async def send(metadata_json, payload):
            async with send_lock:
                await socket.send_str(metadata_json)
                if payload is not None:
                    await socket.send_bytes(payload)

        def initialize():
            backend = backend_factory(runtime)
            return backend, backend.initial_state()

        async with scene_lock:
            backend, initial_state = await worker.run(initialize)
            dispatch = SceneControlDispatcher(backend, initial_state, send, worker,
                                              scene_revision=runtime.scene_revision, settings=settings)
            sessions[socket] = dispatch
        dispatch.start()
        await send(json.dumps(await worker.run(ready, dispatch)), None)
        ack_tasks = set()

        async def acknowledge(message):
            try:
                await dispatch.acknowledge(message)
            except (ValueError, KeyError, TypeError) as error:
                if dispatch.transport_failed:
                    return
                response = dict(type="error", schema=RUNTIME_SCHEMA, message=str(error),
                                packet_id=message.get("packet_id"))
                await send(json.dumps(response, allow_nan=False), None)
            except (ConnectionError, OSError):
                if not dispatch.transport_failed:
                    raise

        try:
            async for incoming in socket:
                if incoming.type != WSMsgType.TEXT:
                    if incoming.type == WSMsgType.ERROR:
                        break
                    continue
                message = {}
                try:
                    message = json.loads(incoming.data)
                    if not isinstance(message, dict):
                        raise ValueError("Expected a message object")
                    kind = message.get("type")
                    if kind == "command":
                        response = dispatch.receive_command(message)
                    elif kind == "played":
                        task = asyncio.create_task(acknowledge(message))
                        ack_tasks.add(task)
                        task.add_done_callback(ack_tasks.discard)
                        continue
                    elif kind == "settings":
                        if (type(message.get("scene_revision")) is not int
                                or message["scene_revision"] != dispatch.scene_revision):
                            raise ValueError("Stale scene revision")
                        values = message["settings"]
                        if not isinstance(values, dict) or set(values) != set(asdict(settings)):
                            raise ValueError("Expected complete packet_frames/lookahead_tokens/dense_debug settings")
                        dispatch.update_settings(RuntimeSettings(**values), message["settings_revision"])
                        response = dict(type="settings_applied", schema=RUNTIME_SCHEMA,
                                        scene_revision=dispatch.scene_revision,
                                        settings_revision=dispatch.settings_revision,
                                        settings=dispatch.settings.contract())
                    elif kind == "pause":
                        dispatch._check_revisions(message)
                        dispatch.set_paused(message["paused"])
                        response = dict(type="pause_applied", schema=RUNTIME_SCHEMA,
                                        scene_revision=dispatch.scene_revision,
                                        settings_revision=dispatch.settings_revision, paused=dispatch.paused)
                    else:
                        raise ValueError("Unknown scene-control-v2 message type")
                    await send(json.dumps(response, allow_nan=False), None)
                except (ValueError, KeyError, TypeError) as error:
                    await send(json.dumps(dict(type="error", schema=RUNTIME_SCHEMA,
                                               message=str(error),
                                               command_id=message.get("command_id")
                                               if isinstance(message, dict) else None)), None)
        finally:
            await dispatch.close()
            if ack_tasks:
                await asyncio.gather(*ack_tasks)
            sessions.pop(socket, None)
        return socket

    async def index(_request):
        return web.FileResponse(web_root / "index.html")

    async def asset(request):
        return web.FileResponse(web_root / request.match_info["name"])

    async def cleanup(_app):
        for dispatch in list(sessions.values()):
            dispatch.stop()
        for socket, dispatch in list(sessions.items()):
            await dispatch.close()
            await socket.close()
        await worker.run(runtime.close)
        worker.close()

    app.router.add_get("/", index)
    app.router.add_get("/scene-control-v2/", index)
    app.router.add_get("/scene-control-v2/{name:app\\.js|styles\\.css}", asset)
    app.router.add_static("/vendor/", vendor_root, show_index=False)
    app.router.add_get(API + "/config", configuration)
    app.router.add_post(API + "/maps/select", select_map)
    app.router.add_get(API + "/ws", websocket)
    app.on_cleanup.append(cleanup)
    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--initial-map")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--packet-frames", type=int, choices=(4, 8), default=4)
    parser.add_argument("--lookahead-tokens", type=int, choices=(8, 12, 16), default=8)
    parser.add_argument("--dense-debug", action="store_true")
    parser.add_argument("--port", type=int, default=28945)
    parser.add_argument("--cuda-memory-fraction", type=float,
                        help="Optional per-process CUDA allocation cap, set before model construction")
    parser.add_argument("--native-proof-report", type=Path,
                        help="Run bounded real-model step/history/skin parity proof instead of serving")
    parser.add_argument("--native-proof-packets", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args(argv)
    if args.port in (28936, 28941, 28942, 28943, 28944) or not 1024 <= args.port <= 65535:
        parser.error("Use an independent unreserved port (default 28945)")
    if args.cuda_memory_fraction is not None:
        if not 0 < args.cuda_memory_fraction <= 1 or not args.device.startswith("cuda"):
            parser.error("CUDA memory fraction must be in (0,1] on a CUDA device")
    return args


def native_proof(runtime, settings, packets):
    import torch
    from motionbricks.data.dual_horizon_execution import DualHorizonBackend, policy

    adapter = SOMADualBackend(runtime)
    reference = DualHorizonBackend.from_backend(runtime.create_backend(), adapter.engine)
    reference.configure_policy(policy(settings.lookahead_tokens))
    # This local baseline uses the same four-frame internal publication path as
    # pfnn_tuning._predict_tuned, not the old session's eight-frame wire protocol.
    reference.PUBLISH_FRAMES = 4
    reference._subpacket = True
    state = adapter.initial_state()
    rows = []
    with torch.inference_mode():
        for index in range(packets * settings.packet_frames // 4):
            from motionbricks.scene_control_v2.runtime import Command
            intent = Command(index + 1, (0., 1.6), 0., 0.)
            actual = adapter.step(state, intent, settings)
            expected = reference._governed_step([intent.velocity_xz], [intent.heading])
            torch.testing.assert_close(actual.motion, expected["planned_motion"], atol=0, rtol=0)
            torch.testing.assert_close(actual.state.history, expected["published_history"], atol=0, rtol=0)
            native_payload = adapter.encode(expected["planned_motion"], dense_debug=False).payload
            if adapter.encode(actual.motion, dense_debug=False).payload != native_payload:
                raise ValueError("Official full-skin transport parity failed")
            rows.append(dict(step=index, motion_equal=True, observed_history_equal=True,
                             fullskin_bytes_equal=True, frames=4))
            state = actual.state
            adapter.commit(state)
            reference.history = expected["published_history"].clone()
            reference.governor_state = expected["governor_state_after"]
    return dict(schema=RUNTIME_SCHEMA + ".native_proof", passed=True, settings=settings.contract(),
                steps=rows, actual_optimizer_updates=0, quality_approved=False,
                cuda_allocator=cuda_allocator_report(runtime.inference.device),
                acceptance_required=dict(vertices=18056, substeps=8, peak_penetration_m=.02))


def cuda_allocator_report(device):
    import torch

    return dict(allocated_bytes=torch.cuda.memory_allocated(device),
                reserved_bytes=torch.cuda.memory_reserved(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))


def main(argv=None):
    args = parse_args(argv)
    if args.cuda_memory_fraction is not None:
        import torch
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, torch.device(args.device))
    from scripts.soma_pfnn_maps_server import PFNNMapRuntime

    try:
        runtime = PFNNMapRuntime(args.bundle, args.catalog, args.device,
                                 execution="root-path-live", initial_map=args.initial_map,
                                 planning_policy="full")
    except Exception as error:
        if args.native_proof_report is not None:
            args.native_proof_report.write_text(json.dumps(dict(
                schema=RUNTIME_SCHEMA + ".native_proof", passed=False, phase="model_startup",
                error=repr(error), cuda_allocator=cuda_allocator_report(args.device),
                actual_optimizer_updates=0, quality_approved=False), indent=2), encoding="utf-8")
        raise
    settings = RuntimeSettings(args.packet_frames, args.lookahead_tokens, args.dense_debug)
    if args.native_proof_report is not None:
        try:
            report = native_proof(runtime, settings, args.native_proof_packets)
            args.native_proof_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception as error:
            args.native_proof_report.write_text(json.dumps(dict(
                schema=RUNTIME_SCHEMA + ".native_proof", passed=False, phase="native_proof",
                error=repr(error), cuda_allocator=cuda_allocator_report(args.device),
                actual_optimizer_updates=0, quality_approved=False), indent=2), encoding="utf-8")
            raise
        finally:
            runtime.close()
        return
    print(json.dumps(dict(server="scene_control_v2", port=args.port, settings=settings.contract(),
                          quality_approved=False)), flush=True)
    web.run_app(create_app(runtime, settings=settings), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
