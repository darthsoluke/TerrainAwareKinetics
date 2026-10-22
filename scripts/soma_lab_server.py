"""Standalone browser control of native MotionBricks and official SOMA skinning."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from aiohttp import web, WSMsgType


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "out/vendor/kimodo"))

from motionbricks.data.soma_lab import SomaLabSession, pack_geometry_frames
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner


class SomaLabRuntime:
    fps = 30

    def __init__(self, bundle_path, device, clips_root, review_root=None):
        from scripts.portable_native_mb_soma import load_bundle
        from motionbricks.data.native_mb_live import NativeSomaLiveBackend

        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.inference, self.bundle = load_bundle(bundle_path, device)
        self.backend_class = NativeSomaLiveBackend
        self.terrain_backend_builder = None
        self.terrain_inference = None
        self.terrain_scene = None
        self.terrain_initial_history = None
        self.terrain_view = None
        self.terrain_quality = None
        skeleton = self.inference.motion_rep.skeleton
        self.skinner = move_skinner(load_official_skinner(skeleton), torch.device(device))
        self.skinner.skeleton_skin.float().to(device)
        self.faces = self.skinner.faces.cpu().numpy().astype(np.uint32)
        self.vertex_count = len(self.skinner.bind_vertices)
        with torch.inference_mode():
            self.initial_root, self.initial_vertices, self.initial_joints = self.skin_features_with_joints(
                self.bundle["initial_history"][:, -1:].to(device))
            warmup = self.create_backend()
            warmup.predict_plan([[0, 0]], [0])
        self.clips = self.load_clips(clips_root)
        if review_root is not None:
            from motionbricks.data.soma_lab_review import load_review_clips

            self.clips.update(load_review_clips(review_root, self.vertex_count, self.review_terrain_mesh))

    @staticmethod
    def review_terrain_mesh(heightfield, origin, spacing):
        from motionbricks.data.lafan_soma_retarget import Y_TO_Z
        from motionbricks.data.soma_lab_review import native_terrain_to_parc
        from parc.util.terrain_util import convert_heightfield_to_voxelized_trimesh

        parc_hf, parc_origin = native_terrain_to_parc(heightfield, origin, spacing)
        vertices, faces = convert_heightfield_to_voxelized_trimesh(
            parc_hf, float(parc_origin[0]), float(parc_origin[1]), spacing)
        vertices = vertices @ Y_TO_Z
        return {"vertices": vertices.reshape(-1).tolist(), "faces": faces.reshape(-1).tolist()}

    def create_backend(self, mode="live"):
        if mode == "live":
            return self.backend_class(self.inference, self.bundle, reuse_motion_buffer=True)
        if mode == "terrain":
            if self.terrain_backend_builder is None:
                raise ValueError("Terrain Live is unavailable: matching online assets were not loaded")
            return self.terrain_backend_builder(
                self.terrain_inference, self.terrain_initial_history.clone(), self.terrain_scene)
        raise ValueError("Unknown live backend mode")

    def configure_terrain_live(self, backend_builder, scene, initial_history, inference,
                               terrain_quality=None, terrain_view=None):
        """Inject a verified online model loader without coupling it to the flat bundle loader."""
        from motionbricks.data.native_mb_online_conditions import online_protocol

        if not callable(backend_builder) or not isinstance(scene, dict):
            raise ValueError("Terrain Live requires an injected backend builder and scene mapping")
        if inference is None:
            raise ValueError("Terrain Live requires its matched online inference pipeline")
        protocol = online_protocol(inference._root_model.args.get('online_conditioning'))
        if inference._pose_model.args.get('online_conditioning') != protocol['version']:
            raise ValueError("Terrain Live checkpoints do not match the online conditioning protocol")

        device = torch.device(inference.device)
        heightfield = torch.as_tensor(scene.get("heightfield_y_m"), device=device, dtype=torch.float32).detach()
        origin = torch.as_tensor(scene.get("min_point_world_xyz_m"), device=device, dtype=torch.float32).detach()
        spacing = torch.as_tensor(scene.get("dx_m"), device=device, dtype=torch.float32).detach()
        if (heightfield.ndim != 2 or min(heightfield.shape) < 1 or not torch.isfinite(heightfield).any()
                or origin.shape != (3,) or not torch.isfinite(origin).all()
                or spacing.ndim != 0 or not torch.isfinite(spacing) or spacing.item() <= 0):
            raise ValueError("Terrain Live scene geometry is invalid")
        prepared_scene = {"heightfield_y_m": heightfield, "min_point_world_xyz_m": origin, "dx_m": spacing}
        display_valid = torch.isfinite(heightfield)
        if "heightfield_valid" in scene:
            valid = torch.as_tensor(scene["heightfield_valid"], device=device, dtype=torch.bool).detach()
            if valid.shape != heightfield.shape:
                raise ValueError("Terrain Live validity grid must match the heightfield")
            prepared_scene["heightfield_valid"] = valid
            display_valid &= valid
        if not display_valid.any():
            raise ValueError("Terrain Live scene contains no known heightfield cells")

        history = torch.as_tensor(initial_history, device=device, dtype=torch.float32).detach()
        expected_width = len(inference.global_motion_rep.indices["all"])
        if history.shape != (1, 4, expected_width) or not torch.isfinite(history).all():
            raise ValueError("Terrain Live initial context must be four finite native full-motion frames")
        self.terrain_backend_builder = backend_builder
        self.terrain_inference = inference
        self.terrain_scene = prepared_scene
        self.terrain_initial_history = history.clone()
        view_heightfield = torch.where(display_valid, heightfield, torch.nan).cpu().numpy()
        safe_view_heightfield = np.where(np.isfinite(view_heightfield), view_heightfield, None).tolist()
        self.terrain_view = terrain_view or {
            "hf": safe_view_heightfield, "min_point": origin.cpu().tolist(),
            "dx": float(spacing.cpu()),
        }
        self.terrain_quality = {
            "approval_status": "experimental_not_approved",
            "actuating": False,
            "display_only": True,
            "quality_note": "Experimental / not approved",
        }
        if terrain_quality:
            self.terrain_quality.update(terrain_quality)
        self.terrain_quality["approval_status"] = "experimental_not_approved"
        self.terrain_quality["actuating"] = False
        self.terrain_quality["display_only"] = True
        with torch.inference_mode():
            self.terrain_initial_root, self.terrain_initial_vertices, self.terrain_initial_joints = self.skin_features_with_joints(
                history[:, -1:], inference)

    @torch.inference_mode()
    def skin_features(self, features, inference=None):
        roots, vertices, _ = self.skin_features_with_joints(features, inference)
        return roots, vertices

    @torch.inference_mode()
    def skin_features_with_joints(self, features, inference=None):
        inference = inference or self.inference
        pose = inference.motion_rep.inverse(features, False, return_all=False)
        vertices = self.skinner.skin(pose["global_joint_rots"][0], pose["posed_joints"][0],
                                     rot_is_global=True)
        return (features[0, :, :3].cpu().numpy(), vertices.cpu().numpy(),
                pose["posed_joints"][0].cpu().numpy())

    def load_clips(self, clips_root):
        from scripts.build_parc_native_dataset import parc_heightfield_to_native
        from motionbricks.data.lafan_soma_retarget import Y_TO_Z

        sys.path.insert(0, str(ROOT / "out/vendor/PARC"))
        sys.modules.setdefault("parc", importlib.import_module("PARC"))
        from parc.util.terrain_util import convert_heightfield_to_voxelized_trimesh

        report_path = clips_root / "report.json"
        if not report_path.exists():
            return {}
        report = json.loads(report_path.read_text(encoding="utf-8"))
        clips = {}
        labels = {"terraces": "Terraces", "platform_2": "Platforms",
                  "three_stair_flights": "Three Stair Flights"}
        for item in report["items"]:
            if item.get("execution_status") != "processed" or item.get("screen_status") != "pass":
                continue
            path = clips_root / item["behavior"] / "retarget.npz"
            from motionbricks.data.bones_contract import file_sha256
            if file_sha256(path) != item["shard_sha256"]:
                raise ValueError("Replay shard changed since its source screening")
            with np.load(path, allow_pickle=False) as source:
                local = source["local_rot_mats"].copy()
                roots = source["root_positions"].copy()
                terrain, origin, dx = parc_heightfield_to_native(source["hf"], source["min_point"], source["dx"])
                source_origin = source["min_point"]
                mesh_vertices, mesh_faces = convert_heightfield_to_voxelized_trimesh(
                    source["hf"], float(source_origin[0]), float(source_origin[1]), dx)
                mesh_vertices = mesh_vertices @ Y_TO_Z
                fps = float(source["source_fps"])
            if fps != self.fps or local.shape != (len(roots), 30, 3, 3):
                raise ValueError("Replay motion does not match native SOMA30 / 30 Hz")
            key = item["behavior"]
            clips[key] = {
                "id": key, "label": labels.get(key, key), "fps": fps,
                "frame_count": len(roots), "vertex_count": self.vertex_count,
                "frame_stride_floats": 3 + self.vertex_count * 3,
                "terrain": {"hf": terrain.tolist(), "min_point": origin.tolist(), "dx": dx,
                            "mesh": {"vertices": mesh_vertices.reshape(-1).tolist(),
                                     "faces": mesh_faces.reshape(-1).tolist()},
                            "mesh_source": "PARC.convert_heightfield_to_voxelized_trimesh"},
                "local": local, "roots": roots,
            }
        return clips

    @torch.inference_mode()
    def clip_frames(self, clip_id, start, count):
        clip = self.clips[clip_id]
        if start < 0 or count < 1 or count > 16 or start >= clip["frame_count"]:
            raise ValueError("Invalid bounded replay range")
        stop = min(start + count, clip["frame_count"])
        device = self.inference.device
        local = torch.as_tensor(clip["local"][start:stop], device=device)
        roots = torch.as_tensor(clip["roots"][start:stop], device=device)
        positions = roots[:, None].expand(-1, 30, -1)
        vertices = self.skinner.skin(local, positions, rot_is_global=False).cpu().numpy()
        return pack_geometry_frames(clip["roots"][start:stop], vertices)

    def config(self):
        config = {
            "fps": self.fps, "vertex_count": self.vertex_count,
            "initial_vertices": self.initial_vertices[0].reshape(-1).tolist(),
            "initial_root": self.initial_root[0].tolist(), "faces": self.faces.reshape(-1).tolist(),
            "modes": [{"id": "live", "label": "Live"}, {"id": "replay", "label": "Replay"}],
            "live_model": {"task": "flat locomotion", "terrain_conditioned": False,
                           "source": "historical flat LAFAN1 bundle; not the new PARC terrain model"},
            "clips": [{k: c[k] for k in ("id", "label", "frame_count", "fps", "vertex_count")}
                      for c in self.clips.values()],
        }
        terrain_available = self.terrain_backend_builder is not None
        if hasattr(self, "inference"):
            skeleton = self.inference.motion_rep.skeleton
            config["skeleton"] = {"names": list(skeleton.bone_order_names),
                                  "parents": skeleton.joint_parents.cpu().tolist()}
        config["terrain_mode"] = {
            "id": "terrain", "label": "Terrain Preview", "available": terrain_available,
            "approval_status": "experimental_not_approved",
            "actuating": False,
            "display_only": True,
            "prefetch": {"version": "played-buffer-v2", "played_frame_count": 2},
        }
        if terrain_available:
            config["terrain_mode"].update(self.terrain_quality)
            config["terrain_scene"] = self.terrain_view
            config["modes"].append({"id": "terrain", "label": "Terrain Preview (Experimental)"})
            if (getattr(self.terrain_inference, '_terrain_execution', None) is not None
                    or getattr(self,'rig_transport',False)):
                skin = self.skinner
                influences = skin.lbs_indices.shape[-1]
                if skin.lbs_indices.shape[0] != self.vertex_count or not 4 <= influences <= 8:
                    raise ValueError('Rig transport requires all official SOMA skin weights')
                config['soma_skin'] = {'format': 'soma77-rig-v1', 'rig_count': 77,
                    'influences_per_vertex': influences,
                    'bind_vertices': skin.bind_vertices.cpu().flatten().tolist(),
                    'inverse_bind': skin.bind_rig_transform_inv.cpu().flatten().tolist(),
                    'indices': skin.lbs_indices.cpu().flatten().tolist(),
                    'weights': skin.lbs_weights.cpu().flatten().tolist()}
        return config

    def ready(self, mode="live"):
        if mode == "terrain":
            if self.terrain_backend_builder is None:
                raise ValueError("Terrain Live is unavailable: matching online assets were not loaded")
            root, vertices = self.terrain_initial_root, self.terrain_initial_vertices
        elif mode == "live":
            root, vertices = self.initial_root, self.initial_vertices
        else:
            raise ValueError("Unknown live backend mode")
        result = {"type": "ready", "mode": mode, "fps": self.fps, "vertex_count": self.vertex_count,
                "root": root[0].tolist(), "vertices": vertices[0].reshape(-1).tolist(),
                "joints_world_xyz_m": (self.terrain_initial_joints if mode == "terrain"
                                       else self.initial_joints)[0].tolist()}
        if mode == 'terrain':
            heading = self.terrain_initial_history[0, -1, 3:5]
            result['heading'] = float(torch.atan2(heading[1], heading[0]))
        return result


def load_terrain_live_bundle(bundle_path, device, scene_path=None):
    """Load the parent's portable terrain inference bundle without source clips."""
    from scripts.portable_parc_online_soma import load_terrain_bundle

    inference, bundle = load_terrain_bundle(bundle_path, device)
    if not isinstance(bundle, dict) or not {"initial_history", "scene"}.issubset(bundle):
        raise ValueError("Portable Terrain Live bundle must contain initial_history and scene")
    forbidden_inputs = ("reference", "future", "clip")
    if any(any(token in str(key).lower() for token in forbidden_inputs) for key in bundle):
        raise ValueError("Portable Terrain bundles cannot contain reference, future, or clip inputs")
    scene = bundle["scene"]
    required = {"heightfield_y_m", "min_point_world_xyz_m", "dx_m"}
    if not isinstance(scene, dict) or required - scene.keys():
        raise ValueError("Portable Terrain Live bundle has no complete static scene")
    try:
        history = torch.as_tensor(bundle["initial_history"])
    except (TypeError, ValueError) as error:
        raise ValueError("Portable Terrain Live history must be a numeric tensor") from error
    if history.shape[:2] != (1, 4):
        raise ValueError("Portable Terrain Live bundle must contain exactly four raw history frames")
    if scene_path is not None:
        from motionbricks.data.bones_contract import file_sha256
        scene_path = Path(scene_path)
        receipt = json.loads(scene_path.with_suffix('.json').read_text())
        if file_sha256(scene_path) != receipt.get('scene_sha256'):
            raise ValueError('Live scene artifact hash mismatch')
        with np.load(scene_path, allow_pickle=False) as data:
            allowed = required | {'initial_history', 'metadata_json'}
            if set(data.files) != allowed:
                raise ValueError('Live scene must contain terrain and four-frame history only')
            metadata = json.loads(str(data['metadata_json']))
            scene_datasets = {bundle['metadata']['dataset_manifest_sha256']}
            policy = bundle['metadata'].get('source_policy')
            if policy:
                from motionbricks.data.mixed_control_training import validate_source_policy
                validate_source_policy(policy)
                scene_datasets.add(policy['parc_manifest_sha256'])
            authored = metadata.get('schema') == 'motionbricks.authored_test_scene.v1'
            dataset_key = 'seed_dataset_manifest_sha256' if authored else 'dataset_manifest_sha256'
            if (metadata.get('schema') not in ('parc.live_scene.v1','motionbricks.authored_test_scene.v1')
                    or metadata.get(dataset_key) not in scene_datasets
                    or metadata.get('history_frames') != 4 or metadata.get('future_motion_included') is not False):
                raise ValueError('Live scene and model observation contracts differ')
            scene = {key: torch.from_numpy(data[key].copy()) for key in required}
            history = torch.from_numpy(data['initial_history'].copy())
        if history.shape != bundle['initial_history'].shape or not torch.isfinite(history).all():
            raise ValueError('Live scene initial history must contain exactly four matching frames')
    return inference, scene, history


def create_native_terrain_backend(inference, initial_history, scene):
    from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend

    return NativeTerrainOnlineBackend(inference, initial_history, scene)


def create_app(runtime, web_root, *, executor=None):
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="soma-inference")
    device = torch.device(getattr(getattr(runtime, 'inference', None), 'device', 'cpu'))
    inference_stream = (torch.cuda.Stream(device=device, priority=-1)
        if device.type=='cuda' and not getattr(runtime,'prefer_default_cuda_stream',False) else None)
    app = web.Application(client_max_size=4096)
    activity = {'last_terrain_request': -float('inf')}

    async def compute(function, *args):
        queued = time.perf_counter()
        def measured():
            started = time.perf_counter()
            tracing=getattr(runtime,'trace_worker',False)
            if tracing:
                import faulthandler
                print(json.dumps({'worker_begin':function.__name__}),flush=True)
                faulthandler.dump_traceback_later(10,repeat=False)
            try:
                if inference_stream is None:
                    result = function(*args)
                else:
                    with torch.cuda.stream(inference_stream):
                        result = function(*args)
            finally:
                if tracing:
                    faulthandler.cancel_dump_traceback_later()
                    print(json.dumps({'worker_end':function.__name__,'elapsed_ms':(time.perf_counter()-started)*1000}),flush=True)
            if isinstance(result, tuple) and len(result) in (2, 3) and isinstance(result[-2], dict):
                result[-2]['server_queue_ms'] = (started - queued) * 1000
                result[-2]['server_compute_ms'] = (time.perf_counter() - started) * 1000
            return result
        return await asyncio.get_running_loop().run_in_executor(executor, measured)

    async def index(request):
        return web.FileResponse(web_root / "index.html")

    async def config(request):
        return web.json_response(await compute(runtime.config))

    async def select_map(request):
        select=getattr(runtime,'select_map',None)
        if not callable(select):raise web.HTTPNotFound()
        try:
            value=await request.json()
            if not isinstance(value,dict) or set(value)!={'map_id'}:
                raise ValueError('Expected one map_id')
            def select_and_configure():
                select(value['map_id'])
                partial=getattr(runtime,'selection_config',None)
                if request.query.get('scene_only')=='1' and callable(partial):
                    return partial(cached_scene=request.query.get('cached')=='1')
                return runtime.config()
            return web.json_response(await compute(select_and_configure))
        except (ValueError,TypeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error

    async def javascript(request):
        return web.FileResponse(web_root / "app.js")

    async def stylesheet(request):
        return web.FileResponse(web_root / "styles.css")

    async def tuning_config(request):
        getter=getattr(runtime,'tuning_config',None)
        if not callable(getter):raise web.HTTPNotFound()
        return web.json_response(await compute(getter))

    async def tuning_asset(request):
        if not callable(getattr(runtime,'tuning_config',None)):raise web.HTTPNotFound()
        name={'/settings':'tuning.html','/tuning.js':'tuning.js','/tuning.css':'tuning.css',
            '/tuning-contract.js':'tuning-contract.js'}[request.path]
        return web.FileResponse(web_root/name)

    async def clip_metadata(request):
        clip = runtime.clips.get(request.match_info["clip_id"])
        if clip is None:
            raise web.HTTPNotFound()
        return web.json_response({k: v for k, v in clip.items() if k not in ("local", "roots")})

    async def clip_frames(request):
        clip_id = request.match_info["clip_id"]
        if clip_id not in runtime.clips:
            raise web.HTTPNotFound()
        try:
            start, count = int(request.query.get("start", 0)), int(request.query.get("count", 8))
            payload = await compute(runtime.clip_frames, clip_id, start, count)
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.Response(body=payload, content_type="application/octet-stream")

    async def websocket(request):
        socket = web.WebSocketResponse(max_msg_size=4096, heartbeat=20)
        await socket.prepare(request)
        session = await compute(SomaLabSession, runtime, getattr(runtime, 'default_mode', 'live'))
        await socket.send_json(runtime.ready(session.mode))
        async def send_plan(metadata, payload):
            started = time.perf_counter()
            encoded = json.dumps(metadata)
            serialized = time.perf_counter()
            await socket.send_str(encoded)
            if payload is not None:
                await socket.send_bytes(payload)
            if 'client_timing' in metadata:
                await socket.send_json(dict(type='plan_timing',request_id=metadata['request_id'],
                    json_encode_ms=(serialized-started)*1000,
                    socket_send_ms=(time.perf_counter()-serialized)*1000))
        async for message in socket:
            if message.type != WSMsgType.TEXT:
                if message.type == WSMsgType.ERROR:
                    break
                continue
            command = {}
            try:
                command = json.loads(message.data)
                if not isinstance(command, dict):
                    raise ValueError("Expected a command object")
                if session.mode == 'terrain' and command.get('type') in ('plan', 'prefetch', 'played'):
                    activity['last_terrain_request'] = time.monotonic()
                if command.get("type") == "reset":
                    mode = command.get("mode", "live")
                    session = await compute(SomaLabSession, runtime, mode, command.get('spawn_id'))
                    ready_args = (session.mode,session.spawn_id) if session.spawn_id is not None else (session.mode,)
                    await socket.send_json(await compute(runtime.ready,*ready_args))
                elif command.get("type") == "plan":
                    metadata, payload = await compute(session.plan, command)
                    await send_plan(metadata, payload)
                elif command.get("type") == "prefetch":
                    metadata, payload = await compute(session.prefetch, command)
                    await send_plan(metadata, payload)
                elif command.get("type") == "played":
                    if 'next_plan' in command:
                        next_plan = command['next_plan']
                        if not isinstance(next_plan, dict) or next_plan.get('type') != 'plan':
                            raise ValueError('Played continuation must contain a plan command')
                        # Commit and plan on one worker wake-up. Inference still
                        # starts strictly after the full playback acknowledgement.
                        def advance():
                            acknowledgement = session.acknowledge_played(command)
                            metadata, payload = session.plan(next_plan)
                            return acknowledgement, metadata, payload
                        acknowledgement, metadata, payload = await compute(advance)
                        await socket.send_json(acknowledgement)
                        await send_plan(metadata, payload)
                    else:
                        acknowledgement = await compute(session.acknowledge_played, command)
                        await socket.send_json(acknowledgement)
                else:
                    raise ValueError("Unknown command type")
            except ConnectionResetError:
                # Closing a viewer during an in-flight plan is not an inference failure.
                break
            except (ValueError, KeyError, TypeError) as error:
                response = {"type": "error", "message": str(error)}
                request_id = command.get("request_id")
                if isinstance(request_id, int) and not isinstance(request_id, bool):
                    response["request_id"] = request_id
                await socket.send_json(response)
            except Exception as error:
                print(f"Inference error: {error}", file=sys.stderr, flush=True)
                await socket.send_json({"type": "error", "message": "Inference failed"})
                break
        return socket

    async def keep_gpu_ready():
        while True:
            await asyncio.sleep(.08)
            if time.monotonic() - activity['last_terrain_request'] < .4:
                current_scene_warmup=getattr(runtime,'keep_ready_current_scene',None)
                if callable(current_scene_warmup):
                    await compute(current_scene_warmup)
                    continue
                execution = getattr(getattr(runtime, 'terrain_inference', None), '_terrain_execution', None)
                if execution is not None and getattr(runtime, 'terrain_keep_ready', False):
                    await compute(execution.keep_ready)

    async def startup(application):
        warm=getattr(runtime,'warm_live_entry',None)
        if callable(warm):await compute(warm)
        application['gpu_ready_task'] = asyncio.create_task(keep_gpu_ready())

    async def cleanup(application):
        task = application.get('gpu_ready_task')
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        executor.shutdown(wait=True, cancel_futures=True)

    app.router.add_get("/", index)
    app.router.add_get("/api/config", config)
    app.router.add_post("/api/maps/select", select_map)
    app.router.add_get("/api/clips/{clip_id}", clip_metadata)
    app.router.add_get("/api/clips/{clip_id}/frames", clip_frames)
    app.router.add_get("/ws", websocket)
    app.router.add_static("/vendor/", web_root / "vendor", show_index=False)
    app.router.add_get("/app.js", javascript)
    app.router.add_get("/styles.css", stylesheet)
    app.router.add_get('/api/tuning',tuning_config)
    app.router.add_get('/settings',tuning_asset)
    app.router.add_get('/tuning.js',tuning_asset)
    app.router.add_get('/tuning.css',tuning_asset)
    app.router.add_get('/tuning-contract.js',tuning_asset)
    app.on_cleanup.append(cleanup)
    app.on_startup.append(startup)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "out/native_mb_lafan_paper_20261006/ue-trial50k-v1/native-soma-50k.pt")
    parser.add_argument("--clips", type=Path, default=ROOT / "out/parc_curated_20261009/reference_soma_v3")
    parser.add_argument("--review", type=Path, help="optional exported PARC source/codec review directory")
    parser.add_argument("--terrain-bundle", type=Path,
                        help="optional portable planar-online bundle; enables experimental Terrain Live")
    parser.add_argument('--terrain-scene', type=Path, help='verified terrain plus four observed initial frames')
    parser.add_argument('--terrain-execution', choices=('eager', 'cuda-graph'), default='cuda-graph')
    parser.add_argument('--no-terrain-keep-ready', action='store_true')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=28780)
    args = parser.parse_args()
    if args.terrain_scene and not args.terrain_bundle:
        parser.error('--terrain-scene requires --terrain-bundle')
    runtime = SomaLabRuntime(args.bundle, args.device, args.clips, args.review)
    if args.terrain_bundle:
        inference, scene, history = load_terrain_live_bundle(args.terrain_bundle, args.device, args.terrain_scene)
        runtime.configure_terrain_live(
            create_native_terrain_backend, scene, history, inference,
            terrain_quality={'vertical_frame': inference._pose_model.args.get('vertical_frame'),
                             'model_file': args.terrain_bundle.name,
                             'scene_file': args.terrain_scene.name if args.terrain_scene else None})
        if args.terrain_execution == 'cuda-graph':
            from motionbricks.data.soma_terrain_execution import TerrainExecution
            print('Compiling and verifying native terrain execution', flush=True)
            execution = TerrainExecution(inference, runtime.terrain_scene, runtime.skinner,
                                         runtime.terrain_initial_history)
            object.__setattr__(inference, '_terrain_execution', execution)
            runtime.terrain_quality['execution'] = 'verified-static-cuda-v1'
            runtime.terrain_quality['execution_parity'] = execution.validation
            runtime.terrain_keep_ready = not args.no_terrain_keep_ready
            runtime.terrain_quality['active_session_gpu_warmup'] = runtime.terrain_keep_ready
    print(json.dumps({"ready": True, "port": args.port, "fps": runtime.fps,
                      "skin": "official SOMASkin", "vertices": runtime.vertex_count,
                      "live_backend": "NativeSomaLiveBackend",
                      "terrain_live": runtime.terrain_backend_builder is not None,
                      "terrain_approval": "experimental_not_approved",
                      "replay_clips": list(runtime.clips)}), flush=True)
    web.run_app(create_app(runtime, ROOT / "web/soma_lab"), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
