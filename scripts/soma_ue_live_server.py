"""Loopback SOMA inference with separate observed and committed planning context."""

import argparse
from collections import OrderedDict
from copy import deepcopy
import json
import math
from pathlib import Path
import socket
import socketserver
import time

import numpy as np
import torch

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.bones_rollout import SomaLocomotionBackend, UE_PUBLISH_FRAMES
from motionbricks.data.soma_replay import skeleton_manifest
from scripts.portable_bones_control import load_bundle


SCHEMA = "motionbricks.soma_control.v1"
COMMITTED_SCHEMA = "motionbricks.soma_control.v2"
PUBLISH_FRAMES = UE_PUBLISH_FRAMES
COORDINATES = "RH_Y_UP_Z_FORWARD_X_LEFT_METERS"
MAX_REQUEST_BYTES = 65536


def vector(value, size, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Invalid " + label)
    return result


class SomaLiveSession:
    def __init__(self, inference, bundle, model_id, manifest, export_representation=None, *, backend=None):
        self.inference, self.model_id, self.manifest = inference, model_id, manifest
        self.export_representation = export_representation or inference.motion_rep
        device = next(inference._root_model.parameters()).device
        # Interactive control must not choose a new random gait on every replan.
        self.backend = backend if backend is not None else SomaLocomotionBackend(
            inference, bundle["initial_history"].to(device), argmax=True)
        self.fps = int(inference.motion_rep.fps)
        if self.fps not in (30, 60):
            raise ValueError("Unsupported native motion clock")
        self.session_id, self.request_id, self.sequence, self.history_tick = None, 0, 0, -1
        self.plans = OrderedDict()
        self.plan_origins = {}
        self.executed_tick = -1

    def history_motion(self, records, history_tick):
        if len(records) != 4:
            raise ValueError("Need four consecutive context records")
        frames, roots = [], []
        for offset, record in enumerate(records):
            sequence, index = int(record["sequence"]), int(record["frame"])
            if sequence not in self.plans:
                raise ValueError("History references an expired or foreign plan")
            start, motion = self.plans[sequence]
            if not 0 <= index < motion.shape[1] or start + index != history_tick - 3 + offset:
                raise ValueError("History frames must be consecutive ticks")
            frame = motion[:, index:index + 1]
            root = vector(record["root_m"], 3, "context root")
            frames.append(frame)
            roots.append(root)
        result = torch.cat(frames, 1).detach()
        roots_array = np.stack(roots)
        supplied = torch.as_tensor(roots_array, device=result.device, dtype=result.dtype)
        # One transfer/check for four records; preserve all native body features.
        if (np.abs(roots_array[:, 1] - result[0, :, 1].cpu().numpy()) > 1e-4).any():
            raise ValueError("The flat preview cannot rewrite predicted hip height")
        result[..., :3] = supplied
        return result

    def apply_history(self, records, history_tick):
        if not self.plans:
            if records or history_tick != -1:
                raise ValueError("Initialization cannot claim previously generated frames")
            return
        if len(records) != 4 or history_tick < self.history_tick + 4:
            raise ValueError("Need four newly executed history frames, not predicted future")
        self.backend.history = self.history_motion(records, history_tick)
        self.history_tick = history_tick

    def apply_committed_context(self, request):
        """Condition a future handoff on an immutable prefix, not pretend observation.

        The engine promises to play all eight current frames before the handoff.
        Observed roots only establish a planar coordinate offset; unknown future
        capsule collisions are not predicted here and are fed back next request.
        """
        tick, executed = int(request["history_tick"]), int(request["executed_tick"])
        records, observed = request.get("history", []), request.get("executed_history", [])
        if request.get("publish_frames") != PUBLISH_FRAMES:
            raise ValueError("Unsupported immutable prefix length")
        if not self.plans:
            if request.get("history_kind") != "initial" or records or observed or tick != -1 or executed != -1:
                raise ValueError("Initialization cannot claim committed/observed frames")
            return
        start, plan = self.plans[self.sequence]
        if (request.get("history_kind") != "committed" or tick != start + PUBLISH_FRAMES - 1
                or executed < self.executed_tick or executed < start - 1 or executed > tick - 4):
            raise ValueError("Invalid committed handoff or observed clock")
        if len(records) != 4 or any(int(r["sequence"]) != self.sequence for r in records):
            raise ValueError("Committed context must come from the current immutable plan")
        committed = self.history_motion(records, tick)
        if executed == -1:
            if observed or self.sequence != 1:
                raise ValueError("Missing observed history")
            shift = plan.new_zeros(3)
        else:
            actual = self.history_motion(observed, executed)
            expected = self.plan_origins[self.sequence] if executed < start else plan[0, executed - start, :3]
            shift = actual[0, -1, :3] - expected
            shift[1] = 0
        expected_roots = plan[:, PUBLISH_FRAMES - 4:PUBLISH_FRAMES, :3] + shift
        if not torch.allclose(committed[..., :3], expected_roots, atol=1e-4, rtol=0):
            raise ValueError("Committed roots cannot alter the model's displacement")
        self.backend.history = committed
        self.history_tick, self.executed_tick = tick, executed

    @torch.inference_mode()
    def respond(self, request):
        if (request.get("schema") not in (SCHEMA, COMMITTED_SCHEMA) or request.get("model_id") != self.model_id
                or request.get("source_id") != self.manifest["source_id"]):
            raise ValueError("Schema/model/skeleton identity mismatch")
        session = request.get("session_id")
        request_id = int(request["request_id"])
        if not isinstance(session, str) or not 1 <= len(session) <= 128:
            raise ValueError("Invalid PIE session")
        if self.session_id is None:
            self.session_id = session
        if session != self.session_id or request_id != self.request_id + 1:
            raise ValueError("Stale session or out-of-order command")
        velocity = vector(request["velocity_ue_mps"], 3, "velocity")
        facing = vector(request["facing_ue"], 3, "facing")
        if (abs(velocity[2]) > 1e-6 or abs(facing[2]) > 1e-6
                or np.linalg.norm(velocity) > 3.001 or np.linalg.norm(facing[:2]) < 1e-6):
            raise ValueError("Commands must be finite planar intent, speed <= 3 m/s")
        client_time = float(request["client_time_s"])
        if not math.isfinite(client_time):
            raise ValueError("Invalid client time")
        committed_protocol = request["schema"] == COMMITTED_SCHEMA
        if not committed_protocol and self.fps != 60:
            raise ValueError("Native 30 Hz requires the committed playback protocol")
        if committed_protocol:
            self.apply_committed_context(request)
        else:
            self.apply_history(request.get("history", []), int(request["history_tick"]))
        self.request_id = request_id
        previous_root_tensor = self.backend.history[0, -1, :3].clone()
        previous_root = previous_root_tensor.cpu().tolist()
        result = self.backend.predict_plan([[-velocity[1], velocity[0]]], [math.atan2(-facing[1], facing[0])])
        export_started = time.perf_counter()
        motion = result["planned_motion"]
        # Export only the immutable published block; do not skin/serialize unused futures.
        published = motion[:, :PUBLISH_FRAMES] if committed_protocol else motion
        if self.export_representation is not self.inference.motion_rep:
            published = published.cpu()
        pose = self.export_representation.inverse(published, False, return_all=False, return_quat=True)
        positions = pose["posed_joints"][0].cpu().numpy()
        quats = pose["global_joint_rots"][0].cpu().numpy()[..., [1, 2, 3, 0]]
        contacts = pose["foot_contacts"][0].cpu().numpy()
        roots = published[0, :, :3].cpu().numpy()
        self.sequence += 1
        start_tick = self.history_tick + 1
        self.plans[self.sequence] = (start_tick, motion)
        self.plan_origins[self.sequence] = previous_root_tensor
        while len(self.plans) > 8:
            expired, _ = self.plans.popitem(last=False)
            self.plan_origins.pop(expired, None)
        frames = [{"root_m": root.tolist(), "positions_m": p.tolist(), "rotations_xyzw": q.tolist(),
                   "contacts": c.tolist()} for root, p, q, c in zip(roots, positions, quats, contacts)]
        return {"schema": request["schema"], "session_id": session, "request_id": request_id,
                "model_id": self.model_id, "source_id": self.manifest["source_id"],
                "coordinates": COORDINATES, "sequence": self.sequence, "start_tick": start_tick,
                "fps": self.fps, "client_time_s": client_time, "previous_root_m": previous_root,
                "bone_names": [joint["name"] for joint in self.manifest["joints"]],
                "parents": [joint["parent_index"] for joint in self.manifest["joints"]],
                "frames": frames, "inference_ms": result["generation_seconds"] * 1000,
                "export_ms": (time.perf_counter() - export_started) * 1000,
                "root_stage": "released_MB" if getattr(self.inference, "use_native_root_stage", True) else "legacy_prefix_override",
                "neural_inference": True, "quality_approved": False, "token_sampling": "argmax",
                "post_processing": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--port", type=int, default=28766)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--protocol-version", type=int, choices=(2,), default=2)
    parser.add_argument("--export-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profile", type=Path, help="One diagnostic plan trace; profiled timings are not a benchmark")
    parser.add_argument("--decoder-cuda-graph", action="store_true")
    parser.add_argument("--backbone-cuda-graphs", action="store_true")
    parser.add_argument("--inference-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    inference, bundle = load_bundle(args.bundle, "cuda:0" if args.inference_device == "cuda" else "cpu")
    if args.decoder_cuda_graph:
        from motionbricks.data.soma_cuda_graph import install_decoder_graph
        install_decoder_graph(inference)
    if args.backbone_cuda_graphs:
        from motionbricks.data.soma_cuda_graph import install_backbone_graphs
        install_backbone_graphs(inference)
    export_representation = deepcopy(inference.motion_rep).to("cpu") if args.export_device == "cpu" else None
    model_id, manifest = file_sha256(args.bundle), skeleton_manifest()
    if bundle["bone_names"] != [joint["name"] for joint in manifest["joints"]]:
        raise ValueError("Portable model and official source hierarchy disagree")
    with torch.inference_mode():
        warmup = SomaLocomotionBackend(inference, bundle["initial_history"].to(inference.device), argmax=True)
        for _ in range(5):
            warmup.predict_plan([[0, 0]], [0])
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(event):
        with args.log.open("a", encoding="utf-8") as output:
            output.write(json.dumps(dict(time_s=time.time(), **event), allow_nan=False) + "\n")

    profile_done = False

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            nonlocal profile_done
            torch.manual_seed(2333)
            session = SomaLiveSession(inference, bundle, model_id, manifest, export_representation)
            self.connection.settimeout(15)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while line := self.rfile.readline(MAX_REQUEST_BYTES + 1):
                    if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
                        raise ValueError("Request exceeds bounded message size")
                    request = json.loads(line)
                    started = time.perf_counter()
                    profiled = args.profile is not None and not profile_done and int(request["request_id"]) == 10
                    if profiled:
                        profile_done = True
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                               torch.profiler.ProfilerActivity.CUDA],
                                                    record_shapes=True) as profile:
                            with torch.profiler.record_function("soma_plan"):
                                reply = session.respond(request)
                        args.profile.parent.mkdir(parents=True, exist_ok=True)
                        profile.export_chrome_trace(str(args.profile))
                        args.profile.with_suffix(".txt").write_text(
                            profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=35), encoding="utf-8")
                    else:
                        reply = session.respond(request)
                    reply["backend_ms"] = (time.perf_counter() - started) * 1000
                    self.wfile.write(json.dumps(reply, separators=(",", ":"), allow_nan=False).encode() + b"\n")
                    self.wfile.flush()
                    log({"event": "plan", "session": session.session_id, "request": session.request_id,
                         "history_tick": session.history_tick, "start_tick": reply["start_tick"],
                         "sequence": session.sequence, "velocity": request["velocity_ue_mps"],
                         "backend_ms": reply["backend_ms"], "inference_ms": reply["inference_ms"],
                         "profiled": profiled,
                         "export_ms": reply["export_ms"], "export_device": args.export_device,
                         "root_stage": reply["root_stage"],
                         "planning_context_kind": request.get("history_kind", "executed"),
                         "executed_tick": request.get("executed_tick", session.history_tick)})
            except (ConnectionError, TimeoutError) as error:
                log({"event": "disconnect", "session": session.session_id, "reason": str(error)})
            except Exception as error:
                log({"event": "error", "session": session.session_id, "reason": str(error)})
                try:
                    self.wfile.write(json.dumps({"error": str(error)}).encode() + b"\n")
                    self.wfile.flush()
                except ConnectionError:
                    pass

    class Server(socketserver.TCPServer):
        allow_reuse_address = True

    with Server(("127.0.0.1", args.port), Handler) as server:
        log({"event": "ready", "model_id": model_id, "source_id": manifest["source_id"],
             "port": args.port, "token_sampling": "argmax", "protocol_version": 2,
             "decoder_cuda_graph": args.decoder_cuda_graph,
             "backbone_cuda_graphs": args.backbone_cuda_graphs,
             "inference_device": args.inference_device, "cpu_threads": args.cpu_threads,
             "runtime_sha256": {name: file_sha256(Path(__file__).resolve().parents[1] / name) for name in
                ("scripts/soma_ue_live_server.py", "motionbricks/data/bones_control.py", "motionbricks/data/bones_rollout.py",
                 "motionbricks/data/soma_cuda_graph.py", "motionbricks/motion_backbone/neural_modules/root_backbone.py",
                 "motionbricks/motion_backbone/neural_modules/pose_backbone.py")},
             "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
             "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
             "publish_frames": PUBLISH_FRAMES})
        print(json.dumps({"ready": True, "port": args.port, "model_id": model_id}), flush=True)
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
