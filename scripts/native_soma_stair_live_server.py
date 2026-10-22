"""Independent loopback endpoint for real-time, source-authored MB stair skills."""

import argparse
import json
import math
from pathlib import Path
import socket
import socketserver
import time

import numpy as np
import torch

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.native_mb_stair_live import NativeStairLiveBackend
from motionbricks.data.native_mb_frame import NativeAnchoredIdle, load_authored_idle_reference
from motionbricks.data.native_mb_smart_object import NativeSmartObject
from motionbricks.data.soma_replay import skeleton_manifest
from scripts.portable_native_mb_soma import load_bundle
from scripts.soma_ue_live_server import COMMITTED_SCHEMA, MAX_REQUEST_BYTES, SomaLiveSession


def make_backend(inference, bundle, course, exit_skill=None, exit_reference=None):
    exit_idle = None
    if exit_skill is not None:
        exit_infer, exit_bundle = exit_skill
        source_indices, exit_indices = inference.global_motion_rep.indices, exit_infer.global_motion_rep.indices
        if (bundle['bone_names'] != exit_bundle['bone_names'] or bundle['fps'] != exit_bundle['fps']
                or source_indices.keys() != exit_indices.keys()
                or any(not np.array_equal(value, exit_indices[key]) for key, value in source_indices.items())):
            raise ValueError('Stair and exit skills must share the native source representation')
        endpoint = course['goals'][-1]['features'][-1]
        exit_idle = NativeAnchoredIdle(NativeSmartObject(exit_infer), exit_infer.global_motion_rep,
            exit_reference if exit_reference is not None else exit_bundle['reference_clips']['idle'][:4].to(exit_infer.device)[None],
            [endpoint[0], endpoint[2]], math.atan2(*course['forward_xz']),
            course['exit_platform_y_m'])
    return NativeStairLiveBackend(inference, bundle, course, exit_locomotion=exit_idle)


class StairLiveSession(SomaLiveSession):
    def __init__(self, inference, bundle, model_id, manifest, courses, exit_skill=None, exit_model_id=None, exit_reference=None):
        self.courses, self.bundle, self.course_id = courses, bundle, None
        self.exit_skill, self.exit_model_id = exit_skill, exit_model_id
        self.exit_reference = exit_reference
        super().__init__(inference, bundle, model_id, manifest,
                         backend=make_backend(inference, bundle, courses["up"], exit_skill, exit_reference))

    def history_motion(self, records, history_tick):
        # Unlike the flat collision mode, this experiment must execute raw XYZ exactly once.
        for record in records:
            sequence, index = int(record["sequence"]), int(record["frame"])
            if sequence not in self.plans or not 0 <= index < self.plans[sequence][1].shape[1]:
                raise ValueError("Unknown stair history")
            raw = self.plans[sequence][1][0, index, :3]
            supplied = torch.as_tensor(record["root_m"], device=raw.device, dtype=raw.dtype)
            if supplied.shape != (3,) or not torch.allclose(raw, supplied, atol=1e-4, rtol=0):
                raise ValueError("Full-3D stair execution cannot rewrite predicted XYZ")
        super().history_motion(records, history_tick)
        # Feedback tolerance is validation only, never a root-edit permission.
        return torch.cat([self.plans[int(r["sequence"])][1][:, int(r["frame"]):int(r["frame"]) + 1]
                          for r in records], dim=1).clone()

    def respond(self, request):
        course = request.get("stair_course")
        if request.get("schema") != COMMITTED_SCHEMA or course not in self.courses:
            raise ValueError("Stair inference requires v2 committed playback and an authored course")
        if self.course_id is None:
            self.backend = make_backend(self.inference, self.bundle, self.courses[course], self.exit_skill, self.exit_reference)
        elif course != self.course_id:
            raise ValueError("Reset the session before switching stair courses")
        reply = super().respond(request)
        self.course_id = course
        reply.update(root_execution="full_3d", stair_course=course,
                     exit_model_id=self.exit_model_id,
                     primitive_state=self.backend.primitive_state, active_goal_index=self.backend.active_goal_index)
        return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--courses", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=28771)
    parser.add_argument("--exit-bundle", type=Path,
                        help="Optional native idle skill on the explicitly authored exit platform")
    parser.add_argument("--exit-reference", type=Path, help="Observed relaxed idle input with source provenance")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid loopback port")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    exit_skill = load_bundle(args.exit_bundle, "cuda:0") if args.exit_bundle else None
    exit_model_id = file_sha256(args.exit_bundle) if args.exit_bundle else None
    if args.exit_reference and not exit_skill:
        parser.error('--exit-reference requires --exit-bundle')
    exit_reference = load_authored_idle_reference(args.exit_reference, exit_skill[0].global_motion_rep,
        exit_skill[0].device) if args.exit_reference else None
    bank = json.loads(args.courses.read_text(encoding="utf-8"))
    if bank.get("schema") != "motionbricks.stair_courses.v1":
        raise ValueError("Invalid authored course bank")
    for key, actual in (("root_checkpoint_sha256", bundle["metadata"]["root_checkpoint_sha256"]),
                        ("pose_checkpoint_sha256", bundle["metadata"]["pose_checkpoint_sha256"]),
                        ("tokenizer_checkpoint_sha256", bundle["metadata"]["tokenizer"]["checkpoint_sha256"])):
        if bank.get(key) != actual:
            raise ValueError("Authored course checkpoint binding mismatch: " + key)
    courses = {c["id"]: c for c in bank["courses"]}
    if set(courses) != {"up", "down"}:
        raise ValueError("Both authored courses must be present")
    model_id, manifest = file_sha256(args.bundle), skeleton_manifest()
    if bundle["bone_names"] != [j["name"] for j in manifest["joints"]]:
        raise ValueError("UE and stair source skeleton disagree")
    for course in courses.values():
        warmup = make_backend(inference, bundle, course, exit_skill, exit_reference)
        warmup.predict_plan([course["forward_xz"]], [0.])
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(event):
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time_s=time.time(), **event), allow_nan=False) + "\n")

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            torch.manual_seed(2333)
            session = StairLiveSession(inference, bundle, model_id, manifest, courses, exit_skill, exit_model_id, exit_reference)
            self.connection.settimeout(60)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                while line := self.rfile.readline(MAX_REQUEST_BYTES + 1):
                    if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
                        raise ValueError("Request exceeds bounded size")
                    request = json.loads(line)
                    started = time.perf_counter()
                    reply = session.respond(request)
                    reply["backend_ms"] = (time.perf_counter() - started) * 1000
                    self.wfile.write(json.dumps(reply, separators=(",", ":"), allow_nan=False).encode() + b"\n")
                    self.wfile.flush()
                    log({"event": "plan", "session": session.session_id, "sequence": session.sequence,
                         "stair_model_id": model_id, "exit_model_id": exit_model_id,
                         "course": session.course_id, "velocity": request["velocity_ue_mps"],
                         "backend_ms": reply["backend_ms"], "generation": session.backend.last_generation,
                         "first_root": reply["frames"][0]["root_m"], "last_root": reply["frames"][-1]["root_m"]})
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
        log({"event": "ready", "model_id": model_id, "course_bank_sha256": file_sha256(args.courses),
             "exit_model_id": exit_model_id, "exit_policy": "native_anchored_idle" if exit_skill else "legacy_endpoint",
             "exit_reference_sha256": file_sha256(args.exit_reference) if args.exit_reference else None,
             "port": args.port, "fps": 30, "root_execution": "full_3d", "post_processing": None,
             "quality_approved": False, "metadata": bundle["metadata"], "gpu": torch.cuda.get_device_name()})
        print(json.dumps({"ready": True, "port": args.port, "model_id": model_id}), flush=True)
        server.serve_forever(poll_interval=.1)


if __name__ == "__main__":
    main()
