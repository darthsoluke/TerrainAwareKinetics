"""Loopback UE trial of the native MB Root/Pose/Tokenizer SOMA 30 Hz chain."""

import argparse
import json
from pathlib import Path
import socket
import socketserver
import time

import torch

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.native_mb_live import NativeSomaLiveBackend
from motionbricks.data.soma_replay import skeleton_manifest
from scripts.portable_native_mb_soma import load_bundle
from scripts.soma_ue_live_server import MAX_REQUEST_BYTES, SomaLiveSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--port", type=int, default=28769)
    parser.add_argument("--disable-motion-buffer", action="store_true",
                        help="Diagnostic baseline: regenerate every eight published frames")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise ValueError("Invalid loopback port")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    model_id, manifest = file_sha256(args.bundle), skeleton_manifest()
    if bundle["bone_names"] != [joint["name"] for joint in manifest["joints"]]:
        raise ValueError("UE and trained source skeleton disagree")
    with torch.inference_mode():
        warmup = NativeSomaLiveBackend(inference, bundle, reuse_motion_buffer=not args.disable_motion_buffer)
        for _ in range(3):
            warmup.predict_plan([[0, 0]], [0])
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(event):
        with args.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time_s=time.time(), **event), allow_nan=False) + "\n")

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            torch.manual_seed(2333)
            session = SomaLiveSession(inference, bundle, model_id, manifest,
                                      backend=NativeSomaLiveBackend(inference, bundle,
                                                                   reuse_motion_buffer=not args.disable_motion_buffer))
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
                    log({"event": "plan", "session": session.session_id, "request": session.request_id,
                         "start_tick": reply["start_tick"], "sequence": session.sequence,
                         "velocity": request["velocity_ue_mps"], "style": session.backend.last_style,
                         "backend_ms": reply["backend_ms"], "inference_ms": reply["inference_ms"],
                         "generation": session.backend.last_generation,
                         "fps": reply["fps"], "frames": len(reply["frames"])})
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
        log({"event": "ready", "model_id": model_id, "port": args.port, "fps": 30,
             "metadata": bundle["metadata"], "quality_approved": False,
             "post_processing": None, "primitive": "native_sparse_locomotion_played_context",
             "reuse_motion_buffer": not args.disable_motion_buffer,
             "condition_contract": warmup.condition_contract,
             "runtime_sha256": {name: file_sha256(Path(__file__).resolve().parents[1] / name) for name in (
                 "scripts/native_soma_ue_live_server.py", "scripts/portable_native_mb_soma.py",
                 "motionbricks/data/native_mb_live.py", "motionbricks/data/native_mb_locomotion.py")},
             "gpu": torch.cuda.get_device_name(), "precision": "FP32, TF32 disabled"})
        print(json.dumps({"ready": True, "port": args.port, "model_id": model_id}), flush=True)
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
