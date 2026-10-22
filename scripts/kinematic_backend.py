"""Run the official MotionBricks models as a loopback-only kinematic service."""

import argparse
import json
from pathlib import Path
import socket
import time
import uuid

import torch

from motionbricks.motion_backbone.demo.kinematic_backend import KinematicBackend, SCHEMA


MAX_REQUEST_BYTES = 16384


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate(backend, output):
    skeleton = backend._motion_rep.skeleton
    parents = skeleton.joint_parents.cpu()
    rest = skeleton.neutral_joints.cpu()
    expected = (rest[1:] - rest[parents[1:]]).norm(dim=-1)
    plans = []
    previous = None
    for mode, velocity, facing in [("idle", (0, 0, 0), (1, 0, 0)), ("walk", (.7, 0, 0), (1, 0, 0)),
                                   ("walk", (.5, .5, 0), (.707, .707, 0)), ("idle", (0, 0, 0), (.707, .707, 0))]:
        plan = backend.plan(mode, velocity, facing)
        data = torch.tensor(plan["frames"])
        xyz, quat = data[..., :3], data[..., 3:]
        lengths = (xyz[:, 1:] - xyz[:, parents[1:]]).norm(dim=-1)
        max_error = (lengths - expected).abs().max().item()
        joint_motion = (xyz[-1] - xyz[0]).norm(dim=-1).max().item()
        boundary = 0.0 if previous is None else (xyz[0] - previous).norm(dim=-1).max().item()
        assert max_error < 1.e-4, "FK bone-length error: %.9fm" % max_error
        assert torch.allclose(quat.norm(dim=-1), torch.ones_like(quat[..., 0]), atol=1.e-5), "Quaternion normalization"
        assert boundary < .15, "Large autoregressive boundary discontinuity"
        assert len(plan["frames"]) >= 2 and plan["neural_inference"] and not plan["physics"]
        assert len(plan["frames"]) >= 37, "Navigation horizon must cover at least 1.2 seconds"
        plans.append({k: plan[k] for k in ("sequence", "mode", "inference_ms", "fps") } | {
            "frames": len(plan["frames"]), "max_bone_length_error_m": max_error,
            "max_joint_motion_m": joint_motion, "boundary_error_m": boundary,
            "root_start_m": xyz[0, 0].tolist(), "root_end_m": xyz[-1, 0].tolist()})
        previous = xyz[-1]
        write_json(output / ("plan_%02d.json" % plan["sequence"]), plan)
    for invalid in [("missing", (.7, 0, 0), (1, 0, 0)), ("walk", (float("nan"), 0, 0), (1, 0, 0)),
                    ("walk", (0, 0, 1), (1, 0, 0)), ("walk", (.7, 0, 0), (0, 0, 0))]:
        try:
            backend.plan(*invalid)
        except ValueError:
            continue
        raise AssertionError("Invalid command accepted")

    # Change intent from a playback frame, not the future end of a queued plan.
    backend.reset()
    initial = backend.plan("walk", (.7, 0, 0), (1, 0, 0))
    for index in range(6):
        replanned = backend.plan("idle", (0, 0, 0), (1, 0, 0), initial["sequence"], 8)
        assert torch.allclose(torch.tensor(initial["frames"][8]), torch.tensor(replanned["frames"][0]), atol=1.e-6)
        assert initial["sequence"] in backend.plan_contexts, "Playing context evicted by unused replacement plans"
    for sequence, frame in [(999, 0), (initial["sequence"], -1), (initial["sequence"], 999), (None, 1), (True, 0)]:
        try:
            backend.plan("idle", (0, 0, 0), (1, 0, 0), sequence, frame)
        except ValueError:
            continue
        raise AssertionError("Invalid playback context accepted")
    report = {"passed": True, "midchunk_replan_continuous": True, "playing_context_preserved": True,
              "invalid_playback_context_rejected": True, "physics_calls": 0, "source_id": backend.manifest["source_id"],
              "model_id": backend.model_id, "checkpoints": backend.checkpoints, "plans": plans}
    validate_locomotion(backend, output)
    report["ground_locomotion_passed"] = True
    write_json(output / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def validate_locomotion(backend, output):
    cases = [
        ("idle", (0, 0, 0), (1, 0, 0), "walk", "standing", "idle"),
        ("walk", (.7, 0, 0), (1, 0, 0), "walk", "standing", "walk"),
        ("slow", (.35, 0, 0), (1, 0, 0), "slow_walk", "standing", "slow_walk"),
        ("fast", (1, 0, 0), (1, 0, 0), "fast_walk", "standing", "walk"),
        ("left", (0, -.7, 0), (1, 0, 0), "walk", "standing", "walk_left"),
        ("right", (0, .7, 0), (1, 0, 0), "walk", "standing", "walk_right"),
        ("backward", (-.7, 0, 0), (1, 0, 0), "walk", "standing", "walk"),
        ("turn", (0, 0, 0), (0, 1, 0), "walk", "standing", "idle"),
        ("crouch_idle", (0, 0, 0), (1, 0, 0), "walk", "crouched", "walk_stealth"),
        ("crouch_walk", (.7, 0, 0), (1, 0, 0), "walk", "crouched", "walk_stealth"),
    ]
    results = []
    for label, velocity, facing, gait, stance, expected_mode in cases:
        backend.reset()
        origin = None
        previous = None
        for _ in range(4):
            plan = backend.plan("walk", velocity, facing, locomotion={"gait": gait, "stance": stance})
            xyz = torch.tensor(plan["frames"])[..., :3]
            if origin is None:
                origin = xyz[0, 0].clone()
            if previous is not None:
                assert torch.allclose(xyz[0], previous, atol=1.e-6), label + " boundary"
            previous = xyz[-1].clone()
            assert plan["mode"] == expected_mode, label + " mode"
            assert plan["neural_inference"] and not plan["physics"]
            assert len(plan["frames"]) >= 33, label + " insufficient horizon"
        # Motion X maps to -UE Y, motion Z maps to UE X.
        delta_ue = (xyz[-1, 0] - origin)[[2, 0, 1]] * torch.tensor([1., -1., 1.])
        if any(velocity):
            assert torch.dot(delta_ue[:2], torch.tensor(velocity[:2], dtype=delta_ue.dtype)).item() > .05, label + " wrong translation"
        root_q = torch.tensor(plan["frames"])[-1, 0, 3:]
        x, y, z, w = root_q
        heading_ue = -torch.atan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y)).item()
        result = {"case": label, "mode": plan["mode"], "state": plan["locomotion_state"],
                  "resolved_speed_mps": plan["resolved_speed_mps"], "root_delta_ue_m": delta_ue.tolist(),
                  "root_height_m": xyz[-10:, 0, 1].mean().item(), "heading_ue_radians": heading_ue}
        expected_heading = torch.atan2(torch.tensor(float(facing[1])), torch.tensor(float(facing[0]))).item()
        heading_error = abs((heading_ue - expected_heading + torch.pi) % (2 * torch.pi) - torch.pi)
        assert heading_error < .2, label + " facing follows translation instead of requested body direction"
        if not any(velocity):
            assert delta_ue[:2].norm().item() < .05, label + " stationary root drift"
        print(json.dumps(result), flush=True)
        results.append(result)
        write_json(output / ("locomotion_" + label + ".json"), plan)
    heights = {result["case"]: result["root_height_m"] for result in results}
    assert heights["crouch_idle"] < heights["idle"] - .15, "Crouch template did not lower the pelvis"
    assert heights["crouch_walk"] < heights["walk"] - .15, "Crouch walk pose did not lower the pelvis"
    backend.reset()
    start = backend.plan(locomotion={"gait": "walk", "stance": "standing"})
    assert start["locomotion_state"] == "start"
    stop = backend.plan(velocity_ue_mps=(0, 0, 0), locomotion={"gait": "walk", "stance": "standing"})
    assert stop["locomotion_state"] == "stop"
    backend.reset()
    turn = backend.plan(velocity_ue_mps=(0, 0, 0), facing_ue=(0, 1, 0), locomotion={"gait": "walk", "stance": "standing"})
    assert turn["locomotion_state"] == "turn_in_place"
    for gait in ("run", "sprint", "jump", "fall", "land"):
        try:
            backend.plan(locomotion={"gait": gait, "stance": "standing"})
        except ValueError:
            continue
        raise AssertionError("Unsupported gait accepted: " + gait)
    write_json(output / "locomotion_validation.json", {"passed": True, "physics_calls": 0, "cases": results})


def serve(backend, port, output, response_delay_ms=0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(1)
        write_json(output / "ready.json", {"port": port, "schema": SCHEMA, "source_id": backend.manifest["source_id"]})
        print("READY 127.0.0.1:%d source=%s" % (port, backend.manifest["source_id"]), flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(30)
                backend.reset()
                session = uuid.uuid4().hex
                last_request = 0
                with connection.makefile("rb") as stream:
                    while True:
                        try:
                            line = stream.readline(MAX_REQUEST_BYTES + 1)
                        except OSError:
                            break
                        if not line:
                            break
                        try:
                            if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
                                raise ValueError("Request too large or incomplete")
                            request = json.loads(line)
                            if not isinstance(request, dict):
                                raise ValueError("Expected a command object")
                            if request.get("schema") != SCHEMA or request.get("source_id") != backend.manifest["source_id"]:
                                raise ValueError("Protocol/source hierarchy mismatch")
                            request_id = request["request_id"]
                            if type(request_id) is not int or request_id != last_request + 1:
                                raise ValueError("Request order mismatch")
                            response = backend.plan(request["mode"], request["velocity_ue_mps"], request["facing_ue"],
                                                    request.get("context_sequence"), request.get("context_frame"), request.get("locomotion"))
                            response.update(request_id=request_id, session_id=session)
                            last_request = request_id
                            print("PLAN %d mode=%s frames=%d inference=%.1fms" % (request_id, response["mode"], len(response["frames"]), response["inference_ms"]), flush=True)
                        except (ValueError, KeyError, TypeError, RuntimeError) as error:
                            response = {"schema": SCHEMA, "error": str(error)}
                        try:
                            if response_delay_ms:
                                time.sleep(response_delay_ms / 1000.0)
                            connection.sendall(json.dumps(response, separators=(",", ":"), allow_nan=False).encode() + b"\n")
                        except (ConnectionError, OSError):
                            break
                        if "error" in response:
                            break


def protocol_test(port, output):
    source = json.loads((output / "g1skel34.json").read_text(encoding="utf-8"))["source_id"]
    base = {"schema": SCHEMA, "source_id": source, "request_id": 1, "mode": "walk",
            "velocity_ue_mps": [.7, 0, 0], "facing_ue": [1, 0, 0]}
    rejected = 0
    for request in [dict(base, schema="legacy.qpos"), dict(base, source_id="wrong"),
                    dict(base, request_id=2), dict(base, mode="missing"),
                    dict(base, velocity_ue_mps=[0, 0, 1]), dict(base, locomotion={"gait": "jump", "stance": "standing"}),
                    dict(base, locomotion={"gait": "walk", "stance": "flying"}),
                    [], {"padding": "x" * MAX_REQUEST_BYTES}]:
        with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
            connection.sendall(json.dumps(request).encode() + b"\n")
            with connection.makefile("rb") as stream:
                reply = json.loads(stream.readline(4 * 1024 * 1024))
            assert "error" in reply, "Invalid wire request accepted"
            rejected += 1
    # Exercise fragmented TCP writes and session continuation using real inference.
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        with connection.makefile("rb") as stream:
            wire = json.dumps(base).encode() + b"\n"
            for part in (wire[:13], wire[13:]):
                connection.sendall(part)
            first = json.loads(stream.readline(4 * 1024 * 1024))
            connection.sendall(json.dumps(dict(base, request_id=2, mode="idle", velocity_ue_mps=[0, 0, 0],
                                              context_sequence=first["sequence"], context_frame=8)).encode() + b"\n")
            second = json.loads(stream.readline(4 * 1024 * 1024))
            assert first["neural_inference"] and not first["physics"] and first["source_id"] == source
            assert second["session_id"] == first["session_id"] and second["sequence"] == 2
            assert torch.allclose(torch.tensor(first["frames"][8]), torch.tensor(second["frames"][0]), atol=1.e-6)
            connection.sendall(json.dumps(dict(base, request_id=3)).encode() + b"\n")
            third = json.loads(stream.readline(4 * 1024 * 1024))
            assert third["session_id"] == first["session_id"] and third["sequence"] == 3
            assert torch.allclose(torch.tensor(second["frames"][-1]), torch.tensor(third["frames"][0]), atol=1.e-6)
            connection.sendall(json.dumps(dict(base, request_id=4, velocity_ue_mps=[0, .7, 0],
                                              locomotion={"gait": "walk", "stance": "standing"})).encode() + b"\n")
            lateral = json.loads(stream.readline(4 * 1024 * 1024))
            assert lateral["mode"] == "walk_right" and lateral["locomotion_state"] == "strafe_right"
            assert lateral["resolved_speed_mps"] <= .201 and lateral["neural_inference"] and not lateral["physics"]
    report = {"passed": True, "invalid_requests_rejected": rejected, "fragmented_request_passed": True,
              "continuous_neural_session_passed": True, "midchunk_replanning_passed": True, "source_id": source}
    write_json(output / "protocol_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "serve", "protocol-test"), default="serve")
    parser.add_argument("--result-dir", default="out")
    parser.add_argument("--output", type=Path, default=Path("out/kinematic_backend"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--port", type=int, default=28765)
    parser.add_argument("--response-delay-ms", type=int, default=0, help="Diagnostic latency injection; no change to neural inference")
    args = parser.parse_args()
    if args.response_delay_ms < 0:
        parser.error("--response-delay-ms must be nonnegative")
    if args.mode == "protocol-test":
        protocol_test(args.port, args.output)
        return
    # Fail-fast proof: none of these physics APIs may be called, even during setup.
    import mujoco
    def no_physics(*_args, **_kwargs):
        raise AssertionError("Physics called by the pure kinematic backend")
    for name in ("MjModel", "MjData", "mj_forward", "mj_step", "mj_inverse"):
        setattr(mujoco, name, no_physics)
    started = time.perf_counter()
    backend = KinematicBackend(args.result_dir)
    write_json(args.manifest or args.output / "g1skel34.json", backend.manifest)
    print("Loaded official models in %.1fs" % (time.perf_counter() - started), flush=True)
    if args.mode == "validate":
        validate(backend, args.output)
    else:
        # Warm CUDA before announcing readiness, not during the first playable frame.
        backend.plan("idle", (0, 0, 0), (1, 0, 0))
        backend.reset()
        serve(backend, args.port, args.output, args.response_delay_ms)


if __name__ == "__main__":
    main()
