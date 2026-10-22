"""Exercise the real socket/committed-clock stair endpoint without a renderer."""

import argparse
import json
from pathlib import Path
import socket
import time
import uuid

import numpy as np

from motionbricks.data.soma_replay import skeleton_manifest
from scripts.soma_ue_live_server import COMMITTED_SCHEMA


def probe(course, model_id, port, commands):
    replies, results = [], []
    forward = np.asarray(course["initial_facing_ue"])
    right = np.asarray([-forward[1], forward[0], 0.])
    session = uuid.uuid4().hex
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        stream = connection.makefile("rwb")
        for index, command in enumerate(commands):
            velocity = {"W": forward * 1.2, "S": -forward * 1.2, "A": -right, "D": right, "release": forward * 0}[command]
            request = dict(schema=COMMITTED_SCHEMA, session_id=session, request_id=index + 1,
                           model_id=model_id, source_id=skeleton_manifest()["source_id"],
                           stair_course=course["id"], velocity_ue_mps=velocity.tolist(),
                           facing_ue=forward.tolist(), client_time_s=time.monotonic(), publish_frames=8,
                           history_kind="initial", history_tick=-1, executed_tick=-1, history=[], executed_history=[])

            def records(reply):
                return [dict(sequence=reply["sequence"], frame=i, root_m=reply["frames"][i]["root_m"]) for i in range(4, 8)]

            if replies:
                previous = replies[-1]
                request.update(history_kind="committed", history_tick=previous["start_tick"] + 7, history=records(previous))
            if len(replies) >= 2:
                observed = replies[-2]
                request.update(executed_tick=observed["start_tick"] + 7, executed_history=records(observed))
            stream.write(json.dumps(request).encode() + b"\n")
            stream.flush()
            reply = json.loads(stream.readline(1024 * 1024))
            if "error" in reply:
                raise RuntimeError(reply["error"])
            assert reply["root_execution"] == "full_3d" and reply["neural_inference"] and reply["post_processing"] is None
            assert reply["model_id"] == model_id and reply["stair_course"] == course["id"]
            assert reply["start_tick"] == index * 8 and len(reply["frames"]) == 8
            roots = np.asarray([f["root_m"] for f in reply["frames"]])
            assert np.isfinite(roots).all()
            results.append(dict(command=command, state=reply["primitive_state"], goal=reply["active_goal_index"],
                                root=roots[-1].tolist(), backend_ms=reply["backend_ms"]))
            replies.append(reply)
    return dict(course=course["id"], source_initial_root=course["initial_root_m"],
                final_root=results[-1]["root"], plans=results,
                max_backend_ms=max(r["backend_ms"] for r in results),
                p95_backend_ms=float(np.percentile([r["backend_ms"] for r in results], 95)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--courses", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=28771)
    args = parser.parse_args()
    courses = json.loads(args.courses.read_text())["courses"]
    result = {"quality_approved": False, "scope": "Live protocol and input wiring, not motion quality approval", "trials": []}
    for course in courses:
        for commands in (["W"] * 45,
                         ["W"] * 3 + ["release"] * 2 + ["A", "D", "S"] * 2,
                         ["W"] * 45 + ["A"] * 3 + ["D"] * 3 + ["S"] * 3 + ["release"] * 3):
            trial = probe(course, args.model_id, args.port, commands)
            if len(commands) == 57:
                assert all(p["state"] == "exit_idle" for p in trial["plans"][45:51]), "Platform A/D did not use exit skill"
                assert any(p["state"] == "backtrack" for p in trial["plans"][51:54]), "Reverse did not return to stair skill"
            result["trials"].append(trial)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps([{k: v for k, v in trial.items() if k != "plans"} for trial in result["trials"]]))
