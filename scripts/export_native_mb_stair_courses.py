"""Author sparse stair Smart Object conditions, never a runtime animation clip."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "motionbricks.stair_courses.v1"


def export(dataset, output):
    dataset, output = Path(dataset), Path(output)
    report_path = dataset / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    windows = {w["artifact"]: w for w in report["windows"]}
    courses = []
    for direction, artifact in (("up", "00_up_start000000"), ("down", "03_down_start000000")):
        window = windows[artifact]
        source_path = dataset / (artifact + ".npz")
        with np.load(source_path, allow_pickle=False) as data:
            if float(data["fps"]) != 30:
                raise ValueError("Courses require native 30 Hz data")
            features = data["features_source"].astype(np.float32).copy()
        if features.ndim != 2 or features.shape[1] != 366 or not np.isfinite(features).all():
            raise ValueError("Invalid official SOMA global features")
        # Only a planar origin change: absolute source heights and every body channel stay intact.
        origin = features[3, [0, 2]].copy()
        features[:, 0] -= origin[0]
        features[:, 2] -= origin[1]
        endpoints = window["all_landing_endpoints"]
        if not endpoints or endpoints != sorted(set(endpoints)) or endpoints[0] < 7 or endpoints[-1] >= len(features):
            raise ValueError("Invalid authored landing endpoints")
        forward = features[endpoints[-1], [0, 2]] - features[3, [0, 2]]
        forward /= np.linalg.norm(forward)
        patches = []
        for patch in window["support_patches"]:
            patches.append({"id": patch["id"], "confidence": patch["confidence"],
                            "polygon_ue_cm": [[float((z - origin[1]) * 100),
                                               float(-(x - origin[0]) * 100), float(patch["height_m"] * 100)]
                                              for x, z in patch["polygon_xz_m"]]})
        courses.append({
            "id": direction, "fps": 30, "initial_root_m": features[3, :3].tolist(),
            "initial_facing_ue": [float(forward[1]), float(-forward[0]), 0.],
            "forward_xz": forward.tolist(), "scene_elevation_cm": 0.,
            "initial_history": features[:4].tolist(),
            "goals": [{"source_endpoint": int(end), "features": features[end - 3:end + 1].tolist()}
                      for end in endpoints],
            "support_patches": patches,
            "source": window["source"], "source_artifact_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "planar_origin_m": origin.tolist(),
            "conditions_role": "Authored source landing goals; not unseen-terrain validation. Only four seed frames and sparse goals are shipped.",
            "support_evidence": "Weak observed source footprints. Display slabs are illustrative, not measured full terrain.",
        })
    result = {"schema": SCHEMA, "courses": courses, "quality_approved": False,
              "root_checkpoint_sha256": report["bundle"]["root_checkpoint_sha256"],
              "pose_checkpoint_sha256": report["bundle"]["pose_checkpoint_sha256"],
              "tokenizer_checkpoint_sha256": report["bundle"]["tokenizer"]["checkpoint_sha256"],
              "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "scope": "Live authored Smart Object traversal; not autonomous terrain perception or collision avoidance."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.dataset, args.output)
    print(json.dumps({"output": str(args.output), "goals": {c["id"]: len(c["goals"]) for c in result["courses"]}}))
