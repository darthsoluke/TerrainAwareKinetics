"""Read-only actual-G1/native-pose reconstruction check, without simulation."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter
from scripts.audit_mb_tracker_binary import decode, read_index
from motionbricks.mb_sonic.scene import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    adapter = CalibratedSomaG1Adapter.load(settings["calibration_path"],
        robot_model_path=settings["robot_model"], representation=build_native_representation())
    _, ticks, artifacts, _, _ = read_index(args.recording)
    rows = list(ticks.items())
    selected = np.unique(np.linspace(0, len(rows)-1, 12).astype(int))
    report = []
    for index in selected:
        key, receipt = rows[index]
        path = Path(artifacts[key]["path"])
        if sha256(path) != artifacts[key]["sha256"]:
            raise ValueError("Actual-state record hash changed")
        with np.load(path, allow_pickle=False) as archive:
            record = decode(json.loads(str(archive["metadata_json"]))["record"], archive)
        batch = record["physics"]
        q = batch["qpos"][[-1]]
        rotations = Rotation.from_quat(batch["body_quat_wxyz"][[-1]][..., [1, 2, 3, 0]].reshape(-1, 4))
        rotations = rotations.as_matrix().reshape(1, -1, 3, 3)
        start = perf_counter()
        estimate = adapter.inverse(q[:, :3], rotations)
        recovered, residual = adapter._map_anatomical(estimate.root_positions_m,
            estimate.world_positions_m, adapter.names, root_rotations=estimate.global_rot_mats[:, 0])
        elapsed = perf_counter()-start
        error = recovered[:, 7:] - q[:, 7:]
        report.append(dict(time_s=float(batch["time_s"][-1]),
            root_error_m=float(np.max(abs(recovered[:, :3]-q[:, :3]))),
            maximum_joint_error_rad=float(np.max(abs(error))), joint_error_rad=error[0].tolist(),
            milliseconds=elapsed*1000, anatomical_projection_residual=residual))
    result = dict(schema="motionbricks.executed_feedback_reconstruction.v1", optimizer_updates=0,
        new_physics_steps=0, gpu_seconds=0, source_recording_sha256=sha256(args.recording), samples=report)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps([dict(time_s=r["time_s"], maximum_joint_error_rad=r["maximum_joint_error_rad"],
        milliseconds=r["milliseconds"]) for r in report], indent=2))


if __name__ == "__main__":
    main()
