"""Compare saved generator/reference and physical execution without resimulation."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.audit_mb_tracker_binary import decode, read_index, robot_contract
from scripts.replay_mb_sonic_tracking_isolation import read_episode, cached_schedule, select_reference
from motionbricks.mb_sonic.scene import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata, ticks, artifacts, _, _ = read_index(args.recording)
    episode = read_episode(args.recording)
    scene_sha = metadata["manifest_sha256"]
    schedule, sources, _ = cached_schedule(episode, scene_sha, metadata["robot"]["sha256"])
    bounds = robot_contract(metadata)["limits"]
    rows, reference_violations = [], 0
    for _, reference, _ in schedule:
        reference_violations += int(((reference.joint_pos < bounds[:, 0]) |
                                     (reference.joint_pos > bounds[:, 1])).sum())
    for key, receipt in ticks.items():
        artifact = artifacts[key]
        path = Path(artifact["path"])
        if sha256(path) != artifact["sha256"]:
            raise ValueError("Saved physics artifact changed")
        with np.load(path, allow_pickle=False) as archive:
            record = decode(json.loads(str(archive["metadata_json"]))["record"], archive)
        batch = record["physics"]
        time = float(batch["time_s"][0])
        reference, _ = select_reference(schedule, float(receipt["reference_time_s"]))
        index = int(np.argmin(abs(reference.timestamps_s - float(receipt["reference_time_s"]))))
        actual = batch["qpos"][0]
        target = reference.qpos[index]
        delta = actual[:3] - target[:3]
        up = Rotation.from_quat(actual[[4, 5, 6, 3]]).apply([0., 0., 1.])
        rows.append(dict(time_s=time, command_id=receipt["command_id"],
            root_xy_error_m=float(np.linalg.norm(delta[:2])), root_z_error_m=float(delta[2]),
            root_tilt_rad=float(np.arccos(np.clip(up[2], -1, 1))),
            joint_tracking_rmse_rad=float(np.sqrt(np.mean((batch["joint_positions"][0] - reference.joint_pos[index])**2)))))
    thresholds = {}
    for field, values in (("root_xy_error_m", [.5, 1., 2., 3.]), ("root_tilt_rad", [.5, .8, 1.])):
        for value in values:
            thresholds[f"first_{field}_over_{value}"] = next((r for r in rows if r[field] > value), None)
    result = dict(schema="motionbricks.mb_tracker.closed_loop_diagnosis.v1", recording=str(args.recording.resolve()),
        recording_sha256=sha256(args.recording), script_sha256=sha256(__file__), optimizer_updates=0,
        new_physics_steps=0, gpu_seconds=0, compared_control_ticks=len(rows), reference_plan_count=len(schedule),
        all_reference_numeric_sources=sources, reference_position_violation_count=reference_violations,
        first_threshold_crossings=thresholds, final_sample=rows[-1],
        interpretation="Measured world-reference drift and physical tilt; replayed references were not injected into physical state.",
        per_control_tick=rows)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ("compared_control_ticks", "reference_plan_count", "reference_position_violation_count", "first_threshold_crossings", "final_sample")}, indent=2))


if __name__ == "__main__":
    main()
