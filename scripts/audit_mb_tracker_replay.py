"""Read a completed physical replay once; no inference, simulation or training."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from motionbricks.mb_sonic.reference import G1_JOINT_NAMES, file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = np.zeros(29, dtype=np.int64)
    minimum_margin, root_min, penetration = np.inf, np.inf, 0.
    worst, samples, force_max, warnings = None, 0, 0., 0
    scene_body_contact_samples = 0
    metadata, model = None, None
    with args.recording.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "metadata":
                metadata = row
                model = mujoco.MjModel.from_xml_path(metadata["robot"]["path"])
                ids = np.array([model.joint(name).id for name in G1_JOINT_NAMES])
                limits = model.jnt_range[ids]
                fall_bodies = {model.body(name).id for name in ("pelvis", "torso_link")}
            if row["kind"] == "policy_tick":
                states = row["physics_states"]
            elif row["kind"] in ("explicit_initial_reset", "actual_history_warmup"):
                states = [row["actual"]]
            else:
                continue
            for state in states:
                q = np.asarray(state["joint_positions"])
                margin = np.minimum(q-limits[:, 0], limits[:, 1]-q)
                counts += margin < -1e-6
                index = int(margin.argmin())
                if margin[index] < minimum_margin:
                    minimum_margin = float(margin[index])
                    worst = dict(time_s=state["time_s"], joint=G1_JOINT_NAMES[index],
                        position_rad=float(q[index]), limits_rad=limits[index].tolist())
                root_min = min(root_min, float(state["qpos"][2]))
                warnings = max(warnings, sum(value["number"] for value in state["solver_warnings"].values()))
                force_max = max(force_max, float(np.abs(state["actuator_forces"]).max()))
                samples += 1
                body_contact = False
                for contact in state["contacts"]:
                    if 0 in contact["body_ids"]:
                        penetration = max(penetration, -contact["distance"])
                        if contact["wrench_contact"][0] > 1 and fall_bodies.intersection(contact["body_ids"]):
                            body_contact = True
                scene_body_contact_samples += body_contact
    result = dict(schema="motionbricks.tracker_replay.actual_audit.v1", optimizer_updates=0,
        new_simulation=False, recording=str(args.recording.resolve()), recording_sha256=file_sha256(args.recording),
        tracker=metadata.get("tracker", "sonic"), robot=metadata["robot"], physical_state_samples=samples,
        maximum_joint_overtravel_rad=max(0., -minimum_margin), minimum_joint_margin_rad=minimum_margin,
        worst_joint_sample=worst, violation_pairs=int(counts.sum()),
        joint_violating_duration_s=dict(zip(G1_JOINT_NAMES, (counts*metadata["physics_dt_s"]).tolist())),
        minimum_root_z_m=root_min, pelvis_or_torso_scene_contact_samples=scene_body_contact_samples,
        maximum_scene_contact_penetration_m=penetration, maximum_abs_motor_force_nm=force_max,
        maximum_solver_warning_count=warnings)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: result[key] for key in ("tracker", "physical_state_samples", "maximum_joint_overtravel_rad",
        "worst_joint_sample", "minimum_root_z_m", "pelvis_or_torso_scene_contact_samples",
        "maximum_scene_contact_penetration_m", "maximum_solver_warning_count")}, indent=2))


if __name__ == "__main__":
    main()
