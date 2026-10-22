"""Read-only saved-record forensics; no inference, simulation or model loading."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from motionbricks.mb_sonic.reference import G1_JOINT_NAMES, MUJOCO_FROM_ISAACLAB


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "out/mb_sonic_e0_e3_20261018"


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[..., [1, 2, 3, 0]])


def value(archive, field):
    return archive[field["array"]] if isinstance(field, dict) and "array" in field else np.asarray(field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=ASSETS / "wasd_v16/episode_0001_000.jsonl")
    parser.add_argument("--output", type=Path, default=ASSETS / "tracking_isolation_v1/first_instability.json")
    parser.add_argument("--maximum-time-s", type=float, default=22.)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    xml_path = ROOT.parent / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
    tree = ET.parse(xml_path)
    ranges = {joint.attrib["name"]: np.fromstring(joint.attrib["range"], sep=" ")
              for joint in tree.iter("joint") if "name" in joint.attrib and "range" in joint.attrib}
    limits = np.array([ranges[name] for name in G1_JOINT_NAMES])
    body_names = ["world", *[body.attrib["name"] for body in tree.find("worldbody").iter("body")]]
    actuator_names = [element.attrib["joint"] for element in tree.find("actuator")]
    actuator_ids = [actuator_names.index(name) for name in G1_JOINT_NAMES]
    ticks = {row["numeric_record_id"]: row for row in rows if row["kind"] == "policy_tick_binary"}
    artifacts = {row["record_id"]: row for row in rows if row["kind"] == "numeric_artifact_written"}
    ordered = sorted((key for key in ticks), key=lambda key: ticks[key]["reference_time_s"])
    states, first_violation = {}, None

    def capture(state, archive=None, path=None, tick_id=None):
        nonlocal first_violation
        q = value(archive, state["qpos"]).copy()
        joints = value(archive, state["joint_positions"]).copy()
        index = round(state["time_s"] * 200)
        states[index] = dict(time_s=state["time_s"], qpos=q, joints=joints,
                             state=state, path=path, tick_id=tick_id)
        bad = np.flatnonzero((joints < limits[:, 0]) | (joints > limits[:, 1]))
        if first_violation is None and len(bad):
            j = int(bad[0])
            first_violation = dict(time_s=state["time_s"], joint=G1_JOINT_NAMES[j],
                angle_rad=float(joints[j]), limits_rad=limits[j].tolist(), tick_id=tick_id,
                margin_rad=float(min(joints[j]-limits[j, 0], limits[j, 1]-joints[j])))

    for row in rows:
        if row["kind"] == "actual_history_warmup":
            capture(row["actual"])
    scanned_ticks = []
    for key in ordered:
        if ticks[key]["reference_time_s"] >= args.maximum_time_s:
            break
        path = artifacts[key]["path"]
        with np.load(path, allow_pickle=False) as archive:
            record = json.loads(str(archive["metadata_json"]))["record"]["tick"]
            for state in record["physics_states"]:
                capture(state, archive, path, key)
        scanned_ticks.append(key)
        if first_violation and record["physics_states"][-1]["time_s"] >= first_violation["time_s"] + 1:
            break
    if first_violation is None:
        raise ValueError("No joint violation found inside the declared scan bound")
    sequence = sorted(states.values(), key=lambda item: item["time_s"])
    q = np.array([item["qpos"] for item in sequence])
    times = np.array([item["time_s"] for item in sequence])
    initial_joints = np.array([item["joints"] for item in sequence if item["time_s"] <= 10.])
    initial_violations = int(((initial_joints < limits[:, 0]) | (initial_joints > limits[:, 1])).sum())
    orientations = rotation(q[:, 3:7])
    euler = orientations.as_euler("xyz")
    tilt = np.arccos(np.clip(orientations.as_matrix()[:, 2, 2], -1, 1))
    masks = {"root_below_0_65m": q[:, 2] < .65, "root_below_0_55m": q[:, 2] < .55,
             "tilt_above_20deg": tilt > np.deg2rad(20), "tilt_above_30deg": tilt > np.deg2rad(30),
             "absolute_roll_above_20deg": np.abs(euler[:, 0]) > np.deg2rad(20),
             "absolute_pitch_above_20deg": np.abs(euler[:, 1]) > np.deg2rad(20)}
    events = {}
    for name, mask in masks.items():
        events[name] = None
        for i in range(len(mask)-40):
            if mask[i:i+41].all() and np.isclose(times[i+40]-times[i], .2, atol=1e-8, rtol=0):
                events[name] = dict(onset_time_s=float(times[i]), confirmed_time_s=float(times[i+40]),
                    root_height_m=float(q[i, 2]), root_rpy_rad=euler[i].tolist(), tick_id=sequence[i]["tick_id"])
                break

    def state_value(item, key):
        if item["path"] is None:
            return np.asarray(item["state"][key])
        with np.load(item["path"], allow_pickle=False) as archive:
            return value(archive, item["state"][key]).copy()

    first_uses = [row for row in rows if row["kind"] == "reference_packet_used" and row["first_command_use"]]
    uses = {round(row["used_time_window"][0]*50): row for row in rows if row["kind"] == "reference_packet_used"}
    snapshots, timeline, plan_ids = [], [], set()
    selected_times = [17.88, 18.14, 18.6, 19., 19.4, 19.8, 20., 20.12, 20.28, 20.36, 20.48]
    selected_keys = {min(scanned_ticks, key=lambda key: abs(ticks[key]["reference_time_s"]-t)) for t in selected_times}
    stable_start, stable_end = first_violation["time_s"]-2, first_violation["time_s"]
    joint_ids = [G1_JOINT_NAMES.index(name) for name in (
        "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_knee_joint", "right_knee_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]
    target_first, target_first_by_joint = None, {}
    stable_stats = dict(ticks=0, reference_actual_joint_error_max_rad=0., reference_actual_joint_error_squared_sum=0.,
                        pd_target_outside_xml_pairs=0, maximum_abs_torque_nm=0.)
    for key in scanned_ticks:
        t = ticks[key]["reference_time_s"]
        if t < stable_start-.3 and key not in selected_keys:
            continue
        with np.load(artifacts[key]["path"], allow_pickle=False) as archive:
            record = json.loads(str(archive["metadata_json"]))["record"]["tick"]
            ref, policy = record["adapted_reference"], record["policy_command"]
            reference_q = value(archive, ref["qpos"])[0]
            target = value(archive, policy["joint_target"])
            action = value(archive, policy["action_isaac"])[MUJOCO_FROM_ISAACLAB]
            actual = states[round(t*200)]
            torque = value(archive, record["physics_states"][0]["actuator_forces"])[actuator_ids]
            outside = np.flatnonzero((target < limits[:, 0]) | (target > limits[:, 1]))
            for j in outside:
                target_first_by_joint.setdefault(G1_JOINT_NAMES[j], dict(time_s=t,
                    value_rad=float(target[j]), reference_rad=float(reference_q[j+7]),
                    actual_rad=float(actual["joints"][j]), limits_rad=limits[j].tolist()))
            if target_first is None and len(outside):
                target_first = dict(time_s=t, scope="inspected focused window, not entire episode",
                    joints=[G1_JOINT_NAMES[i] for i in outside], values_rad=target[outside].tolist())
            error = actual["joints"]-reference_q[7:]
            if stable_start <= t < stable_end:
                stable_stats["ticks"] += 1
                stable_stats["reference_actual_joint_error_max_rad"] = max(stable_stats["reference_actual_joint_error_max_rad"], float(np.abs(error).max()))
                stable_stats["reference_actual_joint_error_squared_sum"] += float(np.sum(error**2))
                stable_stats["pd_target_outside_xml_pairs"] += len(outside)
                stable_stats["maximum_abs_torque_nm"] = max(stable_stats["maximum_abs_torque_nm"], float(np.abs(torque).max()))
            receipt = uses[round(t*50)]
            plan_ids.add(receipt["plan_id"])
            if round(t*50) % 5 == 0 or key in selected_keys:
                timeline.append(dict(time_s=t, command_seq=receipt["generated_from_command_seq"],
                    plan_id=receipt["plan_id"], root_height_m=float(actual["qpos"][2]),
                    root_rpy_rad=rotation(actual["qpos"][3:7]).as_euler("xyz").tolist(),
                    joint_names=[G1_JOINT_NAMES[i] for i in joint_ids], reference_rad=reference_q[7:][joint_ids].tolist(),
                    actual_rad=actual["joints"][joint_ids].tolist(), pd_target_rad=target[joint_ids].tolist(),
                    actuator_torque_nm=torque[joint_ids].tolist()))
            if key not in selected_keys:
                continue
            encoder = value(archive, policy["encoder_input"]).reshape(-1)
            columns = encoder[601:661].reshape(10, 3, 2)[0].astype(float)
            encoded_r = np.column_stack((columns, np.cross(columns[:, 0], columns[:, 1])))
            actual_r = rotation(actual["qpos"][3:7]).as_matrix()
            reference_r = rotation(reference_q[3:7]).as_matrix()
            alignment = actual_r @ encoded_r @ reference_r.T
            actual_body_rot = state_value(actual, "body_quat_wxyz")
            actual_body_pos = state_value(actual, "body_positions_world")
            ref_rot = value(archive, ref["body_quat_w"])[0]
            ref_pos = value(archive, ref["body_pos_w"])[0]
            bodies = {}
            for name in ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"):
                ai, ri = body_names.index(name), ref["body_names"].index(name)
                ar, rr = rotation(actual_body_rot[ai]).as_matrix(), rotation(ref_rot[ri]).as_matrix()
                bodies[name] = dict(actual_position_m=actual_body_pos[ai].tolist(), reference_position_m=ref_pos[ri].tolist(),
                    actual_quaternion_wxyz=actual_body_rot[ai].tolist(), reference_quaternion_wxyz=ref_rot[ri].tolist(),
                    raw_world_angle_error_rad=float(Rotation.from_matrix(ar.T@rr).magnitude()),
                    encoder_heading_aligned_angle_error_rad=float(Rotation.from_matrix(ar.T@alignment@rr).magnitude()))
            snapshots.append(dict(time_s=t, first_substep_time_s=record["physics_states"][0]["time_s"],
                tick_id=key, plan_id=receipt["plan_id"], command=receipt["actual_raw_intent"],
                reference_root_qpos=reference_q[:7].tolist(), actual_root_qpos=actual["qpos"][:7].tolist(),
                joints={name:dict(reference_rad=float(reference_q[i+7]), actual_rad=float(actual["joints"][i]),
                    pd_target_rad=float(target[i]), action_mujoco_order=float(action[i]),
                    first_substep_actuator_torque_nm=float(torque[i]), limits_rad=limits[i].tolist())
                    for i,name in enumerate(G1_JOINT_NAMES)}, bodies=bodies,
                encoder_current_relative_root_matrix=encoded_r.tolist(),
                encoder_current_relative_root_angle_rad=float(Rotation.from_matrix(encoded_r).magnitude()),
                implied_heading_alignment_rpy_rad=Rotation.from_matrix(alignment).as_euler("xyz").tolist(),
                encoder_joint_position_max_error_rad=float(np.abs(encoder[4:294].reshape(10,29)[0][MUJOCO_FROM_ISAACLAB]-reference_q[7:]).max()),
                timing_ns=record["timing_ns"]))
    stable_stats["joint_tracking_rms_rad"] = float(np.sqrt(stable_stats.pop("reference_actual_joint_error_squared_sum")/(stable_stats["ticks"]*29)))
    plans = []
    for plan_id in sorted(plan_ids):
        artifact = artifacts[plan_id]
        with np.load(artifact["path"], allow_pickle=False) as archive:
            record = json.loads(str(archive["metadata_json"]))["record"]
            source, adapted = record["raw_source"], record["adapted_reference"]
            ts = value(archive, source["timestamps_s"])
            root_positions = value(archive, source["root_positions_m"])
            joints = value(archive, adapted["qpos"])[:, 7:]
            near = np.argsort(np.abs(ts-first_violation["time_s"]))[:2]
            plans.append(dict(plan_id=plan_id, numeric_path=artifact["path"], source_time_bounds_s=ts[[0,-1]].tolist(),
                raw_source_root_brackets_near_first_violation=[dict(time_s=float(ts[i]), soma_y_up_xyz_m=root_positions[i].tolist()) for i in sorted(near)],
                source_kind=source["provenance"]["kind"],
                source_constraint_audit=adapted["adaptation"]["source_constraint_audit"],
                dense_constraint_audit=adapted["adaptation"]["dense_constraint_audit"],
                reference_constraints=adapted["adaptation"]["reference_constraints"],
                right_ankle_pitch_range_rad=[float(joints[:,4+6].min()),float(joints[:,4+6].max())],
                current_root_height_range_m=[float(value(archive,adapted["qpos"])[:,2].min()),float(value(archive,adapted["qpos"])[:,2].max())]))
    violation_state = states[round(first_violation["time_s"]*200)]
    violation_torque = state_value(violation_state, "actuator_forces")[actuator_ids]
    violation_velocity = state_value(violation_state, "joint_velocities")
    first_violation["root_qpos"] = violation_state["qpos"][:7].tolist()
    first_violation["actual_named_joints"] = {name: dict(angle_rad=float(violation_state["joints"][i]),
        velocity_rad_s=float(violation_velocity[i]), recorded_actuator_force_nm=float(violation_torque[i]))
        for i, name in enumerate(G1_JOINT_NAMES)}
    first_snapshot = next(s for s in snapshots if s["tick_id"] == first_violation["tick_id"])
    preceding = first_snapshot["joints"][first_violation["joint"]]
    report = dict(schema="motionbricks.mb_sonic.first_instability_forensics.v1", optimizer_updates=0,
        gpu_used=False, inference=False, physical_replay=False, log=str(args.log.resolve()),
        coverage=dict(initial_search_s=10., initial_10s_joint_violation_count=initial_violations, scan_until_s=float(times[-1]),
            physical_states=len(sequence), tick_archives=len(scanned_ticks), body_policy_detail_scope="selected snapshots and two seconds preceding first joint violation; not full episode"),
        original_robot_xml=str(xml_path), first_actual_joint_violation=first_violation,
        sustained_events=events, sustained_definition="threshold continuously exceeded for 0.200s (41 samples at 200Hz)",
        preceding_2s=dict(time_bounds_s=[stable_start,stable_end], **stable_stats),
        first_pd_target_outside_limits_in_focused_window=target_first,
        first_pd_target_outside_by_joint_in_focused_window=target_first_by_joint,
        first_command_uses=[dict(seq=x["generated_from_command_seq"],sim_time_s=x["used_time_window"][0],
            velocity_xy_mps=x["actual_raw_intent"]["velocity_xy_mps"],yaw_rate_radps=x["actual_raw_intent"]["yaw_rate_radps"],
            input_to_motor_ms=x["input_to_motor_ns"]/1e6) for x in first_uses],
        conventions=dict(joints="original XML named MuJoCo 29-DOF order", torque="first substep recorded motor actuator force, not reference torque",
            encoder="saved mode-0 first 6D root orientation, reconstructed from first two matrix columns",
            heading_alignment="inferred actual_R @ saved_encoder_relative_R @ reference_R.T; no runtime state changes",
            event_thresholds="diagnostic thresholds chosen before trajectory extraction, not acceptance gates"),
        snapshots=snapshots, timeline=timeline, active_plans=plans,
        leading_evidence=dict(first_limit_controller_time_s=first_snapshot["time_s"],
            first_limit_controller_joint=preceding,
            right_foot_angle_error_before_first_limit_rad=first_snapshot["bodies"]["right_ankle_roll_link"]["encoder_heading_aligned_angle_error_rad"],
            torso_angle_error_before_first_limit_rad=first_snapshot["bodies"]["torso_link"]["encoder_heading_aligned_angle_error_rad"],
            encoder_root_angle_before_first_limit_rad=first_snapshot["encoder_current_relative_root_angle_rad"],
            all_focused_native_and_dense_audits_pass=all(p["source_constraint_audit"]["status"] == "PASS"
                and p["dense_constraint_audit"]["status"] == "PASS" for p in plans),
            focused_reference_qp_total=sum(p["reference_constraints"]["qp_solves"] for p in plans),
            interpretation="Reference stays legal without QP. Large foot/torso tracking errors and an out-of-range decoder PD target precede first actual limit violation and sustained root collapse. Root-only encoder angle is much smaller than torso error; inspect official anchor/body and action semantics before attributing the fall to reference ROM."),
        causal_scope="Temporal evidence separates upstream legal references from decoder targets and physical tracking; no counterfactual rollout establishes a unique causal bug")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output.resolve()),first=first_violation,sustained=events,
        preceding_2s=report["preceding_2s"],snapshots=len(snapshots),plans=len(plans)),indent=2))


if __name__ == "__main__":
    main()
