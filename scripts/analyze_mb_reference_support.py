"""Read recorded v13 targets and execution; no model, physics, or GPU imports."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


OFFSETS = np.array([0., .02, .04, .08, .16])
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(value, archive):
    if isinstance(value, dict):
        if "array" in value:
            return archive[value["array"]]
        return {key: decode(item, archive) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item, archive) for item in value]
    return value


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]])


def tilt(q):
    q = np.asarray(q)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1]**2 + q[..., 2]**2), -1, 1)))


def angle(a, b):
    return np.degrees((rotation(a).inv() * rotation(b)).magnitude())


def heading(q):
    forward = rotation(q).apply([1., 0., 0.])
    return Rotation.from_rotvec([0., 0., np.arctan2(forward[1], forward[0])])


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def stats(values):
    a = np.asarray(values)
    return dict(min=float(a.min()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()), mean=float(a.mean()))


def sample(ref, times):
    idx = np.rint((np.asarray(times) - ref["timestamps_s"][0]) / .02).astype(int)
    if np.any(idx < 0) or np.any(idx >= len(ref["timestamps_s"])):
        raise ValueError("Requested sample is outside the saved reference")
    if not np.allclose(ref["timestamps_s"][idx], times, atol=1e-8, rtol=0):
        raise ValueError("Saved 50Hz reference does not contain requested timestamps")
    return {key: ref[key][idx] for key in
            ("qpos", "qvel", "body_pos_w", "body_quat_w", "body_lin_vel_w")}


def transition(old, new, time, foot_ids, torso):
    a, b = sample(old, time + OFFSETS), sample(new, time + OFFSETS)
    dq, dv = b["qpos"][:, 7:] - a["qpos"][:, 7:], b["qvel"][:, 6:] - a["qvel"][:, 6:]
    foot = np.linalg.norm(b["body_pos_w"][:, foot_ids] - a["body_pos_w"][:, foot_ids], axis=-1)
    joint = int(np.argmax(abs(dq[0])))
    future, future_joint = np.unravel_index(np.argmax(abs(dq[1:])), dq[1:].shape)
    return dict(time_s=float(time), max_component_details=dict(
        current_joint=old["joint_names"][joint], current_joint_old_rad=float(a["qpos"][0, 7 + joint]),
        current_joint_new_rad=float(b["qpos"][0, 7 + joint]),
        future_joint=old["joint_names"][future_joint], future_offset_s=float(OFFSETS[1:][future]),
        future_joint_old_rad=float(a["qpos"][future + 1, 7 + future_joint]),
        future_joint_new_rad=float(b["qpos"][future + 1, 7 + future_joint])),
        current_root_position_jump_m=float(np.linalg.norm(b["qpos"][0, :3] - a["qpos"][0, :3])),
        current_root_orientation_jump_deg=float(angle(a["qpos"][0, 3:7], b["qpos"][0, 3:7])),
        current_torso_orientation_jump_deg=float(angle(a["body_quat_w"][0, torso], b["body_quat_w"][0, torso])),
        current_joint_position_jump_max_rad=float(abs(dq[0]).max()),
        current_joint_position_jump_rms_rad=rms(dq[0]),
        current_joint_velocity_jump_max_rad_s=float(abs(dv[0]).max()),
        current_foot_position_jump_max_m=float(foot[0].max()),
        tracker_future_joint_position_jump_max_rad=float(abs(dq[1:]).max()),
        tracker_future_joint_position_jump_rms_rad=rms(dq[1:]),
        tracker_future_joint_velocity_jump_max_rad_s=float(abs(dv[1:]).max()),
        tracker_future_torso_orientation_jump_max_deg=float(angle(a["body_quat_w"][1:, torso], b["body_quat_w"][1:, torso]).max()),
        future_foot_position_jump_max_m=float(foot[1:].max()))


def summarize(rows, jumps):
    scalar_keys = ("reference_root_tilt_deg", "reference_torso_tilt_deg", "actual_root_tilt_deg",
        "actual_torso_tilt_deg", "root_orientation_error_deg", "torso_orientation_error_deg",
        "root_xy_error_m", "root_z_error_m", "joint_tracking_rmse_rad", "reference_joint_velocity_max_rad_s",
        "future_torso_tilt_max_deg", "future_joint_velocity_max_rad_s", "reference_root_horizontal_speed_mps",
        "reference_foot_speed_max_mps", "reference_root_to_midfeet_xy_m")
    result = dict(count=len(rows), sample_times_s=[rows[0]["time_s"], rows[-1]["time_s"]],
        metrics={key: stats([r[key] for r in rows]) for key in scalar_keys},
        endpoint_samples=[rows[0], rows[-1]],
        foot_origin_height_m={side: stats([r["reference_foot_positions_m"][i][2] for r in rows])
                              for i, side in enumerate(("left", "right"))},
        raw_predicted_contact_channels={str(i): stats([r["raw_predicted_contacts"][i] for r in rows]) for i in range(4)},
        raw_predicted_contact_threshold_0_5_active_fraction=np.mean(
            np.array([r["raw_predicted_contacts"] for r in rows]) > .5, axis=0).tolist(),
        raw_predicted_contact_threshold_0_5_change_count=np.count_nonzero(np.diff(
            np.array([r["raw_predicted_contacts"] for r in rows]) > .5, axis=0), axis=0).tolist(),
        reference_joint_position_range_rad=(np.ptp([r["reference_joint_positions_rad"] for r in rows], axis=0)).tolist(),
        reference_joint_velocity_rms_rad_s=rms([r["reference_joint_velocities_rad_s"] for r in rows]))
    result["tilt_exceedance_samples"] = {key: {str(limit): sum(r[key] > limit for r in rows)
        for limit in (15, 30, 45, 60)} for key in
        ("reference_root_tilt_deg", "reference_torso_tilt_deg", "actual_root_tilt_deg", "actual_torso_tilt_deg")}
    for key in ("reference_root_tilt_deg", "reference_torso_tilt_deg", "actual_root_tilt_deg", "actual_torso_tilt_deg"):
        result[key + "_max_sample"] = max(rows, key=lambda row: row[key])
    result["adoption_count"] = len(jumps)
    fields = [key for key, value in jumps[0].items() if key != "time_s" and isinstance(value, (int, float))]
    result["same_absolute_time_adoption_discontinuities"] = {
        key: dict(distribution=stats([j[key] for j in jumps]),
                  largest=sorted(jumps, key=lambda j: j[key], reverse=True)[:3]) for key in fields}
    changes = np.diff([r["reference_joint_positions_rad"] for r in rows], axis=0)
    same = np.array([a["plan_id"] == b["plan_id"] for a, b in zip(rows, rows[1:])])
    result["within_one_plan_20ms_joint_change_max_rad"] = stats(np.max(abs(changes[same]), axis=1))
    result["adoption_20ms_joint_change_max_rad"] = stats(np.max(abs(changes[~same]), axis=1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    recording = args.recording.resolve(strict=True)
    log_hash = digest(recording)
    lines = [json.loads(line) for line in recording.read_text(encoding="utf-8").splitlines()]
    metadata = next(r for r in lines if r["kind"] == "metadata")
    artifacts = {r["record_id"]: r for r in lines if r["kind"] == "numeric_artifact_written"}
    uses = [r for r in lines if r["kind"] == "reference_packet_used"]
    ticks = [r for r in lines if r["kind"] == "policy_tick_binary"]
    initial = next(r["actual"] for r in lines if r["kind"] == "initial_measured_state_refresh")
    terminal = next(r["actual"] for r in lines if r["kind"] == "explicit_stop")
    robot = Path(metadata["robot"]["path"])
    if digest(robot) != metadata["robot"]["sha256"]:
        raise ValueError("Recorded robot XML changed")
    body_names = ["world"] + [b.attrib["name"] for b in ET.parse(robot).find("worldbody").iter("body")]
    actual_ids = [body_names.index(name) for name in FEET]
    actual_torso_id = body_names.index("torso_link")
    sources, refs, raw_contacts, jumps, rows = [], {}, {}, [], []

    def load(record_id):
        receipt = artifacts[record_id]
        path = Path(receipt["path"])
        actual_hash = digest(path)
        if actual_hash != receipt["sha256"]:
            raise ValueError(f"Recorded NPZ hash changed: {record_id}")
        sources.append(dict(record_id=record_id, path=str(path.resolve()), sha256=actual_hash))
        return np.load(path, allow_pickle=False)

    previous_actual = dict(qpos=np.array(initial["qpos"]),
        body_positions_world=np.array(initial["body_positions_world"]),
        body_quat_wxyz=np.array(initial["body_quat_wxyz"]))
    previous_id = None
    for receipt, use in zip(ticks, uses, strict=True):
        time = float(receipt["reference_time_s"])
        plan_id = use["plan_id"]
        if plan_id not in refs:
            with load(plan_id) as archive:
                record = json.loads(str(archive["metadata_json"]))["record"]
                ref = decode(record["adapted_reference"], archive)
                raw = decode(record["raw_source"], archive)
            refs[plan_id] = ref
            raw_contacts[plan_id] = (raw["timestamps_s"], raw["raw_features"][:, -4:])
        ref = refs[plan_id]
        torso = ref["body_names"].index("torso_link")
        feet = [ref["body_names"].index(name) for name in FEET]
        if previous_id is None:
            heading_alignment = heading(previous_actual["body_quat_wxyz"][actual_torso_id]) * heading(ref["body_quat_w"][0, torso]).inv()
        if previous_id is not None and previous_id != plan_id:
            jump = transition(refs[previous_id], ref, time, feet, torso)
            jump.update(old_plan_id=previous_id, new_plan_id=plan_id)
            old_times, old_contacts = raw_contacts[previous_id]
            new_times, new_contacts = raw_contacts[plan_id]
            jump["raw_contact_same_time_interp_jump_max"] = float(max(
                abs(np.interp(time, new_times, new_contacts[:, i]) - np.interp(time, old_times, old_contacts[:, i]))
                for i in range(4)))
            jumps.append(jump)
        with load(receipt["numeric_record_id"]) as archive:
            record = json.loads(str(archive["metadata_json"]))["record"]
            saved = decode(record["tick"]["adapted_reference"], archive)
            inputs = decode(record["tick"]["policy_command"]["inputs"], archive)
            physics = record["physics"]
            current = sample(ref, time + OFFSETS)
            if record["tick"]["raw_reference"]["plan_id"] != plan_id:
                raise ValueError("Tick and consumed reference identities differ")
            for key in ("qpos", "qvel", "body_pos_w", "body_quat_w"):
                if not np.array_equal(current[key], saved[key]):
                    raise ValueError(f"Full plan and actual consumed window differ: {key}")
            if not np.array_equal(inputs["mimic_future_dof_pos"][0], current["qpos"][1:, 7:].astype(np.float32)):
                raise ValueError("Recorded tracker future positions differ")
            if not np.array_equal(inputs["mimic_future_dof_vel"][0], current["qvel"][1:, 6:].astype(np.float32)):
                raise ValueError("Recorded tracker future velocities differ")
            next_actual = {key: decode(physics[key], archive)[-1].copy() for key in
                           ("qpos", "body_positions_world", "body_quat_wxyz")}
        target = current["qpos"][0]
        actual = previous_actual["qpos"]
        raw_times, contacts = raw_contacts[plan_id]
        native_idx = int(np.argmin(abs(raw_times - time)))
        root_delta = actual[:3] - target[:3]
        foot_pos = current["body_pos_w"][0, feet]
        torso_q = current["body_quat_w"][0, torso]
        actual_torso_q = previous_actual["body_quat_wxyz"][actual_torso_id]
        aligned_torso = heading_alignment * rotation(torso_q)
        row = dict(time_s=time, plan_id=plan_id, command_seq=use["generated_from_command_seq"],
            reference_root_tilt_deg=float(tilt(target[3:7])), reference_torso_tilt_deg=float(tilt(torso_q)),
            actual_root_tilt_deg=float(tilt(actual[3:7])), actual_torso_tilt_deg=float(tilt(actual_torso_q)),
            root_orientation_error_deg=float(angle(target[3:7], actual[3:7])),
            torso_orientation_error_deg=float(np.degrees((aligned_torso.inv() * rotation(actual_torso_q)).magnitude())),
            root_xy_error_m=float(np.linalg.norm(root_delta[:2])), root_z_error_m=float(root_delta[2]),
            reference_root_position_m=target[:3].tolist(), actual_root_position_m=actual[:3].tolist(),
            reference_root_quat_wxyz=target[3:7].tolist(), reference_torso_quat_wxyz=torso_q.tolist(),
            reference_joint_positions_rad=target[7:].tolist(), reference_joint_velocities_rad_s=current["qvel"][0, 6:].tolist(),
            reference_waist_yaw_roll_pitch_deg=np.degrees(target[19:22]).tolist(),
            joint_tracking_rmse_rad=rms(actual[7:] - target[7:]),
            reference_joint_velocity_max_rad_s=float(abs(current["qvel"][0, 6:]).max()),
            future_torso_tilt_max_deg=float(tilt(current["body_quat_w"][1:, torso]).max()),
            future_joint_velocity_max_rad_s=float(abs(inputs["mimic_future_dof_vel"]).max()),
            recorded_tracker_future_torso_tilt_deg=tilt(inputs["mimic_future_anchor_rot"][0][:, [3, 0, 1, 2]]).tolist(),
            reference_root_horizontal_speed_mps=float(np.linalg.norm(current["qvel"][0, :2])),
            reference_foot_positions_m=foot_pos.tolist(),
            actual_foot_positions_m=previous_actual["body_positions_world"][actual_ids].tolist(),
            reference_foot_speed_max_mps=float(np.linalg.norm(current["body_lin_vel_w"][0, feet], axis=-1).max()),
            reference_root_to_midfeet_xy_m=float(np.linalg.norm(target[:2] - foot_pos[:, :2].mean(axis=0))),
            raw_predicted_contacts=contacts[native_idx].tolist(), raw_contact_native_sample_time_s=float(raw_times[native_idx]),
            commanded_velocity_xy_mps=use["actual_raw_intent"]["velocity_xy_mps"],
            commanded_yaw_rate_rad_s=use["actual_raw_intent"]["yaw_rate_radps"])
        rows.append(row)
        previous_actual, previous_id = next_actual, plan_id

    end = float(terminal["time_s"])
    windows = {"first_2s": (0., 2.), "last_3s": (end - 3., end)}
    summaries = {}
    for name, (start, stop) in windows.items():
        selected = [r for r in rows if start - 1e-8 <= r["time_s"] < stop - 1e-8]
        selected_jumps = [j for j in jumps if start - 1e-8 <= j["time_s"] < stop - 1e-8]
        summaries[name] = summarize(selected, selected_jumps)
        summaries[name]["interval_s_half_open"] = [start, stop]
        summaries[name]["timeline_bins"] = []
        for a in np.arange(start, stop - 1e-8, .5):
            batch = [r for r in selected if a - 1e-8 <= r["time_s"] < a + .5 - 1e-8]
            summaries[name]["timeline_bins"].append(dict(interval_s=[float(a), float(min(a + .5, stop))],
                **{key: stats([r[key] for r in batch]) for key in
                   ("reference_root_tilt_deg", "reference_torso_tilt_deg", "actual_root_tilt_deg",
                    "actual_torso_tilt_deg", "future_joint_velocity_max_rad_s", "joint_tracking_rmse_rad")}))
    last_intent = [r for r in lines if r["kind"] == "intent_received"][-1]
    seq = last_intent["raw_intent"]["command_seq"]
    stopped = [r for r in rows if r["command_seq"] == seq]
    crossings = {}
    for key in ("reference_torso_tilt_deg", "reference_root_tilt_deg", "actual_root_tilt_deg", "actual_torso_tilt_deg"):
        for threshold in (15, 20, 30, 45, 60):
            crossings[f"{key}_first_over_{threshold}"] = next((r for r in rows if r[key] > threshold), None)
    snapshot = recording.parent.parent / "protomotions_live_launch_v13" / "source"
    semantics_paths = [snapshot / "motionbricks/mb_sonic" / (name + ".py")
                       for name in ("protomotions_policy", "retarget", "reference")]
    semantics_paths.append(Path(__file__).resolve().parent.parent / "motionbricks/motionlib/core/motion_reps/dual_root_global_joints.py")
    result = dict(schema="motionbricks.reference_support_forensics.v1", recording=str(recording),
        recording_sha256=log_hash, script_sha256=digest(__file__),
        new_physics_steps=0, gpu_seconds=0, optimizer_updates=0,
        scope="New target/support distinction; existing v13 binary audit and loop diagnosis not rerun",
        methods=dict(actual_alignment="Exact pre-control state from previous tick's last recorded substep; initial measured snapshot at t=0",
            windows="First [0,2) and final [terminal_time-3,terminal_time); terminal endpoint separately recorded",
            inclination="angle between each body local +Z and world +Z; quaternion wxyz; degrees",
            adoption="Old/new full saved plans sampled at identical absolute t+[0,.02,.04,.08,.16]; not inter-tick differences",
            foot_targets="Saved FK ankle-roll body origins and velocities; origin heights are not sole clearance or a COM support polygon",
            contact_channels="Last 4 unnormalized GLOBAL366 channels, left foot/toe then right foot/toe; nearest native sample per tick, interpolated only for same-time adoption comparison",
            target_velocities="Saved forward differences at 50Hz, not inferred from actual motion"),
        provenance=dict(robot=dict(path=str(robot), sha256=digest(robot)),
            source_files=[dict(path=str(p), sha256=digest(p)) for p in semantics_paths],
            numeric_sources=sources, verified_npz_count=len(sources), consumed_plan_count=len(refs),
            prior_reports=[dict(path=str(p), sha256=digest(p)) for p in
                (recording.with_name(recording.stem + "_audit_v1.json"), recording.with_name(recording.stem + "_loop_diagnosis_v1.json"))]),
        tracker_semantics=dict(future_offsets_s=OFFSETS[1:].tolist(), anchor="torso_link",
            future_inputs=["mimic_future_anchor_rot", "mimic_future_dof_pos", "mimic_future_dof_vel"],
            no_direct_input=["world_root_position", "world_root_linear_velocity", "foot_position", "foot_contact_channel", "commanded_velocity"],
            heading_alignment="One fixed yaw alignment at reset; tilt preserved",
            recorded_future_joint_inputs_match_full_reference=True,
            torso_error_comparison="Reference torso rotated by fixed initial yaw alignment; root error uses original world reference",
            reference_joint_position_order=ref["joint_names"]),
        command_after_last_key=dict(command_seq=seq, raw_intent=last_intent["raw_intent"],
            first_consumed_time_s=stopped[0]["time_s"], last_consumed_time_s=stopped[-1]["time_s"],
            controlled_duration_s=end - stopped[0]["time_s"], consumed_ticks=len(stopped),
            adopted_plans=len({r["plan_id"] for r in stopped}),
            raw_command_pairs_counts={str(k): v for k, v in Counter((tuple(r["commanded_velocity_xy_mps"]), r["commanded_yaw_rate_rad_s"]) for r in stopped).items()},
            target_root_xy_displacement_m=(np.array(stopped[-1]["reference_root_position_m"][:2]) - stopped[0]["reference_root_position_m"][:2]).tolist(),
            target_root_xy_displacement_norm_m=float(np.linalg.norm(np.array(stopped[-1]["reference_root_position_m"][:2]) - stopped[0]["reference_root_position_m"][:2])),
            target_root_horizontal_speed_mps=stats([r["reference_root_horizontal_speed_mps"] for r in stopped]),
            target_root_speed_over_0_1_mps_tick_count=sum(r["reference_root_horizontal_speed_mps"] > .1 for r in stopped),
            first_and_last_samples=[stopped[0], stopped[-1]]),
        windows=summaries, first_threshold_crossings=crossings,
        focal_samples=[next(r for r in rows if abs(r["time_s"] - t) < 1e-8)
            for t in (32.38, 32.64, 33.92, 34.32, 36.2, 36.5, 36.58, 36.64, 36.74, 37.06, 37.18)],
        terminal=dict(time_s=end, reason=terminal["terminal_reason"],
            actual_root_tilt_deg=float(tilt(np.array(terminal["qpos"])[3:7])),
            actual_torso_tilt_deg=float(tilt(np.array(terminal["body_quat_wxyz"])[actual_torso_id])),
            last_precontrol_sample=rows[-1]),
        window_samples=[r for r in rows if r["time_s"] < 2 - 1e-8 or r["time_s"] >= end - 3 - 1e-8],
        window_adoptions=[j for j in jumps if j["time_s"] < 2 - 1e-8 or j["time_s"] >= end - 3 - 1e-8])
    result["root_cause_assessment"] = dict(
        classification="Adopted non-upright and discontinuous reference, followed by additional actual tracking divergence",
        confidence="high for the measured distinction and time ordering; medium for causal attribution between target lean and adoption jumps",
        evidence=[
            dict(finding="First 2s is upright-to-mildly-leaning on both sides, not the terminal tipping pattern",
                reference_root_max_deg=summaries["first_2s"]["metrics"]["reference_root_tilt_deg"]["max"],
                reference_torso_max_deg=summaries["first_2s"]["metrics"]["reference_torso_tilt_deg"]["max"],
                actual_root_max_deg=summaries["first_2s"]["metrics"]["actual_root_tilt_deg"]["max"]),
            dict(finding="Adopted torso target leans first and is directly supplied as the future tracker anchor",
                reference_first_over_30_deg_s=crossings["reference_torso_tilt_deg_first_over_30"]["time_s"],
                actual_torso_at_that_time_deg=crossings["reference_torso_tilt_deg_first_over_30"]["actual_torso_tilt_deg"],
                actual_first_over_30_deg_s=crossings["actual_torso_tilt_deg_first_over_30"]["time_s"],
                final_window_reference_torso_median_deg=summaries["last_3s"]["metrics"]["reference_torso_tilt_deg"]["median"],
                final_window_reference_torso_max_deg=summaries["last_3s"]["metrics"]["reference_torso_tilt_deg"]["max"]),
            dict(finding="Target replacement changes joint and foot trajectories at the same absolute timestamp",
                largest_final_window_current_joint_and_foot_jump=max(
                    (j for j in jumps if j["time_s"] >= end - 3 - 1e-8), key=lambda j: j["current_foot_position_jump_max_m"]),
                largest_final_window_future_joint_jump=max(
                    (j for j in jumps if j["time_s"] >= end - 3 - 1e-8), key=lambda j: j["tracker_future_joint_position_jump_max_rad"]),
                timing="The 34.32s large jump precedes a partial posture recovery at 34.7-35.2s; final tipping grows after 36.5s"),
            dict(finding="Final actual posture diverges beyond an already leaning target",
                last_precontrol_pair={key: rows[-1][key] for key in
                    ("time_s", "reference_root_tilt_deg", "reference_torso_tilt_deg", "actual_root_tilt_deg", "actual_torso_tilt_deg", "joint_tracking_rmse_rad")},
                terminal_root_tilt_deg=result["terminal"]["actual_root_tilt_deg"]),
            dict(finding="Zero command does not yield a stationary reference",
                zero_command_interval_s=[stopped[0]["time_s"], end],
                target_root_xy_displacement_norm_m=result["command_after_last_key"]["target_root_xy_displacement_norm_m"],
                final_window_reference_horizontal_speed_max_mps=summaries["last_3s"]["metrics"]["reference_root_horizontal_speed_mps"]["max"]),
        ],
        attribution="Recorded target degradation precedes measured tipping. Relative causal contributions of leaning torso targets, adoption discontinuities, and tracker response require an intervention; this closed record separates their timing but not their individual causal weights.",
        contact_conclusion="Predicted contact channels and FK foot targets vary; there is no G1 contact-target input to this tracker. Foot motion can affect it through the corresponding future joint positions/velocities.")
    if digest(recording) != log_hash:
        raise ValueError("Original JSONL changed during analysis")
    result["recording_unchanged"] = True
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    compact = dict(command_after_last_key=result["command_after_last_key"],
        windows={name: dict(count=s["count"], metrics=s["metrics"],
            foot_origin_height_m=s["foot_origin_height_m"],
            adoption_count=s["adoption_count"],
            discontinuity_maxima={key: value["distribution"]["max"] for key, value in s["same_absolute_time_adoption_discontinuities"].items()},
            largest_discontinuities={key: value["largest"][0] for key, value in s["same_absolute_time_adoption_discontinuities"].items()
                if key in ("current_joint_position_jump_max_rad", "current_foot_position_jump_max_m", "tracker_future_joint_position_jump_max_rad", "tracker_future_torso_orientation_jump_max_deg")},
            timeline_bins=s["timeline_bins"]) for name, s in summaries.items()},
        threshold_times={key: None if value is None else value["time_s"] for key, value in crossings.items()},
        threshold_postures={key: None if value is None else {k: value[k] for k in
            ("time_s", "reference_root_tilt_deg", "reference_torso_tilt_deg", "actual_root_tilt_deg", "actual_torso_tilt_deg", "joint_tracking_rmse_rad")}
            for key, value in crossings.items() if key.endswith(("_30", "_45", "_60"))},
        root_cause_assessment=result["root_cause_assessment"],
        terminal=result["terminal"], verified_npz_count=len(sources))
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
