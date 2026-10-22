"""Bounded zero-training check against the pinned ProtoMotions deployment helpers."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from motionbricks.mb_sonic.protomotions_policy import FrozenProtoMotionsPolicy, PROTOMOTIONS_ONNX_SHA256
from motionbricks.mb_sonic.reference import G1Reference, file_sha256
from motionbricks.mb_sonic.physics_numerics import configure_physics_numerics
from motionbricks.mb_sonic.sonic_policy import ObservationHistory, Proprioception


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--motion-index", type=int, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numerics-config", type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 20:
        parser.error("Native validation is bounded to 20 simulation seconds")
    repo = args.repository.resolve(strict=True)
    sys.path.insert(0, str(repo))
    from deployment.motion_utils import MotionPlayer
    from deployment.test_tracker_mujoco import load_mujoco_model, read_robot_state, set_initial_pose
    from deployment.tracker_inputs import align_future_motion_refs, build_tracker_onnx_inputs, init_tracker_alignment

    args.output.mkdir(parents=True, exist_ok=False)
    model_root = repo/"data/pretrained_models/motion_tracker/g1-bones-deploy"
    onnx = model_root/"compiled_models/unified_pipeline.onnx"
    yaml_path = model_root/"compiled_models/unified_pipeline.yaml"
    policy = FrozenProtoMotionsPolicy(onnx, yaml_path, onnx_sha256=PROTOMOTIONS_ONNX_SHA256,
                                     metadata_sha256="57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139")
    player = MotionPlayer(repo/"data/motion_for_trackers/g1_bones_seed_mini.pt", motion_index=args.motion_index)
    ticks = round(args.seconds/.02)
    if ticks+8 >= player.total_frames:
        raise ValueError("The requested duration must leave eight real future reference samples")
    body_names = tuple(policy.metadata["robot"]["body_names"])
    frames = [player.get_state_at_frame(index) for index in range(player.total_frames)]
    pos, rot, joints, velocities = (np.stack([frame[name] for frame in frames])
                                   for name in ("body_pos", "body_rot", "dof_pos", "dof_vel"))
    quaternion_norm_error = float(np.abs(np.linalg.norm(rot, axis=-1)-1).max())
    rot = rot/np.linalg.norm(rot, axis=-1, keepdims=True)
    qpos = np.concatenate((pos[:, 0], rot[:, 0, [3, 0, 1, 2]], joints), axis=1)
    qvel = np.c_[np.zeros((len(frames), 6)), velocities]
    reference = G1Reference(np.arange(len(frames))*.02, qpos, qvel, body_names, pos,
        rot[..., [3, 0, 1, 2]], np.stack([frame["body_vel"] for frame in frames]),
        np.stack([frame["body_ang_vel"] for frame in frames]), "native_floor", "official_mini",
        (len(frames)-1)*.02)
    xml = repo/"protomotions/data/assets"/policy.metadata["robot"]["mjcf_path"]
    motors = ET.parse(xml).getroot().findall("./actuator/motor")
    if tuple(row.attrib["joint"] for row in motors) != reference.joint_names:
        raise ValueError("Native XML actuator names/order differ")
    original_force_ranges = np.stack([np.fromstring(row.attrib["ctrlrange"], sep=" ") for row in motors])
    model, data = load_mujoco_model(str(xml), policy.kp.tolist(), policy.kd.tolist(), policy.physics_dt)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    # The upstream runner leaves effort limits unset; retain the XML's real motor bounds.
    model.actuator_forcelimited[:] = 1
    model.actuator_forcerange[:] = original_force_ranges
    numerics = configure_physics_numerics(model, .001,
        json.loads(args.numerics_config.read_text(encoding="utf-8")) if args.numerics_config else {})
    set_initial_pose(model, data, player)
    history = ObservationHistory()
    anchor = model.body(policy.anchor_body_name).id
    ref_anchor = body_names.index(policy.anchor_body_name)
    policy.reset_heading(data.xquat[anchor], reference.body_quat_w[0, ref_anchor])
    aligned = init_tracker_alignment(data.xpos[anchor].astype(np.float32),
        data.xquat[anchor, [1, 2, 3, 0]].astype(np.float32), pos[0, ref_anchor], rot[0, ref_anchor])
    q, v, targets, forces, control_times, step_times = [], [], [], [], [], []
    first_limit, first_ground_contact, input_error = None, None, 0.
    lower = model.jnt_range[[model.joint(name).id for name in reference.joint_names], 0]
    upper = model.jnt_range[[model.joint(name).id for name in reference.joint_names], 1]
    started = perf_counter()
    for tick in range(ticks):
        if perf_counter()-started > 90:
            raise TimeoutError("Native CPU validation exceeded its 90 second wall bound")
        history.append(Proprioception(data.time, data.qpos[7:], data.qvel[6:], data.qpos[3:7], data.qvel[3:6], np.zeros(29)))
        window = policy.reference_window(reference, data.time)
        state = read_robot_state(data, ref_anchor, 0)
        future = player.get_future_references(tick, [1, 2, 4, 8])
        future["body_rot"] = rot[tick+np.array([1, 2, 4, 8])]
        official_inputs = build_tracker_onnx_inputs(
            onnx_input_names=list(policy.metadata["_runtime"]["onnx_in_names"]),
            onnx_name_to_key=policy.metadata["_runtime"]["onnx_name_to_in_key"],
            dof_pos=state["dof_pos"], dof_vel=state["dof_vel"], anchor_rot=state["body_rot"][ref_anchor],
            root_local_ang_vel=state["root_local_ang_vel"],
            future_refs=align_future_motion_refs(future, aligned),
            anchor_body_index=ref_anchor, num_dofs=29, prev_actions=policy.previous_target)
        before = perf_counter()
        command = policy.infer(window, history, anchor_quat_wxyz=data.xquat[anchor])
        control_times.append((perf_counter()-before)*1000)
        input_error = max(input_error, *(float(np.abs(command.inputs[name]-value).max())
                                        for name, value in official_inputs.items()))
        if input_error > 2e-6:
            raise ValueError(f"Adapter inputs differ from official helpers: {input_error}")
        data.ctrl[:] = command.joint_target
        targets.append(command.joint_target.copy())
        before = perf_counter()
        for _ in range(20):
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            q.append(data.qpos.copy())
            v.append(data.qvel.copy())
            forces.append(data.actuator_force.copy())
            margin = np.minimum(data.qpos[7:]-lower, upper-data.qpos[7:])
            if margin.min() < -1e-6 and first_limit is None:
                index = int(margin.argmin())
                first_limit = dict(time_s=float(data.time), joint=reference.joint_names[index], margin_rad=float(margin[index]))
            if np.any(data.qfrc_applied) or np.any(data.xfrc_applied) or np.any(data.qfrc_actuator[:6]):
                raise ValueError("Unexpected external or Root actuation")
            for ci in range(data.ncon):
                contact = data.contact[ci]
                bodies = [model.geom_bodyid[g] for g in contact.geom]
                if 0 in bodies and (model.body("pelvis").id in bodies or anchor in bodies):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, wrench)
                    if wrench[0] > 1 and first_ground_contact is None:
                        first_ground_contact = float(data.time)
        step_times.append((perf_counter()-before)*1000)
        if first_ground_contact is not None:
            break
    wall = perf_counter()-started
    def stats(values):
        return dict(p50_ms=float(np.percentile(values, 50)), p99_ms=float(np.percentile(values, 99)), maximum_ms=float(np.max(values)))
    report = dict(schema="motionbricks.protomotions.native_contract.v1", optimizer_updates=0,
        physics_numerics=numerics,
        motion_index=args.motion_index, actual_seconds=float(data.time), requested_seconds=args.seconds,
        first_actual_limit=first_limit, first_pelvis_or_torso_floor_contact_s=first_ground_contact,
        minimum_root_z_m=float(np.asarray(q)[:, 2].min()), input_max_difference_from_official=input_error,
        source_quaternion_max_norm_error=quaternion_norm_error, reference_quaternions_unit_normalized=True,
        physical_steps=len(q), controller_ticks=len(targets), physics_dt_s=.001, control_dt_s=.02,
        wall_time_s=wall, realtime_factor=float(data.time)/wall,
        adapter_and_onnx_timing=stats(control_times), physics_and_audit_timing=stats(step_times),
        maximum_abs_motor_force_nm=float(np.abs(forces).max()), forces_bounded_by_original_xml=True,
        maximum_actual_joint_overtravel_rad=float(max(0., -np.minimum(np.asarray(q)[:, 7:]-lower, upper-np.asarray(q)[:, 7:]).min())),
        onnx_sha256=file_sha256(onnx), metadata_sha256=file_sha256(yaml_path),
        robot_xml_sha256=file_sha256(xml), source_sha256=file_sha256(__file__),
        completed=len(targets)==ticks and first_ground_contact is None)
    np.savez(args.output/"states.npz", qpos=q, qvel=v, targets=targets, actuator_forces=forces)
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
