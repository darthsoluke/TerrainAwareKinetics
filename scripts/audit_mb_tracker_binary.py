"""Read-only audit of completed columnar physics tick artifacts; never steps MuJoCo."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from motionbricks.mb_sonic.columnar_physics import CONTACT_COLUMN_SHAPES, DENSE_FIELDS, SCHEMA, WARNING_NAMES
from motionbricks.mb_sonic.reference import G1_JOINT_NAMES, file_sha256


def decode(value, archive):
    if isinstance(value, dict):
        if "array" in value:
            array = archive[value["array"]]
            if array.dtype.str != value["dtype"] or list(array.shape) != value["shape"]:
                raise ValueError("Numeric array descriptor differs from its payload")
            return array
        return {key: decode(item, archive) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item, archive) for item in value]
    return value


def read_index(path):
    metadata, ticks, artifacts, terminals = [], {}, {}, []
    counts = Counter()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["kind"]
            counts[kind] += 1
            if kind == "metadata":
                metadata.append(row)
            elif kind == "policy_tick_binary":
                key = row["numeric_record_id"]
                if key in ticks:
                    raise ValueError(f"Duplicate policy tick receipt: {key}")
                ticks[key] = row
            elif kind == "numeric_artifact_written":
                key = row["record_id"]
                if key in artifacts:
                    raise ValueError(f"Duplicate numeric completion receipt: {key}")
                artifacts[key] = row
            elif kind in ("terminal_runtime_failure", "runtime_failure", "episode_completed"):
                terminals.append(row)
    if len(metadata) != 1 or not ticks:
        raise ValueError("Expected one metadata header and at least one binary policy tick")
    return metadata[0], ticks, artifacts, dict(counts), terminals


def robot_contract(metadata):
    source = metadata["robot"]
    path = Path(source["path"]).resolve(strict=True)
    if file_sha256(path) != source["sha256"]:
        raise ValueError("Recorded robot XML SHA256 differs from the current file")
    model = mujoco.MjModel.from_xml_path(str(path))
    names = tuple(metadata["actuator_control"]["joint_names"])
    if names != G1_JOINT_NAMES:
        raise ValueError("The named 29-joint control order differs from the audited G1 order")
    ids = np.array([model.joint(name).id for name in names])
    if (np.any(model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE)
            or not np.all(model.jnt_limited[ids])):
        raise ValueError("All 29 body joints must have original XML hinge limits")
    motors = []
    for joint in ids:
        matches = np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                & (model.actuator_trnid[:, 0] == joint))
        if len(matches) != 1:
            raise ValueError("Expected one original motor for each named body joint")
        motors.append(int(matches[0]))
    motors = np.asarray(motors)
    if (not np.all(model.actuator_ctrllimited[motors])
            or not np.all(model.actuator_gear[motors] == [1, 0, 0, 0, 0, 0])):
        raise ValueError("Expected original unit-gear, torque-limited G1 motors")
    bounds = model.actuator_ctrlrange[motors].copy()
    for limited, ranges in ((model.jnt_actfrclimited[ids], model.jnt_actfrcrange[ids]),
                            (model.actuator_forcelimited[motors], model.actuator_forcerange[motors])):
        mask = limited.astype(bool)
        bounds[mask, 0] = np.maximum(bounds[mask, 0], ranges[mask, 0])
        bounds[mask, 1] = np.minimum(bounds[mask, 1], ranges[mask, 1])
    if not np.all(bounds[:, 0] < bounds[:, 1]):
        raise ValueError("Original XML motor force intersections must be nonempty")
    declared = metadata["actuator_control"]
    declared_bounds = np.column_stack((declared["motor_force_lower"], declared["motor_force_upper"]))
    return dict(model=model, joint_ids=ids, motor_ids=motors, limits=model.jnt_range[ids].copy(),
                force_bounds=bounds, declared_bounds_match=bool(np.array_equal(bounds, declared_bounds)),
                fall_body_ids=np.array([model.body(name).id for name in ("pelvis", "torso_link")]),
                path=str(path), sha256=source["sha256"])


class PhysicsAudit:
    def __init__(self, contract, dt):
        self.contract, self.dt = contract, float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Recorded physics timestep must be finite and positive")
        self.frames = self.contacts = self.fall_contacts = self.fall_frames = 0
        self.nonfinite = Counter()
        self.ground_unknown = self.ground_height_missing = 0
        self.versions, self.times = [], []
        self.minimum_margin = np.full(29, np.inf)
        self.joint_excess = self.motor_excess = 0.
        self.joint_count = self.motor_count = 0
        self.first_joint = self.first_motor = self.first_fall = None
        self.minimum_root_z, self.maximum_penetration = np.inf, 0.
        self.penetration_location = None
        self.aux = {name: dict(nonzero_count=0, maximum_abs=0.) for name in
                    ("root_qfrc_actuator", "qfrc_applied", "xfrc_applied")}
        self.warnings = {name: dict(maximum_number=0, nonzero_frames=0) for name in WARNING_NAMES}
        self.terminals = []
        self.motor_peak = np.zeros(29)

    def add(self, batch, receipt, record_id):
        model = self.contract["model"]
        if batch["schema"] != SCHEMA:
            raise ValueError("Unsupported columnar physics schema")
        time, version = batch["time_s"], batch["state_version"]
        n = len(time)
        if n == 0 or time.shape != (n,) or version.shape != (n,) or version.dtype.kind not in "iu":
            raise ValueError("Invalid columnar clock arrays")
        shapes = dict(qpos=(model.nq,), qvel=(model.nv,), joint_positions=(29,), joint_velocities=(29,),
            body_positions_world=(model.nbody, 3), body_quat_wxyz=(model.nbody, 4),
            actuator_controls=(model.nu,), actuator_forces=(model.nu,), generalized_actuator_forces=(model.nv,),
            root_qfrc_actuator=(6,), qfrc_applied=(model.nv,), xfrc_applied=(model.nbody, 6))
        for name in DENSE_FIELDS:
            if batch[name].shape != (n, *shapes[name]):
                raise ValueError(f"Invalid dense column shape: {name}")
            self.nonfinite[name] += int((~np.isfinite(batch[name])).sum())
        for name in ("root_ground_known", "root_ground_height_m", "root_ground_height_none", "captured_at_monotonic_ns"):
            if batch[name].shape != (n,):
                raise ValueError(f"Invalid state column shape: {name}")
        if (batch["root_ground_known"].dtype.kind != "b" or batch["root_ground_height_none"].dtype.kind != "b"
                or len(batch["terminal_reason"]) != n):
            raise ValueError("Invalid ground validity or terminal columns")
        if (int(version[-1]) != receipt["state_version"]
                or not np.isclose(time[-1], receipt["actual_time_s"], atol=1e-9, rtol=0)
                or int(batch["captured_at_monotonic_ns"][-1]) != receipt["captured_at_monotonic_ns"]):
            raise ValueError("Tick receipt differs from its final actual state")
        for name, expected in (("joint_positions", batch["qpos"][:, model.jnt_qposadr[self.contract["joint_ids"]]]),
                               ("joint_velocities", batch["qvel"][:, model.jnt_dofadr[self.contract["joint_ids"]]]),
                               ("root_qfrc_actuator", batch["generalized_actuator_forces"][:, :6])):
            if not np.array_equal(batch[name], expected, equal_nan=True):
                raise ValueError(f"Recorded derived actual column disagrees: {name}")
        self.nonfinite["time_s"] += int((~np.isfinite(time)).sum())
        known = batch["root_ground_known"]
        self.ground_unknown += int((~known).sum())
        self.ground_height_missing += int(batch["root_ground_height_none"].sum())
        self.nonfinite["known_ground_height"] += int((known & ~np.isfinite(batch["root_ground_height_m"])).sum())
        self.versions.append(version.copy())
        self.times.append(time.copy())
        location = lambda i: dict(record_id=record_id, state_version=int(version[i]), time_s=float(time[i]))
        q = batch["joint_positions"]
        bounds = self.contract["limits"]
        margin = np.minimum(q-bounds[:, 0], bounds[:, 1]-q)
        valid = np.isfinite(margin)
        self.minimum_margin = np.minimum(self.minimum_margin, np.min(np.where(valid, margin, np.inf), axis=0))
        bad = valid & (margin < 0)
        self.joint_count += int(bad.sum())
        if bad.any():
            self.joint_excess = max(self.joint_excess, float(-margin[bad].min()))
            i, j = np.argwhere(bad)[0]
            if self.first_joint is None:
                self.first_joint = dict(**location(i), joint_name=G1_JOINT_NAMES[j], position_rad=float(q[i, j]),
                    lower_rad=float(bounds[j, 0]), upper_rad=float(bounds[j, 1]), excess_rad=float(-margin[i, j]))
        z = batch["qpos"][:, 2]
        if np.isfinite(z).any():
            self.minimum_root_z = min(self.minimum_root_z, float(z[np.isfinite(z)].min()))
        for name, metrics in self.aux.items():
            a = batch[name]
            metrics["nonzero_count"] += int(np.count_nonzero(a))
            if np.isfinite(a).any():
                metrics["maximum_abs"] = max(metrics["maximum_abs"], float(np.abs(a[np.isfinite(a)]).max()))
        values, names = batch["solver_warning_values"], batch["solver_warning_names"]
        if values.shape != (n, 7, 2) or values.dtype.kind not in "iu" or set(names) != set(WARNING_NAMES) or len(names) != 7:
            raise ValueError("Invalid solver warning number/lastinfo columns")
        for j, name in enumerate(names):
            numbers = values[:, j, 0]
            self.warnings[name]["maximum_number"] = max(self.warnings[name]["maximum_number"], int(numbers.max()))
            self.warnings[name]["nonzero_frames"] += int(np.count_nonzero(numbers))
        force = batch["actuator_forces"][:, self.contract["motor_ids"]]
        fb = self.contract["force_bounds"]
        excess = np.maximum(fb[:, 0]-force, force-fb[:, 1])
        bad = np.isfinite(excess) & (excess > 0)
        self.motor_count += int(bad.sum())
        self.motor_peak = np.maximum(self.motor_peak, np.max(np.where(np.isfinite(force), abs(force), 0), axis=0))
        if bad.any():
            self.motor_excess = max(self.motor_excess, float(excess[bad].max()))
            i, j = np.argwhere(bad)[0]
            if self.first_motor is None:
                self.first_motor = dict(**location(i), joint_name=G1_JOINT_NAMES[j], force_nm=float(force[i, j]),
                    lower_nm=float(fb[j, 0]), upper_nm=float(fb[j, 1]), excess_nm=float(excess[i, j]))
        contacts = batch["contacts"]
        if set(contacts) != {"offsets", *CONTACT_COLUMN_SHAPES}:
            raise ValueError("Contact columns must all be retained, without unknown columns")
        offsets = contacts["offsets"]
        if (offsets.dtype.kind not in "iu" or offsets.shape != (n+1,) or offsets[0] != 0
                or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("Invalid contact-to-frame offsets")
        count = int(offsets[-1])
        for name, shape in CONTACT_COLUMN_SHAPES.items():
            a = contacts[name]
            if a.shape != (count, *shape):
                raise ValueError(f"Invalid contact column shape: {name}")
            self.nonfinite["contact."+name] += int((~np.isfinite(a)).sum())
        for name in ("geom_ids", "body_ids"):
            if contacts[name].dtype.kind not in "iu" or np.any(contacts[name] < 0):
                raise ValueError("Contact IDs must be nonnegative integers")
        if np.any(contacts["body_ids"] >= model.nbody):
            raise ValueError("Contact body IDs do not belong to the declared robot/world")
        frame = np.repeat(np.arange(n), np.diff(offsets).astype(np.int64))
        ids = contacts["body_ids"]
        world = (ids[:, 0] == 0) ^ (ids[:, 1] == 0)
        other = np.where(ids[:, 0] == 0, ids[:, 1], ids[:, 0])
        fall = world & np.isin(other, self.contract["fall_body_ids"]) & (contacts["wrench_contact"][:, 0] > 1.)
        self.fall_contacts += int(fall.sum())
        self.fall_frames += len(np.unique(frame[fall]))
        if fall.any() and self.first_fall is None:
            row = int(np.flatnonzero(fall)[0])
            self.first_fall = dict(**location(frame[row]), contact_row=row, body_ids=ids[row].tolist(),
                geom_ids=contacts["geom_ids"][row].tolist(), normal_force_n=float(contacts["wrench_contact"][row, 0]))
        distance = contacts["distance"]
        rows = np.flatnonzero(world & np.isfinite(distance))
        if len(rows):
            row = int(rows[np.argmin(distance[rows])])
            depth = max(0., float(-distance[row]))
            if depth > self.maximum_penetration:
                self.maximum_penetration = depth
                self.penetration_location = dict(**location(frame[row]), contact_row=row,
                    geom_ids=contacts["geom_ids"][row].tolist(), body_ids=ids[row].tolist(), penetration_m=depth)
        self.terminals.extend(dict(**location(i), reason=reason) for i, reason in enumerate(batch["terminal_reason"]) if reason)
        self.frames += n
        self.contacts += count

    def result(self):
        version = np.concatenate(self.versions) if self.versions else np.empty(0, dtype=np.int64)
        time = np.concatenate(self.times) if self.times else np.empty(0)
        finite = np.isfinite(time)
        continuity = dict(unique_state_versions=len(np.unique(version)), unique_times=len(np.unique(time)),
            repeated_state_versions=len(version)-len(np.unique(version)), repeated_times=len(time)-len(np.unique(time)),
            state_version_gap_count=int(np.count_nonzero(np.diff(version) != 1)),
            time_step_error_count=int(np.count_nonzero(~np.isclose(np.diff(time), self.dt, atol=1e-9, rtol=0))),
            state_time_grid_error_count=int(np.count_nonzero(~np.isclose(time, version*self.dt, atol=1e-8, rtol=0))),
            timestamp_comparison_atol_s=1e-9, state_time_grid_atol_s=1e-8,
            first_state_version=int(version[0]) if len(version) else None,
            last_state_version=int(version[-1]) if len(version) else None,
            first_time_s=float(time[0]) if len(time) and finite[0] else None,
            last_time_s=float(time[-1]) if len(time) and finite[-1] else None)
        checks = dict(nonempty=self.frames > 0, all_numeric_columns_finite=not any(self.nonfinite.values()),
            state_time_continuous=not any(continuity[k] for k in ("repeated_state_versions", "repeated_times",
                "state_version_gap_count", "time_step_error_count", "state_time_grid_error_count")),
            joints_within_XML=self.joint_count == 0, all_auxiliary_forces_zero=all(x["nonzero_count"] == 0 for x in self.aux.values()),
            ground_all_known=self.ground_unknown == 0 and self.ground_height_missing == 0,
            solver_warning_numbers_zero=all(x["nonzero_frames"] == 0 for x in self.warnings.values()),
            motor_forces_within_XML=self.motor_count == 0, motor_metadata_bounds_match_XML=self.contract["declared_bounds_match"],
            no_pelvis_or_torso_world_contact_over_1N=self.fall_contacts == 0, no_terminal_states=not self.terminals)
        return dict(controlled_substeps=self.frames, continuity=continuity, checks=checks,
            nonfinite_element_counts=dict(self.nonfinite),
            joints=dict(violation_sample_joint_count=self.joint_count, maximum_excess_rad=self.joint_excess,
                first_violation=self.first_joint, minimum_margin_by_joint_rad={n:float(v) if np.isfinite(v) else None
                    for n, v in zip(G1_JOINT_NAMES, self.minimum_margin)}, bound_tolerance_rad=0.),
            root_minimum_z_m=float(self.minimum_root_z) if np.isfinite(self.minimum_root_z) else None,
            auxiliary_forces=self.aux, root_ground_unknown_frames=self.ground_unknown,
            root_ground_height_missing_frames=self.ground_height_missing, solver_warnings=self.warnings,
            contacts=dict(total_contact_rows=self.contacts, retained_columns=["offsets", *CONTACT_COLUMN_SHAPES],
                frame_assignment="offsets[i]:offsets[i+1]; every row checked", fall_contact_rows=self.fall_contacts,
                fall_frames=self.fall_frames, first_fall=self.first_fall,
                maximum_scene_penetration_m=self.maximum_penetration, maximum_penetration_location=self.penetration_location),
            motors=dict(violation_sample_motor_count=self.motor_count, maximum_excess_nm=self.motor_excess,
                first_violation=self.first_motor, bound_tolerance_nm=0.,
                bounds_by_joint_nm={n:b.tolist() for n,b in zip(G1_JOINT_NAMES, self.contract["force_bounds"])},
                maximum_abs_force_by_joint_nm=dict(zip(G1_JOINT_NAMES, self.motor_peak.tolist()))),
            terminal_states=self.terminals)


def audit_recording(path):
    before = file_sha256(path)
    metadata, ticks, artifacts, counts, terminals = read_index(path)
    contract = robot_contract(metadata)
    audit = PhysicsAudit(contract, metadata["physics_dt_s"])
    completed = {key: value for key, value in artifacts.items() if key.startswith("tick:")}
    missing, orphan = sorted(set(ticks)-set(completed)), sorted(set(completed)-set(ticks))
    verified = []
    for key in sorted(completed, key=lambda name: int(name.split(":", 1)[1])):
        item = completed[key]
        entry = dict(record_id=key, path=item["path"], expected_sha256=item["sha256"], sha256_verified=False)
        try:
            artifact = Path(item["path"]).resolve(strict=True)
            entry["actual_sha256"] = file_sha256(artifact)
            entry["sha256_verified"] = entry["actual_sha256"] == item["sha256"]
            if not entry["sha256_verified"]:
                raise ValueError("Completed tick NPZ SHA256 mismatch")
            if artifact.stat().st_size != item["bytes"]:
                raise ValueError("Completed tick NPZ byte count mismatch")
            with np.load(artifact, allow_pickle=False) as archive:
                saved = json.loads(str(archive["metadata_json"]))
                if saved["schema"] != "motionbricks.mb_sonic.numeric_record.v1" or saved["record_id"] != key:
                    raise ValueError("Numeric record schema/identity mismatch")
                record = saved["record"]
                if record["kind"] != "full_policy_tick_columnar":
                    raise ValueError(f"Unsupported tick kind: {record['kind']}; columnar physics required")
                if key not in ticks:
                    raise ValueError("Completed tick artifact has no policy_tick_binary receipt")
                tick = record["tick"]
                if tick["scene_id"] != ticks[key]["scene_id"] or tick["command_id"] != ticks[key]["command_id"]:
                    raise ValueError("Tick scene/command identity mismatch")
                audit.add(decode(record["physics"], archive), ticks[key], key)
                entry["audited"] = True
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            entry["error"] = dict(type=type(error).__name__, message=str(error))
        verified.append(entry)
    after = file_sha256(path)
    metrics = audit.result()
    complete = not missing and not orphan and before == after and all(x.get("audited", False) for x in verified)
    return dict(schema="motionbricks.mb_tracker_binary_audit.v1", recording=str(path.resolve()),
        recording_sha256_before=before, recording_sha256_after=after, recording_unchanged=before == after,
        audit_script_sha256=file_sha256(Path(__file__)), new_physics_steps=0, gpu_seconds=0, optimizer_updates=0,
        scope="completed binary controlled-tick substeps; no resimulation, warmup reconstruction or online quality certification",
        robot=dict(path=contract["path"], sha256=contract["sha256"], loaded_only_for_IDs_and_original_bounds=True),
        tracker_backend=metadata["tracker_backend"], physics_dt_s=metadata["physics_dt_s"],
        JSONL_kind_counts=counts, binary_tick_receipts=len(ticks), completed_tick_NPZ_count=len(completed),
        SHA_verified_tick_NPZ_count=sum(x["sha256_verified"] for x in verified),
        audited_tick_NPZ_count=sum(x.get("audited", False) for x in verified),
        missing_completed_tick_ids=missing, orphan_completed_tick_ids=orphan, artifact_checks=verified,
        audit_completed=complete, recorded_physics_checks_pass=complete and all(metrics["checks"].values()),
        episode_terminal_events=terminals, **metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Audit output already exists; overwriting evidence is prohibited")
    report = audit_recording(args.recording.resolve(strict=True))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({key:report[key] for key in ("audit_completed", "recorded_physics_checks_pass",
        "completed_tick_NPZ_count", "SHA_verified_tick_NPZ_count", "controlled_substeps", "root_minimum_z_m")}, indent=2))
    return 0 if report["recorded_physics_checks_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
