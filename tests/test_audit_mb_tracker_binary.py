import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_mb_tracker_binary as audit


@pytest.fixture
def contract():
    return dict(model=SimpleNamespace(nq=36, nv=35, nu=29, nbody=3,
                    jnt_qposadr=np.arange(7, 36), jnt_dofadr=np.arange(6, 35)),
        joint_ids=np.arange(29), motor_ids=np.arange(29), limits=np.tile([-1., 1.], (29, 1)),
        force_bounds=np.tile([-2., 2.], (29, 1)), declared_bounds_match=True,
        fall_body_ids=np.array([1, 2]), path="unit_fixture.xml", sha256="fixture")


def batch(start=1, count=4):
    model_shapes = dict(qpos=(36,), qvel=(35,), joint_positions=(29,), joint_velocities=(29,),
        body_positions_world=(3, 3), body_quat_wxyz=(3, 4), actuator_controls=(29,),
        actuator_forces=(29,), generalized_actuator_forces=(35,), root_qfrc_actuator=(6,),
        qfrc_applied=(35,), xfrc_applied=(3, 6))
    value = {name:np.zeros((count, *shape)) for name, shape in model_shapes.items()}
    value["qpos"][:, 2] = .8
    value.update(schema=audit.SCHEMA, time_s=np.arange(start, start+count)*.001,
        state_version=np.arange(start, start+count), captured_at_monotonic_ns=np.arange(start, start+count),
        root_ground_known=np.ones(count, dtype=bool), root_ground_height_none=np.zeros(count, dtype=bool),
        root_ground_height_m=np.zeros(count), solver_warning_names=list(audit.WARNING_NAMES),
        solver_warning_values=np.zeros((count, 7, 2), dtype=np.int64), terminal_reason=[None]*count,
        contacts=dict(offsets=np.zeros(count+1, dtype=np.int64),
            **{name:np.zeros((0, *shape), dtype=np.int64 if name.endswith("ids") else float)
               for name, shape in audit.CONTACT_COLUMN_SHAPES.items()}))
    return value


def receipt(value):
    return dict(state_version=int(value["state_version"][-1]), actual_time_s=float(value["time_s"][-1]),
                captured_at_monotonic_ns=int(value["captured_at_monotonic_ns"][-1]))


def test_descriptor_preserves_bits_and_rejects_mismatched_dtype_shape():
    source = np.array([1., np.inf, np.nan], dtype=np.float64)
    descriptor = dict(array="x", shape=[3], dtype=source.dtype.str)
    actual = audit.decode(descriptor, dict(x=source))
    assert np.array_equal(actual.view(np.uint64), source.view(np.uint64))
    for changed in (dict(descriptor, shape=[2]), dict(descriptor, dtype="<f4")):
        with pytest.raises(ValueError, match="descriptor"):
            audit.decode(changed, dict(x=source))


def test_all_substeps_strict_joint_motor_bounds_and_first_violation(contract):
    value = batch()
    value["joint_positions"][1, 3] = 1.+1e-12
    value["qpos"][1, 10] = value["joint_positions"][1, 3]
    value["actuator_forces"][2, 4] = -2.2
    check = audit.PhysicsAudit(contract, .001)
    check.add(value, receipt(value), "tick:4")
    result = check.result()
    assert result["joints"]["violation_sample_joint_count"] == 1
    assert result["joints"]["first_violation"]["state_version"] == 2
    assert result["joints"]["first_violation"]["joint_name"] == "left_knee_joint"
    assert result["motors"]["first_violation"]["state_version"] == 3
    assert result["motors"]["maximum_excess_nm"] == pytest.approx(.2)
    assert not result["checks"]["joints_within_XML"]
    assert result["root_minimum_z_m"] == .8


def test_contact_offsets_assign_all_columns_and_both_body_orders(contract):
    value = batch()
    c = value["contacts"]
    c.update(offsets=np.array([0, 0, 2, 3, 4]),
        geom_ids=np.array([[0, 2], [1, 3], [4, 0], [0, 5]]),
        body_ids=np.array([[0, 1], [1, 2], [2, 0], [0, 1]]), distance=np.array([-.02, -.5, -.01, .01]),
        position_world=np.zeros((4, 3)), frame_contact_from_world=np.tile(np.eye(3), (4, 1, 1)),
        wrench_contact=np.array([[2., 0, 0, 0, 0, 0], [99., 0, 0, 0, 0, 0],
                                 [3., 0, 0, 0, 0, 0], [1., 100., 0, 0, 0, 0]]))
    check = audit.PhysicsAudit(contract, .001)
    check.add(value, receipt(value), "tick:4")
    result = check.result()["contacts"]
    assert result["total_contact_rows"] == 4
    assert result["fall_contact_rows"] == result["fall_frames"] == 2
    assert result["first_fall"]["state_version"] == 2
    assert result["maximum_scene_penetration_m"] == .02
    assert set(result["retained_columns"]) == set(c)


def test_cross_tick_continuity_and_duplicates(contract):
    check = audit.PhysicsAudit(contract, .001)
    for value in (batch(), batch(4)):
        check.add(value, receipt(value), "unit")
    result = check.result()
    assert result["continuity"]["repeated_state_versions"] == 1
    assert result["continuity"]["repeated_times"] == 1
    assert result["continuity"]["state_version_gap_count"] == 1
    assert not result["checks"]["state_time_continuous"]


def test_forces_ground_warning_numbers_and_nonfinite_are_not_hidden(contract):
    value = batch()
    value["generalized_actuator_forces"][0, 0] = value["root_qfrc_actuator"][0, 0] = 1.
    value["xfrc_applied"][1, 2, 4] = np.nan
    value["root_ground_known"][2] = False
    value["solver_warning_values"][3, 5] = [2, 99]
    check = audit.PhysicsAudit(contract, .001)
    check.add(value, receipt(value), "tick:4")
    result = check.result()
    assert result["nonfinite_element_counts"]["xfrc_applied"] == 1
    assert not result["checks"]["all_auxiliary_forces_zero"]
    assert not result["checks"]["ground_all_known"]
    assert result["solver_warnings"]["mjWARN_BADQACC"] == dict(maximum_number=2, nonzero_frames=1)


def test_corrupt_contact_offsets_rejected(contract):
    value = batch()
    value["contacts"]["offsets"] = np.array([0, 0, 1, 0, 0])
    with pytest.raises(ValueError, match="offsets"):
        audit.PhysicsAudit(contract, .001).add(value, receipt(value), "unit")


def write_recording(tmp_path, kind="full_policy_tick_columnar"):
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            name = f"array_{len(arrays)}"
            arrays[name] = value
            return dict(array=name, dtype=value.dtype.str, shape=list(value.shape))
        if isinstance(value, dict):
            return {key:encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value

    value = batch()
    record = dict(schema="motionbricks.mb_sonic.numeric_record.v1", record_id="tick:4",
                  record=dict(kind=kind, tick=dict(scene_id="000", command_id=1), physics=encode(value)))
    path = tmp_path/"tick.npz"
    np.savez(path, metadata_json=np.asarray(json.dumps(record)), **arrays)
    lines = [dict(kind="metadata", physics_dt_s=.001, tracker_backend="unit_fixture"),
        dict(kind="policy_tick_binary", numeric_record_id="tick:4", scene_id="000", command_id=1, **receipt(value)),
        dict(kind="numeric_artifact_written", record_id="tick:4", path=str(path),
             sha256=audit.file_sha256(path), bytes=path.stat().st_size)]
    recording = tmp_path/"recording.jsonl"
    recording.write_text("\n".join(json.dumps(row) for row in lines)+"\n", encoding="utf-8")
    return recording, path


def test_complete_index_sha_and_columnar_audit(tmp_path, monkeypatch, contract):
    monkeypatch.setattr(audit, "robot_contract", lambda _:contract)
    recording, _ = write_recording(tmp_path)
    result = audit.audit_recording(recording)
    assert result["audit_completed"] and result["recorded_physics_checks_pass"]
    assert result["SHA_verified_tick_NPZ_count"] == 1
    assert result["controlled_substeps"] == 4
    assert result["new_physics_steps"] == result["optimizer_updates"] == result["gpu_seconds"] == 0


def test_sha_failure_does_not_consume_unverified_payload(tmp_path, monkeypatch, contract):
    monkeypatch.setattr(audit, "robot_contract", lambda _:contract)
    recording, artifact = write_recording(tmp_path)
    with artifact.open("ab") as stream:
        stream.write(b"changed")
    result = audit.audit_recording(recording)
    assert not result["audit_completed"]
    assert result["SHA_verified_tick_NPZ_count"] == result["controlled_substeps"] == 0
    assert "SHA256" in result["artifact_checks"][0]["error"]["message"]


def test_legacy_dict_schema_is_explicitly_unsupported(tmp_path, monkeypatch, contract):
    monkeypatch.setattr(audit, "robot_contract", lambda _:contract)
    recording, _ = write_recording(tmp_path, kind="full_policy_tick")
    result = audit.audit_recording(recording)
    assert result["SHA_verified_tick_NPZ_count"] == 1
    assert not result["audit_completed"]
    assert "Unsupported tick kind: full_policy_tick" in result["artifact_checks"][0]["error"]["message"]


def test_output_never_overwrites(tmp_path):
    output = tmp_path/"existing.json"
    output.write_text("retained", encoding="utf-8")
    with pytest.raises(SystemExit):
        audit.main(["--recording", str(tmp_path/"absent.jsonl"), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == "retained"
