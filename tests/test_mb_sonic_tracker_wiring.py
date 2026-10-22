"""CPU actuator/observation wiring; no ONNX policy, service, browser or GPU launch."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from motionbricks.mb_sonic import runtime as module
from motionbricks.mb_sonic.protomotions_policy import reference_window
from motionbricks.mb_sonic.reference import G1_JOINT_NAMES, load_official_reference
from motionbricks.mb_sonic.runtime import SonicPhysicsRuntime, consumer_horizon_contract
from motionbricks.mb_sonic.scene import export_scene, sha256
from motionbricks.mb_sonic.sonic_policy import DEFAULT_ANGLES, KP, KD


ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT/"motionbricks/out/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml"
METADATA_SHA = "57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139"
ROBOT = ROOT/"gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
REFERENCE = ROOT/"gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428"


class ProtoFixture:
    """Only records the call contract; it is not a production tracker or ONNX mock."""
    physics_dt = .001
    control_mode = "implicit_pd"
    minimum_history_frames = 1
    anchor_body_name = "torso_link"
    reference_offsets_s = np.array([0., .02, .04, .08, .16])

    def __init__(self):
        assert sha256(METADATA) == METADATA_SHA
        self.metadata = yaml.safe_load(METADATA.read_text())
        self.kp = np.array(self.metadata["control"]["stiffness"])
        self.kd = np.array(self.metadata["control"]["damping"])
        self.resets, self.calls = [], []

    def reference_window(self, reference, time_s):
        return reference_window(reference, time_s, self.reference_offsets_s)

    def reset_heading(self, actual, reference):
        self.resets.append((actual.copy(), reference.copy()))

    def infer(self, reference, history, *, anchor_quat_wxyz):
        self.calls.append((reference, history.entries[-1], anchor_quat_wxyz.copy()))
        return SimpleNamespace(joint_target=reference.joint_pos[0].copy(), raw_action=np.zeros(29),
            processed_action_mujoco=reference.joint_pos[0].copy(), inputs={}, timing_ns={})


@pytest.fixture(scope="module")
def scene_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("tracker_wiring")
    source = root/"flat.npz"
    np.savez(source, heightfield_y_m=np.zeros((5, 5), np.float32),
             min_point_world_xyz_m=np.array([-10., 0., -10.]), dx_m=np.array(5.),
             metadata_json=np.array('{"coordinate_frame":"soma_world_y_up"}'))
    manifest = export_scene(source, root/"scene", scene_id="tracker_unit",
                            friction=(1., .005, .0001), friction_source="explicit_unit_fixture")
    return module.load_physics_scene(manifest, ROBOT)


def initialized(scene_model, *, backend="protomotions"):
    scene, model, _ = scene_model
    policy = ProtoFixture() if backend == "protomotions" else None
    physics = SonicPhysicsRuntime(scene, model, policy, tracker_backend=backend,
                                  physics_dt=.001 if backend == "protomotions" else .005)
    source = load_official_reference(REFERENCE, scene_id=scene.manifest.scene_id)
    qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
    qpos[:7], qvel[:6] = source.qpos[0, :7], source.qvel[0, :6]
    qpos[physics.qpos_indices], qvel[physics.qvel_indices] = source.joint_pos[0], source.joint_vel[0]
    physics.reset(qpos, qvel)
    return physics, policy, source


def test_proto_implicit_pd_preserves_raw_cache_hands_limits_and_original_force_bounds(scene_model):
    _, original, _ = scene_model
    before = {name: getattr(original, name).copy() for name in (
        "actuator_gainprm", "actuator_biasprm", "actuator_ctrllimited", "actuator_forcelimited",
        "actuator_forcerange", "jnt_stiffness", "dof_damping", "dof_frictionloss", "jnt_range")}
    physics, policy, _ = initialized(scene_model)
    assert physics.model is not original
    for name, expected in before.items():
        np.testing.assert_array_equal(getattr(original, name), expected)
    ids = physics.actuator_ids
    np.testing.assert_array_equal(physics.model.actuator_gainprm[ids, 0], policy.kp)
    np.testing.assert_array_equal(physics.model.actuator_biasprm[ids, 1], -policy.kp)
    np.testing.assert_array_equal(physics.model.actuator_biasprm[ids, 2], -policy.kd)
    assert (physics.model.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_AFFINE).all()
    assert not physics.model.actuator_ctrllimited[ids].any()
    assert physics.model.actuator_forcelimited[ids].all()
    np.testing.assert_array_equal(physics.model.actuator_forcerange[ids, 0], physics.torque_lower)
    np.testing.assert_array_equal(physics.model.actuator_forcerange[ids, 1], physics.torque_upper)
    other_dofs = np.setdiff1d(np.arange(original.nv), physics.qvel_indices)
    for name in ("dof_damping", "dof_frictionloss"):
        np.testing.assert_array_equal(getattr(physics.model, name)[other_dofs], before[name][other_dofs])
        np.testing.assert_array_equal(getattr(physics.model, name)[physics.qvel_indices], 0)
    other_joints = np.setdiff1d(np.arange(original.njnt), physics.joint_ids)
    np.testing.assert_array_equal(physics.model.jnt_stiffness[other_joints], before["jnt_stiffness"][other_joints])
    np.testing.assert_array_equal(physics.model.jnt_stiffness[physics.joint_ids], 0)
    np.testing.assert_array_equal(physics.model.jnt_range, before["jnt_range"])
    assert not np.array_equal(policy.kp, KP) and not np.array_equal(policy.kd, KD)
    again, _, _ = initialized(scene_model)
    assert again.model is not physics.model


def test_proto_target_angles_are_unclipped_but_mujoco_motor_forces_remain_bounded(scene_model):
    physics, _, _ = initialized(scene_model)
    target = np.full(29, 100.)
    frames = physics._advance_pd(target, substeps=1)
    np.testing.assert_array_equal(frames[0].actuator_controls[physics.actuator_ids], target)
    forces = frames[0].actuator_forces[physics.actuator_ids]
    assert (forces <= physics.torque_upper+1e-9).all() and (forces >= physics.torque_lower-1e-9).all()
    assert not np.array_equal(frames[0].joint_positions, target)
    np.testing.assert_array_equal(frames[0].root_qfrc_actuator, 0)
    np.testing.assert_array_equal(frames[0].qfrc_applied, 0)
    np.testing.assert_array_equal(frames[0].xfrc_applied, 0)


def test_proto_real_warmup_has180native_steps_and_only_one_actual_proprio_sample(scene_model):
    physics, policy, _ = initialized(scene_model)
    initial = physics.data.qpos[physics.qpos_indices].copy()
    frames = physics.warmup_history()
    assert len(frames) == 180 and physics.time_s == pytest.approx(.18)
    assert physics.warmup_history() == () and not policy.calls
    assert len(physics.history.entries) == 1
    current = physics.history.entries[-1]
    assert current.time_s == pytest.approx(.18) and not hasattr(current, "last_action_isaac")
    assert not hasattr(physics, "last_action_isaac")
    np.testing.assert_array_equal(physics.warmup_joint_target, initial)
    np.testing.assert_array_equal(current.joint_positions, frames[-1].joint_positions)
    np.testing.assert_array_equal(current.base_angular_velocity, frames[-1].qvel[3:6])
    assert physics.control_metadata["warmup_target_source"] == "actual_time_zero_reset_joints"


def test_proto_anchor_is_actual_torso_and_sparse_future_is_not_sonic_window(scene_model):
    physics, policy, source = initialized(scene_model)
    physics.warmup_history()
    window, command = physics.begin_control(source)
    np.testing.assert_allclose(window.timestamps_s, .18+np.array([0., .02, .04, .08, .16]))
    np.testing.assert_array_equal(policy.resets[0][0], physics.data.xquat[physics.anchor_body_id])
    anchor = window.body_names.index("torso_link")
    np.testing.assert_array_equal(policy.resets[0][1], window.body_quat_w[0, anchor])
    physics.begin_control(source)
    assert len(policy.resets) == 1 and not hasattr(command, "action_isaac")
    assert physics.reference_lookahead_s == .16
    short = replace(source, trusted_until_s=.34)
    assert physics.begin_control(short)[0].timestamps_s[-1] == pytest.approx(.34)
    with pytest.raises(ValueError):
        physics.begin_control(replace(source, trusted_until_s=.339))


@pytest.mark.parametrize("dt", [.005, .0025, .0005])
def test_proto_rejects_other_physics_timestep_before_model_changes(scene_model, dt):
    scene, model, _ = scene_model
    with pytest.raises(ValueError, match="1ms"):
        SonicPhysicsRuntime(scene, model, ProtoFixture(), tracker_backend="protomotions", physics_dt=dt)


def test_sonic_default_pd_warmup_and_window_contract_stay_original(scene_model):
    physics, _, _ = initialized(scene_model, backend="sonic")
    frames = physics.warmup_history()
    assert len(frames) == 36 and physics.time_s == pytest.approx(.18)
    assert len(physics.history.entries) == 10 and physics.reference_lookahead_s == .9
    np.testing.assert_array_equal(physics.warmup_joint_target, DEFAULT_ANGLES)
    np.testing.assert_array_equal(physics.kp, KP)
    np.testing.assert_array_equal(physics.kd, KD)


def test_tracker_specific_coverage_changes_not_mb_consumer_horizon():
    settings = dict(consumer_horizon_s=1.3, mb_replan_interval_s=.1, max_plan_wall_age_s=.1)
    sonic = consumer_horizon_contract(settings)
    proto = consumer_horizon_contract(dict(settings, tracker_backend="protomotions"))
    assert sonic["required_future_s"] == .9 and proto["required_future_s"] == .16
    assert sonic["consumer_horizon_s"] == proto["consumer_horizon_s"] == settings["consumer_horizon_s"] == 1.3


def test_loader_passes_exact_proto_asset_pins_without_loading_sonic(monkeypatch):
    from motionbricks.mb_sonic import protomotions_policy
    calls = []

    def factory(*args, **kwargs):
        calls.append((args, kwargs))
        return "isolated_loader_fixture"

    monkeypatch.setattr(protomotions_policy, "FrozenProtoMotionsPolicy", factory)
    monkeypatch.setattr(module, "FrozenSonicPolicy", lambda *a, **k: pytest.fail("Do not load SONIC"))
    settings = dict(tracker_backend="protomotions", policy_threads=2, protomotions_policy=dict(
        onnx_path="not_loaded_fixture.onnx", metadata_path=str(METADATA),
        onnx_sha256="a59baa3e04a951e5cf0b4cc68f24ebaafa9272714226618b99a5017dfc805b4c",
        metadata_sha256=METADATA_SHA))
    policy, path, digest = module._load_tracker_policy(settings)
    assert policy == "isolated_loader_fixture" and path == METADATA and digest == METADATA_SHA
    assert calls[0][1]["threads"] == 2 and calls[0][1]["metadata_sha256"] == METADATA_SHA
    settings["protomotions_policy"]["metadata_sha256"] = "0"*64
    with pytest.raises(ValueError, match="explicit pin"):
        module._load_tracker_policy(settings)
    assert len(calls) == 1


def test_matching_robot_pairs_use_canonical_ground_and_friction_without_extra_floor():
    directory = ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/protomotions_g1_v1"
    binding = json.loads((directory/"binding.json").read_text())
    scene, model, receipt = module.load_physics_scene(
        ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/scenes/000/scene_manifest.json",
        binding["target_robot"]["path"], contact_pair_binding=binding["external_contact_pairs"])
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert model.npair == 14 and model.geom("mb_ground").id >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
    assert ((model.pair_geom1 == model.geom("mb_ground").id)
            | (model.pair_geom2 == model.geom("mb_ground").id)).all()
    slide, torsion, roll = scene.manifest.friction["coefficients"]
    np.testing.assert_allclose(model.pair_friction, np.tile([slide, slide, torsion, roll, roll], (14, 1)))
    assert receipt["contact_pair_binding"]["extra_ground"] is False


def test_matching_robot_visual_primitives_are_preserved_without_gpu():
    from collections import OrderedDict
    directory = ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/protomotions_g1_v1"
    binding = json.loads((directory/"binding.json").read_text())
    scene, model, _ = module.load_physics_scene(
        ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/scenes/000/scene_manifest.json",
        binding["target_robot"]["path"], contact_pair_binding=binding["external_contact_pairs"])
    lab = module.SonicLabRuntime.__new__(module.SonicLabRuntime)
    key = sha256(scene.manifest_path)
    lab.physics = SimpleNamespace(scene=scene, model=model, manifest_sha256=key)
    lab._scene_asset_limit = 134217728
    lab._scene_cache = OrderedDict({key: dict(geometry=None, assets=None, geom_ids=None)})
    _, visuals = lab._geometry_config()
    assert len(visuals) == int(((model.geom_bodyid > 0) & (model.geom_group == 1)).sum())
    assert {row["kind"] for row in visuals} == {"mesh", "sphere", "cylinder"}


def test_proto_can_start_with_its_one_real_initial_measurement_without_sonic_warmup(scene_model):
    scene, model, receipt = scene_model
    policy = ProtoFixture()
    physics = SonicPhysicsRuntime(scene, model, policy, tracker_backend="protomotions",
        physics_dt=.001, warmup_control_ticks=0, robot_model_sha256=receipt["sha256"])
    source = load_official_reference(REFERENCE, scene_id=scene.manifest.scene_id)
    qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
    qpos[:7] = source.qpos[0, :7]
    qpos[physics.qpos_indices] = source.joint_pos[0]
    physics.reset(qpos, qvel)
    assert not physics.needs_warmup and physics.warmup_history() == ()
    assert len(physics.history.entries) == 1 and physics.history.entries[0].time_s == 0
    window, _ = physics.begin_control(source)
    assert window.timestamps_s[0] == 0 and window.timestamps_s[-1] == pytest.approx(.16)
    assert physics.control_metadata["warmup_duration_s"] == 0


def test_batched_contact_snapshot_matches_native_tuple_force_bits(scene_model):
    physics, _, _ = initialized(scene_model)
    before = physics.snapshot()
    physics.contact_layout = "batch"
    after = physics.snapshot()
    assert len(after.contacts) == len(before.contacts)
    for old, new in zip(before.contacts, after.contacts):
        assert old.geom_ids == new.geom_ids and old.body_ids == new.body_ids
        assert old.distance == new.distance
        for name in ("position_world", "frame_contact_from_world", "wrench_contact"):
            assert getattr(old, name).tobytes() == getattr(new, name).tobytes()


def test_b1_disabled_actual_diagnostics_retains_only_measured_snapshot_not_duplicate_contacts(scene_model):
    from motionbricks.mb_sonic.reference import ExecutedHistory
    physics, _, _ = initialized(scene_model)
    state = physics.snapshot()
    lab = module.SonicLabRuntime.__new__(module.SonicLabRuntime)
    lab.mode, lab.settings = "mb", dict(estimate_actual_diagnostics=False)
    lab._executed = ExecutedHistory(physics.scene_id)
    lab._append_executed(state)
    newer = replace(state, time_s=.001, state_version=1)
    lab._append_executed(newer)
    assert len(lab._executed.samples) == 1
    saved = lab._executed.samples[0]
    assert set(saved) == {"timestamp_s", "qpos", "captured_at_monotonic_ns", "state_version"}
    np.testing.assert_array_equal(saved["qpos"], newer.qpos)
    assert len(state.contacts) == len(newer.contacts)


def test_flat_ground_box_has_exact_finite_bounds_and_identical_probe_contact_depth():
    directory = ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/protomotions_g1_v1"
    binding = json.loads((directory/"binding.json").read_text())
    path = ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/scenes/000/scene_manifest.json"
    scene, native, _ = module.load_physics_scene(path, binding["target_robot"]["path"],
        contact_pair_binding=binding["external_contact_pairs"])
    _, box, receipt = module.load_physics_scene(path, binding["target_robot"]["path"],
        contact_pair_binding=binding["external_contact_pairs"], flat_ground_primitive="exact_finite_box_v1")
    g = box.geom("mb_ground").id
    assert box.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    np.testing.assert_allclose(box.geom_pos[g, :2]-box.geom_size[g, :2], scene.bounds_xy_m[0], atol=1e-12)
    np.testing.assert_allclose(box.geom_pos[g, :2]+box.geom_size[g, :2], scene.bounds_xy_m[1], atol=1e-12)
    assert box.geom_pos[g, 2]+box.geom_size[g, 2] == pytest.approx(scene.height[0, 0])
    assert receipt["ground_collision_implementation"] == "exact_finite_box_v1"
    center = scene.bounds_xy_m.mean(0)
    points = [center, scene.bounds_xy_m[0]+.1, scene.bounds_xy_m[1]-.1,
              scene.bounds_xy_m[1]+.1]
    for point in points:
        distances = []
        for model in (native, box):
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            sphere = model.geom("left_foot1_collision").id
            radius = model.geom_size[sphere, 0]
            target = np.r_[point, scene.height[0, 0]+radius-.003]
            data.qpos[:3] += target-data.geom_xpos[sphere]
            mujoco.mj_forward(model, data)
            ground = model.geom("mb_ground").id
            contacts = [contact.dist for contact in data.contact if set(contact.geom) == {sphere, ground}]
            distances.append(min(contacts) if contacts else None)
        assert (distances[0] is None) == (distances[1] is None)
        if distances[0] is not None:
            assert distances[0] == pytest.approx(distances[1], abs=1e-12)


def test_nonflat_ground_keeps_original_full_heightfield():
    directory = ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/protomotions_g1_v1"
    binding = json.loads((directory/"binding.json").read_text())
    _, model, receipt = module.load_physics_scene(
        ROOT/"motionbricks/out/mb_sonic_e0_e3_20261018/scenes/003/scene_manifest.json",
        binding["target_robot"]["path"], contact_pair_binding=binding["external_contact_pairs"],
        flat_ground_primitive="exact_finite_box_v1")
    assert model.geom_type[model.geom("mb_ground").id] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert receipt["ground_collision_implementation"] == "full_resolution_hfield"
