"""Only000/003 x200paired diagnostic steps; no tracker inference or quality replay."""

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from motionbricks.mb_sonic.cached_forward_step import cached_forward_step


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ["MB_CACHED_FORWARD_REPORT"]) if "MB_CACHED_FORWARD_REPORT" in os.environ else None
pytestmark = pytest.mark.skipif(OUTPUT is None, reason="Explicit bounded numerical-proof allocation required")
RUNTIME = ROOT/"motionbricks/mb_sonic/runtime.py"
METADATA = ROOT/"out/vendor/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml"
METADATA_SHA = "57d81b60dda9aeb30c57d3428a822f203a83bca399c5abe82ffea7a9aaf8b139"
RECORDING = ROOT/"out/mb_sonic_e0_e3_20261018/wasd_v16/episode_0001_000.jsonl"
ROBOT = ROOT.parent/"gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
BINDING = Path(os.environ["MB_CACHED_FORWARD_ROBOT_BINDING"]) if "MB_CACHED_FORWARD_ROBOT_BINDING" in os.environ else None


def file_identity(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def same_bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def contact_state(model, data):
    values = {}
    names = [name for name, item in mujoco.MjContact.__dict__.items()
             if isinstance(item, property) or type(item).__name__ == "property"]
    for index, contact in enumerate(data.contact):
        for name in names:
            value = getattr(contact, name)
            if isinstance(value, (int, float, np.ndarray, np.generic)):
                values[f"contact{index}.{name}"] = np.asarray(value).copy()
        force = np.empty(6)
        mujoco.mj_contactForce(model, data, index, force)
        values[f"contact{index}.wrench"] = force
    return values


def snapshot(model, data):
    result = {name: getattr(data, name).copy() for name in (
        "qpos", "qvel", "act", "qacc", "qacc_warmstart", "xpos", "xquat", "geom_xpos", "geom_xmat",
        "actuator_force", "qfrc_actuator", "qfrc_applied", "xfrc_applied", "efc_force", "sensordata")}
    result.update(time=np.asarray(data.time), ncon=np.asarray(data.ncon), nefc=np.asarray(data.nefc),
        warnings=np.array([(warning.number, warning.lastinfo) for warning in data.warning], dtype=np.int64))
    result.update(contact_state(model, data))
    return result


def statistics(values):
    values = np.asarray(values)/1e6
    return dict(n=len(values), total_ms=float(values.sum()), p50_ms=float(np.percentile(values, 50)),
                p99_ms=float(np.percentile(values, 99)), max_ms=float(values.max()))


@pytest.fixture(scope="module")
def report():
    if OUTPUT.exists():
        pytest.fail("Bounded diagnostic output already exists; do not silently rerun/overwrite")
    metadata = file_identity(METADATA)
    assert metadata["sha256"] == METADATA_SHA
    result = dict(schema="motionbricks.cached_forward_step.diagnostic.v1", status="NOT_RUN",
        scope="Two canonical G1models,200paired1ms indexes each; baseline and candidate MjData at each index",
        paired_step_indexes=400, integrated_substeps_across_both_data=800,
        mujoco_version=mujoco.__version__, metadata=metadata, robot=file_identity(ROBOT),
        runtime_before=file_identity(RUNTIME), helper=file_identity(Path(cached_forward_step.__code__.co_filename)),
        sources=["https://github.com/google-deepmind/mujoco/blob/3.8.0/src/engine/engine_forward.c#L1882-L1969",
                 "https://mujoco.readthedocs.io/en/3.8.0/APIreference/APIfunctions.html#mj-step"],
        source_contract="mj_step checks qpos/qvel, forwards, checks qacc and integrates; candidate only skips cached position/velocity stages, never the constraint solve.",
        gpu_seconds=0, model_inferences=0, optimizer_updates=0, runtime_path_executed=False, scenes=[])
    if BINDING is not None:
        binding = json.loads(BINDING.read_text())
        result["source_robot"] = result["robot"]
        result["robot"] = file_identity(Path(binding["target_robot"]["path"]))
        result["robot_binding"] = file_identity(BINDING)
    yield result
    result["runtime_after"] = file_identity(RUNTIME)
    result["runtime_source_unchanged"] = result["runtime_before"] == result["runtime_after"]
    result["status"] = "PASS" if len(result["scenes"]) == 2 and all(
        row["bitwise_pass"] for row in result["scenes"]) and result["runtime_source_unchanged"] else "FAIL"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print("Bounded cached-forward report:", OUTPUT, result["status"])


@pytest.mark.parametrize("scene_id", ["000", "003"])
def test_bitwise_cached_forward_and_postforward_for200native_steps(report, scene_id):
    from motionbricks.mb_sonic.runtime import SonicPhysicsRuntime, load_physics_scene
    from motionbricks.mb_sonic.reference import G1_JOINT_NAMES

    metadata = yaml.safe_load(METADATA.read_text())
    policy = SimpleNamespace(physics_dt=.001, control_mode="implicit_pd", minimum_history_frames=1,
        anchor_body_name="torso_link", reference_offsets_s=np.array([0., .02, .04, .08, .16]),
        kp=np.array(metadata["control"]["stiffness"]), kd=np.array(metadata["control"]["damping"]))
    manifest = ROOT/f"out/mb_sonic_e0_e3_20261018/scenes/{scene_id}/scene_manifest.json"
    if BINDING is None:
        scene, original, _ = load_physics_scene(manifest, ROBOT)
    else:
        binding = json.loads(BINDING.read_text())
        scene, original, _ = load_physics_scene(manifest, binding["target_robot"]["path"],
            contact_pair_binding=binding["external_contact_pairs"])
    physics = SonicPhysicsRuntime(scene, original, policy, tracker_backend="protomotions", physics_dt=.001)
    model = physics.model
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert (model.nq, model.nv, model.nu) == ((50, 49, 43) if BINDING is None else (36, 35, 29))
    with RECORDING.open() as stream:
        initial = next(row["actual"] for row in map(json.loads, stream) if row["kind"] == "explicit_initial_reset")
    baseline, candidate = mujoco.MjData(model), mujoco.MjData(model)
    if BINDING is not None:
        source_model = mujoco.MjModel.from_xml_path(str(ROBOT))
        qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
        qpos[:7], qvel[:6] = initial["qpos"][:7], initial["qvel"][:6]
        for name in G1_JOINT_NAMES:
            a, b = source_model.joint(name).id, model.joint(name).id
            qpos[model.jnt_qposadr[b]] = initial["qpos"][source_model.jnt_qposadr[a]]
            qvel[model.jnt_dofadr[b]] = initial["qvel"][source_model.jnt_dofadr[a]]
        initial = dict(qpos=qpos, qvel=qvel)
    for data in (baseline, candidate):
        data.qpos[:] = initial["qpos"]
        data.qvel[:] = initial["qvel"]
        data.ctrl[physics.actuator_ids] = np.asarray(initial["qpos"])[physics.qpos_indices]
        mujoco.mj_forward(model, data)
    initial_state = snapshot(model, baseline)
    other_state = snapshot(model, candidate)
    assert initial_state.keys() == other_state.keys()
    assert all(same_bits(value, other_state[name]) for name, value in initial_state.items())
    reference_timing, candidate_timing, mismatches = [], [], {}
    contact_samples, checked_contact_fields = 0, set()
    base_target = baseline.qpos[physics.qpos_indices].copy()
    commands = []
    first_mismatch = None
    for index in range(200):
        if index % 20 == 0:
            target = base_target+.01*np.sin(index//20+np.arange(29)*.13)
            commands.append(dict(at_step=index, target=target.tolist()))
            baseline.ctrl[physics.actuator_ids] = target
            candidate.ctrl[physics.actuator_ids] = target
        # Alternate call order without changing either trajectory or its inputs.
        def reference_step():
            started = perf_counter_ns()
            mujoco.mj_step(model, baseline)
            mujoco.mj_forward(model, baseline)
            reference_timing.append(perf_counter_ns()-started)

        def candidate_step():
            started = perf_counter_ns()
            cached_forward_step(model, candidate)
            candidate_timing.append(perf_counter_ns()-started)

        if index % 2:
            candidate_step(); reference_step()
        else:
            reference_step(); candidate_step()
        before, after = snapshot(model, baseline), snapshot(model, candidate)
        contact_samples += baseline.ncon
        checked_contact_fields.update(name.split(".", 1)[1] for name in before if name.startswith("contact"))
        for name in before.keys() | after.keys():
            if name not in before or name not in after or not same_bits(before[name], after[name]):
                mismatch = mismatches.setdefault(name, dict(steps=0, maximum_abs_difference=0.))
                mismatch["steps"] += 1
                if name in before and name in after and before[name].shape == after[name].shape:
                    difference = np.abs(before[name].astype(float)-after[name].astype(float))
                    mismatch["maximum_abs_difference"] = max(mismatch["maximum_abs_difference"], float(difference.max(initial=0)))
                if first_mismatch is None:
                    first_mismatch = dict(step=index+1, field=name)
    row = dict(scene_id=scene_id, manifest=file_identity(manifest), bitwise_pass=not mismatches,
        native_steps_per_trajectory=200, physics_dt_s=model.opt.timestep, integrator="implicitfast",
        final_time_s=candidate.time, contact_samples_checked=int(contact_samples),
        contact_fields_checked=sorted(checked_contact_fields), mismatches=mismatches, first_mismatch=first_mismatch,
        baseline=statistics(reference_timing), candidate=statistics(candidate_timing),
        baseline_step_ns=reference_timing, candidate_step_ns=candidate_timing,
        median_speedup=float(np.median(reference_timing)/np.median(candidate_timing)),
        final_warning_counts=[int(item.number) for item in candidate.warning],
        controlled_joints=list(G1_JOINT_NAMES), commands=commands,
        initial_state_source=file_identity(RECORDING),
        initialization="Both data copy actualv16timezero qpos/qvel once and complete one initial forward; no later state writes except ctrl.")
    report["scenes"].append(row)
    print(json.dumps({key: row[key] for key in ("scene_id", "bitwise_pass", "first_mismatch", "baseline", "candidate", "median_speedup")}, indent=2))
    assert not mismatches, first_mismatch


def test_unsupported_integrator_and_state_mutating_callback_raise_without_fallback():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint/><geom size=".1" mass="1"/></body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="implicitfast"):
        cached_forward_step(model, data)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    original = mujoco.get_mjcb_control()
    calls = []

    def mutating_control(_model, current):
        calls.append(True)
        current.qpos[0] += 1

    try:
        mujoco.set_mjcb_control(mutating_control)
        with pytest.raises(ValueError, match="callbacks"):
            cached_forward_step(model, data)
        assert not calls and data.time == 0 and data.qpos[0] == 0
    finally:
        mujoco.set_mjcb_control(original)
