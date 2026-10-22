"""Bounded CPU tracking replay, without MB generation or wall-clock admission.

--seconds is the absolute simulation end, including the recorded PD warmup.
Each invocation runs exactly one explicitly selected reference variant.
"""

import argparse
from dataclasses import asdict, fields, is_dataclass, replace
import inspect
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter_ns
import traceback

import mujoco
import numpy as np
import onnxruntime

from motionbricks.mb_sonic.reference import G1Reference, G1_JOINT_NAMES, load_official_reference
from motionbricks.mb_sonic.protomotions_policy import (
    FrozenProtoMotionsPolicy, PROTOMOTIONS_ONNX_SHA256, PROTOMOTIONS_REVISION,
)
from motionbricks.mb_sonic.runtime import PhysicsRecorder, SonicPhysicsRuntime, load_physics_scene
from motionbricks.mb_sonic.scene import sha256
from motionbricks.mb_sonic.physics_numerics import configure_physics_numerics
from motionbricks.mb_sonic.cached_forward_step import cached_forward_step
from motionbricks.mb_sonic.sonic_policy import FrozenSonicPolicy, HISTORY_FRAMES, POLICY_DT
from scripts.run_mb_sonic_physics import CONFIG_SHA256, DECODER_SHA256, ENCODER_SHA256, REVISION


def read_episode(path):
    metadata, resets, warmup, adoptions, uses, artifacts = [], [], [], [], [], {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            kind = row["kind"]
            if kind == "metadata":
                metadata.append(row)
            elif kind == "explicit_initial_reset":
                resets.append(row["actual"])
            elif kind == "actual_history_warmup":
                warmup.append(row["actual"])
            elif kind == "reference_packet_adopted":
                adoptions.append(row)
            elif kind == "reference_packet_used":
                uses.append(row)
            elif kind == "numeric_artifact_written":
                artifacts[row["record_id"]] = row
    if len(metadata) != 1 or len(resets) != 1 or resets[0]["time_s"] != 0.:
        raise ValueError("Replay requires one recorded explicit time-zero reset and metadata header")
    return dict(metadata=metadata[0], initial=resets[0], warmup=warmup,
                adoptions=adoptions, uses=uses, artifacts=artifacts)


def read_adapted_reference(receipt, plan_id, expected_manifest, expected_robot):
    path = Path(receipt["path"]).resolve(strict=True)
    if sha256(path) != receipt["sha256"]:
        raise ValueError(f"Recorded numeric artifact SHA256 differs: {plan_id}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata["schema"] != "motionbricks.mb_sonic.numeric_record.v1" or metadata["record_id"] != plan_id:
            raise ValueError("Unknown or mismatched recorded numeric artifact")
        record = metadata["record"]
        packet = record["packet"]
        if (record["kind"] != "mb_raw_and_adapted_reference" or packet["plan_id"] != plan_id
                or packet["scene_manifest_sha256"] != expected_manifest
                or packet["robot_model_sha256"] != expected_robot):
            raise ValueError("Recorded reference belongs to a different plan, robot or canonical scene")

        def decode(value):
            if isinstance(value, dict):
                if "array" in value and "dtype" in value and "shape" in value:
                    array = archive[value["array"]]
                    if array.dtype.str != value["dtype"] or list(array.shape) != value["shape"]:
                        raise ValueError("Recorded numeric array descriptor differs from its payload")
                    return array.copy()
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        values = decode(record["adapted_reference"])
        values["body_names"], values["joint_names"] = tuple(values["body_names"]), tuple(values["joint_names"])
        reference = G1Reference(**values)
        for key, expected in (("future_timestamps", reference.timestamps_s), ("root_pose", reference.qpos[:, :7]),
                              ("body_joint_positions", reference.joint_pos), ("body_joint_velocities", reference.joint_vel)):
            if not np.array_equal(decode(packet[key]), expected):
                raise ValueError(f"Recorded packet/reference payload differs: {key}")
    return reference, dict(path=str(path), sha256=receipt["sha256"], plan_id=plan_id,
                          command_seq=packet["generated_from_command_seq"])


def cached_schedule(episode, expected_manifest, expected_robot):
    adopted = {row["plan_id"]: row for row in episode["adoptions"]}
    first_use = {}
    for row in episode["uses"]:
        first_use.setdefault(row["plan_id"], row)
    if not first_use:
        raise ValueError("The selected episode has no actually consumed adapted-reference stream")
    schedule, receipts = [], []
    for plan_id, use in first_use.items():
        adoption = adopted[plan_id]
        reference, receipt = read_adapted_reference(episode["artifacts"][plan_id], plan_id,
                                                    expected_manifest, expected_robot)
        time_s = float(use["used_time_window"][0])
        if (time_s < adoption["sim_time_s"]-1e-8
                or receipt["command_seq"] != use["generated_from_command_seq"]):
            raise ValueError("Recorded adoption/use identity or simulation-clock order differs")
        schedule.append((time_s, reference, dict(receipt,
            recorded_adoption_sim_time_s=adoption["sim_time_s"], recorded_first_use_sim_time_s=time_s,
            recorded_adoption_monotonic_ns=adoption["reference_adopted_monotonic_ns"])))
        receipts.append(receipt)
    if any(right[0] <= left[0] for left, right in zip(schedule, schedule[1:])):
        raise ValueError("Consumed reference transitions must have strictly increasing actual times")
    return schedule, receipts, float(episode["uses"][-1]["used_time_window"][0])+POLICY_DT


def select_reference(schedule, time_s):
    index = int(np.searchsorted([item[0] for item in schedule], time_s+1e-8, side="right")-1)
    if index < 0:
        raise ValueError("No genuinely consumed reference exists at the requested replay time")
    return schedule[index][1:]


def distribution(values):
    return ({"count": len(values), "p50_ms": float(np.percentile(values, 50)),
             "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
             "maximum_ms": float(np.max(values))} if values else None)


def save_failure_evidence(path, runtime, pending_tick):
    """Keep the exact failed numbers outside JSON, including an unwritten tick."""
    arrays, finite = {}, {}

    def encode(value):
        if is_dataclass(value):
            return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, np.ndarray) or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            key = f"array_{len(arrays):04d}"
            array = np.array(value, copy=True)
            arrays[key] = array
            valid = np.isfinite(array)
            finite[key] = dict(finite=bool(valid.all()), nonfinite_count=int(array.size-valid.sum()))
            return dict(array=key, dtype=array.dtype.str, shape=list(array.shape), **finite[key])
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    names = ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "actuator_force",
             "qfrc_actuator", "qfrc_passive", "qfrc_bias", "qfrc_constraint",
             "qfrc_applied", "xfrc_applied", "xpos", "xquat")
    raw = dict(time_s=runtime.time_s, terminal_reason=runtime.terminal_reason,
        mujoco_data={name: getattr(runtime.data, name) for name in names}, pending_tick=pending_tick,
        solver_warnings={mujoco.mjtWarning(index).name: dict(number=int(item.number), lastinfo=int(item.lastinfo))
                         for index, item in enumerate(runtime.data.warning)})
    metadata = dict(schema="motionbricks.mb_sonic.replay_failure.v1", record=encode(raw))
    with path.open("xb") as stream:
        np.savez(stream, metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)), **arrays)
    return dict(path=str(path.resolve()), sha256=sha256(path), arrays=finite,
                finite=all(item["finite"] for item in finite.values()))


def finite_summary(summary):
    nonfinite_fields = []

    def encode(value, location):
        if isinstance(value, dict):
            return {key: encode(item, f"{location}.{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item, f"{location}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            nonfinite_fields.append(location)
            return None
        return value

    result = encode(summary, "summary")
    result["nonfinite_summary_fields"] = nonfinite_fields
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--episode-record", type=Path, required=True)
    parser.add_argument("--variant", choices=("cached_adapted", "official_csv"), required=True)
    parser.add_argument("--tracker", choices=("sonic", "protomotions"), default="sonic")
    parser.add_argument("--proto-model", type=Path)
    parser.add_argument("--proto-metadata", type=Path)
    parser.add_argument("--proto-metadata-sha256")
    parser.add_argument("--execution-binding", type=Path)
    parser.add_argument("--official-reference-dir", type=Path)
    parser.add_argument("--seconds", type=float, required=True, help="Absolute simulation end, including warmup; at most 30")
    parser.add_argument("--max-wall-seconds", type=float, default=180.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    args = parser.parse_args(argv)
    setup_started_ns = perf_counter_ns()
    if (not np.isfinite(args.seconds) or not 0 < args.seconds <= 30
            or not np.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 300):
        parser.error("Explicit replay bounds are 0<simulation<=30s and 0<wall<=300s")
    if (args.variant == "official_csv") != (args.official_reference_dir is not None):
        parser.error("Only official_csv requires --official-reference-dir")
    proto_assets = (args.proto_model, args.proto_metadata, args.proto_metadata_sha256)
    if args.tracker == "protomotions" and not all(proto_assets):
        parser.error("ProtoMotions requires --proto-model, --proto-metadata and --proto-metadata-sha256")
    if args.tracker == "sonic" and any(value is not None for value in proto_assets):
        parser.error("ProtoMotions assets cannot be supplied to a SONIC replay")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    episode = read_episode(args.episode_record)
    recorded = episode["metadata"]
    manifest_hash, robot_hash = sha256(args.scene_manifest), sha256(settings["robot_model"])
    if (manifest_hash != settings["scene_manifest_sha256"]["000"]
            or recorded["manifest_sha256"] != manifest_hash or recorded["robot"]["sha256"] != robot_hash
            or robot_hash != settings["robot_model_sha256"]
            or settings["physics_dt_s"] != .005 or recorded["physics_dt_s"] != .005):
        raise ValueError("Replay must bind the exact canonical 000 scene, robot and recorded 5ms source episode")
    policy_dir = Path(settings["policy_dir"])
    for name, expected in (("encoder", ENCODER_SHA256), ("decoder", DECODER_SHA256)):
        if settings[name+"_sha256"] != expected or recorded["policy"][name]["sha256"] != expected:
            raise ValueError("Replay requires the matched original SONIC pair used by the episode")
    observation_config = policy_dir/"observation_config.yaml"
    if (settings["observation_config_sha256"] != CONFIG_SHA256 or sha256(observation_config) != CONFIG_SHA256
            or recorded["observation_config"]["sha256"] != CONFIG_SHA256):
        raise ValueError("Replay observation YAML does not match the recorded original SONIC pair")
    warmup = episode["warmup"]
    expected_warmup = (HISTORY_FRAMES-1)*round(POLICY_DT/.005)
    if len(warmup) != expected_warmup:
        raise ValueError("A complete real 36-step warmup is required; history padding/restoration is forbidden")
    control_start = float(warmup[-1]["time_s"])
    count = round((args.seconds-control_start)/POLICY_DT)
    if count < 1 or not np.isclose(control_start+count*POLICY_DT, args.seconds, atol=1e-8, rtol=0):
        raise ValueError("The explicit simulation end must lie on a covered native control tick after warmup")
    if args.variant == "cached_adapted":
        schedule, source_receipts, recorded_end = cached_schedule(episode, manifest_hash, robot_hash)
        if args.seconds > recorded_end+1e-8:
            raise ValueError(f"Requested end exceeds the actually recorded control stream ({recorded_end:.6f}s)")
    else:
        official = load_official_reference(args.official_reference_dir, scene_id="000")
        shifted = replace(official, timestamps_s=official.timestamps_s+control_start,
                          trusted_until_s=official.trusted_until_s+control_start)
        schedule = [(control_start, shifted, dict(kind="official_csv", provenance=official.provenance,
                     source_to_simulation_time_offset_s=control_start))]
        source_receipts = [official.provenance]
    if args.tracker == "sonic":
        policy = FrozenSonicPolicy(policy_dir/"model_encoder.onnx", policy_dir/"model_decoder.onnx",
            observation_config, encoder_sha256=ENCODER_SHA256, decoder_sha256=DECODER_SHA256,
            threads=settings["policy_threads"])
        physics_dt = float(settings.get("comparison_physics_dt_s", .005))
    else:
        policy = FrozenProtoMotionsPolicy(args.proto_model, args.proto_metadata,
            onnx_sha256=PROTOMOTIONS_ONNX_SHA256, metadata_sha256=args.proto_metadata_sha256,
            threads=settings["policy_threads"])
        physics_dt = policy.physics_dt
    for index in range(count):
        time_s = control_start+index*POLICY_DT
        reference, _ = select_reference(schedule, time_s)
        if args.tracker == "sonic":
            reference.sonic_window(time_s)
        else:
            policy.reference_window(reference, time_s)
    binding = None
    if args.execution_binding is None:
        scene, model, robot = load_physics_scene(args.scene_manifest, settings["robot_model"])
    else:
        binding = json.loads(args.execution_binding.read_text(encoding="utf-8"))
        if (binding["schema"] != "motionbricks.protomotions.robot_binding.v1"
                or binding["source_robot"]["sha256"] != robot_hash
                or sha256(binding["target_robot"]["path"]) != binding["target_robot"]["sha256"]
                or not binding["joint_ranges_equal"]):
            raise ValueError("Execution robot binding differs from the recorded source or prepared target")
        scene, model, robot = load_physics_scene(args.scene_manifest, binding["target_robot"]["path"],
            contact_pair_binding=binding["external_contact_pairs"])
        source_model = load_physics_scene(args.scene_manifest, settings["robot_model"])[1]
        source_qids = np.array([source_model.jnt_qposadr[source_model.joint(name).id] for name in G1_JOINT_NAMES])
        source_vids = np.array([source_model.jnt_dofadr[source_model.joint(name).id] for name in G1_JOINT_NAMES])
        initial_q, initial_v = model.qpos0.copy(), np.zeros(model.nv)
        initial_q[:7], initial_v[:6] = episode["initial"]["qpos"][:7], episode["initial"]["qvel"][:6]
        for index, name in enumerate(G1_JOINT_NAMES):
            target = model.joint(name).id
            initial_q[model.jnt_qposadr[target]] = episode["initial"]["qpos"][source_qids[index]]
            initial_v[model.jnt_dofadr[target]] = episode["initial"]["qvel"][source_vids[index]]
        converted = []
        for when, reference, receipt in schedule:
            provenance = dict(kind="cross_model_reference_replay", source_provenance=dict(reference.provenance),
                source_robot_sha256=robot_hash, execution_robot_sha256=robot["sha256"],
                binding_sha256=sha256(args.execution_binding), all_reference_numeric_channels_unchanged=True)
            converted.append((when, replace(reference, provenance=provenance), receipt))
        schedule = converted
    numerics_receipt = configure_physics_numerics(model, physics_dt, settings)
    if scene.manifest.scene_id != "000":
        raise ValueError("This tracking-isolation allocation is restricted to canonical scene 000")
    ground_geoms = {model.geom(row["geom_name"]).id for row in scene.manifest.objects}
    fall_bodies = {model.body(name).id for name in ("pelvis", "torso_link")}

    def walking_failure(state):
        if not state.root_ground_known:
            return "unknown_canonical_ground_domain"
        for contact in state.contacts:
            a, b = contact.geom_ids
            ba, bb = contact.body_ids
            if contact.wrench_contact[0] > 1 and ((a in ground_geoms and bb in fall_bodies)
                                                 or (b in ground_geoms and ba in fall_bodies)):
                return "walking_task:pelvis_or_torso_scene_contact"
        return None

    runtime = SonicPhysicsRuntime(scene, model, policy, tracker_backend=args.tracker, physics_dt=physics_dt,
        physics_execution=settings.get("physics_execution", "native"),
        robot_model_sha256=robot["sha256"], failure_evaluator=walking_failure)
    initial = runtime.reset(episode["initial"]["qpos"] if binding is None else initial_q,
                            episode["initial"]["qvel"] if binding is None else initial_v)
    args.output.mkdir(parents=True, exist_ok=False)
    source_dir = args.output/"sources"
    source_dir.mkdir()
    source_paths = {Path(inspect.getfile(value)).resolve() for value in
                    (SonicPhysicsRuntime, FrozenSonicPolicy, FrozenProtoMotionsPolicy, configure_physics_numerics, cached_forward_step,
                     G1Reference, load_physics_scene, sha256)}
    source_paths.update((Path(__file__).resolve(), Path(sys.modules["scripts.run_mb_sonic_physics"].__file__).resolve(),
                         Path(inspect.getfile(load_official_reference)).resolve()))
    source_hashes = {}
    for path in sorted(source_paths):
        shutil.copyfile(path, source_dir/path.name)
        source_hashes[str(path)] = sha256(path)
    metadata = dict(schema="motionbricks.mb_sonic.tracking_isolation.v1", allocation_id=args.allocation_id,
        variant=args.variant, tracker=args.tracker, optimizer_updates=0, gpu_seconds=0, mb_generation=False,
        wall_clock_admission=False,
        reference_timestamps="unchanged recorded absolute times" if args.variant == "cached_adapted" else "official CSV plus explicit warmup clock offset",
        reference_switches="actual first policy-use joined to actual plan adoption" if args.variant == "cached_adapted" else "official CSV clock offset only",
        root_reanchoring=False, replay_source_sha256=source_hashes, config_sha256=sha256(args.config),
        episode_record=dict(path=str(args.episode_record.resolve()), sha256=sha256(args.episode_record)),
        recorded_metadata=recorded, source_receipts=source_receipts,
        scene_manifest=asdict(scene.manifest), manifest_sha256=manifest_hash, robot=robot,
        execution_robot_binding=binding,
        sonic_revision=REVISION if args.tracker == "sonic" else None,
        protomotions_revision=PROTOMOTIONS_REVISION if args.tracker == "protomotions" else None,
        policy=policy.asset_metadata, observation_config_sha256=CONFIG_SHA256 if args.tracker == "sonic" else None,
        versions=dict(python=sys.version, numpy=np.__version__, mujoco=mujoco.__version__, onnxruntime=onnxruntime.__version__),
        physics_dt_s=physics_dt, recorded_physics_dt_s=.005, controller_hz=50,
        physics_integrator=mujoco.mjtIntegrator(runtime.model.opt.integrator).name,
        physics_numerics=numerics_receipt,
        requested_simulation_end_s=args.seconds,
        warmup=dict(reset="same recorded time-zero qpos/qvel; no further state/time assignment",
            controller="sonic_explicit_pd" if args.tracker == "sonic" else "protomotions_implicit_pd",
            controller_contract=runtime.control_metadata,
            end_time_s=control_start, bitwise_match_required=(binding is None and args.tracker == "sonic" and physics_dt == .005 and settings.get("physics_integrator", "Euler") == "Euler"),
            comparison="recorded SONIC 5ms checkpoints; Proto differences are measured, not corrected"),
        max_wall_seconds=args.max_wall_seconds, warmup_qpos_atol=1e-10, warmup_qvel_atol=1e-9,
        first_instability="first native terminal reason or XML joint overshoot beyond 1e-6 rad",
        termination="native numeric/solver failure; walking pelvis/torso ground contact over 1N; explicit bounds",
        root_error_semantics="diagnostic only; original encoder ignores world Root XYZ/height",
        timing_scope="step: selected tracker+physics+history; loop: includes synchronous recording; no producer/browser")
    metadata["setup_wall_time_s"] = (perf_counter_ns()-setup_started_ns)/1e9
    chain_ms, loop_ms, joint_errors = [], [], []
    first_instability, failure, selected_plan = None, None, None
    warmup_q_error, warmup_v_error, steps, warmup_comparisons = 0., 0., 0, 0
    controlled_initial, pending_tick, failure_evidence = None, None, None
    force_max = dict(qfrc_applied=0., xfrc_applied=0., root_qfrc_actuator=0.)
    wall_started = perf_counter_ns()

    with PhysicsRecorder(args.output/"recording.jsonl", metadata) as recorder:
        def audit(state, phase):
            nonlocal first_instability
            for key in force_max:
                force_max[key] = max(force_max[key], float(np.abs(getattr(state, key)).max()))
            margin = float(np.minimum(state.joint_positions-runtime.joint_limits[:, 0],
                                      runtime.joint_limits[:, 1]-state.joint_positions).min())
            reason = state.terminal_reason or ("XML_joint_limit_overshoot" if margin < -1e-6 else None)
            if reason and first_instability is None:
                first_instability = dict(reason=reason, phase=phase, time_s=state.time_s,
                    state_version=state.state_version, minimum_joint_margin_rad=margin)
                recorder.write(dict(kind="first_instability", **first_instability, actual=asdict(state)))

        try:
            recorder.write(dict(kind="explicit_initial_reset", actual=asdict(initial)))
            audit(initial, "explicit_reset")
            for state in runtime.warmup_history():
                if (perf_counter_ns()-wall_started)/1e9 >= args.max_wall_seconds:
                    raise TimeoutError("Explicit CPU replay wall budget exhausted")
                steps += 1
                checkpoint = round(state.time_s/.005)
                if np.isclose(state.time_s, checkpoint*.005, atol=1e-10, rtol=0):
                    expected = warmup[checkpoint-1]
                    warmup_comparisons += 1
                    if binding is None:
                        q_error = state.qpos-np.asarray(expected["qpos"])
                        v_error = state.qvel-np.asarray(expected["qvel"])
                    else:
                        q_error = np.r_[state.qpos[:7]-expected["qpos"][:7],
                                        state.joint_positions-np.asarray(expected["qpos"])[source_qids]]
                        v_error = np.r_[state.qvel[:6]-expected["qvel"][:6],
                                        state.joint_velocities-np.asarray(expected["qvel"])[source_vids]]
                    warmup_q_error = max(warmup_q_error, float(np.abs(q_error).max()))
                    warmup_v_error = max(warmup_v_error, float(np.abs(v_error).max()))
                recorder.write(dict(kind="actual_history_warmup", actual=asdict(state)))
                audit(state, "warmup")
                if state.terminal_reason:
                    break
            if not runtime.terminal_reason:
                if (steps != round(control_start/physics_dt) or warmup_comparisons != len(warmup)
                        or not np.isclose(runtime.time_s, control_start, atol=1e-10, rtol=0)):
                    raise ValueError("Native controller warmup did not reach the recorded 0.18s start")
                if metadata["warmup"]["bitwise_match_required"] and (warmup_q_error > 1e-10 or warmup_v_error > 1e-9):
                    raise ValueError("Recorded reset/warmup was not reproduced; do not substitute recorded state")
                controlled_initial = runtime.snapshot()
                recorder.write(dict(kind="controlled_initial_state", tracker=args.tracker,
                    actual=asdict(controlled_initial), source="native controller warmup; not restored state"))
                recorder.write(dict(kind="rebuilt_policy_history", entries=[asdict(entry) for entry in runtime.history.entries]))
                for _ in range(count):
                    if (perf_counter_ns()-wall_started)/1e9 >= args.max_wall_seconds:
                        raise TimeoutError("Explicit CPU replay wall budget exhausted")
                    began = perf_counter_ns()
                    reference, provenance = select_reference(schedule, runtime.time_s)
                    identity = provenance.get("plan_id", "official_csv")
                    if identity != selected_plan:
                        recorder.write(dict(kind="replay_reference_selected", simulation_time_s=runtime.time_s,
                                            reference=provenance))
                        selected_plan = identity
                    pending_tick = None
                    tick = runtime.step(reference, raw_reference=provenance,
                        issued_at_monotonic_ns=began, phase="cpu_tracking_isolation_"+args.tracker+"_"+args.variant)
                    pending_tick = tick
                    steps += len(tick.physics_states)
                    chain_ms.append(tick.timing_ns["chain_total"]/1e6)
                    for state in tick.physics_states:
                        audit(state, "controlled")
                    recorder.tick(tick)
                    loop_ms.append((perf_counter_ns()-began)/1e6)
                    index = int(np.searchsorted(reference.timestamps_s, tick.actual.time_s-1e-9))
                    if index < len(reference.timestamps_s):
                        joint_errors.append(float(np.mean((tick.actual.joint_positions-reference.joint_pos[index])**2)))
                    if runtime.terminal_reason:
                        break
        except Exception as error:
            failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
            try:
                failure_evidence = save_failure_evidence(args.output/"failure_actual.npz", runtime, pending_tick)
                recorder.write(dict(kind="replay_failure", **failure, actual_evidence=failure_evidence))
            except Exception as recording_error:
                failure["evidence_recording_error"] = dict(type=type(recording_error).__name__,
                    message=str(recording_error), traceback=traceback.format_exc())
        finally:
            wall_s = (perf_counter_ns()-wall_started)/1e9
            actual_steps = round(runtime.time_s/physics_dt) if np.isfinite(runtime.time_s) else None
            summary = finite_summary(dict(schema=metadata["schema"], allocation_id=args.allocation_id, variant=args.variant,
                tracker=args.tracker, physics_dt_s=physics_dt,
                optimizer_updates=0, gpu_seconds=0, controller_ticks=len(chain_ms), physics_steps=actual_steps,
                recorded_physics_steps=steps,
                simulation_time_s=runtime.time_s, wall_time_s=wall_s, realtime_factor=runtime.time_s/wall_s,
                setup_wall_time_s=metadata["setup_wall_time_s"],
                total_wall_time_s=metadata["setup_wall_time_s"]+wall_s,
                terminal_reason=runtime.terminal_reason, failure=failure, failure_evidence=failure_evidence,
                first_instability=first_instability,
                warmup_max_qpos_error=warmup_q_error, warmup_max_qvel_error=warmup_v_error,
                warmup_comparison_samples=warmup_comparisons,
                warmup_bitwise_match_required=metadata["warmup"]["bitwise_match_required"],
                controlled_initial_root_pose=None if controlled_initial is None else controlled_initial.qpos[:7].tolist(),
                controlled_initial_sim_time_s=None if controlled_initial is None else controlled_initial.time_s,
                maximum_actual_auxiliary_forces=force_max, step_timing=distribution(chain_ms),
                including_recording_timing=distribution(loop_ms),
                joint_tracking_rmse_rad=float(np.sqrt(np.mean(joint_errors))) if joint_errors else None,
                completed_requested_duration=len(chain_ms) == count and not runtime.terminal_reason and failure is None,
                display_fps=None, display_latency_ms=None, online_admission_pass=None))
            try:
                recorder.write(dict(kind="summary", **summary))
            finally:
                (args.output/"summary.json").write_text(json.dumps(summary, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, allow_nan=False, indent=2))
    return 0 if summary["completed_requested_duration"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
