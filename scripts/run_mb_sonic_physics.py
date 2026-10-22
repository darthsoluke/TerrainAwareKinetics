"""Bounded CPU-only E1: frozen SONIC follows an independent official G1 clip."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter_ns

import mujoco
import numpy as np

from motionbricks.mb_sonic.reference import load_official_reference
from motionbricks.mb_sonic.runtime import PhysicsRecorder, SonicPhysicsRuntime, load_physics_scene
from motionbricks.mb_sonic.scene import sha256
from motionbricks.mb_sonic.sonic_policy import FrozenSonicPolicy, POLICY_DT


ENCODER_SHA256 = "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
DECODER_SHA256 = "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"
CONFIG_SHA256 = "466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b"
REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--robot-model", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    config = args.policy_dir / "observation_config.yaml"
    if sha256(config) != CONFIG_SHA256:
        raise ValueError("Use the matched pinned original observation_config.yaml")
    scene, model, robot_metadata = load_physics_scene(args.scene_manifest, args.robot_model)
    policy = FrozenSonicPolicy(args.policy_dir / "model_encoder.onnx",
                               args.policy_dir / "model_decoder.onnx", config,
                               encoder_sha256=ENCODER_SHA256, decoder_sha256=DECODER_SHA256,
                               threads=args.threads)
    source = load_official_reference(args.reference_dir, scene_id=scene.manifest.scene_id)
    count = int(np.floor(args.seconds / POLICY_DT + 1e-9))
    if count < 1 or count * POLICY_DT + 0.9 > source.trusted_until_s + 1e-9:
        raise ValueError("Requested duration plus 0.9s future horizon exceeds official source coverage")
    # Walking-specific failure, not a root-world-height test shared with crawling.
    floor_geoms = {model.geom(obj["geom_name"]).id for obj in scene.manifest.objects}
    fall_bodies = {model.body(name).id for name in ("pelvis", "torso_link")}

    def walking_failure(state):
        for contact in state.contacts:
            if (contact.wrench_contact[0] > 1 and
                    ((contact.geom_ids[0] in floor_geoms and contact.body_ids[1] in fall_bodies) or
                     (contact.geom_ids[1] in floor_geoms and contact.body_ids[0] in fall_bodies))):
                return "walking_task:pelvis_or_torso_scene_contact"
        return None

    runtime = SonicPhysicsRuntime(scene, model, policy, physics_dt=args.physics_dt,
                                  robot_model_sha256=robot_metadata["sha256"],
                                  failure_evaluator=walking_failure)
    initial_qpos = model.qpos0.copy()
    initial_qvel = np.zeros(model.nv)
    initial_qpos[:7] = source.qpos[0, :7]
    initial_qpos[runtime.qpos_indices] = source.joint_pos[0]
    initial_qvel[:6] = source.qvel[0, :6]
    initial_qvel[runtime.qvel_indices] = source.joint_vel[0]
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(schema="motionbricks.mb_sonic.physics_record.v1", allocation_id=args.allocation_id,
                    optimizer_updates=0, execution_provider="CPUExecutionProvider",
                    sonic_revision=REVISION, policy=policy.asset_metadata,
                    observation_config=dict(path=str(config.resolve()), sha256=CONFIG_SHA256),
                    robot=robot_metadata, mujoco_version=mujoco.__version__,
                    scene_manifest=asdict(scene.manifest), manifest_sha256=runtime.manifest_sha256,
                    source=source.provenance, physics_dt_s=args.physics_dt, controller_hz=50,
                    startup="explicit_source_frame0_reset_then_real_default_pose_PD_history_warmup",
                    hands="original_model_preserved; no_policy_hand_targets; passive_zero_motor_control",
                    failure_criterion="walking_pelvis_or_torso_scene_contact_over_1N; numeric_solver_failure")
    chain_ms, joint_errors, root_errors, timings = [], [], [], []
    wall_started = perf_counter_ns()
    with PhysicsRecorder(args.output / "recording.jsonl", metadata) as recorder:
        recorder.write(dict(kind="explicit_initial_reset", actual=asdict(runtime.reset(initial_qpos, initial_qvel))))
        for frame in runtime.warmup_history():
            recorder.write(dict(kind="actual_history_warmup", actual=asdict(frame)))
        source_clock_origin_s = runtime.time_s
        recorder.write(dict(kind="reference_clock", source_start_s=0.0,
                            simulation_start_s=source_clock_origin_s))
        reference = replace(source, timestamps_s=source.timestamps_s + source_clock_origin_s,
                            trusted_until_s=source.trusted_until_s + source_clock_origin_s)
        command_issued_ns = perf_counter_ns()
        for _ in range(count):
            if runtime.terminal_reason:
                break
            tick_start = perf_counter_ns()
            window = reference.sonic_window(runtime.time_s)
            raw = dict(kind="official_g1_csv_identity_adaptation", provenance=source.provenance,
                       source_times_s=window.timestamps_s-source_clock_origin_s,
                       qpos=window.qpos, qvel=window.qvel)
            tick = runtime.step(reference, raw_reference=raw,
                                issued_at_monotonic_ns=command_issued_ns, phase="official_walking_reference")
            recorder.tick(tick)
            elapsed = (perf_counter_ns() - tick_start) / 1e6
            timings.append(elapsed)
            chain_ms.append(tick.timing_ns["chain_total"] / 1e6)
            source_index = round((tick.actual.time_s-source_clock_origin_s) / POLICY_DT)
            joint_errors.append(float(np.sqrt(np.mean((tick.actual.joint_positions-source.joint_pos[source_index])**2))))
            root_errors.append(float(np.linalg.norm(tick.actual.qpos[:3]-source.qpos[source_index, :3])))
        wall_s = (perf_counter_ns() - wall_started) / 1e9
        summary = dict(schema="motionbricks.mb_sonic.physics_summary.v1", allocation_id=args.allocation_id,
                       optimizer_updates=0, gpu_seconds=0, controller_ticks=len(chain_ms),
                       physics_steps=round(runtime.time_s/args.physics_dt), simulation_time_s=runtime.time_s,
                       source_clock_origin_s=source_clock_origin_s, wall_time_s=wall_s,
                       realtime_factor=runtime.time_s/wall_s, terminal_reason=runtime.terminal_reason,
                       completed_requested_duration=len(chain_ms)==count and runtime.terminal_reason is None,
                       chain_p99_ms=float(np.percentile(chain_ms, 99)) if chain_ms else None,
                       including_recording_p99_ms=float(np.percentile(timings, 99)) if timings else None,
                       joint_tracking_rmse_rad=float(np.sqrt(np.mean(np.square(joint_errors)))) if joint_errors else None,
                       root_tracking_rmse_m=float(np.sqrt(np.mean(np.square(root_errors)))) if root_errors else None,
                       root_error_semantics="diagnostic_root_deviation; original_G1_encoder_does_not_read_world_root_translation_height",
                       display_latency_ms=None, display_fps=None,
                       scene_manifest_id=runtime.scene_id, scene_manifest_sha256=runtime.manifest_sha256)
        recorder.write(dict(kind="summary", **summary))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["completed_requested_duration"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
