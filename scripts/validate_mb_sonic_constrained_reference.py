"""Bounded CPU-only reference-solver checks; no neural weights or physical rollout."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
from importlib.machinery import SourceFileLoader
import json
from pathlib import Path
import sys
from time import perf_counter_ns
import traceback

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from motionbricks.mb_sonic.reference import G1_JOINT_NAMES, SomaReference, load_soma_reference


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "out/mb_sonic_e0_e3_20261018"
OLD_ADAPTER_SHA256 = "dfc6e4290335fc1dea8596864115195b27048348ca073d4a2562fea4aa0803a4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original_adapter(representation):
    path = ASSETS / "waist_marker_fix_v1/retarget_before.py.txt"
    if digest(path) != OLD_ADAPTER_SHA256:
        raise ValueError("Original adapter snapshot differs from the declared frozen source")
    name = "motionbricks.mb_sonic._constrained_validation_old_adapter"
    loader = SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module.CalibratedSomaG1Adapter.load(
        ASSETS / "retarget_calibration_v1.json",
        robot_model_path=ROOT.parent / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
        representation=representation)


def load_failed_source(path):
    with np.load(path, allow_pickle=False) as archive:
        record = json.loads(str(archive["metadata_json"]))["record"]
        values = {key: archive[value["array"]].copy()
                  if isinstance(value, dict) and "array" in value else value
                  for key, value in record["raw_source"].items()}
    values["bone_names"] = tuple(values["bone_names"])
    return SomaReference(**values)


def model_snapshot(model):
    """Fingerprint numeric model/option storage outside measured solve intervals."""
    result = {}
    for prefix, obj in (("model", model), ("option", model.opt)):
        for name in dir(obj):
            if name.startswith("_"):
                continue
            value = getattr(obj, name)
            key = prefix + "." + name
            if isinstance(value, np.ndarray):
                result[key] = (value.dtype.str, value.shape, hashlib.sha256(value.tobytes()).hexdigest())
            elif prefix == "option" and isinstance(value, (bool, int, float, np.generic)):
                result[key] = value.item() if isinstance(value, np.generic) else value
    return result


def constraints(kinematics, qpos, times):
    margin = np.minimum(qpos[:, 7:] - kinematics.limits[:, 0], kinematics.limits[:, 1] - qpos[:, 7:])
    speeds = np.abs(np.diff(qpos[:, 7:], axis=0) / np.diff(times)[:, None])
    excess = speeds - kinematics.velocity_limits
    return dict(minimum_position_margin_rad=float(margin.min()),
                position_violation_pairs=int((margin < 0).sum()),
                position_violations_gt_1e_8_rad=int((margin < -1e-8).sum()),
                maximum_velocity_ratio=float((speeds / kinematics.velocity_limits).max()),
                velocity_violation_pairs=int((excess > 0).sum()),
                velocity_violations_gt_1e_8_rad_s=int((excess > 1e-8).sum()),
                maximum_velocity_excess_rad_s=float(excess.max()),
                first_row_velocity="not defined without an external preceding sample",
                all_finite=bool(np.isfinite(qpos).all()))


def distribution(values):
    values = np.asarray(values, float)
    return dict(rms=float(np.sqrt(np.mean(values ** 2))), maximum=float(values.max()))


def quality(kinematics, nominal, solved):
    old_pos, old_rot = kinematics.fk(nominal)
    new_pos, new_rot = kinematics.fk(solved)
    result = {}
    for group, names in (
        ("feet", ("left_ankle_roll_link", "right_ankle_roll_link")),
        ("wrists", ("left_wrist_yaw_link", "right_wrist_yaw_link")),
        ("torso", ("torso_link",)),
    ):
        ids = [kinematics.body_names.index(name) for name in names]
        distance = np.linalg.norm(new_pos[:, ids] - old_pos[:, ids], axis=-1)
        angle = Rotation.from_matrix((new_rot[:, ids] @ old_rot[:, ids].swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude()
        result[group] = dict(position_displacement_m=distribution(distance),
                             orientation_tradeoff_rad=distribution(angle), bodies=list(names))
    result["joint_change_rad"] = distribution(solved[:, 7:] - nominal[:, 7:])
    result["maximum_abs_joint_change_rad"] = float(np.abs(solved[:, 7:] - nominal[:, 7:]).max())
    return result


def timings(values):
    values = np.asarray(values, float)
    return dict(repetitions=len(values), p50_ms=float(np.percentile(values, 50)),
                p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
                maximum_ms=float(values.max()))


def run_case(name, qpos, times, kinematics, solver_class, iterations, repetitions, output,
             *, packet_adapter=None, packet_source=None):
    nominal = np.array(qpos, dtype=np.float64, copy=True)
    timestamps = np.array(times, dtype=np.float64, copy=True)
    before_bytes, time_bytes = nominal.tobytes(), timestamps.tobytes()
    nominal.flags.writeable = timestamps.flags.writeable = False
    source_before = None if packet_source is None else {
        key: array.tobytes() for key, array in vars(packet_source).items() if isinstance(array, np.ndarray)}
    before_model = model_snapshot(kinematics.model)
    started = perf_counter_ns()
    solver = solver_class(kinematics, iterations=iterations)
    constructed = perf_counter_ns()
    def invoke():
        if packet_adapter is None:
            return solver.solve(nominal, timestamps)
        return packet_adapter.retarget_packet(packet_source, start_time_s=54.58,
                                              consumer_horizon_s=1.3, constraints=solver)

    value = invoke()
    cold_done = perf_counter_ns()
    packet_fields = ("timestamps_s", "qpos", "qvel", "body_pos_w", "body_quat_w",
                     "body_lin_vel_w", "body_ang_vel_w")
    if packet_adapter is None:
        solved, solver_report = value
    else:
        packet = value
        solved, solver_report = packet.qpos, dict(packet.adaptation)
        assert len(solved) == 66 and packet.timestamps_s.tobytes() == timestamps.tobytes()
        assert np.isfinite(packet.qvel).all()
    assert nominal.tobytes() == before_bytes and timestamps.tobytes() == time_bytes, "Solver mutated input"
    assert model_snapshot(kinematics.model) == before_model, "Solver mutated kinematic model/option storage"
    assert np.asarray(solved).shape == nominal.shape and np.isfinite(solved).all()
    root_exact = solved[:, :7].tobytes() == nominal[:, :7].tobytes()
    hot_ms, stable, largest_repeat_difference = [], True, 0.
    for _ in range(repetitions):
        begin = perf_counter_ns()
        value = invoke()
        hot_ms.append((perf_counter_ns() - begin) / 1e6)
        if packet_adapter is None:
            repeated, _ = value
        else:
            repeated = value.qpos
            stable &= all(getattr(value, field).tobytes() == getattr(packet, field).tobytes()
                          for field in packet_fields)
        stable &= repeated.tobytes() == solved.tobytes()
        largest_repeat_difference = max(largest_repeat_difference, float(np.abs(repeated - solved).max()))
        assert repeated[:, :7].tobytes() == nominal[:, :7].tobytes(), "Hot solve changed root"
        assert nominal.tobytes() == before_bytes and timestamps.tobytes() == time_bytes, "Hot solve mutated input"
    model_unchanged = model_snapshot(kinematics.model) == before_model
    after = constraints(kinematics, solved, timestamps)
    result = dict(name=name, frames=len(nominal), time_bounds_s=timestamps[[0, -1]].tolist(),
        constraints_before=constraints(kinematics, nominal, timestamps), constraints_after=after,
        root_bitwise_exact=root_exact, input_bitwise_unchanged=True, kinematic_model_unchanged=model_unchanged,
        nominal_output_bitwise_equal=solved.tobytes() == nominal.tobytes(),
        quality_against_original_nominal=quality(kinematics, nominal, solved), solver_report=solver_report,
        cold_constructor_ms=(constructed - started) / 1e6, cold_solve_ms=(cold_done - constructed) / 1e6,
        cold_total_ms=(cold_done - started) / 1e6, hot=timings(hot_ms), hot_bitwise_stable=stable,
        maximum_hot_difference_vs_cold=largest_repeat_difference,
        constraints_pass=bool(root_exact and model_unchanged and after["position_violation_pairs"] == 0
                              and after["velocity_violations_gt_1e_8_rad_s"] == 0),
        overall_motion_quality_approved=False)
    arrays = dict(timestamps_s=timestamps, original_qpos=nominal, solved_qpos=solved)
    if packet_adapter is not None:
        source_unchanged = all(getattr(packet_source, key).tobytes() == value
                               for key, value in source_before.items())
        expected_qdot = np.diff(solved[:, 7:], axis=0) / np.diff(timestamps)[:, None]
        expected_qdot = np.concatenate((expected_qdot, expected_qdot[-1:]))
        joint_qvel_exact = packet.joint_vel.tobytes() == expected_qdot.tobytes()
        result.update(timing_scope="full_retarget_packet_including_mapping_audits_FK_and_constraints",
                      source_arrays_bitwise_unchanged=source_unchanged,
                      joint_qvel_matches_dense_difference_bitwise=joint_qvel_exact,
                      full_reference_arrays_hot_bitwise_stable=stable,
                      packet_start_time_s=54.58, consumer_horizon_s=1.3,
                      raw_source_frames=len(packet_source.timestamps_s))
        result["constraints_pass"] &= source_unchanged and joint_qvel_exact and stable
        arrays.update({field: getattr(packet, field) for field in packet_fields if field != "qpos"})
    with (output / (name + ".npz")).open("xb") as stream:
        np.savez(stream, **arrays, report_json=np.asarray(json.dumps(result, allow_nan=False)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS / "constrained_reference_v1")
    parser.add_argument("--iterations", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--hot-repetitions", type=int, choices=range(1, 21), default=20)
    parser.add_argument("--dense-packet", action="store_true",
                        help="Also validate the current adapter's actual failed 66-frame consumer packet")
    args = parser.parse_args(argv)
    # Native representation loads skeleton geometry only, not neural model weights.
    import torch
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.mb_sonic.constrained_reference import ConstrainedReferenceSolver

    torch.set_num_threads(2)
    representation = build_native_representation()
    adapter = load_original_adapter(representation)
    failed_path = ASSETS / "wasd_v13/episode_0003_numeric/e44758570cbf01c506a803d51339bf4b843fd595a5f5568d9d06c35b64a655ac.npz"
    healthy_path = ASSETS / "mb_proposal_v1/proposal.npz"
    sources = dict(failed=load_failed_source(failed_path), healthy=load_soma_reference(healthy_path))
    cases = {}
    for name, source in sources.items():
        qpos, _ = adapter._map_anatomical(source.root_positions_m, source.world_positions_m, source.bone_names,
                                         root_rotations=source.global_rot_mats[:, 0])
        cases[name] = (qpos, source.timestamps_s)
    boundary = cases["healthy"][0].copy()
    boundary[:, 7 + G1_JOINT_NAMES.index("waist_pitch_joint")] = .65
    cases["synthetic_waist_065"] = (boundary, cases["healthy"][1])
    packet_adapter = None
    if args.dense_packet:
        from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter

        packet_adapter = CalibratedSomaG1Adapter.load(
            ASSETS / "retarget_calibration_v1.json", robot_model_path=adapter.kinematics.path,
            representation=representation)
        source = sources["failed"]
        native_q, _ = packet_adapter._map_anatomical(
            source.root_positions_m, source.world_positions_m, source.bone_names,
            root_rotations=source.global_rot_mats[:, 0])
        assert native_q.tobytes() == cases["failed"][0].tobytes(), "Current adapter changed original native mapping"
        times = 54.58 + np.arange(66) / 50
        nominal = np.empty((66, 36))
        nominal[:, :3] = np.stack([np.interp(times, source.timestamps_s, native_q[:, i]) for i in range(3)], -1)
        old_module = sys.modules[type(adapter).__module__]
        nominal[:, 3:7] = old_module._quat(Slerp(source.timestamps_s, Rotation.from_quat(
            native_q[:, [4, 5, 6, 3]]))(times).as_matrix())
        nominal[:, 7:] = np.stack([np.interp(times, source.timestamps_s, native_q[:, i]) for i in range(7, 36)], -1)
        cases["failed_dense66_packet"] = (nominal, times)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ["report.json", "report.md", *(case + ".npz" for case in cases)]:
        if (output / name).exists():
            raise FileExistsError(f"Preserve existing evidence: {output / name}")
    core_path = ROOT / "motionbricks/mb_sonic/constrained_reference.py"
    adapter_path = ROOT / "motionbricks/mb_sonic/retarget.py"
    report = dict(schema="motionbricks.mb_sonic.constrained_reference_cpu_validation.v1",
        iterations=args.iterations, hot_repetitions=args.hot_repetitions, optimizer_updates=0,
        gpu_used=False, neural_model_weights_loaded=False, physical_rollout=False,
        quality_approved=False, online_30fps_validation="NOT_RUN; offline component timing is not page/frame-rate proof",
        velocity_numerical_audit_tolerance_rad_s=1e-8,
        cross_packet_boundary_validation="NOT_RUN; each solve covers its own timestamps only",
        sources=dict(old_adapter_sha256=OLD_ADAPTER_SHA256,
                     original_calibration_sha256=digest(ASSETS / "retarget_calibration_v1.json"),
                     failed_packet_sha256=digest(failed_path), healthy_packet_sha256=digest(healthy_path),
                     core_sha256=digest(core_path), current_adapter_sha256=digest(adapter_path),
                     harness_sha256=digest(__file__), robot_xml_sha256=digest(adapter.kinematics.path)),
        packages={name: importlib.metadata.version(name) for name in ("mink", "qpsolvers", "daqp")},
        synthetic_case="Explicit unit boundary: healthy reference with waist_pitch=0.65 on every row; not generated or physically executed",
        cases=[], errors=[])
    for name, (qpos, times) in cases.items():
        try:
            dense = name == "failed_dense66_packet"
            result = run_case(name, qpos, times, packet_adapter.kinematics if dense else adapter.kinematics,
                              ConstrainedReferenceSolver, args.iterations, args.hot_repetitions, output,
                              packet_adapter=packet_adapter if dense else None,
                              packet_source=sources["failed"] if dense else None)
            report["cases"].append(result)
            print(json.dumps(dict(case=name, constraints_pass=result["constraints_pass"],
                                  hot=result["hot"], quality=result["quality_against_original_nominal"])), flush=True)
        except Exception as error:
            report["errors"].append(dict(case=name, type=type(error).__name__, message=str(error),
                                         traceback=traceback.format_exc()))
            break
    report["status"] = "CONSTRAINT_CHECK_PASS" if len(report["cases"]) == len(cases) and not report["errors"] and all(
        case["constraints_pass"] for case in report["cases"]) else "FAIL"
    report["core_source_unchanged"] = digest(core_path) == report["sources"]["core_sha256"]
    report["current_adapter_source_unchanged"] = digest(adapter_path) == report["sources"]["current_adapter_sha256"]
    if not report["core_source_unchanged"] or not report["current_adapter_source_unchanged"]:
        report["status"] = "FAIL"
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    lines = ["# Constrained Reference CPU Validation", "", "Status: " + report["status"],
             "Quality approval and online 30 fps: NOT_RUN. Model optimizer updates: 0.", "",
             "| Case | Cold Solve ms | Hot P50 ms | Hot P99 ms | Foot RMS m | Wrist RMS m | Torso RMS rad |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"]:
        quality_report = case["quality_against_original_nominal"]
        lines.append(f"| {case['name']} | {case['cold_solve_ms']:.6f} | {case['hot']['p50_ms']:.6f} | "
                     f"{case['hot']['p99_ms']:.6f} | {quality_report['feet']['position_displacement_m']['rms']:.9f} | "
                     f"{quality_report['wrists']['position_displacement_m']['rms']:.9f} | "
                     f"{quality_report['torso']['orientation_tradeoff_rad']['rms']:.9f} |")
    lines += ["", "Full constraints, root/input/model invariants, nominal/corrected arrays, errors and source hashes are preserved alongside this report.",
              "The synthetic boundary is not a generated motion. No clipping, fallback, smoothing, root correction or physical control is performed by this harness."]
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps(dict(status=report["status"], report=str(output / "report.json"))), flush=True)
    return 0 if report["status"] == "CONSTRAINT_CHECK_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
