"""Compare immutable retarget repairs using already recorded physical rollouts."""

import argparse
import json
from pathlib import Path

import numpy as np

from motionbricks.mb_sonic.reference import file_sha256
from motionbricks.mb_sonic.reference_ground import ReferenceFootGround
from motionbricks.mb_sonic.retarget import G1Kinematics
from motionbricks.mb_sonic.runtime import load_physics_scene
from scripts.isolate_mb_continuous_reference import load_fixed_reference, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.root
    baseline, _ = load_fixed_reference(root/'continuous_reference_isolation_v1')
    initial_rows = []
    rows = []
    for name in ('continuous_reference_isolation_v1', 'retarget_native_axes_v1',
                 'retarget_native_ground_v1', 'retarget_native_support_v1'):
        case = root/name
        tracking = case/('tracking_v2' if name == 'continuous_reference_isolation_v1' else 'tracking')
        reference, receipt = load_fixed_reference(case)
        review = json.loads((case/'review/review.json').read_text())
        physical = json.loads((tracking/'audit.json').read_text())
        retarget = json.loads((case/'retarget_complete.json').read_text())
        recording = tracking/'tracking.jsonl'
        metadata = json.loads(recording.read_text().splitlines()[0])
        initial = next(json.loads(line)['actual'] for line in recording.read_text().splitlines()
            if json.loads(line)['kind'] == 'explicit_initial_reset')
        initial_rows.append(initial)
        np.testing.assert_array_equal(reference.qpos[0], baseline.qpos[0])
        _, model, _ = load_physics_scene(case/'scene/scene_manifest.json', config['robot_model'],
            contact_pair_binding=config['robot_contact_pairs'], flat_ground_primitive=config['flat_ground_primitive'])
        kin = G1Kinematics(config['robot_model'])
        ids = list(kin.joint_ids.values())
        qadr = kin.model.jnt_qposadr[ids]
        selected = np.r_[np.arange(6), kin.model.jnt_dofadr[ids]]
        geometry = ReferenceFootGround(model).audit(reference.qpos, qadr, selected)
        report_path = case/'review/repair_reference_geometry.json'
        save_json(report_path, geometry)
        solve_ms = np.array([r['wall_ns']/1e6 for r in retarget['reports']])
        rows.append(dict(case=name, source_sha256=review['source_sha256'],
            reference_sha256=receipt['reference_sha256'], recording_sha256=file_sha256(recording),
            initial_reference_qpos_equal=True,
            reference_ground_minimum_m=geometry['minimum_distance_m'],
            reference_peak_penetration_m=max(0., -geometry['minimum_distance_m']),
            physical_peak_penetration_m=physical['contacts']['maximum_scene_penetration_m'],
            actual_joint_acceleration_rms_rad_s2=review['actual_joints']['acceleration_rms'],
            actual_pose_8_14Hz_energy_m2=review['actual_pose_root_relative']['energy_8_14'],
            final_planar_root_error_m=float(np.linalg.norm(review['final_reference_root_error_xyz_m'][:2])),
            actual_forward_mps=review['actual_root']['mean_forward_mps'],
            actual_lateral_range_m=review['actual_root']['lateral_range_m'],
            solver_p50_p95_p99_ms=np.percentile(solve_ms, [50, 95, 99]).tolist(),
            frozen_tracker_sha256=config['protomotions_policy']['onnx_sha256'],
            physical_audit_passed=physical['recorded_physics_checks_pass'],
            controlled_substeps=physical['controlled_substeps'],
            completed=json.loads((tracking/'summary.json').read_text())['completed']))
    for actual in initial_rows[1:]:
        for key in ('qpos', 'qvel'):
            np.testing.assert_array_equal(actual[key], initial_rows[0][key])
    if len({row['source_sha256'] for row in rows}) != 1:
        raise ValueError('Comparisons did not use the identical source')
    report = dict(schema='mb_retarget_repair_review.v1', comparisons=rows,
        identical_source=True, identical_initial_qpos_qvel=True,
        optimizer_updates=0, filters=False, actuator_residual=False,
        deployed=False, browser_30fps_tested=False,
        decision='hold: partial quality improvement; remaining drift and reference solve latency fail interactive gate',
        root_drift_next_scope='contact-compatible retarget trajectory and tracker root-observation contract',
        source_code_capture='post_execution_snapshot; baseline original source snapshot retained separately')
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
