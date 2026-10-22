"""Static stair authoring from source sole sweeps, never from model predictions.

For an ascending axis s, a vertex with y < tread_height - tolerance must
remain BEFORE that riser. These half-space constraints give a lower bound on
each riser's s. Footprint heights are weak estimates, not measured terrain.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.soma_skin_reference import load_official_skinner
from scripts.portable_native_mb_soma import file_sha256
from scripts.prepare_native_mb_stairs import _skin_sole_regions
from scripts.review_native_mb_stair_live_contact import scene_sole_report, scene_clearance


def source_sweep_treads(soles, ascending_xz, heights, *, tolerance_m=.01,
                        width_m=1.1, landing_length_m=1.2):
    direction = np.asarray(ascending_xz, dtype=np.float64)
    points = np.asarray(soles, dtype=np.float64).reshape(-1, 3)
    levels = np.asarray(sorted(set(heights)), dtype=np.float64)
    if (direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1.)
            or not np.isfinite(points).all() or not len(points) or not np.isfinite(levels).all()
            or not len(levels) or not np.isfinite([tolerance_m, width_m, landing_length_m]).all()
            or tolerance_m < 0 or width_m <= 0 or landing_length_m <= 0):
        raise ValueError('Need finite source geometry and a unit ascending direction')
    s = points[:, [0, 2]] @ direction
    right = np.array([-direction[1], direction[0]])
    if np.max(np.abs(points[:, [0, 2]] @ right)) > width_m / 2:
        raise ValueError('Source feet fall outside the predeclared stair width')
    starts = []
    for height in levels:
        lower = points[:, 1] < height - tolerance_m
        if not lower.any():
            raise ValueError('No lower-surface evidence for this riser')
        starts.append(float(s[lower].max() + .001))
    if any(b - a < .08 for a, b in zip(starts, starts[1:])):
        raise ValueError('Source does not support ordered treads at least 8 cm deep')
    stops = starts[1:] + [max(starts[-1] + landing_length_m, float(s.max() + .15))]
    result = []
    for index, (height, start, stop) in enumerate(zip(levels, starts, stops)):
        polygon = []
        for distance, lateral in ((start, -width_m / 2), (stop, -width_m / 2),
                                  (stop, width_m / 2), (start, width_m / 2)):
            x, z = direction * distance + right * lateral
            polygon.append([float(z * 100), float(-x * 100), float(height * 100)])
        result.append(dict(id=f'source_sweep_tread_{index:02d}', top_polygon_ue_cm=polygon,
                           ascending_start_m=start, ascending_stop_m=stop, height_m=float(height)))
    return result


def author(courses_path, source_dir, audits_dir, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory')
    bank = json.loads(Path(courses_path).read_text())
    rep = build_native_representation()
    skinner = load_official_skinner(rep.skeleton)
    surfaces = official_sole_surfaces(rep.skeleton)[0]
    report = dict(quality_approved=False, measured_terrain=False, courses_sha256=file_sha256(courses_path),
        source_code_sha256=file_sha256(__file__),
        protocol=dict(source_only=True, generated_output_used=False, height_tolerance_m=.01,
                      width_m=1.1, landing_length_m=1.2,
                      scope='Weak source-compatible authored test geometry; not recovered measured terrain'), courses=[])
    for course in bank['courses']:
        up = course['id'] == 'up'
        artifact = '00_up_start000000.npz' if up else '03_down_start000000.npz'
        source_path = Path(source_dir) / artifact
        if file_sha256(source_path) != course['source_artifact_sha256']:
            raise ValueError('Source reference hash does not match course')
        audit_path = Path(audits_dir) / ('contact_audit.json' if up else 'contact_audit_down_root_v2.json')
        audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
        if audit['course'] != course['id']:
            raise ValueError('Scene course mismatch')
        with np.load(source_path, allow_pickle=False) as data:
            features = torch.from_numpy(data['features_source'].copy())
        features[:, 0] -= course['planar_origin_m'][0]
        features[:, 2] -= course['planar_origin_m'][1]
        motion = rep.inverse(features, False, return_all=True, return_quat=False)
        soles, _, _ = _skin_sole_regions(motion['local_joint_rots'], motion['posed_joints'], skinner, surfaces)
        old_surfaces = audit['scene_top_surfaces']
        heights = sorted(set(float(p['top_polygon_ue_cm'][0][2]) / 100 for p in old_surfaces
                             if float(p['top_polygon_ue_cm'][0][2]) > 5.))
        direction = np.asarray(course['forward_xz']) * (1 if up else -1)
        treads = source_sweep_treads(np.concatenate(list(soles.values()), axis=1), direction, heights)
        ground = [p for p in old_surfaces if abs(p['top_polygon_ue_cm'][0][2]) < 1e-5]
        new_surfaces = ground + treads
        course['scene_treads'] = treads
        course['exit_platform_y_m'] = heights[-1] if up else 0.
        origins = []
        for frame in [3] + [goal['source_endpoint'] for goal in course['goals']]:
            points = np.concatenate([points[frame] for points in soles.values()])
            ue = np.stack([points[:, 2], -points[:, 0], points[:, 1]], -1) * 100
            gap = scene_clearance(ue, new_surfaces) / 100
            near = np.isfinite(gap) & (np.abs(gap) <= .02)
            if not near.any():
                raise ValueError(f'No source-supported authored frame origin at {course["id"]}:{frame}')
            origins.append(float((points[:, 1] - gap)[near].min()))
        course['input_frame_origins_y_m'] = origins
        source_scores = scene_sole_report(soles, new_surfaces)
        if source_scores['any_foot_below_top_by_3cm_frames']:
            raise ValueError(f"Authored scene still contradicts its source: {source_scores}")
        report['courses'].append(dict(course=course['id'], source_sha256=file_sha256(source_path),
            original_scene_sha256=file_sha256(audit_path), old_source=scene_sole_report(soles, old_surfaces),
            source=source_scores, scene_top_surfaces=new_surfaces))
    bank['scene_authoring'] = report['protocol']
    output.mkdir(parents=True)
    (output / 'courses.json').write_text(json.dumps(bank, separators=(',', ':'), allow_nan=False))
    report['output_courses_sha256'] = file_sha256(output / 'courses.json')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--courses', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--audits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = author(args.courses, args.source, args.audits, args.output)
    print(json.dumps([{k: row[k] for k in ('course', 'old_source', 'source')} for row in result['courses']], indent=2))
