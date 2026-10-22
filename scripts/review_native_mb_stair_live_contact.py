"""Replay recorded live INPUTS through MB and inspect unmodified SOMA soles.

This is an offline diagnostic, not animation playback or a terrain correction.
Unknown footprint coverage remains unknown. Surface distances do not infer force.
"""

import argparse
from contextlib import nullcontext
import json
import time
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_stair_live import NativeStairLiveBackend
from motionbricks.data.native_mb_live import NativeSomaLiveBackend
from motionbricks.data.native_mb_frame import NativePlatformLocomotion, NativeAnchoredIdle, load_authored_idle_reference
from motionbricks.data.native_mb_smart_object import NativeSmartObject
from motionbricks.data.native_mb_root_conditioning import decoder_root_interpolation
from motionbricks.data.soma_replay import skeleton_manifest
from motionbricks.data.soma_skin_reference import load_official_skinner
from motionbricks.data.stair_support import bounded_membership, query_support
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from scripts.portable_native_mb_soma import file_sha256, load_bundle
from scripts.prepare_native_mb_stairs import _skin_sole_regions


def scene_clearance(points_ue_cm, surfaces):
    """Signed distance to the topmost authored horizontal block, NOT weak labels."""
    points = np.asarray(points_ue_cm, dtype=np.float64)
    height = np.full(points.shape[:-1], -np.inf)
    for surface in surfaces:
        polygon = np.asarray(surface['top_polygon_ue_cm'], dtype=np.float64)
        if polygon.shape != (4, 3) or not np.isfinite(polygon).all() or np.ptp(polygon[:, 2]) > 1e-4:
            raise ValueError('Only finite horizontal rectangular course tops are supported')
        mask = bounded_membership(points[..., :2].reshape(-1, 2), polygon[:, :2]).reshape(height.shape)
        height[mask] = np.maximum(height[mask], polygon[0, 2])
    return np.where(np.isfinite(height), points[..., 2] - height, np.nan)


def summarize_ue_probe(input_path, audit_path, output_path):
    inputs = json.loads(Path(input_path).read_text())
    audit = json.loads(Path(audit_path).read_text(encoding='utf-8-sig'))
    if inputs['source_id'] != audit['source_id'] or len(inputs['frames']) != len(audit['frames']):
        raise ValueError('UE contact audit does not correspond to the source fixture')
    rows = []
    for source, target in zip(inputs['frames'], audit['frames']):
        if (source['kind'], source['frame_index']) != (target['kind'], target['frame_index']):
            raise ValueError('Retarget audit frame alignment mismatch')
        row = {key: source[key] for key in ('kind', 'frame_index')}
        for name in ('source', 'manny'):
            closest = []
            for side in ('left', 'right'):
                if name == 'source':
                    xyz = np.asarray(source[f'sole_{side}_m'])
                    points = np.stack([xyz[:, 2], -xyz[:, 0], xyz[:, 1]], -1) * 100
                else:
                    points = np.asarray(target[f'target_sole_{side}_cm'])
                gap = scene_clearance(points, audit['scene_top_surfaces'])
                value = float(np.nanmin(gap)) if np.isfinite(gap).any() else None
                row[f'{name}_{side}_lowest_gap_cm'] = value
                closest.append(value)
            row[f'{name}_bilateral_clearance_cm'] = min(closest) if all(v is not None for v in closest) else None
        rows.append(row)
    report = dict(course=audit['course'], quality_approved=False,
                  meaning='Distances to actual illustrative UE blocks, not measured training terrain or load-bearing contacts.',
                  summary={}, frames=rows)
    for kind in ('authored', 'generated'):
        report['summary'][kind] = {}
        for name in ('source', 'manny'):
            values = np.array([np.nan if r[f'{name}_bilateral_clearance_cm'] is None else r[f'{name}_bilateral_clearance_cm']
                               for r in rows if r['kind'] == kind])
            finite = np.isfinite(values)
            report['summary'][kind][name] = dict(bilateral_clearance_cm=statistics(values),
                both_feet_above_5cm_samples=int((values > 5).sum()),
                any_foot_penetration_over_3cm_samples=int((values < -3).sum()),
                unknown_samples=int((~finite).sum()), sample_count=len(values))
    Path(output_path).write_text(json.dumps(report, indent=2, allow_nan=False))
    return report['summary']


def export_ue_probe(replay_path, courses_path, output_path, course_id='up'):
    """Independent UE retarget audit fixture; never used as live server input."""
    from scipy.spatial.transform import Rotation
    course = next(c for c in json.loads(Path(courses_path).read_text())['courses'] if c['id'] == course_id)
    rep = build_native_representation()
    with np.load(replay_path, allow_pickle=False) as data:
        sampled = np.arange(0, len(data['features']), 4)
        features = np.concatenate([np.asarray(course['initial_history'] + [g['features'][-1] for g in course['goals']]),
                                   data['features'][sampled]], axis=0).astype(np.float32)
        left = np.concatenate([data['authored_left'], data['left'][sampled]])
        right = np.concatenate([data['authored_right'], data['right'][sampled]])
    motion = rep.inverse(torch.from_numpy(features)[None], False, return_all=True, return_quat=False)
    positions = motion['posed_joints'][0].numpy()
    quats = Rotation.from_matrix(motion['global_joint_rots'][0].numpy().reshape(-1, 3, 3)).as_quat().reshape(-1, 30, 4)
    authored_count = 4 + len(course['goals'])
    frames = [dict(frame_index=int(i if i < authored_count else sampled[i - authored_count]),
                   kind='authored' if i < authored_count else 'generated', positions_m=positions[i].tolist(),
                   rotations_xyzw=quats[i].tolist(), sole_left_m=left[i].tolist(), sole_right_m=right[i].tolist())
              for i in range(len(features))]
    Path(output_path).write_text(json.dumps(dict(source_id=skeleton_manifest()['source_id'],
                                               joint_names=rep.skeleton.bone_order_names, frames=frames),
                                          separators=(',', ':'), allow_nan=False))
    return len(frames)


def statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=int(values.size), mean=float(values.mean()) if values.size else None,
                p95=float(np.quantile(values, .95)) if values.size else None,
                max=float(values.max()) if values.size else None)


def scene_sole_report(soles, surfaces):
    closest = []
    for points in soles.values():
        ue = np.stack([points[..., 2], -points[..., 0], points[..., 1]], -1) * 100
        gaps = scene_clearance(ue, surfaces)
        known = np.isfinite(gaps).any(-1)
        values = np.full(len(points), np.nan)
        values[known] = np.nanmin(gaps[known], axis=-1)
        closest.append(values)
    values = np.stack(closest, -1)
    known = np.isfinite(values).all(-1)
    return dict(frames=len(values), both_feet_known_frames=int(known.sum()),
        bilateral_clearance_above_5cm_frames=int((known & (values > 5).all(-1)).sum()),
        any_foot_below_top_by_3cm_frames=int((known & (values < -3).any(-1)).sum()),
        interpretation='Actual scene top geometry only; no force or stance classification.')


def audit_authored_source(courses_path, source_dir, audit_path, output_path):
    """Check static scene compatibility before attributing its collisions to MB."""
    from types import SimpleNamespace
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError('Do not overwrite prior analysis')
    audit = json.loads(Path(audit_path).read_text(encoding='utf-8-sig'))
    course = next(c for c in json.loads(Path(courses_path).read_text())['courses'] if c['id'] == audit['course'])
    name = '00_up_start000000.npz' if course['id'] == 'up' else '03_down_start000000.npz'
    source_path = Path(source_dir) / name
    if file_sha256(source_path) != course['source_artifact_sha256']:
        raise ValueError('Source reference does not match the authored course')
    with np.load(source_path, allow_pickle=False) as data:
        features = torch.from_numpy(data['features_source'].copy())[None]
    features[..., 0] -= course['planar_origin_m'][0]
    features[..., 2] -= course['planar_origin_m'][1]
    rep = build_native_representation()
    skinner = load_official_skinner(rep.skeleton)
    _, soles, _ = inspect(features, SimpleNamespace(global_motion_rep=rep), course, skinner,
                          official_sole_surfaces(rep.skeleton)[0])
    report = dict(course=course['id'], source_sha256=file_sha256(source_path),
        course_sha256=file_sha256(courses_path), scene_sha256=file_sha256(audit_path),
        actual_scene=scene_sole_report(soles, audit['scene_top_surfaces']),
        interpretation='Original source motion versus independently authored static blocks; no model output is used.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def endpoint_report(features, joints, authored_joints, plans, foot_indices):
    rows = []
    for request, block in enumerate(plans):
        if len(block['plans']) > 1:
            raise ValueError('Endpoint ledger expects at most one new phase per eight-frame request')
        for plan in block['plans']:
            end = (request + 1) * 8 - 1 + block['remaining_frames']
            if plan['state'] not in ('advance', 'backtrack') or end >= len(features):
                continue
            target_root = np.asarray(plan['goal_root_m'])
            reference = authored_joints[4 + plan['goal_index'] if plan['goal_index'] >= 0 else 3]
            goal = reference + target_root - reference[0]
            rows.append(dict(state=plan['state'], goal_index=plan['goal_index'], endpoint_frame=end,
                root_error_m=float(np.linalg.norm(features[end, :3] - target_root)),
                max_foot_joint_error_m=float(np.linalg.norm(joints[end, foot_indices] - goal[foot_indices], axis=-1).max())))
    return dict(rows=rows, summary={state: {
        key: statistics([row[key] for row in rows if row['state'] == state])
        for key in ('root_error_m', 'max_foot_joint_error_m')}
        for state in ('advance', 'backtrack')})


def published_phase_states(plans, frame_count):
    """Recover exact phase boundaries, including a boundary inside a request."""
    states = np.full(frame_count, '', dtype='<U16')
    previous_end = -1
    for request, block in enumerate(plans):
        if len(block['plans']) > 1:
            raise ValueError('Phase ledger expects at most one new phase per request')
        for plan in block['plans']:
            end = (request + 1) * 8 - 1 + block['remaining_frames']
            start = end - (plan['frames'] - 4) + 1
            # A changed command may discard an UNPLAYED hold tail. It cannot
            # replace frames already published before the current request.
            if (start < request * 8 or start > previous_end + 1 or end < start or
                    (start != previous_end + 1 and start != request * 8)):
                raise ValueError('Generated phase ledger has gaps or rewrites played frames')
            states[start:min(end + 1, frame_count)] = plan['state']
            previous_end = end
    if (states == '').any():
        raise ValueError('Some published frames have no generating phase')
    return states


def summarize_saved_replay(replay_dir, audit_path, output_path):
    """Apply a later UE scene audit without rewriting a frozen inference run."""
    replay_dir, audit_path, output_path = Path(replay_dir), Path(audit_path), Path(output_path)
    if output_path.exists():
        raise ValueError('Do not overwrite prior analysis')
    report = json.loads((replay_dir / 'report.json').read_text())
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if audit['course'] != report['course']:
        raise ValueError('Scene geometry belongs to a different course')
    with np.load(replay_dir / 'replay.npz', allow_pickle=False) as data:
        states = published_phase_states(report['plans'], len(data['features']))
        soles = {side: data[side] for side in ('left', 'right')}
        summary = dict(quality_approved=False, course=report['course'],
            report_sha256=file_sha256(replay_dir / 'report.json'),
            replay_sha256=file_sha256(replay_dir / 'replay.npz'), scene_audit_sha256=file_sha256(audit_path),
            actual_scene=scene_sole_report(soles, audit['scene_top_surfaces']),
            by_phase={state: scene_sole_report({s: p[states == state] for s, p in soles.items()},
                audit['scene_top_surfaces']) for state in np.unique(states)})
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def inspect(features, inference, course, skinner, surfaces):
    motion = inference.global_motion_rep.inverse(features, False, return_all=True, return_quat=False)
    soles, _, parity = _skin_sole_regions(motion['local_joint_rots'][0].cpu(),
                                        motion['posed_joints'][0].cpu(), skinner, surfaces)
    patches = []
    for patch in course['support_patches']:
        xyz = np.asarray(patch['polygon_ue_cm']) / 100
        patches.append(dict(id=patch['id'], status='weak_support', confidence=patch['confidence'],
                            height_m=float(xyz[:, 2].mean()), polygon_xz_m=np.stack([-xyz[:, 1], xyz[:, 0]], -1).tolist()))
    result, nearest = {}, []
    for side, points in soles.items():
        heights, _, ids = query_support(points[..., [0, 2]], patches)
        gaps = np.where(ids >= 0, points[..., 1] - heights, np.nan)
        known = np.isfinite(gaps).any(-1)
        closest = np.full(len(points), np.nan)
        closest[known] = np.nanmin(gaps[known], axis=-1)
        nearest.append(closest)
        result[side] = dict(known_frame_fraction=float(known.mean()),
                            lowest_known_vertex_gap_m=statistics(closest))
    both = np.stack(nearest, -1)
    fully_known = np.isfinite(both).all(-1)
    result['both_feet_above_5cm_frames'] = int((fully_known & (both > .05).all(-1)).sum())
    result['both_feet_have_known_patch_frames'] = int(fully_known.sum())
    result['frames'] = int(features.shape[1])
    result['selected_lbs_parity_m'] = parity
    return result, soles, motion['posed_joints'][0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--courses', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--session')
    parser.add_argument('--case', choices=('recorded', 'traverse-hold'), default='recorded')
    parser.add_argument('--course', choices=('up', 'down'), default='up')
    parser.add_argument('--scene-audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--context-keyframes', type=int, choices=(1, 4), default=1,
                        help='Diagnostic input-mask comparison; does not modify the live backend')
    parser.add_argument('--pose-iterations', type=int, default=1,
                        help='Original MB iterative token inference, without output filtering')
    parser.add_argument('--exit-platform-y', type=float,
                        help='Explicit scene landing height for native idle handoff, never fitted to predictions')
    parser.add_argument('--exit-bundle', type=Path,
                        help='Explicit diagnostic of flat-skill retention; records both backbone identities')
    parser.add_argument('--anchored-exit-idle', action='store_true',
                        help='Author a real idle reference on the endpoint platform; never edit generated output')
    parser.add_argument('--exit-reference', type=Path)
    parser.add_argument('--authored-input-frames', action='store_true',
                        help='Use explicit source-authored scene origins, held fixed through each plan')
    parser.add_argument('--decoder-root-interpolation', action='store_true',
                        help='Experimental paper-inspired decoder INPUT conditioning, never output correction')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory; previous experiments must remain immutable')
    courses = json.loads(args.courses.read_text())['courses']
    if args.case == 'recorded':
        if args.log is None:
            parser.error('--log is required for recorded input replay')
        entries = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
        entries = [e for e in entries if e.get('event') == 'plan']
        if not entries:
            raise ValueError('No recorded plans')
        session = args.session or entries[-1]['session']
        entries = [e for e in entries if e['session'] == session]
    else:
        course = next(c for c in courses if c['id'] == args.course)
        x, z = course['forward_xz']
        session = f'fixed-input-{args.course}-traverse45-hold30'
        entries = [dict(sequence=i+1, course=args.course, velocity=[z*speed, -x*speed, 0.])
                   for i, speed in enumerate([1.2] * 45 + [0.] * 30)]
    if not entries or [e['sequence'] for e in entries] != list(range(1, len(entries) + 1)):
        raise ValueError('Need a complete contiguous recorded session from its seed')
    course = next(c for c in courses if c['id'] == entries[0]['course'])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_bundle(args.bundle, args.device)
    torch.manual_seed(2333)
    primitive = NativeSmartObject(inference, context_keyframes=args.context_keyframes,
                                  pose_iterations=args.pose_iterations)
    if args.exit_bundle and args.exit_platform_y is None:
        parser.error('--exit-bundle requires --exit-platform-y')
    exit_infer, exit_bundle = load_bundle(args.exit_bundle, args.device) if args.exit_bundle else (inference, bundle)
    exit_locomotion = None if args.exit_platform_y is None else NativePlatformLocomotion(
        NativeSomaLiveBackend(exit_infer, exit_bundle, reuse_motion_buffer=True),
        inference.global_motion_rep, args.exit_platform_y)
    if args.anchored_exit_idle:
        if args.exit_platform_y is None:
            parser.error('--anchored-exit-idle requires --exit-platform-y')
        endpoint = course['goals'][-1]['features'][-1]
        reference = (load_authored_idle_reference(args.exit_reference, exit_infer.global_motion_rep, args.device)
                     if args.exit_reference else exit_bundle['reference_clips']['idle'][:4].to(args.device)[None])
        exit_locomotion = NativeAnchoredIdle(NativeSmartObject(exit_infer), exit_infer.global_motion_rep, reference,
            [endpoint[0], endpoint[2]], float(np.arctan2(*course['forward_xz'])), args.exit_platform_y)
    backend = NativeStairLiveBackend(inference, bundle, course, primitive=primitive,
                                    exit_locomotion=exit_locomotion, use_authored_frames=args.authored_input_frames)
    pieces, errors, plans, generation_times = [], [], [], []
    root_policy = decoder_root_interpolation(inference) if args.decoder_root_interpolation else nullcontext([])
    with torch.inference_mode(), root_policy as root_trace:
        for entry in entries:
            velocity = entry['velocity']
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = backend.predict_plan([[-velocity[1], velocity[0]]], [0.])
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            if backend.last_generation['neural_generation']:
                generation_times.append(time.perf_counter() - start)
            prediction = result['planned_motion']
            if 'last_root' in entry:
                errors.append(float(np.max(np.abs(prediction[0, -1, :3].cpu().numpy() - entry['last_root']))))
            pieces.append(prediction.clone())
            plans.append(backend.last_generation)
            backend.history = prediction[:, -4:].clone()
        generated = torch.cat(pieces, 1)
        skeleton = SOMASkeleton30()
        skinner = load_official_skinner(skeleton)
        surfaces = official_sole_surfaces(skeleton)[0]
        metrics, soles, joints = inspect(generated, inference, course, skinner, surfaces)
        source_goals = torch.tensor(course['initial_history'] + [g['features'][-1] for g in course['goals']],
                                    dtype=torch.float32, device=args.device)[None]
        source_metrics, source_soles, source_joints = inspect(source_goals, inference, course, skinner, surfaces)
    states = published_phase_states(plans, int(generated.shape[1]))
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(session=session, course=course['id'], input_requests=len(entries),
                  context_keyframes=args.context_keyframes, target_keyframes=1,
                  pose_iterations=args.pose_iterations,
                  exit_platform_y_m=args.exit_platform_y,
                  anchored_exit_idle=args.anchored_exit_idle,
                  exit_reference_sha256=file_sha256(args.exit_reference) if args.exit_reference else None,
                  authored_input_frames=args.authored_input_frames,
                  exit_bundle_sha256=file_sha256(args.exit_bundle) if args.exit_bundle else None,
                  generation_seconds_including_warmup=statistics(generation_times),
                  decoder_root_interpolation=args.decoder_root_interpolation, root_condition_trace=root_trace,
                  replay_last_root_max_error_m=max(errors) if errors else None, model=metrics, authored_inputs=source_metrics,
                  quality_approved=False, interpretation='Weak spatial footprint distances, not contact or stance truth.',
                  model_metadata=bundle['metadata'], bundle_sha256=file_sha256(args.bundle),
                  courses_sha256=file_sha256(args.courses), log_sha256=file_sha256(args.log) if args.log else None,
                  recorded_root_comparison='Difference from the original logged model; zero is expected only for its exact replay.',
                  plans=plans)
    report['endpoint_reach'] = endpoint_report(generated[0].cpu().numpy(), joints, source_joints, plans,
        [skeleton.bone_order_names.index(n) for n in ('LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase')])
    if args.scene_audit:
        audit = json.loads(args.scene_audit.read_text(encoding='utf-8-sig'))
        if audit['course'] != course['id']:
            raise ValueError('Scene geometry belongs to a different course')
        report['actual_scene'] = scene_sole_report(soles, audit['scene_top_surfaces'])
        report['actual_scene_by_phase'] = {state: scene_sole_report(
            {side: vertices[states == state] for side, vertices in soles.items()}, audit['scene_top_surfaces'])
            for state in np.unique(states)}
    sources = [Path(__file__)] + [Path(__file__).parents[1] / 'motionbricks/data' / f'{name}.py'
        for name in ('native_mb_root_conditioning', 'native_mb_frame', 'native_mb_smart_object',
                     'native_mb_locomotion', 'native_mb_live', 'native_mb_stair_live')]
    report['source_sha256'] = {p.name: file_sha256(p) for p in sources}
    np.savez_compressed(args.output / 'replay.npz', features=generated[0].cpu().numpy(), joints=joints,
                        left=soles['left'], right=soles['right'], states=np.asarray(states),
                        authored_joints=source_joints, authored_left=source_soles['left'], authored_right=source_soles['right'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ('plans', 'root_condition_trace', 'model_metadata', 'endpoint_reach')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
