"""Bounded official PARC->SOMA retarget preflight, never automatic training admission."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'out/vendor/kimodo'))

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from motionbricks.data.parc_soma_retarget import (
    ParcSomaRetargeter, Y_TO_Z, load_native_pair, official_parc_character,
    scale_scene, source_foot_geometry, terrain_gaps_zup, terrain_surface_distance_zup,
)
from motionbricks.data.soma_skin_reference import load_official_skinner
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


# Fixed engineering screening bounds, not force labels or a physical certificate.
GEOMETRIC_CONTACT_GAP_TOLERANCE_M = .015
BOUNDS = {'root_task_p95_m': .02, 'foot_task_p95_m': .03, 'hand_task_p95_m': .04,
          'added_angular_step_p99_deg': 5., 'added_angular_step_max_deg': 15.,
          'task_orientation_p95_deg': 10., 'angular_step_error_p99_deg': 5.,
          'angular_step_error_max_deg': 15.,
          'sole_penetration_p99_m': .03, 'sole_penetration_max_m': .08,
          'geometric_near_sole_contact_target_top_vertical_gap_p95_m': .03}
AUDIT_ONLY_BOUNDS = {'labelled_sole_surface_distance_p95_m': .03}
CRITERIA_REVISION = {
    'revision': 'source_geometric_near_sole_contact_v1',
    'previous_metric': 'labelled_sole_surface_distance_p95_m',
    'previous_metric_role': 'preserved unchanged for audit; not a sole-support criterion',
    'active_metric': 'geometric_near_sole_contact_target_top_vertical_gap_p95_m',
    'previous_bound_m': .03,
    'active_bound_m': .03,
    'threshold_relaxed': False,
    'pose_changes': False,
    'source_contact_booleans_changed': False,
    'candidate_mask_name': 'geometric_near_sole_contact',
    'candidate_mask_is_loadbearing_truth': False,
    'candidate_mask_excludes_reset_frame0': True,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def stats(value):
    value = np.asarray(value)
    value = value[np.isfinite(value)]
    return {'count': int(value.size), **{key: float(fun(value)) if value.size else None
            for key, fun in (('mean', np.mean), ('p95', lambda a: np.quantile(a, .95)),
                             ('p99', lambda a: np.quantile(a, .99)), ('max', np.max))}}


def geometric_near_sole_contact(body_contacts, source_bottom_gaps_m, source_bottom_known,
                                tolerance_m=GEOMETRIC_CONTACT_GAP_TOLERANCE_M):
    """Return a source-only geometric candidate mask, not load-bearing truth.

    All sampled source bottom corners must be on known terrain, at least one
    must be within the vertical-gap tolerance of its terrain top, and the
    corresponding source foot body must have contact. Frame zero is the reset
    frame and is deliberately excluded. Source contact labels are never edited.
    """
    contacts = np.asarray(body_contacts) > .5
    gaps = np.asarray(source_bottom_gaps_m, dtype=np.float64)
    known = np.asarray(source_bottom_known, dtype=bool)
    if contacts.ndim != 1 or gaps.ndim != 2 or gaps.shape[0] != contacts.size:
        raise ValueError('Expected body contacts [T] and source bottom gaps [T,K]')
    if known.shape != gaps.shape or gaps.shape[1] == 0:
        raise ValueError('Source bottom known mask must match nonempty [T,K] gaps')
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('tolerance_m must be finite and nonnegative')
    known = known & np.isfinite(gaps)
    fully_known = known.all(axis=-1)
    near_top = np.any(known & (np.abs(gaps) <= tolerance_m), axis=-1)
    candidate = contacts & fully_known & near_top
    candidate = candidate.copy()
    if candidate.size:
        candidate[0] = False
    return candidate


def assess_geometric_contact_pairs(target_top_gaps_m):
    gap_stats = stats(target_top_gaps_m)
    if gap_stats['count'] == 0:
        return {'assessment_status': 'not_assessed', 'target_gap_p95_m': None,
                'screen_failure': 'geometric_near_sole_contact_not_assessed_zero_eligible_frames'}
    return {'assessment_status': 'assessed', 'target_gap_p95_m': gap_stats['p95'],
            'screen_failure': None}


def angular_steps(rotations):
    relative = rotations[:-1].swapaxes(-1, -2) @ rotations[1:]
    return np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(relative.shape[:2])


def rotation_tracking_metrics(actual, goal):
    def angles(matrix):
        return np.rad2deg(Rotation.from_matrix(matrix.reshape(-1, 3, 3)).magnitude())
    residual = angles(goal.swapaxes(-1, -2) @ actual)
    actual_step = actual[:-1].swapaxes(-1, -2) @ actual[1:]
    goal_step = goal[:-1].swapaxes(-1, -2) @ goal[1:]
    step_error = angles(goal_step.swapaxes(-1, -2) @ actual_step)
    return {'task_orientation_p95_deg': stats(residual)['p95'],
            'angular_step_error_p99_deg': stats(step_error)['p99'],
            'angular_step_error_max_deg': stats(step_error)['max']}


def skin_regions(skeleton, local, positions, selected):
    skinner = load_official_skinner(skeleton)
    with torch.no_grad():
        full = skinner.skin(torch.from_numpy(local[:2]), torch.from_numpy(positions[:2]), rot_is_global=False)
    for name in ('bind_vertices', 'lbs_indices', 'lbs_weights'):
        setattr(skinner, name, getattr(skinner, name)[selected])
    with torch.no_grad():
        subset = skinner.skin(torch.from_numpy(local[:2]), torch.from_numpy(positions[:2]), rot_is_global=False)
    error = float((subset - full[:, selected]).abs().max())
    if error > 1e-6:
        raise ValueError('Selected LBS disagrees with official full-mesh LBS')
    chunks = []
    for begin in range(0, len(local), 128):
        with torch.no_grad():
            chunks.append(skinner.skin(torch.from_numpy(local[begin:begin + 128]),
                torch.from_numpy(positions[begin:begin + 128]), rot_is_global=False).numpy())
    return np.concatenate(chunks), error


def convert_item(item, args, output, character, skeleton):
    source_path = Path(item['source']['local_file'])
    if sha256(source_path) != item['source']['sha256']:
        raise ValueError('Selected source hash changed')
    pair = load_native_pair(source_path, character)
    directory = output / item['behavior']
    directory.mkdir()
    retarget = ParcSomaRetargeter(character, skeleton, args.gmr, directory / 'calibration', pair['fps'])
    rng = np.random.default_rng(2333)
    fk_error = retarget.validate_fk(Rotation.from_rotvec(rng.normal(0, .2, (30, 3))).as_matrix(),
                                   np.array([.2, 1., -.3]))
    scale = retarget.calibration['uniform_scale']
    names = list(skeleton.bone_order_names)
    reference_local = np.asarray(retarget.calibration['reference_local_rot_mats'])
    initial_root = scale * (Y_TO_Z.T @ pair['root_pos'][0])
    retarget.engine.configuration.update(retarget.qpos_from_motion(reference_local, initial_root))
    local, roots, goals, goal_rotations = [], [], [], []
    started = time.monotonic()
    count = len(pair['root_pos'])
    if args.max_frames:
        count = min(count, args.max_frames)
    for frame in range(count):
        lr, root = retarget.solve(pair['positions_zup'][frame], pair['global_rot_mats'][frame])
        if frame == 0:
            for _ in range(10):
                lr, root = retarget.solve(pair['positions_zup'][frame], pair['global_rot_mats'][frame])
        local.append(lr)
        roots.append(root)
        goals.append([retarget.engine.scaled_human_data[name][0] @ Y_TO_Z for name in names])
        goal_rotations.append([Y_TO_Z.T @ Rotation.from_quat(
            retarget.engine.scaled_human_data[name][1], scalar_first=True).as_matrix() @ Y_TO_Z for name in names])
    local, roots = np.asarray(local, dtype=np.float32), np.asarray(roots, dtype=np.float32)
    with torch.no_grad():
        rotations, positions, _ = skeleton.fk(torch.from_numpy(local), torch.from_numpy(roots))
    positions, rotations = positions.numpy(), rotations.numpy()
    goals, goal_rotations = np.asarray(goals), np.asarray(goal_rotations)
    selected = np.unique(np.concatenate(list(retarget.calibration['sole_indices'].values())))
    vertices, skin_error = skin_regions(skeleton, local, positions, selected)
    hf, origin, dx = scale_scene(pair['hf'], pair['min_point'], pair['dx'], scale)
    gaps, known = terrain_gaps_zup(vertices @ Y_TO_Z.T, hf, origin, dx)
    sdf_options={'device':getattr(args,'sdf_device','cpu'),'chunk_size':getattr(args,'sdf_chunk_size',1024)}
    surface_distances, _ = terrain_surface_distance_zup(vertices @ Y_TO_Z.T, hf, origin, dx,**sdf_options)
    errors = np.linalg.norm(positions - goals, axis=-1)
    feet, hands = [names.index(s + 'Foot') for s in ('Left', 'Right')], [names.index(s + 'Hand') for s in ('Left', 'Right')]
    added_steps = np.maximum(angular_steps(rotations) - angular_steps(goal_rotations), 0.)
    metrics = {'root_task_p95_m': stats(errors[:, 0])['p95'],
               'foot_task_p95_m': stats(errors[:, feet])['p95'],
               'hand_task_p95_m': stats(errors[:, hands])['p95'],
               'added_angular_step_p99_deg': stats(added_steps)['p99'],
               'added_angular_step_max_deg': stats(added_steps)['max'],
               'sole_penetration_p99_m': stats(np.maximum(-surface_distances, 0))['p99'],
               'sole_penetration_max_m': stats(np.maximum(-surface_distances, 0))['max'],
               'sole_terrain_known_fraction': float(known.mean())}
    metrics.update(rotation_tracking_metrics(rotations, goal_rotations))
    per_foot, contact_gaps, paired_target_gaps = {}, [], []
    pairing_masks, pairing_counts = {}, {}
    lookup = {int(vertex): i for i, vertex in enumerate(selected)}
    for side in ('Left', 'Right'):
        ids = [lookup[i] for i in retarget.calibration['sole_indices'][side]]
        body = pair['source_names'].index(side.lower() + '_foot')
        body_contact = pair['body_contacts'][:count, body] > .5
        region_gap = np.min(gaps[:, ids], axis=-1)
        region_distance = np.min(np.abs(surface_distances[:, ids]), axis=-1)
        # Preserve the original all-body-contact diagnostic exactly as an audit
        # metric. It is not a sole-support label and is no longer the paired gate.
        contact_gaps.extend(region_distance[body_contact].tolist())
        corners, _ = source_foot_geometry(character, side)
        source_points = (pair['global_rot_mats'][:count, body, None] @ corners[None, ..., None])[..., 0]
        source_points += pair['positions_zup'][:count, body, None]
        source_points *= scale
        source_gap, source_known = terrain_gaps_zup(source_points, hf, origin, dx)
        source_distance, _ = terrain_surface_distance_zup(source_points, hf, origin, dx,**sdf_options)
        mask = geometric_near_sole_contact(body_contact, source_gap, source_known)
        pairing_masks[f'geometric_near_sole_contact_{side.lower()}'] = mask
        fully_known_source = source_known.all(axis=-1) & np.isfinite(source_gap).all(axis=-1)
        source_near_top = (fully_known_source &
                           np.any(np.abs(source_gap) <= GEOMETRIC_CONTACT_GAP_TOLERANCE_M, axis=-1))
        not_reset_contact = body_contact.copy()
        post_reset_frame = np.ones(count, dtype=bool)
        reset_frame0_contact = bool(body_contact[0]) if count else False
        if not_reset_contact.size:
            not_reset_contact[0] = False
            post_reset_frame[0] = False
        contact_not_near = not_reset_contact & fully_known_source & ~source_near_top
        contact_geometry_unknown = not_reset_contact & ~fully_known_source
        near_top_without_contact = (~body_contact & post_reset_frame & fully_known_source & source_near_top)
        target_top_gap = np.min(np.abs(gaps[:, ids]), axis=-1)
        target_top_gap[~known[:, ids].all(axis=-1)] = np.nan
        paired_target_gaps.extend(target_top_gap[mask].tolist())
        source_speed = np.linalg.norm(np.diff(source_points, axis=0), axis=-1) * pair['fps']
        source_still = (body_contact[:-1, None] & body_contact[1:, None]
                        & source_known[:-1] & source_known[1:]
                        & (np.abs(source_gap[:-1]) <= .015) & (np.abs(source_gap[1:]) <= .015)
                        & (source_speed <= .1))
        # Full patch material-point speeds are diagnostic only, not a slip label:
        # a body contact label does not identify which rolling sole vertices touch.
        patch_speed = np.linalg.norm(np.diff(vertices[:, ids], axis=0), axis=-1) * pair['fps']
        persistent = body_contact[:-1] & body_contact[1:]
        pairing_counts[side] = {
            'source_body_contact_frames_including_reset': int(body_contact.sum()),
            'reset_frame0_excluded': True,
            'reset_frame0_source_contact_boolean_raw_but_unknown': reset_frame0_contact,
            'eligible_geometric_near_sole_contact_frames': int(mask.sum()),
            'body_contact_not_near_known_top_surface_frames': int(contact_not_near.sum()),
            'body_contact_with_unknown_bottom_terrain_frames': int(contact_geometry_unknown.sum()),
            'near_top_without_body_contact_frames_left_undecided': int(near_top_without_contact.sum()),
            'assessment_status': 'assessed' if mask.any() else 'not_assessed',
        }
        per_foot[side] = {'source_labelled_frames': int(body_contact.sum()),
            'geometric_near_sole_contact': {
                'eligible_frames': int(mask.sum()),
                'assessment_status': 'assessed' if mask.any() else 'not_assessed',
                'loadbearing_truth': False,
                'reset_frame0_excluded': True,
                'reset_frame0_source_contact_boolean_raw_but_unknown': reset_frame0_contact,
                'body_contact_not_near_known_top_surface_frames': int(contact_not_near.sum()),
                'body_contact_with_unknown_bottom_terrain_frames': int(contact_geometry_unknown.sum()),
                'near_top_without_body_contact_frames_left_undecided': int(near_top_without_contact.sum()),
            },
            'source_bottom_gap_on_labels_m': stats(np.min(source_gap, axis=-1)[body_contact]),
            'target_lowest_surface_gap_on_source_labels_m': stats(region_gap[body_contact]),
            'source_corner_surface_distance_on_labels_m': stats(np.min(np.abs(source_distance), axis=-1)[body_contact]),
            'target_sole_surface_distance_on_source_labels_m': stats(region_distance[body_contact]),
            'source_nearest_top_vertical_gap_on_geometric_near_sole_contact_m':
                stats(np.min(np.abs(source_gap), axis=-1)[mask]),
            'target_nearest_top_vertical_gap_on_geometric_near_sole_contact_m':
                stats(target_top_gap[mask]),
            'target_patch_penetration_m': stats(np.maximum(-surface_distances[:, ids], 0)),
            'source_quiet_corner_intervals': int(source_still.sum()),
            'target_material_point_speed_diagnostic_mps': stats(patch_speed[persistent]),
            'slip_assessment': 'not_certified; body labels do not specify corresponding touching material points'}
    metrics['labelled_sole_surface_distance_p95_m'] = stats(contact_gaps)['p95']
    pairing_assessment = assess_geometric_contact_pairs(paired_target_gaps)
    metrics['geometric_near_sole_contact_target_top_vertical_gap_p95_m'] = pairing_assessment['target_gap_p95_m']
    failures = [name for name, bound in BOUNDS.items() if metrics[name] is None or metrics[name] > bound]
    eligible_pair_count = sum(item['eligible_geometric_near_sole_contact_frames']
                              for item in pairing_counts.values())
    pairing_status = pairing_assessment['assessment_status']
    if pairing_assessment['screen_failure']:
        failures.append(pairing_assessment['screen_failure'])
    if not known.all():
        failures.append('sole_vertices_outside_known_terrain')
    # A screened pair still requires full collision/contact and motion-encoding checks.
    report = {'behavior': item['behavior'], 'execution_status': 'processed',
              'screen_status': 'pass' if not failures else 'fail', 'screen_failures': failures,
              'training_approved': False, 'source': item['source'], 'source_group': item['source_group'],
              'frames': count, 'source_frames': len(pair['root_pos']), 'fps': pair['fps'],
              'metrics': metrics, 'per_foot': per_foot, 'bounds': BOUNDS,
              'audit_only_bounds': AUDIT_ONLY_BOUNDS, 'criteria_revision': CRITERIA_REVISION,
              'contact_pairing': {
                  'mask_name': 'geometric_near_sole_contact',
                  'assessment_status': pairing_status,
                  'eligible_foot_frames': eligible_pair_count,
                  'candidate_mask_semantics': (
                      'source-only geometric candidate: source foot body contact, all sampled '
                      'bottom corners on known terrain, and at least one corner within 0.015 m '
                      'vertical gap of the terrain top; reset frame 0 excluded. This is NOT '
                      'load-bearing truth. Nonmatching body contacts and near-top frames without '
                      'body contact remain reported and are not promoted.'),
                  'by_foot': pairing_counts,
              },
              'source_contact_semantics': {
                  'source_boolean_preserved': True,
                  'meaning': ('simulation-exported body-level contact boolean; it does not '
                              'identify a sole support region or prove load-bearing contact, '
                              'and can include side/other body contacts'),
                  'reset_frame0': 'contact state is stale/unknown at reset and excluded from the new pairing mask',
                  'legacy_distance_metric': ('labelled_sole_surface_distance_p95_m is retained '
                                             'unchanged as an all-body-contact audit diagnostic, '
                                             'not sole support'),
              },
              'fk_max_error': fk_error, 'selected_lbs_max_error_m': skin_error,
              'calibration': retarget.calibration, 'elapsed_seconds': time.monotonic() - started,
              'not_certified': ['hand surface contact', 'whole-body collision', 'dynamics',
                                'material-point slip', 'MB tokenizer reconstruction'],
              'geometry_measurement': 'official PARC normal-box exterior and inverted-box interior distance; fixed selected SOMA sole vertices',
              'geometry_backend':sdf_options,
              'terrain_transform': 'same uniform world scale as source motion; no per-frame fit',
              'motion_postprocessing': 'none; offline upstream GMR retarget only'}
    shard = directory / 'retarget.npz'
    np.savez_compressed(shard, source_positions_zup=pair['positions_zup'][:count],
        source_rotations_xyzw=pair['rotations_xyzw'][:count], source_names=np.asarray(pair['source_names']),
        source_contacts=pair['body_contacts'][:count], target_positions_yup=positions,
        target_names=np.asarray(names), target_parents=skeleton.joint_parents.numpy(),
        target_sole_vertices_yup=vertices, sole_vertex_ids=selected,
        local_rot_mats=local, root_positions=roots, target_global_rot_mats=rotations,
        task_positions_yup=goals, task_global_rot_mats=goal_rotations,
        **pairing_masks,
        hf=hf, min_point=origin, dx=dx, uniform_scale=scale, source_fps=pair['fps'])
    report['shard'] = str(shard.resolve())
    report['shard_sha256'] = sha256(shard)
    dump(directory / 'report.json', report)
    print(json.dumps({'behavior': item['behavior'], 'status': report['screen_status'],
                      'failures': failures, 'metrics': metrics}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parc', type=Path, default=ROOT / 'out/vendor/PARC')
    parser.add_argument('--gmr', type=Path, default=ROOT / 'out/vendor/GMR')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--behaviors', nargs='+')
    parser.add_argument('--sdf-device',default='cpu')
    parser.add_argument('--sdf-chunk-size',type=int,default=1024)
    args = parser.parse_args()
    if args.max_frames < 0 or args.max_frames == 1:
        raise ValueError('max-frames must be zero (all) or at least two')
    torch.set_num_threads(1)
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    items = selection['selected_items']
    if args.behaviors:
        items = [i for i in items if i['behavior'] in args.behaviors]
        if {i['behavior'] for i in items} != set(args.behaviors):
            raise ValueError('Requested behavior absent from selection')
    if not items:
        raise ValueError('Empty source selection')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'training_approved': False, 'bounds': BOUNDS,
              'audit_only_bounds': AUDIT_ONLY_BOUNDS, 'criteria_revision': CRITERIA_REVISION,
              'source_contact_semantics': {
                  'meaning': ('simulation-exported body-level contact booleans, not sole-support '
                              'or load-bearing truth; side/other body contacts can be present'),
                  'reset_frame0': 'stale/unknown contact state; excluded from geometric pairing',
                  'legacy_distance_metric': ('all-body-contact sole-distance diagnostic is retained '
                                             'unchanged for audit and is not called sole support'),
              },
              'selection_sha256': sha256(args.selection), 'items': [],
              'dependencies': {name: importlib.metadata.version(name) for name in
                               ('mink', 'mujoco', 'qpsolvers', 'daqp', 'torch', 'numpy')},
              'code_sha256': {str(p): sha256(p) for p in (Path(__file__),
                  ROOT / 'motionbricks/data/parc_soma_retarget.py', ROOT / 'motionbricks/data/lafan_soma_retarget.py')}}
    report_path = args.output / 'report.json'
    dump(report_path, report)
    char, soma = official_parc_character(args.parc), SOMASkeleton30()
    for item in items:
        try:
            report['items'].append(convert_item(item, args, args.output, char, soma))
        except Exception as error:
            report['items'].append({'behavior': item['behavior'], 'execution_status': 'failed',
                                    'error': f'{type(error).__name__}: {error}',
                                    'traceback': traceback.format_exc(), 'training_approved': False})
        dump(report_path, report)
    failed_execution = any(i['execution_status'] == 'failed' for i in report['items'])
    report['status'] = 'execution_failed' if failed_execution else 'preflight_complete_not_training_approved'
    dump(report_path, report)
    if failed_execution:
        raise SystemExit('One or more conversions failed; see report.json')


if __name__ == '__main__':
    main()
