"""Corpus selection and duration accounting, independent of PARC's objectives."""

import hashlib
from pathlib import Path

import numpy as np


FAMILIES = ('stairs', 'platforms', 'blocks', 'gaps', 'rough', 'mixed')
POLICY = {
    'version': 'parc-admitted-hours-v1',
    'minimum_train_seconds': 3600., 'target_train_seconds': 7200.,
    'minimum_train_seconds_per_family': 300.,
    'minimum_train_layouts_per_family': 6,
    'minimum_holdout_seconds_per_family': 30.,
    'minimum_holdout_layouts_per_family': 2,
    'max_credited_seconds_per_geometry': 120.,
    'retained_per_layout': 16, 'candidate_pose_min_rms_m': .02,
    'candidate_batch': 128, 'max_attempts_per_layout': 3,
    'scope': 'new complete physics recordings with SOMA screen pass only; base corpus excluded',
    'duration': '(frames - 1) / measured fps; no resampling, looping or augmentation credit',
}


def candidate_descriptor(candidate):
    with np.load(Path(candidate['motion']).parent/'fk.npz', allow_pickle=False) as data:
        body = np.asarray(data['body_world_pos'], dtype=np.float64)
        root = np.asarray(data['root_pos'], dtype=np.float64)
    if len(root) < 2 or body.shape != (len(root), 15, 3) or not np.isfinite(body).all():
        raise ValueError('Invalid official candidate FK descriptor')
    # Same scene and route: retain body coordination differences, not spawn offset.
    pose = body - root[:, None]
    # Keep speed/cadence differences when phase-normalizing candidate lengths.
    # Convert velocities to displacement over fixed intervals for the descriptor.
    root_velocity = np.gradient(root, 1./30., axis=0) * .25
    pose_velocity = np.gradient(pose, 1./30., axis=0) * .1
    features = np.concatenate((pose.reshape(len(root), -1),
                               pose_velocity.reshape(len(root), -1),root_velocity),axis=-1)
    times = np.linspace(0, len(pose)-1, 32)
    return np.stack([np.interp(times, np.arange(len(pose)), column)
                     for column in features.T], axis=-1).reshape(-1)


def diverse_selection(candidates, limit, min_rms_m=.02, descriptor=candidate_descriptor):
    """Quality gates run first; farthest-point selection only ranks passing clips."""
    unique = {row['sha256']: row for row in sorted(candidates, key=lambda r: -r['total_loss'])}
    pool = sorted(unique.values(), key=lambda r: (r['total_loss'], r['name']))
    if not pool:
        return []
    features = np.stack([descriptor(row) for row in pool])
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite selection descriptor')
    chosen = [0]
    nearest = np.full(len(pool), np.inf)
    while len(chosen) < min(limit, len(pool)):
        distances = np.sqrt(np.mean((features - features[chosen[-1]])**2, axis=-1))
        nearest = np.minimum(nearest, distances)
        nearest[chosen] = -1
        index = int(np.argmax(nearest))
        if nearest[index] < min_rms_m:
            break
        chosen.append(index)
    return [pool[index] for index in chosen]


def motion_identity(shard):
    with np.load(shard, allow_pickle=False) as data:
        root = np.asarray(data['root_positions'], dtype=np.float64).copy()
        local = np.asarray(data['local_rot_mats'], dtype=np.float64)
        fps = float(data['source_fps'])
    if len(root) < 65 or local.shape[0] != len(root) or not np.isfinite(root).all() or not np.isfinite(local).all() or fps != 30:
        raise ValueError('Invalid admitted motion timing or geometry')
    root[:, (0, 2)] -= root[0, (0, 2)]
    digest = hashlib.sha256()
    for array in (np.rint(root*1000).astype('<i8'), np.rint(local*10000).astype('<i8')):
        digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest(), (len(root)-1)/fps


def summarize(rows, policy=POLICY):
    buckets = {split: {family: {'seconds': 0., 'credited_seconds': 0., 'clips': 0, 'layouts': 0}
                      for family in FAMILIES} for split in ('train', 'validation', 'test')}
    seen, geometry_owners, groups, duplicates = set(), {}, {}, []
    for row in rows:
        if row.get('disposition') != 'admitted':
            continue
        if row.get('whole_motion') is not True or row.get('origin') != 'parc_generated':
            raise ValueError('Campaign credit requires complete new generated physics')
        family, split = row['terrain_family'], row['split']
        identity = row['motion_content_sha256']
        geometry = row['terrain_content_sha256']
        if geometry in geometry_owners and geometry_owners[geometry] != split:
            raise ValueError('Terrain geometry crossed immutable splits')
        geometry_owners[geometry] = split
        if identity in seen:
            duplicates.append(row['source_id'])
            continue
        seen.add(identity)
        seconds = float(row['admitted_seconds'])
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid duration')
        bucket = buckets[split][family]
        bucket['seconds'] += seconds
        bucket['clips'] += 1
        # Geometry can have multiple routes or seed names. It gets one cap in total.
        key = (split, geometry)
        previous = groups.get(key, 0.)
        credit = min(seconds, max(0., policy['max_credited_seconds_per_geometry'] - previous))
        bucket['credited_seconds'] += credit
        bucket['layouts'] += int(key not in groups)
        groups[key] = previous + seconds
    deficits = []
    for split, families in buckets.items():
        for family, value in families.items():
            seconds = policy['minimum_train_seconds_per_family'] if split == 'train' else policy['minimum_holdout_seconds_per_family']
            layouts = policy['minimum_train_layouts_per_family'] if split == 'train' else policy['minimum_holdout_layouts_per_family']
            if value['credited_seconds'] < seconds or value['layouts'] < layouts:
                deficits.append({'family': family, 'split': split,
                    'missing_seconds': max(0., seconds-value['credited_seconds']),
                    'missing_layouts': max(0, layouts-value['layouts'])})
    train = sum(row['credited_seconds'] for row in buckets['train'].values())
    return {'by_split_family': buckets, 'unique_admitted_clips': len(seen),
        'duplicate_source_ids': duplicates, 'credited_train_seconds': train,
        'admitted_seconds': sum(row['seconds'] for families in buckets.values() for row in families.values()),
        'coverage_deficits': deficits,
        'minimum_met': train >= policy['minimum_train_seconds'] and not deficits,
        'target_met': train >= policy['target_train_seconds'] and not deficits,
        'skill_success_certified': False}


def verify_training_gate(receipt, policy=POLICY):
    coverage = receipt.get('campaign_coverage')
    if receipt.get('campaign_policy') != policy or not isinstance(coverage, dict):
        raise ValueError('Training requires the versioned generated-duration gate')
    if not coverage.get('target_met') or coverage.get('credited_train_seconds', 0) < policy['target_train_seconds']:
        raise ValueError('Generated duration or distribution target has not been met')
    if coverage.get('coverage_deficits'):
        raise ValueError('Generated distribution coverage is incomplete')
