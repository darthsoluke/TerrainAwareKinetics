"""Explicit source binding and balanced real-clip sampling for PARC + BONES."""

from collections import Counter
import math

import torch


SOURCE_POLICY = 'parc-bones-soma-v1'
ADAPTATION = 'matched-mixed-velocity-control-v2'
MIXED_BUNDLE_SCHEMA = 'motionbricks.mixed_online_soma_bundle.v1'


def native_validation_loss(model, stage, batch, index):
    if stage == 'tokenizer':
        return model.training_step(batch, index)
    info = {}
    model.training_step(batch, index, use_outside_training=True, meta_info=info)
    # Official Root/Pose evaluation intentionally returns None; losses are
    # carried by meta_info. Do not mistake that interface for a skipped batch.
    return info.get('losses', {}).get('loss')


def validate_source_policy(policy):
    if (not isinstance(policy, dict) or policy.get('version') != SOURCE_POLICY
            or set(policy.get('allowed_origins', [])) != {'parc_official', 'parc_generated', 'bones_seed'}):
        raise ValueError('Unknown mixed motion source policy')
    for key in ('parc_manifest_sha256', 'bones_source_manifest_sha256', 'bones_selection_sha256'):
        value = policy.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Mixed source policy lacks a source identity')


def sampling_weights(records, *, policy='clips-v1', window_frames=65):
    if policy not in ('clips-v1', 'parc-windows-v1') or window_frames < 1:
        raise ValueError('Unknown sampling policy or invalid window length')
    parc = [row for row in records if row['origin'] in ('parc_official', 'parc_generated')]
    categories = Counter(row['control_category'] for row in records if row['origin'] == 'bones_seed')
    if not parc or not categories or not {'idle', 'start', 'stop'} <= categories.keys():
        raise ValueError('Mixed training needs terrain and genuine idle/start/stop clips')
    if any(row['split'] != 'train' for row in records):
        raise ValueError('Training sampler cannot include held-out records')
    def parc_weight(row):
        if policy == 'clips-v1':
            return 1
        frames = row.get('frames_30hz')
        if type(frames) is not int or frames < window_frames:
            raise ValueError('Window sampling requires a complete trainable frame count')
        return frames - window_frames + 1
    parc_total = sum(parc_weight(row) for row in parc)
    result = []
    for row in records:
        if row['origin'] == 'bones_seed':
            result.append(.5 / len(categories) / categories[row['control_category']])
        elif row['origin'] in ('parc_official', 'parc_generated'):
            result.append(.5 * parc_weight(row) / parc_total)
        else:
            raise ValueError('Undeclared training origin')
    return torch.tensor(result, dtype=torch.float64)


def summarize_validation(rows):
    """Weight native batch means by examples, then match the declared source mixture."""
    groups = {}
    for row in rows:
        count, loss = row['samples'], row['loss']
        if count < 1 or not math.isfinite(loss):
            raise ValueError('Validation needs finite losses and positive sample counts')
        key = row['group']
        group = groups.setdefault(key, {'samples': 0, 'loss_sum': 0., 'domain': row['domain']})
        if group['domain'] != row['domain']:
            raise ValueError('A validation group crosses domains')
        group['samples'] += count
        group['loss_sum'] += loss * count
    for group in groups.values():
        group['mean_loss'] = group.pop('loss_sum') / group['samples']
    parc = [value for value in groups.values() if value['domain'] == 'parc']
    bones = [value for value in groups.values() if value['domain'] == 'bones']
    if not parc or not bones:
        raise ValueError('Validation must cover both PARC and BONES')
    parc_mean = sum(value['mean_loss'] * value['samples'] for value in parc) / sum(value['samples'] for value in parc)
    bones_mean = sum(value['mean_loss'] for value in bones) / len(bones)
    return {'selection_loss': .5 * (parc_mean + bones_mean), 'parc_mean_loss': parc_mean,
            'bones_category_mean_loss': bones_mean, 'groups': groups,
            'samples': sum(value['samples'] for value in groups.values()),
            'aggregation': 'sample-weighted native batch means; half PARC, half available BONES category macro mean'}
