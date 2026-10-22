"""Prove Root input equivalence across a contact-channel-only data revision."""

import hashlib
import json
from pathlib import Path

import numpy as np


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_root_contact_only_revision(source_dataset, target_dataset, root_model):
    """Keep different training identities explicit; never relabel a checkpoint.

    Callers must already verify both datasets against their own manifests. Root
    has no learned dependency on the four contact features in the supported
    original configuration. All non-contact values, splits and stats must match.
    This exception does NOT authorize transferring a Pose/tokenizer dependency.
    """
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    source, target = Path(source_dataset), Path(target_dataset)
    for name in ('accepted.json', 'stats/mean.npy', 'stats/std.npy'):
        if _hash(source / name) != _hash(target / name):
            raise ValueError(f'Root contact-only transfer requires identical {name}')
    for mode, feature in (('global', 'global_root_feature'), ('local', 'local_root_feature'),
                          ('local', 'local_pose_feature')):
        rep = getattr(root_model, mode + '_motion_rep')
        used = extract_feature_from_motion_rep(None, rep, root_model.args[feature], fetch_feat_idx=True)
        if set(used) & set(rep.indices['foot_contacts']):
            raise ValueError('Root consumes contact channels; contact-only transfer is invalid')
    rep = root_model.global_motion_rep
    contacts = np.asarray(rep.indices['foot_contacts'])
    if len(contacts) != 4:
        raise ValueError('Expected the original four native foot channels')
    other = np.setdiff1d(np.arange(len(rep.indices['all'])), contacts)
    rows = json.loads((source / 'accepted.json').read_text())
    changes = []
    for row in rows:
        relative = Path(row['shard'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe dataset shard path')
        left, right = source / relative, target / relative
        if _hash(left) == _hash(right):
            continue
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            x, y = a['global_features'], b['global_features']
            if (x.shape != y.shape or x.dtype != y.dtype or x.ndim != 2
                    or x.shape[1] != len(rep.indices['all']) or not np.isfinite(y).all()
                    or not np.array_equal(x[:, other], y[:, other])):
                raise ValueError(f'Non-contact Root input changed: {relative}')
            changed = int(np.count_nonzero(x[:, contacts] != y[:, contacts]))
            changes.append({'shard': str(relative), 'changed_contact_values': changed,
                            'source_sha256': _hash(left), 'target_sha256': _hash(right)})
    return {'status': 'verified_root_inputs_identical_contact_features_only',
        'source_dataset_manifest_sha256': _hash(source / 'native_manifest.json'),
        'target_dataset_manifest_sha256': _hash(target / 'native_manifest.json'),
        'root_weights_retrained_or_modified': False, 'changed_shards': changes}
