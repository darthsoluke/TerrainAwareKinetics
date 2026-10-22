import json
from types import SimpleNamespace

import numpy as np
import pytest

from motionbricks.data.native_mb_dataset_compatibility import verify_root_contact_only_revision


@pytest.fixture
def pair(tmp_path):
    from test_native_mb_stage_training import _motion_and_representation
    rep, batch = _motion_and_representation(frame_count=75)
    arrays = rep.unnormalize(batch['motion'])[0].numpy()
    source, target = tmp_path / 'source', tmp_path / 'target'
    for path in (source, target):
        (path / 'stats').mkdir(parents=True)
        (path / 'accepted.json').write_text(json.dumps([{'shard': 'motion.npz', 'split': 'train'}]))
        (path / 'native_manifest.json').write_text(json.dumps({'name': path.name}))
        np.save(path / 'stats/mean.npy', [0.])
        np.save(path / 'stats/std.npy', [1.])
        np.savez_compressed(path / 'motion.npz', global_features=arrays)
    root = SimpleNamespace(global_motion_rep=rep, local_motion_rep=rep.dual_rep.local_motion_rep,
        args={'global_root_feature': 'root', 'local_root_feature': 'root',
              'local_pose_feature': 'joint_positions_and_rotations_and_hip_height'})
    return source, target, root, arrays


def test_contact_only_changes_are_proven_not_checkpoint_relabels(pair):
    source, target, root, values = pair
    modified = values.copy()
    modified[:, root.global_motion_rep.indices['foot_contacts']] = 0
    np.savez_compressed(target / 'motion.npz', global_features=modified)
    proof = verify_root_contact_only_revision(source, target, root)
    assert proof['changed_shards'][0]['changed_contact_values'] > 0
    assert proof['root_weights_retrained_or_modified'] is False
    assert proof['source_dataset_manifest_sha256'] != proof['target_dataset_manifest_sha256']


def test_changed_motion_and_stats_are_rejected(pair):
    source, target, root, values = pair
    modified = values.copy()
    modified[10, 0] += .001
    np.savez_compressed(target / 'motion.npz', global_features=modified)
    with pytest.raises(ValueError, match='Non-contact'):
        verify_root_contact_only_revision(source, target, root)
    np.save(target / 'stats/mean.npy', [1.])
    with pytest.raises(ValueError, match='identical stats'):
        verify_root_contact_only_revision(source, target, root)


def test_changed_split_is_rejected(pair):
    source, target, root, _ = pair
    (target / 'accepted.json').write_text('[]')
    with pytest.raises(ValueError, match='identical accepted'):
        verify_root_contact_only_revision(source, target, root)
