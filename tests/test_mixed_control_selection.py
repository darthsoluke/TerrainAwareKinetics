import pytest

from scripts.build_mixed_control_dataset import validate_selection


def test_selected_validation_clip_cannot_be_promoted_to_training():
    source = {'shard': 'shards/a.npz', 'split': 'validation', 'filename': 'idle',
              'recording_group': 'idle-family', 'converted_shard_sha256': 'a' * 64}
    validate_selection([source], [dict(source, control_category='idle')])
    with pytest.raises(ValueError, match='split changed'):
        validate_selection([source], [dict(source, split='train')])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_selection([source], [source, source])
    with pytest.raises(ValueError, match='identity'):
        validate_selection([source], [dict(source, converted_shard_sha256='b' * 64)])
