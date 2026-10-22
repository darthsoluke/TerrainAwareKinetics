from copy import deepcopy

import pytest

from motionbricks.data.unified_terrain import PROFILE, SCHEMA
from scripts.soma_unified_terrain_server import BASELINE_SHA256, validate_candidate


def candidate():
    return dict(schema=SCHEMA, steps=200, adaptation={}, contract=dict(
        bundle_sha256=BASELINE_SHA256, profile=deepcopy(PROFILE), updates=200))


def test_matching_experimental_checkpoint_is_accepted():
    validate_candidate(candidate(), BASELINE_SHA256)


@pytest.mark.parametrize('field', ['schema', 'source', 'profile', 'steps'])
def test_mismatched_checkpoint_is_rejected(field):
    value = candidate()
    if field == 'schema':
        value['schema'] = 'legacy'
    elif field == 'source':
        value['contract']['bundle_sha256'] = 'different'
    elif field == 'profile':
        value['contract']['profile']['refresh_frames'] = 8
    else:
        value['steps'] = 199
    with pytest.raises(ValueError):
        validate_candidate(value, BASELINE_SHA256)
