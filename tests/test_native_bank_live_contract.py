from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.unified_terrain import PROFILE
from scripts.package_approved_parc_live import state_hash
from scripts.soma_native_bank_server import validate_native_export, BASELINE_SHA256, TRAINED_SHA256
from motionbricks.data.unified_committed_playback import CommittedUnifiedBackend


def pair():
    state = {prefix + 'weight': torch.ones(2) for prefix in ('root.', 'pose.', 'codec.', 'adaptation.')}
    saved = dict(schema='motionbricks.unified.native_route_curriculum.v1',
        contract=dict(stage='pose', bundle_sha256=BASELINE_SHA256), actual_updates=50, model_state=state)
    receipt = dict(schema='motionbricks.native_bank_live.v1', original_checkpoint_sha256=TRAINED_SHA256,
        source_bundle_sha256=BASELINE_SHA256, runtime_profile=dict(PROFILE), model_tensors_unchanged=True,
        model_state_sha256=state_hash(state))
    return saved, receipt


def test_full_native_weights_required_and_tensor_changes_rejected():
    saved, receipt = pair(); validate_native_export(saved, receipt)
    partial = deepcopy(saved); del partial['model_state']['pose.weight']
    with pytest.raises(ValueError, match='Full Root'):
        validate_native_export(partial, receipt)
    saved['model_state']['codec.weight'][0] = 2
    with pytest.raises(ValueError, match='tensor digest'):
        validate_native_export(saved, receipt)


def test_inference_profile_cannot_silently_change():
    saved, receipt = pair(); receipt['runtime_profile']['refresh_frames'] = 8
    with pytest.raises(ValueError, match='inference contract'):
        validate_native_export(saved, receipt)


def test_prepared_motion_cannot_commit_against_another_played_history():
    backend = object.__new__(CommittedUnifiedBackend)
    backend.history = torch.zeros(1, 4, 6)
    candidate = dict(generation=dict(prefetch_policy='committed-model-history-v1'),
        planned_motion=torch.ones(1, 8, 6), remaining_motion=torch.zeros(1, 0, 6),
        expected_prior_history=backend.history.clone(), published_history=torch.ones(1, 4, 6),
        command=((0., 1.6), 0.))
    backend.history[0, -1, 0] = .1
    with pytest.raises(ValueError, match='actual playback'):
        backend.commit_prepared_plan(candidate)
    backend.history.zero_()
    backend.commit_prepared_plan(candidate)
    assert backend.history.eq(0).all()
    assert backend._last_published_history.eq(1).all()
    assert backend._future_buffer.shape == (1, 0, 6)


def test_failed_acceleration_cannot_prepare_a_live_plan():
    backend = object.__new__(CommittedUnifiedBackend)
    backend.execution = SimpleNamespace(validation=[dict(tokens_equal=False)])
    history = torch.zeros(1, 4, 6)
    with pytest.raises(ValueError, match='verified acceleration'):
        backend.prepare_next_plan([0., 1.6], 0., history, history)
