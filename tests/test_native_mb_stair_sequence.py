from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.review_native_mb_stair_sequence import (
    all_landing_endpoints, common_scoring_masks, generate_sequence, intent_constraints,
    landing_endpoints, official_intent_constraints, training_condition_counts,
)


def example():
    from test_native_mb_inbetween import FakeInference
    from test_native_mb_stage_training import _motion_and_representation
    rep, batch = _motion_and_representation(frame_count=75)
    raw = rep.unnormalize(batch["motion"])[:, :65]
    infer = FakeInference(rep)
    return infer, rep.change_first_heading(raw, 0., False, False)


def test_intent_packing_matches_existing_four_context_probe():
    from scripts.review_native_mb_stairs import stair_constraints
    infer, raw = example()
    original, _ = stair_constraints(infer, raw)
    inputs = intent_constraints(infer, raw[:, :4], raw[:, 60:65], 16)
    for key in original:
        if original[key] is not None and not key.startswith('has_') and key in ('global_root_values', 'local_root_values', 'local_poses'):
            mask = original['has_' + key]
            torch.testing.assert_close(inputs[key][mask], original[key][mask], rtol=0, atol=1e-6)
        elif original[key] is not None:
            torch.testing.assert_close(inputs[key], original[key])


def test_context_and_goal_masks_do_not_claim_unknown_context_velocity():
    infer, raw = example()
    inputs = intent_constraints(infer, raw[:, :4], raw[:, 28:33], 8, 1, 1)
    assert inputs['has_local_poses'].tolist() == [[False, False, False, True, False, False, False, True]]
    assert inputs['has_local_root_values'].tolist() == [[False, False, True, False, False, False, False, True]]
    assert inputs['local_root_values'][:, 3].count_nonzero() == 0
    assert inputs['allowed_pred_num_tokens'].sum() == 1
    assert inputs['allowed_pred_num_tokens'][0, 8 - infer._args['min_tokens']] == 1


def test_no_source_interiors_can_enter_conditions():
    infer, raw = example()
    original = intent_constraints(infer, raw[:, :4], raw[:, 28:33], 8)
    changed = raw.clone()
    changed[:, 4:28] += 100
    changed[:, 33:] -= 100
    actual = intent_constraints(infer, changed[:, :4], changed[:, 28:33], 8)
    for key, value in original.items():
        if value is not None:
            torch.testing.assert_close(value, actual[key], rtol=0, atol=0)


def test_landing_selection_has_two_legal_horizons_and_uses_evidence():
    events = [{"status": "weak_support", "start_frame": 30, "end_frame_exclusive": 35},
              {"status": "rejected", "start_frame": 5, "end_frame_exclusive": 50}]
    assert landing_endpoints(events, 0, 6, 16) == [31, 63]
    with pytest.raises(ValueError, match='No evidenced'):
        landing_endpoints(events, 0, 10, 16)
    with pytest.raises(ValueError, match='No evidenced'):
        landing_endpoints(events[1:], 0, 6, 16)


def test_scoring_excludes_union_of_goal_blocks_and_interval_endpoints():
    frames, intervals = common_scoring_masks([31, 63])
    assert frames.sum() == 51
    assert not frames[:4].any() and not frames[28:32].any() and not frames[60:].any()
    assert not frames[32]
    np.testing.assert_array_equal(intervals, frames[:-1] & frames[1:])


def test_rollout_uses_generated_history_and_never_pastes_goal_or_context(monkeypatch):
    import scripts.review_native_mb_stair_sequence as module
    calls = []
    def constraints(infer, history, goal, tokens, context_count, goal_count):
        calls.append(history.clone())
        return {'tokens': tokens}
    monkeypatch.setattr(module, 'intent_constraints', constraints)
    def predict(tokens, **unused):
        value = 10. if len(calls) == 1 else 20.
        output = torch.full((1, tokens * 4, 3), value)
        output[:, :4] = -100
        return output, torch.tensor([tokens])
    infer = SimpleNamespace(predict=predict)
    seed = torch.zeros(1, 4, 3)
    goals = [(31, torch.full((1, 5, 3), 99.)), (63, torch.full((1, 5, 3), 98.))]
    actual, plans = generate_sequence(infer, seed, goals, context_count=4, goal_count=1, iterations=1)
    assert actual.shape == (1, 64, 3)
    assert torch.all(actual[:, :4] == 0)
    assert torch.all(actual[:, 4:32] == 10)
    assert torch.all(actual[:, 32:] == 20)
    assert torch.all(calls[1] == 10)
    assert plans[1]['context_source'] == 'generated_only'
    assert [p['tokens'] for p in plans] == [8, 9]


def test_invalid_native_intent_duration_rejected():
    infer, raw = example()
    with pytest.raises(ValueError, match='contract'):
        intent_constraints(infer, raw[:, :4], raw[:, 28:33], 100)


def test_declared_runtime_condition_counts_come_from_original_curriculum():
    from motionbricks.motion_backbone.models.pose_model import MotionModel as PoseModel
    from motionbricks.motion_backbone.models.root_model import MotionModel as RootModel
    from motionbricks.data.native_mb_paper_protocol import paper_config
    # Use the real methods on light probes; no large network allocation.
    class Pose:
        _construct_keyframe_prob = PoseModel._construct_keyframe_prob
        args = paper_config('pose').model.args
    class Root:
        _construct_keyframe_prob = RootModel._construct_keyframe_prob
        args = paper_config('root').model.args
    infer = SimpleNamespace(_root_model=Root(), _pose_model=Pose())
    result = training_condition_counts(infer, {'root_step': 15000, 'pose_step': 15000})
    assert result == {'root': {'start': [1], 'end': [1]}, 'pose': {'start': [1], 'end': [1]}}


def test_multiple_landing_events_obey_timing_and_preserve_skipped_record():
    events = [{'status': 'weak_support', 'side': side, 'start_frame': a, 'end_frame_exclusive': b}
              for side, a, b in [('left', 0, 29), ('right', 0, 10), ('right', 24, 45), ('left', 43, 63)]]
    endpoints, skipped = all_landing_endpoints(events, 0, 3, 16)
    assert endpoints == [15, 35, 51, 63]
    assert skipped[0]['side'] == 'right'
    durations = np.diff([3] + endpoints) + 4
    assert (durations % 4 == 0).all() and (durations >= 12).all() and (durations <= 64).all()


def test_long_sequence_requires_real_intermediate_events_not_invented_goals():
    with pytest.raises(ValueError, match='cannot cover'):
        all_landing_endpoints([], 0, 3, 16, frames=128)


def test_official_inputs_ignore_redundant_positions_and_unused_future_goal():
    from test_native_mb_locomotion import RecordingInference
    original, raw = example()
    infer = RecordingInference(original.global_motion_rep)
    history, goal = raw[:, :4].clone(), raw[:, 28:33].clone()
    before = official_intent_constraints(infer, history, goal, 8)
    changed = history.clone()
    changed[..., infer.global_motion_rep.indices['ric_data']] += 10
    goal[:, -1] += 100
    after = official_intent_constraints(infer, changed, goal, 8)
    for key in before:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    assert before['has_local_poses'].tolist() == [[False, False, False, True, False, False, False, True]]
    assert before['has_local_root_values'].tolist() == [[False, False, True, False, False, False, False, True]]
