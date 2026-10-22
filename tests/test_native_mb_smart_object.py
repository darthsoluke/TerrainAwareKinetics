from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.native_mb_smart_object import NativeSmartObject, rollout_authored_goals, rollout_fixed_lookahead
from test_native_mb_stair_sequence import example, official_intent_constraints
from test_native_mb_locomotion import RecordingInference


def test_fixed_inputs_match_official_stair_packer():
    original, raw = example()
    infer = RecordingInference(original.global_motion_rep)
    expected = official_intent_constraints(infer, raw[:, :4], raw[:, 28:33], 8)
    primitive = NativeSmartObject(infer, fixed_tokens=8)
    primitive.predict(raw[:, :4], raw[:, 28:32])
    for key in ('global_root_values', 'local_root_values', 'local_poses',
                'has_global_root_values', 'has_local_root_values', 'has_local_poses',
                'allowed_pred_num_tokens'):
        torch.testing.assert_close(primitive.last_constraints[key], expected[key])


def test_root_can_choose_duration_without_a_source_clock():
    original, raw = example()
    infer = RecordingInference(original.global_motion_rep)
    captured = {}
    tokens = infer._args['min_tokens'] + 1
    def predict(*args, **kwargs):
        captured.update(kwargs)
        output = raw[:, :tokens * 4].clone()
        captured['output'] = output
        return output, torch.tensor([tokens])
    infer.predict = predict
    primitive = NativeSmartObject(infer)
    result, count = primitive.predict(raw[:, :4], raw[:, 28:32])
    assert count == tokens * 4 and result is captured['output']
    assert primitive.last_constraints['allowed_pred_num_tokens'].all()
    assert captured['config']['smooth_root_traj'] is False
    assert primitive.last_constraints['has_local_root_values'][0].tolist() == [False, False, True, False, False, False, False, True]


def test_rollout_uses_model_time_and_generated_history_without_output_repairs():
    histories = []
    def predict(history, goal):
        histories.append(history.clone())
        count = 12 if len(histories) == 1 else 20
        output = torch.full((1, count, 3), float(len(histories)))
        output[:, :4] = -99
        return output, count
    seed = torch.zeros(1, 4, 3)
    output, plans = rollout_authored_goals(SimpleNamespace(predict=predict), seed,
        [torch.full((1, 4, 3), 20.), torch.full((1, 4, 3), 30.)])
    assert output.shape == (1, 28, 3)
    assert torch.all(output[:, 4:12] == 1) and torch.all(output[:, 12:] == 2)
    assert torch.all(histories[1] == 1)
    assert plans[1]['first_generated_frame'] == 12
    assert plans[1]['last_generated_frame'] == 27
    assert plans[1]['context_source'] == 'generated_only'


def test_reject_invalid_goal_and_duration():
    original, raw = example()
    infer = RecordingInference(original.global_motion_rep)
    primitive = NativeSmartObject(infer)
    with pytest.raises(ValueError, match='four-frame'):
        primitive.predict(raw[:, :4], raw[:, 28:33])
    infer.predict = lambda *a, **k: (raw[:, :4], torch.tensor([1]))
    with pytest.raises(ValueError, match='duration'):
        primitive.predict(raw[:, :4], raw[:, 28:32])
    with pytest.raises(ValueError, match='at least one'):
        rollout_authored_goals(primitive, raw[:, :4], [])


def test_native_iterations_forward_only_the_upstream_sampling_option():
    original, raw = example()
    infer = RecordingInference(original.global_motion_rep)
    captured = {}
    def predict(*args, **kwargs):
        captured.update(kwargs)
        return raw[:, :32], torch.tensor([8])
    infer.predict = predict
    primitive = NativeSmartObject(infer, fixed_tokens=8, pose_iterations=4)
    output, count = primitive.predict(raw[:, :4], raw[:, 28:32])
    torch.testing.assert_close(output, raw[:, :32])
    assert count == 32
    assert captured['config']['num_inference_step'] == 4
    assert primitive.last_constraints['config'] == captured['config']
    assert captured['config']['smooth_root_traj'] is False
    for bad in (0, -1, 17, 2.5, True):
        with pytest.raises(ValueError, match='iterations'):
            NativeSmartObject(infer, pose_iterations=bad)


def test_lookahead_replaces_unplayed_context_without_replay_or_output_repair(monkeypatch):
    import motionbricks.data.native_mb_smart_object as module
    contexts, predictions, durations = [], [], []
    class Primitive:
        def __init__(self, infer, fixed_tokens):
            self.frames = fixed_tokens * 4
            durations.append(self.frames)
        def predict(self, context, goal):
            contexts.append(context.clone())
            output = torch.arange(self.frames).reshape(1, -1, 1).float() + 100 * len(contexts)
            predictions.append(output)
            return output, self.frames
    monkeypatch.setattr(module, 'NativeSmartObject', Primitive)
    seed = torch.arange(4).reshape(1, 4, 1).float()
    result, plans = rollout_fixed_lookahead(None, seed, [(15, seed), (27, seed), (39, seed)])
    assert durations == [16, 16, 16]
    torch.testing.assert_close(contexts[1], predictions[0][:, 12:16])
    torch.testing.assert_close(contexts[2], predictions[1][:, 12:16])
    expected = torch.cat([seed, predictions[0][:, 4:12], predictions[1][:, :12], predictions[2]], dim=1)
    torch.testing.assert_close(result, expected)
    assert result.shape[1] == 40
    assert [p['first_generated_frame'] for p in plans] == [4, 12, 24]
    assert plans[1]['context_absolute_frames'] == [12, 13, 14, 15]
    assert plans[1]['emitted_prediction_start'] == 0
    assert not plans[1]['goal_committed_from_this_plan']
    assert plans[-1]['goal_committed_from_this_plan']
    ledger = [frame for p in plans for frame in range(
        p['prediction_absolute_start_frame'] + p['emitted_prediction_start'],
        p['prediction_absolute_start_frame'] + p['emitted_prediction_stop'])]
    assert ledger == list(range(4, 40))


def test_lookahead_rejects_empty_and_misaligned_schedule():
    seed = torch.zeros(1, 4, 3)
    with pytest.raises(ValueError, match='at least one'):
        rollout_fixed_lookahead(None, seed, [])
    with pytest.raises(ValueError, match='token-aligned'):
        rollout_fixed_lookahead(None, seed, [(14, seed)])
