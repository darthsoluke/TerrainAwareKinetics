"""Sparse authored Smart Object goals through the original MB runtime packer.

The Root model chooses duration unless a diagnostic fixes it explicitly. This
module does not alter generated root values, joints, rotations, or playback rate.
"""

import torch

from motionbricks.data.native_mb_locomotion import _InferenceInputs
from motionbricks.data.native_mb_frame import translate_global_height
from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent


class NativeSmartObject:
    NUM_FRAMES_PER_TOKEN = 4
    _predict_inbetween_motion_features = full_navigation_agent._predict_inbetween_motion_features

    def __init__(self, infer, *, context_keyframes=1, target_keyframes=1, fixed_tokens=None,
                 pose_iterations=1):
        if context_keyframes not in (1, 4) or target_keyframes not in (1, 4):
            raise ValueError('Use one or four native keyframes per end')
        minimum, maximum = infer._args['min_tokens'], infer._args['max_tokens']
        if fixed_tokens is not None and not minimum <= fixed_tokens <= maximum:
            raise ValueError('Fixed duration is outside the native token range')
        if infer.global_motion_rep.skeleton.root_idx != 0 or infer.global_motion_rep.fps != 30:
            raise ValueError('The original runtime requires root index zero and 30 Hz')
        self.infer = infer
        self._device = torch.device(infer.device)
        self._target_root_realignment = True
        self.SKIP_ENDING_TARGET_COND = False
        self._inferencer = _InferenceInputs(infer, target_keyframes, fixed_tokens, context_keyframes,
                                           pose_iterations=pose_iterations)

    @property
    def last_constraints(self):
        return self._inferencer.last

    @torch.inference_mode()
    def predict(self, played_history, authored_goal_block, *, frame_origin_y_m=0.):
        """Four context frames plus four goal frames (terminal backward derivative).

        A one-keyframe configuration supplies only the last goal pose/position
        and terminal velocity; its interior goal-block poses are masked. The
        The context can be played history or an unplayed prediction buffer.
        Callers must use a consistent cursor policy: discard regenerated played
        history, or replace unplayed context and emit its regenerated frame zero.
        """
        rep = self.infer.global_motion_rep
        if frame_origin_y_m:
            played_history = translate_global_height(played_history, rep, -frame_origin_y_m)
            authored_goal_block = translate_global_height(authored_goal_block, rep, -frame_origin_y_m)
        expected = (1, 4, len(rep.indices['all']))
        for value in (played_history, authored_goal_block):
            if value.shape != expected or not torch.isfinite(value).all():
                raise ValueError('Need finite four-frame history and authored goal block')
            if (value.device.type != self._device.type or
                    (self._device.index is not None and value.device.index != self._device.index)):
                raise ValueError('Inputs must be on the native inference device')
        past = rep.inverse(played_history, False, return_all=True, return_quat=False)
        target = rep.inverse(authored_goal_block, False, return_all=True, return_quat=False)
        planar = target['posed_joints'][:, :, 0] * played_history.new_tensor([1., 0., 1.])
        minimum, maximum = self.infer._args['min_tokens'], self.infer._args['max_tokens']
        prediction, frames = self._predict_inbetween_motion_features({
            'context_global_joint_positions': past['posed_joints'],
            'context_global_joint_rotations': past['global_joint_rots'],
            'target_global_root_positions': planar,
            'target_global_joint_positions': target['posed_joints'] - planar[:, :, None],
            'target_global_joint_rotations': target['global_joint_rots'],
            'allowed_pred_num_tokens': played_history.new_ones((1, maximum - minimum + 1)),
        })
        count = int(frames.item())
        if (count % 4 or not minimum * 4 <= count <= maximum * 4
                or prediction.shape[0] != 1 or prediction.shape[1] < count
                or not torch.isfinite(prediction).all()):
            raise ValueError('Invalid original MB duration or generated output')
        if frame_origin_y_m:
            prediction = translate_global_height(prediction, rep, frame_origin_y_m)
        return prediction, count


def rollout_authored_goals(primitive, seed, goals):
    """Advance an ordered authored goal list at the model's native 30 Hz."""
    history, pieces, plans, played = seed.clone(), [seed.clone()], [], 4
    if not goals:
        raise ValueError('An authored interaction requires at least one goal')
    for index, goal in enumerate(goals):
        prediction, frames = primitive.predict(history, goal)
        committed = prediction[:, 4:frames].clone()
        if committed.shape[1] != frames - 4 or frames <= 4:
            raise ValueError('A prediction must advance time beyond its context')
        plans.append({'goal_index': index, 'first_generated_frame': played,
            'last_generated_frame': played + frames - 5, 'predicted_tokens': frames // 4,
            'context_source': 'initial_seed' if index == 0 else 'generated_only'})
        pieces.append(committed)
        history = torch.cat((history, committed), dim=1)[:, -4:].clone()
        played += frames - 4
    return torch.cat(pieces, dim=1), plans


def rollout_fixed_lookahead(infer, seed, goals):
    """Test original PRED_OFFSETS=4 cursor semantics at authored goal boundaries.

    Goals are (absolute endpoint, four-frame authored block). The first plan
    discards regenerated seed history. Subsequent plans replace the last four
    UNPLAYED frames, then emit from frame zero, without blending or replay.
    This late soft-advance schedule is not the preview's periodic replanning.
    Intermediate goals may be overwritten; they are not hard reach guarantees.
    """
    if not goals or seed.shape[1] != 4:
        raise ValueError('Need four seed frames and at least one authored goal')
    context, pieces, plans, played = seed.clone(), [seed.clone()], [], 4
    for index, (endpoint, goal) in enumerate(goals):
        plan_start = 0 if index == 0 else played
        count = endpoint + 1 - plan_start
        if count <= 8 or count % 4:
            raise ValueError('Lookahead goals must advance by token-aligned intervals')
        primitive = NativeSmartObject(infer, fixed_tokens=count // 4)
        prediction, frames = primitive.predict(context, goal)
        if frames != count:
            raise ValueError('Native duration did not match the fixed authored clock')
        start, stop = (4 if index == 0 else 0), count if index == len(goals) - 1 else count - 4
        committed = prediction[:, start:stop].clone()
        if plan_start + start != played or not len(committed[0]):
            raise ValueError('Lookahead planning must advance playback')
        plans.append({'goal_index': index, 'first_generated_frame': played,
            'prediction_absolute_start_frame': plan_start,
            'last_generated_frame': played + stop - start - 1,
            'authored_endpoint_frame': endpoint, 'predicted_tokens': count // 4,
            'context_absolute_frames': list(range(plan_start, plan_start + 4)),
            'context_source': 'initial_played_seed' if index == 0 else 'generated_unplayed_lookahead',
            'emitted_prediction_start': start, 'emitted_prediction_stop': stop,
            'goal_committed_from_this_plan': index == len(goals) - 1})
        pieces.append(committed)
        context = prediction[:, stop:stop + 4].clone()
        played += stop - start
    result = torch.cat(pieces, dim=1)
    if result.shape[1] != goals[-1][0] + 1:
        raise ValueError('Lookahead playback changed the authored clock')
    return result, plans
