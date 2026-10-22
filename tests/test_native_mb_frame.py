import pytest
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_frame import translate_global_height
from motionbricks.data.native_mb_frame import NativePlatformLocomotion
from motionbricks.data.native_mb_frame import NativeAnchoredIdle


def test_rigid_height_transform_changes_both_position_paths_not_rotations():
    rep = build_native_representation()
    local = torch.eye(3).expand(8, 30, 3, 3).clone()
    root = torch.zeros(8, 3)
    root[:, 1] = 1.0
    root[:, 2] = torch.arange(8) / 30.0
    source = rep({'local_joint_rots': local, 'translation': root},
                 to_normalize=False)[None]
    a = rep.inverse(source, False, return_all=True, return_quat=False)
    moved = translate_global_height(source, rep, 1.7)
    b = rep.inverse(moved, False, return_all=True, return_quat=False)
    shift = source.new_tensor([0, 1.7, 0])
    for key in ('ric_data', 'global_rot_data'):
        torch.testing.assert_close(b['all_posed_joints'][key], a['all_posed_joints'][key] + shift)
    torch.testing.assert_close(b['global_joint_rots'], a['global_joint_rots'])
    torch.testing.assert_close(translate_global_height(moved, rep, -1.7), source)
    for key in ('local_vel', 'foot_contacts', 'global_rot_data'):
        torch.testing.assert_close(moved[..., rep.indices[key]], source[..., rep.indices[key]])


def test_height_transform_rejects_nonfinite_origin():
    rep = build_native_representation()
    with pytest.raises(ValueError):
        translate_global_height(torch.zeros(1, 4, len(rep.indices['all'])), rep, float('nan'))


def test_platform_frame_inverse_preserves_raw_prediction_error_and_rotation():
    rep = build_native_representation()
    history = torch.zeros(1, 4, len(rep.indices['all']))
    history[..., 1] = 2.3
    history[..., rep.indices['ric_data'][1::3]] = 1.6

    class Backend:
        last_generation = {'neural_generation': True}
        def predict_plan(self, velocity, heading):
            torch.testing.assert_close(self.history[..., 1], torch.full((1, 4), 1.0))
            assert velocity == [[0., 0.]] and heading == [.3]
            self.output = self.history[:, -1:].repeat(1, 8, 1) + .07
            return dict(planned_motion=self.output, generation_seconds=.1)

    backend = Backend()
    wrapper = NativePlatformLocomotion(backend, rep, 1.3)
    result = wrapper.predict(history, [[0., 0.]], [.3])
    expected = translate_global_height(backend.output, rep, 1.3)
    torch.testing.assert_close(result['planned_motion'], expected)
    assert result['neural_generation'] is True
    # The known model error is retained, not projected onto either platform or target.
    torch.testing.assert_close(result['planned_motion'][..., 1], torch.full((1, 8), 2.37))
    torch.testing.assert_close(history[..., 1], torch.full((1, 4), 2.3))


def test_platform_origin_is_explicit_and_finite():
    with pytest.raises(ValueError, match='finite'):
        NativePlatformLocomotion(None, None, float('inf'))


def idle_reference(rep):
    rotations = torch.eye(3).expand(8, 30, 3, 3).clone()
    root = torch.zeros(8, 3)
    root[:, 1] = 1.
    return rep({'local_joint_rots': rotations, 'translation': root},
               to_normalize=False)[None, :4]


def test_anchored_idle_authors_input_but_preserves_neural_output_error():
    rep = build_native_representation()
    reference = idle_reference(rep)
    original = reference.clone()
    history = translate_global_height(reference, rep, 1.3)

    class Primitive:
        def predict(self, local_history, goal):
            torch.testing.assert_close(local_history, reference)
            torch.testing.assert_close(goal[0, -1, [0, 2]], torch.tensor([2., -3.]))
            # Explicit model error must survive the inverse scene transform.
            self.output = local_history[:, -1:].repeat(1, 16, 1) + .07
            return self.output, 16

    primitive = Primitive()
    idle = NativeAnchoredIdle(primitive, rep, reference, [2., -3.], .3, 1.3)
    result = idle.predict(history, [[0., 0.]], [.3])
    torch.testing.assert_close(result['planned_motion'],
        translate_global_height(primitive.output[:, 4:12], rep, 1.3))
    torch.testing.assert_close(reference, original)
    assert result['neural_generation'] is True
    with pytest.raises(ValueError, match='free navigation'):
        idle.predict(history, [[.1, 0.]], [.3])


@pytest.mark.parametrize('anchor', [[], [1.], [1., 2., 3.], [float('nan'), 1.]])
def test_anchored_idle_rejects_invalid_anchor(anchor):
    rep = build_native_representation()
    with pytest.raises(ValueError, match='two finite'):
        NativeAnchoredIdle(None, rep, idle_reference(rep), anchor, 0., 0.)


def test_anchored_idle_rejects_nonfinite_reference():
    rep = build_native_representation()
    reference = idle_reference(rep)
    reference[0, 1, 0] = float('nan')
    with pytest.raises(ValueError, match='reference'):
        NativeAnchoredIdle(None, rep, reference, [0., 0.], 0., 0.)


def test_idle_strafe_changes_input_only_not_raw_output():
    rep = build_native_representation()
    reference = idle_reference(rep)
    class Primitive:
        def predict(self, history, goal):
            torch.testing.assert_close(goal[0, -1, [0, 2]], torch.tensor([2.25, 3.]))
            return reference[:, -1:].repeat(1, 12, 1), 12
    idle = NativeAnchoredIdle(Primitive(), rep, reference, [2., 3.], 0., 0.)
    result = idle.predict(reference, [[0., 0.]], [0.], target_offset_xz=[.25, 0.])
    torch.testing.assert_close(result['planned_motion'], reference[:, -1:].repeat(1, 8, 1))
    torch.testing.assert_close(idle.goal[0, -1, [0, 2]], torch.tensor([2., 3.]))
