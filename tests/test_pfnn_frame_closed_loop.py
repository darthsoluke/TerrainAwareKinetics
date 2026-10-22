"""P2 targets must not turn invisible absolute offsets into motion commands."""

import pytest
import torch

from motionbricks.data.pfnn_frame_actor import advance
from motionbricks.data.pfnn_frame_closed_loop import transition_loss, relative_path_loss, closed_loop_loss
from motionbricks.motionlib.core.utils.rotations import matrix_to_cont6d
from test_pfnn_frame_actor import source, cpu_only


def exact_prediction(rep, frames, shift):
    history = frames[:, :4].clone()
    history[..., :3] += shift
    target = rep.inverse(frames[:, 4:], False, return_all=False, joint_positions_from='global_rot_data')
    pred = dict(root_delta=(frames[:, 4, :3] - frames[:, 3, :3]).clone().requires_grad_(),
                root_next=(frames[:, 4, :3] + shift).clone().requires_grad_(),
                local_rot6d=matrix_to_cont6d(target['local_joint_rots'][:, 0]).clone().requires_grad_(),
                contact_logits=torch.zeros(len(frames), 4, requires_grad=True))
    return history, pred


def test_translated_generated_root_does_not_chase_absolute_source(source):
    rep, frames = source
    for shift in (torch.zeros(3), torch.tensor([3., 0., -7.])):
        history, pred = exact_prediction(rep, frames, shift)
        loss = transition_loss(rep, history, advance(rep, history, pred), pred, frames[:, 3], frames[:, 4])
        assert loss['root_delta'] == 0
        assert float(loss['body_fk']) < 1e-8
        assert float(loss['boundary_joint_velocity']) < 1e-8
        loss['total'].backward()
        assert torch.isfinite(pred['root_next'].grad).all()


def test_velocity_loss_sees_a_different_actual_previous_frame(source):
    rep, frames = source
    history, pred = exact_prediction(rep, frames, torch.zeros(3))
    emitted = advance(rep, history, pred)
    baseline = transition_loss(rep, history, emitted, pred, frames[:, 3], frames[:, 4])
    moved = history.clone()
    moved[:, -1, 0] += .01
    changed = transition_loss(rep, moved, emitted, pred, frames[:, 3], frames[:, 4])
    assert float(baseline['boundary_joint_velocity']) < 1e-8
    assert changed['boundary_joint_velocity'] > .01
    torch.testing.assert_close(changed['body_fk'], baseline['body_fk'])


def test_source_displacement_labels_are_detached(source):
    rep, frames = source
    labels = frames.clone().requires_grad_()
    history, pred = exact_prediction(rep, frames, torch.zeros(3))
    history = history.clone().requires_grad_()
    loss = transition_loss(rep, history, advance(rep, history, pred), pred, labels[:, 3], labels[:, 4])
    loss['total'].backward()
    assert labels.grad is None
    assert history.grad is not None and torch.isfinite(history.grad).all()


def test_relative_future_path_is_translation_invariant_and_excludes_tail():
    worlds = torch.zeros(2, 33, 366)
    worlds[..., 0] = torch.arange(33) * .03
    history = worlds[:, 18:22].clone()
    stations = 21 + torch.arange(1, 9) * 3
    known = stations < 33
    path = torch.zeros(2, 8, 2)
    path[:, known] = worlds[:, stations[known]][..., [0, 2]]
    path[:, ~known] = 1000
    offset = torch.tensor([9., -3.])
    history[..., [0, 2]] += offset
    path += offset
    path.requires_grad_()
    loss = relative_path_loss(path, history, worlds, 18)
    assert float(loss) < 1e-9
    loss.backward()
    assert torch.count_nonzero(path.grad[:, ~known]) == 0


@pytest.mark.parametrize('start,steps', [(0, 32), (23, 4), (-1, 4)])
def test_rollout_rejects_unobserved_source_intervals(start, steps):
    with pytest.raises(ValueError, match='actual source window'):
        closed_loop_loss(None, None, None, torch.zeros(2, 33, 366),
                         torch.zeros(2, 26, 4), None, start, steps, torch.ones(2, dtype=torch.bool))


def test_mixed_rollout_uses_generated_history_and_one_continuous_state(source, monkeypatch):
    from types import SimpleNamespace
    import motionbricks.data.pfnn_frame_closed_loop as module
    from motionbricks.data.pfnn_frame_actor import FrameActor
    from motionbricks.data.native_mb_observed_history import reencode_played_history
    from test_pfnn_frame_actor import observation

    rep, frames = source
    worlds = torch.cat((frames, frames[:, -1:].repeat(1, 15, 1)), dim=1)
    actor = FrameActor(rep)
    observed, histories, memories, emitted = [], [], [], []

    def observe(history, command, field, horizon):
        observed.append(history.clone())
        return observation(len(history))

    def fine(sampler, history, path, coarse, field):
        assert path.requires_grad
        return coarse

    root_step, advance_step = actor.root_step, module.advance

    def root(history, command, coarse, state):
        histories.append(history.clone())
        memories.append(state.clone())
        return root_step(history, command, coarse, state)

    def advance_with_record(rep, history, pred):
        value = advance_step(rep, history, pred)
        emitted.append(value)
        return value

    monkeypatch.setattr(actor, 'root_step', root)
    monkeypatch.setattr(module, 'path_observation', fine)
    monkeypatch.setattr(module, 'advance', advance_with_record)
    mask = torch.tensor([True, False])
    losses = closed_loop_loss(actor, rep, SimpleNamespace(observe=observe), worlds,
                              torch.zeros(2, 16, 4), None, 0, 4, mask)
    for i in range(1, 4):
        torch.testing.assert_close(histories[i][0], emitted[i - 1][0])
        teacher = reencode_played_history(worlds[:, i:i + 4], rep)
        torch.testing.assert_close(histories[i][1], teacher[1])
        torch.testing.assert_close(observed[i], histories[i][..., :5])
        assert torch.count_nonzero(memories[i]) > 0
    gradient = torch.autograd.grad(histories[-1][0].square().sum(), emitted[0], retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient[0].abs().sum() > 0
    assert torch.count_nonzero(gradient[1]) == 0
    losses['total'].backward()
    for parameter in actor.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
