"""CPU contracts for the compact native frame actor, with real SOMA FK."""

import inspect

import pytest
import torch
from torch import nn

from motionbricks.data.pfnn_frame_actor import (
    FrameActor, advance, load_environment_encoder, source_one_step_loss,
)
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.core.utils.rotations import (
    angle_to_Y_rotation_matrix, cont6d_to_matrix, matrix_to_cont6d,
)
from motionbricks.motionlib.core.utils.stats import Stats


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("FrameActor tests must remain CPU-only")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old_threads)


@pytest.fixture(scope="module")
def source():
    stats = Stats()
    stats.register_from_tensors(torch.linspace(-.2, .2, 366), torch.linspace(.5, 1.5, 366))
    rep = GlobalRootGlobalJoints(30., SOMASkeleton30(), "somaskel30_global_root_global_joints", stats=stats)
    time = torch.arange(5, dtype=torch.float32)
    rotations = torch.eye(3).expand(2, 5, 30, 3, 3).clone()
    rotations[:, :, 0] = angle_to_Y_rotation_matrix(.4 + .02 * time)
    rotations[:, :, 5] = angle_to_Y_rotation_matrix(-.1 + .04 * time)
    roots = torch.stack((3. + .03 * time, 1.1 + .002 * time, -4. + .015 * time), -1)
    roots = roots[None].expand(2, -1, -1).clone()
    contacts = torch.tensor([1., 0., 1., 0.]).expand(2, 5, 4).clone()
    features, origin = rep(dict(local_joint_rots=rotations, translation=roots, foot_contacts=contacts),
                           to_normalize=False, return_init_heading_info=True, lengths=torch.tensor([5, 5]))
    quat = origin["init_heading_quat"]
    features = rep.change_first_heading(features, 2 * torch.atan2(quat[..., 2], quat[..., 0]), False, False)
    features[..., [0, 2]] += roots[:, :1, [0, 2]]
    return rep, features


def observation(batch=2, *, invalid=False):
    generator = torch.Generator().manual_seed(83)
    return dict(features=torch.randn(batch, 3, 512, 15, generator=generator),
                distance=torch.randn(batch, 3, 512, generator=generator),
                valid=torch.full((batch, 3, 512), not invalid, dtype=torch.bool))


def run(actor, history, command=None, state=None, coarse=None, fine=None):
    batch = history.shape[0]
    if command is None:
        command = history.new_tensor([.4, .7, 1., 0.]).expand(batch, -1)
    if state is None:
        state = actor.initial_state(batch)
    root = actor.root_step(history, command, observation(batch) if coarse is None else coarse, state)
    pred = actor.pose_step(history, command, observation(batch) if fine is None else fine, root)
    return root, pred


def test_shapes_finite_and_initial_rotation_continuation(source):
    rep, frames = source
    actor = FrameActor(rep)
    history = frames[:, :4]
    root, pred = run(actor, history)
    for key, shape in dict(state=(2, 256), root_delta=(2, 3),
                           path_world_xz=(2, 8, 2), embedding=(2, 256)).items():
        assert root[key].shape == shape and torch.isfinite(root[key]).all()
    for key, shape in dict(root_next=(2, 3), local_rot6d=(2, 30, 6),
                           contact_logits=(2, 4), state=(2, 256)).items():
        assert pred[key].shape == shape and torch.isfinite(pred[key]).all()
    actual = rep.inverse(history, False, return_all=False)["local_joint_rots"][:, -1]
    torch.testing.assert_close(cont6d_to_matrix(pred["local_rot6d"]), actual, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(pred["root_next"], history[:, -1, :3] + root["root_delta"])
    assert pred["state"] is root["state"]
    torch.testing.assert_close(actor.path_times_s, torch.arange(1, 9) * .1)


def test_causal_explicit_state_no_command_reset_or_hidden_mutation(source):
    rep, frames = source
    actor = FrameActor(rep)
    history = frames[:, :4]
    state = torch.randn(2, 256)
    before = state.clone()
    original = history.clone()
    root, pred = run(actor, history, state=state)
    again, _ = run(actor, history, state=state)
    torch.testing.assert_close(root["state"], again["state"], rtol=0, atol=0)
    torch.testing.assert_close(state, before, rtol=0, atol=0)
    torch.testing.assert_close(history, original, rtol=0, atol=0)
    zero = history.new_tensor([0., 0., 1., 0.]).expand(2, -1)
    continued, _ = run(actor, history, zero, pred["state"])
    restarted, _ = run(actor, history, zero, actor.initial_state(2))
    assert not torch.allclose(continued["state"], restarted["state"])
    repeated, _ = run(actor, history, zero, pred["state"])
    torch.testing.assert_close(continued["state"], repeated["state"], rtol=0, atol=0)
    assert "nexttruth" not in inspect.signature(actor.root_step).parameters
    assert "nexttruth" not in inspect.signature(actor.pose_step).parameters
    with pytest.raises(ValueError, match="Explicit recurrent state"):
        actor.root_step(history, zero, observation(), None)


def test_fixed_normalization_recenter_and_physical_height(source):
    rep, frames = source
    actor = FrameActor(rep)
    history = frames[:, :4]
    normalized = actor._history_input(history)
    raw = normalized * actor.history_scale + actor.history_mean
    torch.testing.assert_close(raw[..., 1], history[..., 1])
    torch.testing.assert_close(raw[..., [0, 2]], history[..., [0, 2]] - history[:, -1:, [0, 2]])
    torch.testing.assert_close(raw[..., 3:], history[..., 3:])
    assert not any(key.startswith("rep.") for key in actor.state_dict())
    rep_mean = rep.stats.mean.clone()
    actor.history_mean[9] += 1
    torch.testing.assert_close(rep.stats.mean, rep_mean)


def test_xz_translation_equivariance(source):
    rep, frames = source
    actor = FrameActor(rep)
    history = frames[:, :4]
    shift = history.new_tensor([16., 0., -8.])
    moved = history.clone()
    moved[..., :3] += shift
    root, pred = run(actor, history)
    shifted_root, shifted_pred = run(actor, moved)
    torch.testing.assert_close(root["state"], shifted_root["state"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(root["root_delta"], shifted_root["root_delta"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(root["path_world_xz"] + shift[[0, 2]], shifted_root["path_world_xz"])
    torch.testing.assert_close(pred["root_next"] + shift, shifted_pred["root_next"])
    torch.testing.assert_close(pred["local_rot6d"], shifted_pred["local_rot6d"])
    advanced = advance(rep, history, pred)
    advanced_shifted = advance(rep, moved, shifted_pred)
    expected = advanced.clone()
    expected[..., :3] += shift
    torch.testing.assert_close(advanced_shifted[..., :272], expected[..., :272], atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(advanced_shifted[..., -4:], expected[..., -4:])
    # World-coordinate float32 roundoff is amplified by 30 in velocity channels.
    torch.testing.assert_close(advanced_shifted[..., 272:362], expected[..., 272:362], atol=2e-4, rtol=1e-5)


def test_body_command_conversion_and_learned_path(source):
    rep, frames = source
    actor = FrameActor(rep, command_frame="body")
    history = frames[:, :4].clone()
    history[:, -1, 3:5] = history.new_tensor([0., 1.])
    command = history.new_tensor([1., 2., -.3, .9]).expand(2, -1)
    torch.testing.assert_close(actor._command(history, command), history.new_tensor([2., -1., -.3, .9]).expand(2, -1))
    with torch.no_grad():
        actor.path_head.weight.zero_()
        actor.path_head.bias.zero_()
    root, _ = run(actor, history, command)
    torch.testing.assert_close(root["path_world_xz"], history[:, -1, [0, 2]][:, None].expand(2, 8, 2))
    with pytest.raises(ValueError, match="command_frame"):
        FrameActor(rep, command_frame="unknown")


def test_all_unknown_observations_remain_finite(source):
    actor = FrameActor(source[0])
    empty = observation(invalid=True)
    torch.testing.assert_close(actor._terrain(empty), torch.zeros(2, 128))
    root, pred = run(actor, source[1][:, :4], coarse=empty, fine=empty)
    assert all(torch.isfinite(value).all() for value in (*root.values(), *pred.values()))


def test_advance_preserves_actual_pose_and_native_fk_channels(source):
    rep, frames = source
    actor = FrameActor(rep)
    history = frames[:, :4]
    _, pred = run(actor, history)
    emitted = advance(rep, history, pred)
    assert emitted.shape == (2, 4, 366)
    actual_pose = rep.inverse(history, False, return_all=False)
    fk = rep.inverse(emitted, False, return_all=False, joint_positions_from="global_rot_data")
    ric = rep.inverse(emitted, False, return_all=False, joint_positions_from="ric_data")
    torch.testing.assert_close(fk["posed_joints"][:, :3], actual_pose["posed_joints"][:, 1:], atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(fk["local_joint_rots"][:, :3], actual_pose["local_joint_rots"][:, 1:], atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(fk["posed_joints"], ric["posed_joints"], atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(emitted[:, -1, :3], pred["root_next"], atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(fk["local_joint_rots"][:, -1], cont6d_to_matrix(pred["local_rot6d"]), atol=3e-5, rtol=1e-5)
    velocity = (fk["posed_joints"][:, 1:] - fk["posed_joints"][:, :-1]) * rep.fps
    torch.testing.assert_close(emitted[:, :-1, rep.indices["local_vel"]], velocity.flatten(-2), atol=3e-5, rtol=1e-5)


def test_source_loss_matches_moving_truth_not_zero_velocity(source):
    rep, frames = source
    history, truth = frames[:, :4], frames[:, 4].clone().requires_grad_()
    pose = rep.inverse(truth[:, None], False, return_all=False)
    pred = dict(root_next=truth[:, :3].detach().clone(),
                root_delta=truth[:, :3].detach() - history[:, -1, :3],
                local_rot6d=matrix_to_cont6d(pose["local_joint_rots"][:, 0]).detach(),
                contact_logits=(truth[:, rep.indices["foot_contacts"]].detach() * 2 - 1) * 12)
    losses = source_one_step_loss(rep, pred, truth, history)
    for key in ("root_delta", "body_fk", "rotation", "boundary_joint_velocity"):
        assert losses[key].item() < 1e-6, (key, losses[key].item())
    assert not losses["total"].requires_grad
    assert truth.grad is None


def test_gradients_at_initialization_and_after_first_cpu_test_update(source):
    rep, frames = source
    actor = FrameActor(rep)
    coarse, fine = observation(), observation()
    coarse["features"].requires_grad_()
    fine["features"].requires_grad_()
    optimizer = torch.optim.SGD(actor.parameters(), lr=.01)
    _, pred = run(actor, frames[:, :4], coarse=coarse, fine=fine)
    losses = source_one_step_loss(rep, pred, frames[:, 4], frames[:, :4])
    losses["total"].backward()
    for parameter in (actor.root_head.weight, actor.rotation_head.weight,
                      actor.gru.weight_ih, actor.terrain_encoder.point[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert coarse["features"].grad.abs().sum() > 0
    assert fine["features"].grad.abs().sum() > 0  # contact head is already active
    optimizer.step()  # Isolated CPU unit fixture, not an experiment or data run.
    optimizer.zero_grad(set_to_none=True)
    fine["features"].grad = None
    _, pred = run(actor, frames[:, :4], coarse=coarse, fine=fine)
    source_one_step_loss(rep, pred, frames[:, 4], frames[:, :4])["body_fk"].backward()
    assert actor.pose[0].weight.grad.abs().sum() > 0
    assert fine["features"].grad.abs().sum() > 0


def test_state_gradient_crosses_frames(source):
    rep, frames = source
    actor = FrameActor(rep)
    first, pred = run(actor, frames[:, :4])
    first["state"].retain_grad()
    history = advance(rep, frames[:, :4], pred)
    second, _ = run(actor, history, state=first["state"])
    second["root_delta"].square().sum().backward()
    assert first["state"].grad.abs().sum() > 0


def test_immediate_command_and_path_head_gradients(source):
    actor = FrameActor(source[0])
    history = source[1][:, :4]
    command = history.new_tensor([[.4, .7, 1., 0.], [-.2, .5, 0., 1.]]).requires_grad_()
    root, _ = run(actor, history, command)
    root["root_delta"].square().sum().backward(retain_graph=True)
    assert torch.all(command.grad.abs().sum(0) > 0)
    root["path_world_xz"].square().sum().backward()
    assert actor.path_head.weight.grad.abs().sum() > 0


def test_encoder_load_is_scoped_shape_checked_and_frozen(source):
    actor = FrameActor(source[0])
    prefix = "adaptation.encoder."
    checkpoint = {prefix + key: value.clone() + .125 for key, value in actor.terrain_encoder.state_dict().items()}
    checkpoint["root.unrelated"] = torch.tensor(float("nan"))
    before = actor.root_head.weight.detach().clone()
    names = load_environment_encoder(actor, checkpoint)
    assert names and all(name.startswith(prefix) for name in names)
    assert all(not p.requires_grad for p in actor.terrain_encoder.parameters())
    torch.testing.assert_close(before, actor.root_head.weight)
    for key, value in actor.terrain_encoder.state_dict().items():
        torch.testing.assert_close(value, checkpoint[prefix + key])
    load_environment_encoder(actor, checkpoint, freeze=False)
    assert all(p.requires_grad for p in actor.terrain_encoder.parameters())
    bad = dict(checkpoint)
    bad[prefix + "point.0.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shapes"):
        load_environment_encoder(actor, bad)
    with pytest.raises(ValueError, match="keys"):
        load_environment_encoder(actor, {})


def test_tensor_export_root_and_pose_steps(source):
    class Pipeline(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, history, command, coarse, fine, state):
            root = self.actor.root_step(history, command, coarse, state)
            return root, self.actor.pose_step(history, command, fine, root)

    actor = FrameActor(source[0]).eval()
    pipeline = Pipeline(actor)
    args = (source[1][:, :4], torch.tensor([[.4, .7, 1., 0.]]).expand(2, -1),
            observation(), observation(), actor.initial_state(2))
    batch = torch.export.Dim("batch", min=1, max=4)
    shapes = ({0: batch}, {0: batch}, {key: {0: batch} for key in args[2]},
              {key: {0: batch} for key in args[3]}, {0: batch})
    exported = torch.export.export(pipeline, args, dynamic_shapes=shapes)
    expected, actual = pipeline(*args), exported.module()(*args)
    for expected_stage, actual_stage in zip(expected, actual):
        for key in expected_stage:
            torch.testing.assert_close(actual_stage[key], expected_stage[key])
    singleton = (args[0][:1], args[1][:1], {key: value[:1] for key, value in args[2].items()},
                 {key: value[:1] for key, value in args[3].items()}, args[4][:1])
    root, pred = exported.module()(*singleton)
    assert root["state"].shape == (1, 256) and pred["local_rot6d"].shape == (1, 30, 6)
