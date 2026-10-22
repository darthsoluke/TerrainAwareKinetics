"""CPU contracts for the A/B/C ablation actor; real SOMA FK, no CUDA."""

import pytest
import torch

from motionbricks.data.pfnn_frame_actor import advance, source_one_step_loss
from motionbricks.data.pfnn_frame_actor_v2 import BACKBONES, GEOMETRIES, FrameActorV2
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix
from motionbricks.motionlib.core.utils.stats import Stats


FRAMES = 6


@pytest.fixture(scope="module")
def source():
    stats = Stats()
    stats.register_from_tensors(torch.linspace(-.2, .2, 366), torch.linspace(.5, 1.5, 366))
    rep = GlobalRootGlobalJoints(30., SOMASkeleton30(), "somaskel30_global_root_global_joints", stats=stats)
    time = torch.arange(FRAMES + 1, dtype=torch.float32)
    rotations = torch.eye(3).expand(2, FRAMES + 1, 30, 3, 3).clone()
    rotations[:, :, 0] = angle_to_Y_rotation_matrix(.4 + .02 * time)
    rotations[:, :, 5] = angle_to_Y_rotation_matrix(-.1 + .04 * time)
    roots = torch.stack((3. + .03 * time, 1.1 + .002 * time, -4. + .015 * time), -1)
    roots = roots[None].expand(2, -1, -1).clone()
    contacts = torch.tensor([1., 0., 1., 0.]).expand(2, FRAMES + 1, 4).clone()
    features, origin = rep(dict(local_joint_rots=rotations, translation=roots, foot_contacts=contacts),
                           to_normalize=False, return_init_heading_info=True,
                           lengths=torch.full((2,), FRAMES + 1))
    quat = origin["init_heading_quat"]
    features = rep.change_first_heading(features, 2 * torch.atan2(quat[..., 2], quat[..., 0]), False, False)
    features[..., [0, 2]] += roots[:, :1, [0, 2]]
    return rep, features


def observation(batch=2, *, invalid=False):
    generator = torch.Generator().manual_seed(83)
    return dict(features=torch.randn(batch, 3, 512, 15, generator=generator),
                distance=torch.randn(batch, 3, 512, generator=generator),
                valid=torch.full((batch, 3, 512), not invalid, dtype=torch.bool))


def actor_for(rep, backbone, geometry):
    return FrameActorV2(rep, backbone=backbone, geometry=geometry, frames=FRAMES, width=32,
                        layers=2, heads=2, feedforward=64)


def step(actor, history, command=None):
    batch = history.shape[0]
    if command is None:
        command = history.new_tensor([.4, .7, 1., 0.]).expand(batch, -1)
    state = actor.initial_state(batch)
    root = actor.root_step(history, command, observation(batch), state)
    return root, actor.pose_step(history, command, observation(batch), root)


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_every_arm_emits_the_same_contract(source, backbone, geometry):
    rep, frames = source
    actor = actor_for(rep, backbone, geometry)
    root, pred = step(actor, frames[:, :FRAMES])
    assert root["root_delta"].shape == (2, 3)
    assert root["path_world_xz"].shape == (2, 8, 2)
    assert root["embedding"].shape == (2, 256)
    assert pred["local_rot6d"].shape == (2, 30, 6)
    assert pred["contact_logits"].shape == (2, 4)
    for value in (root["root_delta"], root["path_world_xz"], pred["local_rot6d"],
                  pred["contact_logits"], pred["root_next"]):
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_every_arm_produces_finite_losses_and_gradients(source, backbone, geometry):
    rep, frames = source
    actor = actor_for(rep, backbone, geometry)
    history = frames[:, :FRAMES]
    root, pred = step(actor, history)
    losses = source_one_step_loss(rep, pred, frames[:, FRAMES], history)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    moved = [name for name, parameter in actor.named_parameters()
             if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0]
    assert moved, "no parameter received gradient"
    assert any(name.startswith("terrain_encoder.") for name in moved)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_geometry_stage_masks_an_entirely_unknown_observation(source, geometry):
    rep, frames = source
    actor = actor_for(rep, "gru", geometry)
    history = frames[:, :FRAMES]
    command = history.new_tensor([.4, .7, 1., 0.]).expand(2, -1)
    state = actor.initial_state(2)
    root = actor.root_step(history, command, observation(2, invalid=True), state)
    assert torch.isfinite(root["root_delta"]).all()


def test_attention_arm_starts_identical_to_the_mean_arm(source):
    """The attention residual is zero-initialised, so B must reproduce A exactly."""
    rep, frames = source
    torch.manual_seed(7)
    mean_actor = actor_for(rep, "gru", "mean")
    attention_actor = actor_for(rep, "gru", "attention")
    shared = {k: v for k, v in mean_actor.state_dict().items()
              if not k.startswith(("coarse_attention.", "fine_attention."))}
    attention_actor.load_state_dict(shared, strict=False)
    observation_value = observation(2)
    query = torch.randn(2, 256, generator=torch.Generator().manual_seed(11))
    tokens, valid = mean_actor.terrain_encoder(observation_value)
    mean_geometry = mean_actor._geometry(query, tokens, valid, "coarse")
    attention_geometry = attention_actor._geometry(query, tokens, valid, "coarse")
    torch.testing.assert_close(mean_geometry, attention_geometry, rtol=0, atol=0)
    with torch.no_grad():
        attention_actor.coarse_attention.output.bias.fill_(.5)
    moved = attention_actor._geometry(query, tokens, valid, "coarse")
    assert not torch.allclose(moved, mean_geometry)


def test_window_length_mismatch_is_a_loud_error(source):
    rep, frames = source
    actor = actor_for(rep, "gru", "mean")
    with pytest.raises(ValueError, match="history frames"):
        step(actor, frames[:, :FRAMES - 1])


def test_transformer_arm_is_stateless_and_gru_arm_is_not(source):
    rep, frames = source
    assert actor_for(rep, "transformer", "mean").initial_state(2) is None
    assert actor_for(rep, "gru", "mean").initial_state(2).shape == (2, 256)
    with pytest.raises(ValueError, match="recurrent state"):
        actor = actor_for(rep, "gru", "mean")
        actor.root_step(frames[:, :FRAMES], frames.new_tensor([.4, .7, 1., 0.]).expand(2, -1),
                        observation(2), None)


def test_advance_reuses_the_shared_frame_semantics(source):
    rep, frames = source
    actor = actor_for(rep, "transformer", "attention")
    history = frames[:, :FRAMES]
    _, pred = step(actor, history)
    grown = advance(rep, history, pred)
    assert grown.shape == (2, FRAMES, 366)
    assert torch.isfinite(grown).all()
