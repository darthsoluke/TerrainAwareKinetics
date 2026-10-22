from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from motionbricks.data.bones_control import BonesControlRootModel, BonesPoseModel, SomaMotionInference, state_digest
from motionbricks.data.bones_models import ConditionedBonesVQVAE
from motionbricks.data.bones_seed import build_representation


def test_native_root_stage_matches_training_derivatives_without_partial_prefix_override(monkeypatch):
    from motionbricks.data.bones_rollout import control_inputs
    from motionbricks.motion_backbone.inference.motion_inference import motion_inference

    rep, root, pose = models()
    inference = SomaMotionInference({"root": root, "pose": pose},
                                    {"min_tokens": 6, "max_tokens": 16,
                                     "cond_root_feature_is_from_motion_rep": True}, device="cpu")
    observed = torch.zeros(1, 4, len(rep.indices["all"]))
    observed[..., 0] = torch.arange(4) * .02
    observed[..., 1] = 1
    observed[..., 3] = 1
    goal = observed[..., :5].clone()
    values = control_inputs(observed, goal, torch.zeros(1, 2), rep)
    keys = ("global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values",
            "local_poses", "has_local_poses")
    batch = dict(zip(keys, values))
    batch.update(num_tokens=torch.tensor([16]), text_embeddings=None, has_text_embeddings=None,
                 allowed_pred_num_tokens=None)
    trajectory = torch.zeros(1, 64, 5)
    trajectory[..., 0] = torch.arange(64) * .02 + .1
    trajectory[..., 1] = 1
    trajectory[..., 3] = 1
    monkeypatch.setattr(root.backbone_net, "forward", lambda *args, **kwargs:
                        dict(pred_num_tokens=torch.tensor([16]), pred_global_root_values=trajectory.clone()))
    inference.use_native_root_stage = True
    native = inference._predict_root_trajectories(batch, {})
    official = motion_inference._predict_root_trajectories(inference, batch, {})
    torch.testing.assert_close(native[0].flatten(), official[0].flatten())
    for a, b in zip(native[1:], official[1:]):
        torch.testing.assert_close(a, b)
    inference.use_native_root_stage = False
    legacy = inference._predict_root_trajectories(batch, {})
    torch.testing.assert_close(native[1][:, 4:] , legacy[1][:, 4:])
    # The same future and constant 10 cm model bias produce a fake 6 m/s step
    # only when inference rewrites the history prefix before differentiation.
    assert native[2][0, 3, 1].item() == pytest.approx(1.2, abs=1e-4)
    assert legacy[2][0, 3, 1].item() == pytest.approx(7.2, abs=1e-4)


def models(history=False):
    rep = build_representation()
    from motionbricks.motionlib.core.utils.stats import Stats
    for representation in (rep, rep.dual_rep, rep.dual_rep.global_motion_rep, rep.dual_rep.local_motion_rep):
        stats = Stats()
        stats.register_from_tensors(torch.zeros(len(representation.indices["all"])),
                                    torch.ones(len(representation.indices["all"])))
        representation.stats = stats
    project = Path(__file__).resolve().parents[1]
    pose_config = OmegaConf.load(project / "out/motionbricks_pose/version_1/hparams.yaml")
    root_config = OmegaConf.load(project / "out/motionbricks_root/version_1/hparams.yaml")
    for config in (pose_config, root_config):
        with open_dict(config):
            config.data = {"text_embeddings": None}
            config.model.args.min_tokens = 6
            config.model.args.max_tokens = 16
            config.model.args.batchsize_mul_factor = 1
            config.model.args.prob_provide_num_tokens = 1.0
            config.model.backbone_network.args.n_embd = 64
            config.model.backbone_network.args.n_head = 2
    with open_dict(pose_config):
        pose_config.model.backbone_network.args.n_layers = 1
        pose_config.model.pose_vqvae_network.num_heads = 2
        pose_config.model.pose_vqvae_network.nb_code = 16
        pose_config.model.pose_vqvae_network.code_dim = 16
    with open_dict(root_config):
        root_config.model.backbone_network.args.n_layers_shared = 1
        root_config.model.backbone_network.args.n_layers_root_token = 1
        root_config.model.backbone_network.args.width = 16
        root_config.model.backbone_network.args.depth = 1
    tokenizer = ConditionedBonesVQVAE(
        pose_root_mode="pose", motion_rep=rep.dual_rep.local_motion_rep,
        encoder_state_dim=0, decoder_state_dim=0, decoder_target_cond_dim=0, decoder_external_cond_dim=0,
        feature_mode=["joint_positions_and_rotations", "pose", "joint_positions_and_rotations", "root_without_hip_height"],
        code_dim=16, output_emb_width=16, width=16, depth=1, down_t=2, stride_t=2,
        nb_code=16, num_heads=2, kmeans_init=False)
    pose_net = instantiate(pose_config.model.backbone_network, motion_rep=rep, _recursive_=False)
    pose_class = BonesPoseModel
    if history:
        from motionbricks.data.bones_history import BonesHistoryPoseModel
        pose_class = BonesHistoryPoseModel
    pose = pose_class(pose_vqvae_network=tokenizer, root_vqvae_network=None, backbone_network=pose_net,
                          motion_rep=rep, args=OmegaConf.to_container(pose_config.model.args, resolve=True))
    root_net = instantiate(root_config.model.backbone_network, motion_rep=rep, _recursive_=False)
    root = BonesControlRootModel(pose_vqvae_network=None, root_vqvae_network=None, backbone_network=root_net,
                                 motion_rep=rep, args=OmegaConf.to_container(root_config.model.args, resolve=True))
    return rep, root, pose


def source_motion(rep, frames=65):
    rotations = torch.eye(3).expand(frames, 30, 3, 3).clone()
    root = torch.zeros(frames, 3)
    root[:, 1] = 0.9
    root[:, 2] = torch.arange(frames) / 60
    global_rot, positions, _ = rep.skeleton.fk(rotations, root)
    return rep.normalize(rep({"posed_joints": positions, "global_joint_rots": global_rot,
                              "local_joint_rots": rotations, "translation": root,
                              "foot_contacts": torch.ones(frames, 4)}, to_normalize=False))[None]


def test_pose_real_training_updates_backbone_and_freezes_tokenizer():
    torch.set_num_threads(2)
    rep, _, pose = models()
    pose.train()
    assert not pose.pose_net.training
    assert callable(pose.pose_net.requires_grad_)
    assert not any(parameter.requires_grad for parameter in pose.pose_net.parameters())
    motion = source_motion(rep)
    batch = {"motion": motion, "motion_len": torch.tensor([65]),
             "motion_pad_mask": torch.ones(1, 65, dtype=torch.bool), "batch_size": 1}
    before = state_digest(pose.pose_net)
    pose.log = lambda *args, **kwargs: None
    np.random.seed(8)
    loss = pose.training_step(batch, 0)
    assert (batch["input_tokens"] == pose.backbone_net.POSE_MASK_ID).all()
    assert batch["focused_token_mask"].all()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pose.backbone_net.parameters())
    assert all(p.grad is None for p in pose.pose_net.parameters())
    assert state_digest(pose.pose_net) == before
    optimizer = torch.optim.Adam(pose.parameters(), lr=1e-4)
    optimizer.step()
    assert state_digest(pose.pose_net) == before
    restored = models()[2]
    restored.load_state_dict(pose.state_dict(), strict=True)
    assert state_digest(restored) == state_digest(pose)


def test_pose_predictions_ignore_unavailable_future_body():
    rep, _, pose = models()
    pose.eval()
    motion = source_motion(rep)
    local = rep.dual_rep.global_to_local(motion, True, True, lengths=torch.tensor([65]))[:, :64]
    changed = local.clone()
    changed[:, 4:, 4:] += 123
    with torch.inference_mode():
        expected = pose.predict_local(local, motion[:, :64])
        actual = pose.predict_local(changed, motion[:, :64])
    torch.testing.assert_close(expected[0]["pose_logits"], actual[0]["pose_logits"])
    torch.testing.assert_close(expected[1], actual[1])


def test_root_last_history_velocity_uses_past_and_no_future_height():
    _, root, _ = models()
    local = torch.arange(64 * 4).reshape(1, 64, 4).float()
    value, _ = root._sample_the_conditions(local, 64)
    torch.testing.assert_close(value[:, 3, :3], local[:, 2, :3])
    torch.testing.assert_close(value[:, 4:, 3], local[:, 3:4, 3].expand(-1, 4))
    pose = torch.randn(1, 64, 268)
    value, mask = root._sample_the_conditions(pose, 64)
    assert not mask[:, 4:].any()
    assert not value[:, 4:].any()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_control_models_complete_official_inference_with_soma_contract(batch_size):
    rep, root, pose = models()
    inference = SomaMotionInference({"root": root, "pose": pose},
                                    {"min_tokens": 6, "max_tokens": 16,
                                     "cond_root_feature_is_from_motion_rep": "global"}, device="cpu")
    source = rep.unnormalize(source_motion(rep))[0]
    local = rep.dual_rep.global_to_local(source[None, :4], False, False, lengths=torch.tensor([4]))
    global_root = torch.cat((source[None, :4, :5], source[None, 60:64, :5]), 1)
    local_root = torch.cat((local[..., :4], local[..., :4]), 1)
    body = torch.cat((source[None, :4, rep.indices["ric_data"]],
                      source[None, :4, rep.indices["global_rot_data"]]), -1)
    poses = torch.cat((body, torch.zeros_like(body)), 1)
    root_mask = torch.ones(1, 8, dtype=torch.bool)
    pose_mask = root_mask.clone()
    pose_mask[:, 4:] = False
    if batch_size == 2:
        global_root = global_root.repeat(2, 1, 1)
        global_root[1, :, 0] += 3
        local_root, poses = local_root.repeat(2, 1, 1), poses.repeat(2, 1, 1)
        root_mask, pose_mask = root_mask.repeat(2, 1), pose_mask.repeat(2, 1)
    with torch.inference_mode():
        output, lengths = inference.predict(global_root, root_mask, local_root, root_mask,
                                            poses, pose_mask, 16,
                                            config={"pose_token_sampling_use_argmax": True})
    assert output.shape == (batch_size, 64, 366)
    assert (lengths == 16).all()
    assert torch.isfinite(output).all()
    assert torch.isfinite(rep.inverse(output, False, return_all=False)["posed_joints"]).all()
    if batch_size == 2:
        torch.testing.assert_close(output[1, :, 0], output[0, :, 0] + 3, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(output[1, :, 1:], output[0, :, 1:], atol=1e-5, rtol=1e-5)


def test_rollout_retains_only_executed_history_and_never_rewrites_output():
    from motionbricks.data.bones_rollout import SomaLocomotionBackend, control_inputs
    rep, _, _ = models()
    history = rep.unnormalize(source_motion(rep))[:, :4]
    endpoint = history[..., :5].clone()
    endpoint[..., 1] = 999
    inputs = control_inputs(history, endpoint, torch.zeros(1, 2), rep)
    assert (inputs[0][:, 4:, 1] == history[:, -1:, 1]).all()
    assert not inputs[-1][:, 4:].any()
    assert not inputs[-2][:, 4:].any()

    class FakeInference:
        motion_rep = rep

        def predict(self, *inputs, **kwargs):
            output = history[:, :1].repeat(1, 64, 1)
            output[:, 4:8, 0] = inputs[0][:, 3:4, 0] + 0.1
            return output, torch.tensor([[16]])

    backend = SomaLocomotionBackend(FakeInference(), history)
    original = backend.history.clone()
    first = backend.step([[1.0, 0.0]], [0.0])
    published = first["motion"].clone()
    torch.testing.assert_close(history, original)
    torch.testing.assert_close(backend.history, published)
    second = backend.step([[0.0, 0.0]], [1.0])
    torch.testing.assert_close(first["motion"], published)
    assert second["start_frame"] == 4
    assert (second["motion"][..., 0] > published[..., 0]).all()


@pytest.mark.parametrize("policy", ["teacher", "generated"])
def test_adjacent_history_training_updates_pose_without_future_body_inputs(policy):
    from types import SimpleNamespace
    rep, _, pose = models(history=True)
    pose.args.update(history_policy=policy, history_max_probability=1.0, history_ramp_steps=1)
    pose._trainer = SimpleNamespace(global_step=10000)
    pose.log = lambda *args, **kwargs: None
    raw = rep.unnormalize(source_motion(rep, 137))

    class FakeGenerator:
        def predict(self, *inputs, **kwargs):
            assert not inputs[5][:, 4:].any()
            assert not inputs[4][:, 4:].any()
            output = raw[:, :128].clone()
            output[:, 4:8, 0] += 0.2
            return output, torch.tensor([[32]])

    pose.history_generator = lambda device: FakeGenerator()
    batch = {"motion": rep.normalize(raw), "motion_len": torch.tensor([137])}
    before = state_digest(pose.pose_net)
    loss = pose.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pose.backbone_net.parameters())
    assert state_digest(pose.pose_net) == before
    assert pose.history_counts[policy] == 1


def test_target_transport_is_common_rigid_transform_and_keeps_ground_height():
    from motionbricks.data.bones_history import transport_target_to_history
    rep, _, _ = models()
    source = source_motion(rep, 65)
    generated = rep.unnormalize(source[:, :4]).clone()
    generated[..., 0] += 2
    generated[..., 3] = 0
    generated[..., 4] = 1
    transported = rep.unnormalize(transport_target_to_history(source, generated, rep))
    original = rep.unnormalize(source)
    torch.testing.assert_close(transported[..., 1], original[..., 1])
    torch.testing.assert_close(transported[:, :1, [0, 2]], generated[:, :1, [0, 2]], atol=1e-5, rtol=1e-5)
    joints = rep.inverse(transported, False, return_all=True)["posed_joints"]
    original_joints = rep.inverse(original, False, return_all=True)["posed_joints"]
    torch.testing.assert_close(joints[..., 1], original_joints[..., 1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close((joints[:, 1:, 0] - joints[:, :-1, 0]).norm(dim=-1),
                               (original_joints[:, 1:, 0] - original_joints[:, :-1, 0]).norm(dim=-1),
                               atol=1e-5, rtol=1e-5)


def test_padded_tokens_cannot_change_valid_decoded_motion():
    from motionbricks.data.bones_control import decode_tokens
    rep, _, pose = models()
    pose.eval()
    local = rep.dual_rep.global_to_local(source_motion(rep), True, True, lengths=torch.tensor([65]))[:, :64]
    ids = torch.zeros(1, 16, 2, dtype=torch.long)
    changed = ids.clone()
    changed[:, 8:] = pose.backbone_net.get_num_codes(include_aug_tokens=False)[0] - 1
    mask = torch.arange(16)[None] < 8
    pose_mask = torch.zeros(1, 64, dtype=torch.bool)
    pose_mask[:, :4] = True
    with torch.inference_mode():
        first = decode_tokens(pose.pose_net, local, pose_mask, ids, local, mask)
        second = decode_tokens(pose.pose_net, local, pose_mask, changed, local, mask)
    torch.testing.assert_close(first[:, :32], second[:, :32])


def test_stop_goal_brakes_ahead_instead_of_returning_to_current_position():
    from motionbricks.data.bones_rollout import intent_endpoint
    rep, _, _ = models()
    history = rep.unnormalize(source_motion(rep))[:, :4]
    history[..., 0] = 0
    history[..., 2] = torch.arange(4) / 60
    goal, velocities = intent_endpoint(history, [[0.0, 0.0]], [0.0], 64)
    assert (goal[..., 2] > history[:, -1:, 2]).all()
    torch.testing.assert_close(goal[:, -1, 2] - history[:, -1, 2], torch.tensor([1 / 6]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(velocities, torch.zeros_like(velocities))


def test_input_goal_acceleration_is_bounded_without_imposing_generated_root():
    from motionbricks.data.bones_rollout import intent_endpoint
    rep, _, _ = models()
    history = rep.unnormalize(source_motion(rep))[:, :4].clone()
    history[..., [0, 2]] = 0
    goal, velocity = intent_endpoint(history, [[10.0, 0.0]], [0.0], 64)
    assert velocity[:, -1, 0].item() == pytest.approx(3.0)
    assert (goal[:, -1, 0] - history[:, -1, 0]).item() == pytest.approx(1.5)
