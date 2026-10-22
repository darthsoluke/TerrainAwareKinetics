import pytest
import torch

from motionbricks.data.soma_continuation import continuation_losses, context_tokens, decoder_training_loss, decode_logits_straight_through
from scripts.diagnose_soma_continuity import angular_steps, stats
from scripts.train_soma_continuation import training_records
from test_bones_control import models, source_motion


def fixture(frames=12):
    time = torch.arange(frames, dtype=torch.float64) / 60
    positions = torch.zeros(1, frames, 3, 3, dtype=torch.float64)
    positions[..., 0] = (time + 2 * time.square())[None, :, None]
    rotations = torch.eye(3, dtype=torch.float64).expand(1, frames, 3, 3, 3).clone()
    target = {"posed_joints": positions, "local_joint_rots": rotations}
    history = {key: value[:, :4].clone() for key, value in target.items()}
    return target, history


def test_exact_accelerating_motion_is_not_penalized_for_moving():
    target, history = fixture()
    losses = continuation_losses(target, target, history, 60)
    assert all(value.item() == 0 for value in losses.values())
    assert target["posed_joints"].diff(n=2, dim=1).abs().max() > 0


def test_boundary_uses_actual_history_not_decoder_reconstructed_history():
    target, history = fixture()
    prediction = {key: value.clone() for key, value in target.items()}
    # The entire decoded sequence is translated: its internal velocity looks correct.
    prediction["posed_joints"] += .1
    assert torch.allclose(prediction["posed_joints"].diff(dim=1), target["posed_joints"].diff(dim=1))
    losses = continuation_losses(prediction, target, history, 60)
    assert losses["l_continuation_velocity_mps"] > 2
    assert losses["l_continuation_acceleration_mps2"] > 100


def test_labels_are_detached_and_only_published_future_receives_temporal_gradient():
    target, history = fixture()
    target["posed_joints"].requires_grad_()
    history["posed_joints"].requires_grad_()
    positions = target["posed_joints"].detach().clone().requires_grad_()
    prediction = {"posed_joints": positions + .01, "local_joint_rots": target["local_joint_rots"]}
    losses = continuation_losses(prediction, target, history, 60)
    (losses["l_continuation_velocity_mps"] + losses["l_continuation_acceleration_mps2"]).backward()
    assert positions.grad[:, 4:8].abs().sum() > 0
    assert positions.grad[:, :4].abs().sum() == 0
    assert positions.grad[:, 8:].abs().sum() == 0
    assert target["posed_joints"].grad is None
    assert history["posed_joints"].grad is None


def test_predicted_tokens_cannot_read_source_future_body():
    rep, _, pose = models()
    pose.eval()
    raw = source_motion(rep)
    local = rep.dual_rep.global_to_local(raw, True, True, lengths=torch.tensor([65]))[:, :64]
    changed = local.clone()
    changed[:, 4:, 4:] += 100
    first = context_tokens(pose, local, raw[:, :64], True)
    second = context_tokens(pose, changed, raw[:, :64], True)
    torch.testing.assert_close(first, second)


def test_real_decoder_training_path_has_gradients_and_weighted_temporal_terms():
    torch.set_num_threads(2)
    rep, _, pose = models()
    pose.eval()
    pose.requires_grad_(False)
    pose.pose_net.decoder.requires_grad_(True)
    raw = source_motion(rep)
    local = rep.dual_rep.global_to_local(raw, True, True, lengths=torch.tensor([65]))[:, :64]
    confidence = torch.ones(1, 64, 4)
    a, _, _ = decoder_training_loss(pose, local, raw[:, :64], confidence,
                                    {"enabled": False, "predicted_tokens": True})
    b, terms, weights = decoder_training_loss(pose, local, raw[:, :64], confidence,
                                             {"enabled": True, "predicted_tokens": True})
    assert b > a
    torch.testing.assert_close(b, sum(terms[key] * value for key, value in weights.items()))
    b.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pose.pose_net.decoder.parameters())
    for module in (pose.backbone_net, pose.pose_net.encoder, pose.pose_net.quantizer):
        assert all(p.grad is None for p in module.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in pose.pose_net.decoder.parameters())


def test_training_selection_never_uses_validation_or_test():
    records = [{"split": split, "frames_60hz": frames} for split, frames in
               (("train", 129), ("validation", 150), ("test", 140), ("train", 128))]
    assert training_records(records) == [records[0]]
    with pytest.raises(ValueError):
        training_records(records[1:])


def test_angular_metric_does_not_hide_one_bad_joint_in_average():
    rotations = torch.eye(3).expand(2, 30, 3, 3).clone()
    rotations[1, 7] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    result = stats(angular_steps(rotations))
    assert result["max"] == pytest.approx(90)
    assert result["mean"] == pytest.approx(3)


def test_short_boundary_contract_fails_explicitly():
    target, history = fixture(6)
    with pytest.raises(ValueError):
        continuation_losses(target, target, history, 60)


def test_straight_through_forward_is_exact_native_hard_token_decode():
    from motionbricks.data.bones_control import decode_tokens
    rep, _, pose = models()
    pose.eval().requires_grad_(False)
    raw = source_motion(rep)
    local = rep.dual_rep.global_to_local(raw, True, True, lengths=torch.tensor([65]))[:, :64]
    logits = torch.randn(1, 16, 2, 4, requires_grad=True)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[:, :4] = True
    actual = decode_logits_straight_through(pose.pose_net, logits, local, mask)
    expected = decode_tokens(pose.pose_net, local * mask[..., None], mask, logits.argmax(-1), local)
    torch.testing.assert_close(actual, expected, atol=1.e-6, rtol=1.e-6)
    actual.square().mean().backward()
    assert logits.grad.abs().sum() > 0
    assert torch.isfinite(logits.grad).all()
    assert all(parameter.grad is None for parameter in pose.pose_net.parameters())


def test_native_pose_supervision_reaches_transformer_not_frozen_decoder():
    rep, _, pose = models()
    pose.eval().requires_grad_(False)
    pose.backbone_net.requires_grad_(True)
    raw = source_motion(rep)
    local = rep.dual_rep.global_to_local(raw, True, True, lengths=torch.tensor([65]))[:, :64]
    loss, terms, weights = decoder_training_loss(pose, local, raw[:, :64], torch.ones(1, 64, 4),
                         {"enabled": True, "predicted_tokens": True, "train_pose": True})
    torch.testing.assert_close(loss, sum(terms[key] * value for key, value in weights.items()))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pose.backbone_net.parameters())
    assert all(p.grad is None for p in pose.pose_net.parameters())
