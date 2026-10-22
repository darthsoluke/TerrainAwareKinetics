import pytest
import torch

from motionbricks.data.bones_models import BonesTokenizer
from motionbricks.data.soma_continuation import continuation_losses
from motionbricks.data.soma_decoder_refinement import history_in_input_frame, source_target_with_history
from motionbricks.data.soma_joint_rollout import deployed_reference, joint_rollout_loss
from test_soma_joint_rollout import configured
from test_soma_continuation import fixture as pose_fixture


def test_generated_translation_cannot_invent_groundtruth_velocity():
    rep, _, _, _, raw, _ = configured()
    history = raw[:, :4].clone()
    history[..., 0] += .1
    global_target, local_target = source_target_with_history(raw, history, rep)
    normal_global, normal_local = source_target_with_history(raw, raw[:, :4], rep)
    torch.testing.assert_close(local_target, normal_local, atol=1e-5, rtol=1e-5)
    target = rep.inverse(global_target, True, return_all=False, joint_positions_from="ric_data")["posed_joints"]
    normal = rep.inverse(normal_global, True, return_all=False, joint_positions_from="ric_data")["posed_joints"]
    torch.testing.assert_close(target.diff(dim=1), normal.diff(dim=1), atol=2e-7, rtol=1e-5)
    legacy, _ = source_target_with_history(raw, history, rep, legacy_splice=True)
    broken = rep.inverse(legacy, True, return_all=False, joint_positions_from="ric_data")["posed_joints"]
    extra_speed = (broken.diff(dim=1) - target.diff(dim=1))[0, 3, :, 0] * rep.fps
    torch.testing.assert_close(extra_speed, torch.full_like(extra_speed, -6.), atol=1e-4, rtol=0)


def test_generated_body_pose_and_height_are_never_copied_to_targets():
    rep, _, _, _, raw, _ = configured()
    history = raw[:, :4].clone()
    history[..., 1] += .05
    history[..., 3:] += .05
    actual = source_target_with_history(raw, history, rep)
    expected = source_target_with_history(raw, raw[:, :4], rep)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_shared_world_translation_changes_neither_input_nor_target():
    rep, _, _, _, raw, _ = configured()
    history = raw[:, :4].clone()
    history[..., 0] += .2
    shifted, shifted_history = raw.clone(), history.clone()
    shifted[..., [0, 2]] += torch.tensor([7., -3.])
    shifted_history[..., [0, 2]] += torch.tensor([7., -3.])
    for a, b in zip(source_target_with_history(raw, history, rep),
                    source_target_with_history(shifted, shifted_history, rep)):
        torch.testing.assert_close(a, b, atol=8e-5, rtol=1e-5)
    torch.testing.assert_close(history_in_input_frame(history), history_in_input_frame(shifted_history), atol=8e-7, rtol=1e-5)


def test_continuing_drift_is_not_a_velocity_impulse_but_position_error_remains():
    target, history = pose_fixture()
    history["posed_joints"][..., 0] += .1
    smooth = {key: value.clone() for key, value in target.items()}
    smooth["posed_joints"][..., 0] += .1
    terms = continuation_losses(smooth, target, history, 60)
    assert terms["l_continuation_velocity_mps"] < 1e-10
    assert terms["l_continuation_acceleration_mps2"] < 1e-10
    assert terms["l_boundary_position_m"] == pytest.approx(.1)
    snap = continuation_losses(target, target, history, 60)
    assert snap["l_continuation_velocity_mps"] > 1.
    assert snap["l_continuation_acceleration_mps2"] > 80.


def test_reference_rotation_increment_does_not_use_generated_history():
    target, history = pose_fixture()
    rotation = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=torch.float64)
    history["local_joint_rots"][:] = rotation
    smooth = {key: value.clone() for key, value in target.items()}
    smooth["local_joint_rots"][:] = rotation
    assert continuation_losses(smooth, target, history, 60)["l_rotation_increment_chordal"] == 0
    assert continuation_losses(target, target, history, 60)["l_rotation_increment_chordal"] > .1


def test_real_loss_preserves_world_position_error_and_uses_actual_boundary():
    rep, _, pose, _, raw, trainer = configured()
    history = raw[:, :4].clone()
    history[..., 0] += .1
    global_target, local = source_target_with_history(raw, history, rep)
    # This decoded motion continues the shifted input smoothly. It is still 10 cm
    # away from the unmodified source in world space, which must not disappear.
    predicted = local.clone().requires_grad_()
    batch = {"local_motions": local.requires_grad_(), "global_motions": global_target.requires_grad_(),
             "executed_history": history_in_input_frame(history).requires_grad_(),
             "contact_confidence": torch.ones(1, 64, 4, dtype=torch.bool)}
    output = {"recon_state": predicted, "l_commit": predicted.sum() * 0., "perplexity": 0.}
    losses = trainer.loss(batch, output, None)
    assert losses["l_joint_vel"] < 1e-8
    assert float(losses["l_fk_position_m"].detach()) == pytest.approx(.1, abs=2e-6)
    assert float(losses["l_sole_position_m"].detach()) == pytest.approx(.1, abs=2e-6)
    losses["loss"].backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert all(batch[key].grad is None for key in ("local_motions", "global_motions", "executed_history"))


def test_native_training_path_contains_rotation_term_and_preserves_forward():
    rep, root, pose, inference, raw, trainer = configured()
    trainer.args["rollout_rotation_increment_coeff"] = 1.
    history = raw[:, :4].clone()
    history[..., 0] += .03
    reference = deployed_reference(inference, raw, history)
    global_target, local = source_target_with_history(raw, history, rep)
    _, source_local = source_target_with_history(raw, raw[:, :4], rep)
    total, terms = joint_rollout_loss(inference, trainer, raw, history, reference,
        source_local, local, global_target, torch.ones(1, 64, 4, dtype=torch.bool), {})
    assert terms["runtime_decoded_max_error"] < 1e-4
    assert terms["weighted_rotation_increment"] > 0
    torch.testing.assert_close(total.detach(), terms["pose_ce"] + terms["decoded_motion_loss"])
    total.backward()
    for module in (root.backbone_net, pose.backbone_net):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    assert all(p.grad is None for p in pose.pose_net.parameters())


def test_invalid_supervision_policy_fails_before_inference():
    rep, _, _, inference, raw, trainer = configured()
    global_target, local = source_target_with_history(raw, raw[:, :4], rep)
    with pytest.raises(ValueError, match="supervision policy"):
        joint_rollout_loss(inference, trainer, raw, raw[:, :4], {}, local, local, global_target,
                           torch.ones(1, 64, 4), {}, "typo")


def test_geometry_mask_excludes_context_and_empty_selection_has_zero_gradient():
    from motionbricks.data.sole_geometry_losses import sole_geometry_supervision
    from motionbricks.data.sole_region_contact import FRAME_STATES, INTERVAL_STATES
    height = torch.full((1, 12, 2), -.02, requires_grad=True)
    horizontal = torch.zeros(1, 12, 2, 2, requires_grad=True)
    labels = {"frame_state": torch.full((1, 12), FRAME_STATES.index("contact")),
              "interval_state": torch.full((1, 11), INTERVAL_STATES.index("stable_contact")),
              "region_measured": torch.ones(1, 11, dtype=torch.bool),
              "vertex_measured": torch.ones(1, 11, 2, dtype=torch.bool)}
    mask = torch.ones(1, 12, dtype=torch.bool)
    mask[:, :4] = False
    losses = sole_geometry_supervision(height, horizontal, labels, 60, frame_mask=mask)
    assert losses["contact_frames_supervised"] == 8
    assert losses["measured_stable_intervals_supervised"] == 7
    losses["l_penetration_m"].backward()
    assert not height.grad[:, :4].any() and height.grad[:, 4:].abs().sum() > 0
    empty = sole_geometry_supervision(height, horizontal, labels, 60, frame_mask=torch.zeros_like(mask))
    assert all(torch.isfinite(empty[key]) and empty[key] == 0
               for key in ("l_penetration_m", "l_contact_height_m", "l_measured_slip_mps"))


def test_decoder_can_learn_through_native_path_without_changing_encoder_or_codebooks():
    from motionbricks.data.bones_control import state_digest
    rep, root, pose, inference, raw, trainer = configured()
    frozen = (state_digest(pose.pose_net.encoder), state_digest(pose.pose_net.quantizer))
    pose.pose_net.decoder.requires_grad_(True)
    trainer.args["rollout_rotation_increment_coeff"] = 1.
    reference = deployed_reference(inference, raw, raw[:, :4])
    target, local = source_target_with_history(raw, raw[:, :4], rep)
    loss, terms = joint_rollout_loss(inference, trainer, raw, raw[:, :4], reference,
        local, local, target, torch.ones(1, 64, 4, dtype=torch.bool), {})
    loss.backward()
    for module in (root.backbone_net, pose.backbone_net, pose.pose_net.decoder):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    for module in (pose.pose_net.encoder, pose.pose_net.quantizer):
        assert all(p.grad is None for p in module.parameters())
    assert frozen == (state_digest(pose.pose_net.encoder), state_digest(pose.pose_net.quantizer))
    assert terms["runtime_decoded_max_error"] < 1e-4
