import pytest
import torch

from motionbricks.data.bones_rollout import observed_history_features
from test_soma_joint_rollout import configured


def test_observation_uses_executed_fk_not_auxiliary_predictions():
    rep, _, _, _, raw, _ = configured()
    history = raw[:, :4].clone()
    corrupted = history.clone()
    corrupted[..., rep.indices["ric_data"]] += .2
    corrupted[..., rep.indices["local_vel"]] += 2.
    original = corrupted.clone()
    expected = observed_history_features(history, rep, "executed_fk")
    actual = observed_history_features(corrupted, rep, "executed_fk")
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(corrupted, original, atol=0, rtol=0)
    before = rep.inverse(corrupted, False, return_all=False)
    after = rep.inverse(actual, False, return_all=True)
    for key in ("posed_joints", "global_joint_rots", "local_joint_rots", "root_pos"):
        torch.testing.assert_close(before[key], after[key], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(after["posed_joints"], after["all_posed_joints"]["ric_data"], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual[..., rep.indices["foot_contacts"]], history[..., rep.indices["foot_contacts"]])


@pytest.mark.parametrize("yaw", [.7, -1.8, 3.0])
def test_observation_preserves_world_pose_at_nonzero_heading_and_origin(yaw):
    rep, _, _, _, raw, _ = configured()
    history = rep.change_first_heading(raw[:, :4], yaw, False, False)
    history[..., [0, 2]] += torch.tensor([7., -3.])
    features = observed_history_features(history, rep, "executed_fk")
    before = rep.inverse(history, False, return_all=False)
    after = rep.inverse(features, False, return_all=True)
    for key in ("posed_joints", "global_joint_rots", "local_joint_rots", "root_pos"):
        torch.testing.assert_close(before[key], after[key], atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(after["posed_joints"], after["all_posed_joints"]["ric_data"], atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(features[..., :3], history[..., :3], atol=0, rtol=0)


def test_observation_does_not_depend_on_predicted_heading_or_6d_scale():
    rep, _, _, _, raw, _ = configured()
    history = raw[:, :4].clone()
    original = observed_history_features(history, rep, "executed_fk")
    history[..., rep.indices["global_root_heading"]] *= -1
    history[..., rep.indices["global_rot_data"]] *= 2
    actual = observed_history_features(history, rep, "executed_fk")
    torch.testing.assert_close(actual, original, atol=2e-5, rtol=1e-5)


def test_legacy_history_is_unchanged_and_unknown_policy_rejected():
    rep, _, _, _, raw, _ = configured()
    assert observed_history_features(raw[:, :4], rep).data_ptr() == raw.data_ptr()
    with pytest.raises(ValueError):
        observed_history_features(raw[:, :4], rep, "smooth")


def test_observation_is_differentiable_and_has_no_time_shift():
    rep, _, _, _, raw, _ = configured()
    history = raw[:, :4].clone().requires_grad_()
    features = observed_history_features(history, rep, "executed_fk")
    features.square().sum().backward()
    assert history.grad is not None and torch.isfinite(history.grad).all()
    assert history.grad[..., rep.indices["global_rot_data"]].abs().sum() > 0
    torch.testing.assert_close(features[..., :3], history[..., :3], atol=0, rtol=0)


def test_observed_geometry_uses_same_native_forward_and_gradients_in_training():
    from motionbricks.data.soma_joint_rollout import deployed_reference, differentiable_native_prediction
    rep, root, pose, inference, raw, _ = configured()
    inference.history_geometry = "executed_fk"
    history = raw[:, :4].clone()
    history[..., rep.indices["ric_data"]] += .2
    reference = deployed_reference(inference, raw, history)
    clean = deployed_reference(inference, raw, raw[:, :4])
    torch.testing.assert_close(reference["decoded"], clean["decoded"], atol=0, rtol=0)
    prediction, _, terms = differentiable_native_prediction(inference, raw, history, reference)
    assert terms["runtime_decoded_max_error"] < 1e-4
    prediction[:, 4:].square().mean().backward()
    for network in (root.backbone_net, pose.backbone_net):
        grads = [p.grad for p in network.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_bundle_cannot_switch_history_geometry_without_matching_training(tmp_path):
    from scripts.portable_bones_control import load_bundle
    bundle = {"schema": "motionbricks.soma_control_bundle.v1", "history_geometry": "executed_fk"}
    path = tmp_path / "invalid.pt"
    torch.save(bundle, path)
    with pytest.raises(ValueError, match="matching training"):
        load_bundle(path, "cpu")
