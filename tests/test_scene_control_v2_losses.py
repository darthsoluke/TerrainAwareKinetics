import pytest
import torch

from motionbricks.scene_control_v2.losses import (
    codec_loss_terms, contact_surface_transforms, executed_boundary_loss,
)


def make_case(frames=4, channels=1):
    samples = torch.zeros(1, frames, channels + 1, 3, requires_grad=True)
    joints = torch.zeros(1, frames, 2, 3, requires_grad=True)
    prediction = {
        "recon_state": torch.zeros(1, frames, 6, requires_grad=True),
        "l_commit": torch.tensor(0.0, requires_grad=True),
        "joint_positions_m": joints, "body_samples_m": samples,
        "contact_logits": torch.zeros(1, frames, channels, requires_grad=True),
        "contact_signed_distance_m": samples[:, :, :channels, 1],
        "body_signed_distance_m": samples[..., 1],
    }
    contacts = {
        "state": torch.ones(1, frames, channels), "valid": torch.ones(1, frames, channels, dtype=torch.bool),
        "actual_state": torch.ones(1, frames, channels),
        "actual_valid": torch.ones(1, frames, channels, dtype=torch.bool),
        "anchor_valid": torch.ones(1, frames, channels, dtype=torch.bool),
        "object_id": torch.zeros(1, frames, channels, dtype=torch.long),
        "kind": torch.zeros(1, frames, channels, dtype=torch.long),
        "anchor_local_m": torch.zeros(1, frames, channels, 3),
        "normal_local": torch.tensor([0.0, 1.0, 0.0]).expand(1, frames, channels, 3).clone(),
    }
    target = {
        "recon_state": torch.zeros(1, frames, 6),
        "frame_valid": torch.ones(1, frames, dtype=torch.bool),
        "joint_positions_m": torch.zeros(1, frames, 2, 3),
        "joint_valid": torch.ones(1, frames, 2, dtype=torch.bool),
        "contacts": contacts,
        "scene": {"object_ids": torch.tensor([[0]]),
                  "object_to_world": torch.eye(4).expand(1, frames, 1, 4, 4).clone(),
                  "object_valid": torch.ones(1, frames, 1, dtype=torch.bool)},
        "contact_sample_indices": torch.arange(channels),
        "contact_surface_valid": torch.ones(1, frames, channels, dtype=torch.bool),
        "contact_interval_valid": torch.ones(1, frames - 1, channels, dtype=torch.bool),
        "body_noncontact_valid": torch.zeros(1, frames, channels + 1, dtype=torch.bool),
        "executed_joints_m": torch.zeros(1, 2, 2, 3),
        "executed_valid": torch.zeros(1, 2, 2, dtype=torch.bool),
    }
    return prediction, target


def evaluate(prediction, target):
    return codec_loss_terms(prediction, target, fps=30.0, first_new_frame=0)


def test_all_invalid_labels_are_finite_zero_and_differentiable():
    prediction, target = make_case()
    target["frame_valid"].fill_(False)
    target["contacts"]["actual_valid"].fill_(False)
    target["contacts"]["anchor_valid"].fill_(False)
    target["scene"]["object_valid"].fill_(False)
    with torch.no_grad():
        for key in ("recon_state", "joint_positions_m", "body_samples_m", "contact_logits"):
            prediction[key].fill_(float("nan"))
        target["contacts"]["actual_state"].fill_(float("nan"))
        target["contacts"]["anchor_local_m"].fill_(float("nan"))
        target["contacts"]["normal_local"].fill_(float("nan"))
        target["scene"]["object_to_world"].fill_(float("nan"))
        target["executed_joints_m"].fill_(float("nan"))
    result = evaluate(prediction, target)
    assert all(torch.isfinite(value) and value == 0 for value in result.as_dict().values())
    result.weighted_sum(dict.fromkeys(result.as_dict(), 1.0)).backward()
    for key in ("recon_state", "joint_positions_m", "body_samples_m", "contact_logits"):
        assert torch.isfinite(prediction[key].grad).all()


def test_moving_object_is_evaluated_in_its_translated_and_rotated_frame():
    prediction, target = make_case()
    angles = torch.arange(4) * 0.2
    transforms = target["scene"]["object_to_world"]
    transforms[0, :, 0, 0, 0] = angles.cos()
    transforms[0, :, 0, 0, 2] = angles.sin()
    transforms[0, :, 0, 2, 0] = -angles.sin()
    transforms[0, :, 0, 2, 2] = angles.cos()
    transforms[0, :, 0, 0, 3] = torch.arange(4) * 0.1
    anchor = target["contacts"]["anchor_local_m"]
    anchor[..., 0] = 1.0
    world = (transforms[..., :3, :3] @ anchor[..., None]).squeeze(-1) + transforms[..., :3, 3]
    with torch.no_grad():
        prediction["body_samples_m"][:, :, :1] = world
    result = evaluate(prediction, target)
    assert result.contact_anchor_m < 2e-7
    assert result.relative_slip_mps < 4e-6
    target["scene"]["object_to_world"] = torch.eye(4).expand_as(transforms).clone()
    assert evaluate(prediction, target).relative_slip_mps > 1.0


def test_predicted_contact_cannot_close_the_slip_or_geometry_gate():
    prediction, target = make_case()
    with torch.no_grad():
        prediction["body_samples_m"][0, :, 0, 0] = torch.arange(4) * 0.1
        prediction["body_samples_m"][0, :, 0, 1] = 0.02
        prediction["contact_logits"].fill_(-100)
    low = evaluate(prediction, target)
    with torch.no_grad():
        prediction["contact_logits"].fill_(100)
    high = evaluate(prediction, target)
    assert low.relative_slip_mps.detach().item() == pytest.approx(3.0)
    torch.testing.assert_close(low.relative_slip_mps, high.relative_slip_mps)
    torch.testing.assert_close(low.contact_anchor_m, high.contact_anchor_m)
    torch.testing.assert_close(low.contact_gap_m, high.contact_gap_m)
    assert low.contact_bce > 90 and high.contact_bce < 1e-5


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4])
def test_fixed_and_sliding_contact_semantics(kind):
    prediction, target = make_case()
    target["contacts"]["kind"].fill_(kind)
    with torch.no_grad():
        prediction["body_samples_m"][0, :, 0, 0] = torch.arange(4) * 0.1
    result = evaluate(prediction, target)
    if kind < 3:
        assert result.relative_slip_mps > 2.9 and result.contact_anchor_m > 0.1
    else:
        assert result.relative_slip_mps == 0 and result.contact_anchor_m == 0


def test_intended_contact_never_substitutes_missing_actual_evidence():
    prediction, target = make_case()
    target["contacts"]["actual_valid"].fill_(False)
    with torch.no_grad():
        prediction["body_samples_m"].fill_(0.1)
    result = evaluate(prediction, target)
    assert result.contact_bce == result.relative_slip_mps == result.contact_anchor_m == result.contact_gap_m == 0


@pytest.mark.parametrize("change", ["object", "anchor", "event", "invalid", "kind"])
def test_slip_does_not_bridge_a_contact_event_change(change):
    prediction, target = make_case(frames=2)
    with torch.no_grad():
        prediction["body_samples_m"][0, 1, 0, 0] = 0.1
    if change == "object":
        target["scene"]["object_ids"] = torch.tensor([[0, 1]])
        target["scene"]["object_to_world"] = torch.eye(4).expand(1, 2, 2, 4, 4).clone()
        target["scene"]["object_valid"] = torch.ones(1, 2, 2, dtype=torch.bool)
        target["contacts"]["object_id"][0, 1] = 1
    elif change == "anchor":
        target["contacts"]["anchor_local_m"][0, 1, 0, 0] = 0.1
    elif change == "event":
        target["contact_interval_valid"].fill_(False)
    elif change == "invalid":
        target["contacts"]["actual_valid"][0, 1] = False
    else:
        target["contacts"]["kind"][0, 1] = 1
    assert evaluate(prediction, target).relative_slip_mps == 0


def test_no_object_table_does_not_invent_a_static_surface():
    prediction, target = make_case()
    target["scene"]["object_ids"] = torch.empty(1, 0, dtype=torch.long)
    target["scene"]["object_to_world"] = torch.empty(1, 4, 0, 4, 4)
    target["scene"]["object_valid"] = torch.empty(1, 4, 0, dtype=torch.bool)
    _, valid = contact_surface_transforms(target["contacts"], target["scene"])
    assert not valid.any()
    with torch.no_grad():
        prediction["body_samples_m"].fill_(1)
    result = evaluate(prediction, target)
    assert result.contact_anchor_m == result.contact_gap_m == result.relative_slip_mps == 0
    assert result.contact_bce > 0


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_noncontiguous_object_ids_resolve_after_table_reordering(order):
    order = torch.tensor(order)
    contacts = {"object_id": torch.tensor([[[7, 42, -1, 99], [7, 42, -1, 99]],
                                            [[7, 42, -1, 99], [7, 42, -1, 99]]])}
    ids = torch.tensor([[7, 42], [42, 7]])
    transforms = torch.eye(4).expand(2, 2, 2, 4, 4).clone()
    transforms[0, :, 0, 0, 3] = 0.7
    transforms[0, :, 1, 0, 3] = 4.2
    transforms[1, :, 0, 0, 3] = 14.2
    transforms[1, :, 1, 0, 3] = 10.7
    object_valid = torch.ones(2, 2, 2, dtype=torch.bool)
    object_valid[1, 1, 1] = False
    scene = {"object_ids": ids[:, order], "object_to_world": transforms[:, :, order],
             "object_valid": object_valid[:, :, order]}
    gathered, valid = contact_surface_transforms(contacts, scene)
    torch.testing.assert_close(gathered[..., 0, 3], torch.tensor(
        [[[0.7, 4.2, 0., 0.], [0.7, 4.2, 0., 0.]],
         [[10.7, 14.2, 0., 0.], [10.7, 14.2, 0., 0.]]]))
    assert torch.equal(valid, torch.tensor([[[True, True, False, False], [True, True, False, False]],
                                            [[True, True, False, False], [False, True, False, False]]]))


def test_noncontiguous_moving_contact_losses_and_gradients_survive_table_reordering():
    prediction, target = make_case(frames=4, channels=2)
    target["contacts"]["object_id"][:] = torch.tensor([7, 42])
    scene = target["scene"]
    scene["object_ids"] = torch.tensor([[7, 42]])
    scene["object_to_world"] = torch.eye(4).expand(1, 4, 2, 4, 4).clone()
    scene["object_to_world"][0, :, 0, 0, 3] = 0.2 * torch.arange(4)
    scene["object_to_world"][0, :, 1, 0, 3] = 10 + 0.4 * torch.arange(4)
    scene["object_valid"] = torch.ones(1, 4, 2, dtype=torch.bool)
    with torch.no_grad():
        prediction["body_samples_m"][:, :, :2] = scene["object_to_world"][..., :3, 3]
        prediction["body_samples_m"][:, :, :2, 0] += 0.01 * torch.arange(4)[None, :, None]
        prediction["body_samples_m"][:, :, :2, 1] = 0.02
    original = evaluate(prediction, target)
    assert original.contact_anchor_m > 0.02
    assert original.contact_gap_m.detach().item() == pytest.approx(0.02)
    assert original.relative_slip_mps.detach().item() == pytest.approx(0.3, abs=2e-5)
    scene["object_ids"] = scene["object_ids"][:, [1, 0]]
    scene["object_to_world"] = scene["object_to_world"][:, :, [1, 0]]
    scene["object_valid"] = scene["object_valid"][:, :, [1, 0]]
    reordered = evaluate(prediction, target)
    for name in ("contact_anchor_m", "contact_gap_m", "relative_slip_mps"):
        torch.testing.assert_close(getattr(original, name), getattr(reordered, name))
    (reordered.contact_anchor_m + reordered.contact_gap_m + reordered.relative_slip_mps).backward()
    assert torch.isfinite(prediction["body_samples_m"].grad).all()
    assert prediction["body_samples_m"].grad[:, :, :2].abs().sum() > 0


def test_penetration_only_uses_verified_noncontact_body_and_scene_validity():
    prediction, target = make_case()
    with torch.no_grad():
        prediction["body_samples_m"][..., 1] = -0.02
    target["body_noncontact_valid"][:, :, 1] = True
    result = evaluate(prediction, target)
    assert result.penetration_m.detach().item() == pytest.approx(0.02)
    target["body_noncontact_valid"].fill_(False)
    assert evaluate(prediction, target).penetration_m == 0


def test_boundary_uses_real_prefix_not_source_future_or_predicted_history():
    prediction, target = make_case()
    target["executed_valid"].fill_(True)
    target["executed_joints_m"][:, -2, :, 0] = 0.0
    target["executed_joints_m"][:, -1, :, 0] = 0.1
    with torch.no_grad():
        prediction["joint_positions_m"][:, :2, :, 0] = 100.0
        prediction["joint_positions_m"][:, 2, :, 0] = 0.2
    actual = executed_boundary_loss(prediction["joint_positions_m"], target["executed_joints_m"],
                                    target["executed_valid"], target["frame_valid"],
                                    fps=30, first_new_frame=2)
    assert actual == 0
    with torch.no_grad():
        prediction["joint_positions_m"][:, 2, :, 0] = 0.3
    actual = executed_boundary_loss(prediction["joint_positions_m"], target["executed_joints_m"],
                                    target["executed_valid"], target["frame_valid"],
                                    fps=30, first_new_frame=2)
    assert actual.detach().item() == pytest.approx(3.0)
    actual.backward()
    assert prediction["joint_positions_m"].grad[:, 2].abs().sum() > 0
    assert prediction["joint_positions_m"].grad[:, :2].abs().sum() == 0


def test_all_geometry_and_contact_gradients_reach_predictions_not_labels():
    prediction, target = make_case()
    with torch.no_grad():
        prediction["body_samples_m"][0, :, 0, 0] = torch.arange(4) * 0.01
        prediction["body_samples_m"][..., 1] = -0.02
        prediction["joint_positions_m"].fill_(0.01)
        prediction["recon_state"].fill_(0.01)
    target["joint_positions_m"].requires_grad_(True)
    target["contacts"]["anchor_local_m"].requires_grad_(True)
    target["scene"]["object_to_world"].requires_grad_(True)
    target["body_noncontact_valid"][:, :, 1] = True
    result = evaluate(prediction, target)
    result.weighted_sum(dict.fromkeys(result.as_dict(), 1.0)).backward()
    for key in ("recon_state", "joint_positions_m", "body_samples_m", "contact_logits"):
        assert torch.isfinite(prediction[key].grad).all()
        assert prediction[key].grad.abs().sum() > 0
    assert target["joint_positions_m"].grad is None
    assert target["contacts"]["anchor_local_m"].grad is None
    assert target["scene"]["object_to_world"].grad is None


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_weights_reject_nonfinite_and_negative_values(bad):
    prediction, target = make_case()
    result = evaluate(prediction, target)
    weights = dict.fromkeys(result.as_dict(), 1.0)
    weights["fk_m"] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        result.weighted_sum(weights)
    with pytest.raises(ValueError, match="every"):
        result.weighted_sum({})
