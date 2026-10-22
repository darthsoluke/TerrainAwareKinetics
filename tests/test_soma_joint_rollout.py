import pytest
import torch

from motionbricks.data.bones_control import state_digest
from motionbricks.data.bones_models import BonesTokenizer
from motionbricks.data.soma_decoder_refinement import source_target_with_history
from motionbricks.data.soma_joint_rollout import (deployed_reference, differentiable_native_prediction,
                                               joint_rollout_loss)
from test_soma_pose_rollout import fixture


def configured(joint=True):
    rep, root, pose, inference, raw = fixture()
    inference.requires_grad_(False).eval()
    root.backbone_net.requires_grad_(joint)
    pose.backbone_net.requires_grad_(True)
    trainer = BonesTokenizer(pose_vqvae_network=pose.pose_net, root_vqvae_network=None,
        motion_rep=rep, optimizer=None, scheduler=None,
        args={"commit_loss_coeff": .02, "joint_vel_loss_coeff": 2.,
              "fk_position_coeff": 5., "sole_position_coeff": 2.}).eval()
    return rep, root, pose, inference, raw, trainer


def test_paired_kernel_is_scoped_and_restored_on_failure():
    from motionbricks.data.soma_joint_rollout import paired_attention_kernel
    previous = torch.backends.mha.get_fastpath_enabled()
    with pytest.raises(RuntimeError, match="deliberate"):
        with paired_attention_kernel():
            assert not torch.backends.mha.get_fastpath_enabled()
            raise RuntimeError("deliberate")
    assert torch.backends.mha.get_fastpath_enabled() == previous


@pytest.mark.parametrize("joint", [False, True])
def test_native_forward_gradient_and_frozen_tokenizer(joint):
    torch.manual_seed(2333)
    rep, root, pose, inference, raw, trainer = configured(joint)
    before = (state_digest(root), state_digest(pose.pose_net))
    reference = deployed_reference(inference, raw, raw[:, :4])
    global_target, local = source_target_with_history(raw, raw[:, :4], rep)
    total, terms = joint_rollout_loss(inference, trainer, raw, raw[:, :4], reference,
        local, local, global_target, torch.ones(1, 64, 4, dtype=torch.bool), {})
    assert torch.isfinite(total) and terms["runtime_decoded_max_error"] < 1e-4
    total.backward()
    root_grads = [p.grad for p in root.parameters() if p.grad is not None]
    pose_grads = [p.grad for p in pose.backbone_net.parameters() if p.grad is not None]
    assert bool(root_grads) == joint
    assert pose_grads and sum(float(g.abs().sum()) for g in pose_grads) > 0
    if joint:
        assert all(torch.isfinite(g).all() for g in root_grads)
        assert sum(float(g.abs().sum()) for g in root_grads) > 0
    assert all(p.grad is None for p in pose.pose_net.parameters())
    assert before == (state_digest(root), state_digest(pose.pose_net))
    assert not reference["history"].requires_grad


def test_decoded_motion_itself_reaches_both_backbones_without_token_ce():
    _, root, pose, inference, raw, _ = configured()
    reference = deployed_reference(inference, raw, raw[:, :4])
    prediction, _, _ = differentiable_native_prediction(inference, raw, raw[:, :4], reference)
    prediction[:, 4:].square().mean().backward()
    for network in (root.backbone_net, pose.backbone_net):
        gradients = [p.grad for p in network.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_future_body_and_height_are_not_request_inputs():
    _, _, _, inference, raw, _ = configured()
    first = deployed_reference(inference, raw, raw[:, :4])
    modified = raw.clone()
    modified[:, 4:, 5:] += 100
    modified[:, 4:, 1] += 10
    second = deployed_reference(inference, modified, raw[:, :4])
    for key in ("root", "logits", "decoded", "history"):
        torch.testing.assert_close(first[key], second[key], atol=0., rtol=0.)


def test_source_targets_do_not_receive_gradients_and_total_contains_decoded_loss():
    rep, _, _, inference, raw, trainer = configured()
    reference = deployed_reference(inference, raw, raw[:, :4])
    global_target, local = source_target_with_history(raw, raw[:, :4], rep)
    local.requires_grad_()
    global_target.requires_grad_()
    total, terms = joint_rollout_loss(inference, trainer, raw, raw[:, :4], reference,
        local, local, global_target, torch.ones(1, 64, 4, dtype=torch.bool), {})
    torch.testing.assert_close(total.detach(), terms["pose_ce"] + terms["decoded_motion_loss"])
    total.backward()
    assert local.grad is None and global_target.grad is None


def test_hooks_removed_when_native_prediction_raises(monkeypatch):
    _, root, pose, inference, raw, _ = configured()
    reference = deployed_reference(inference, raw, raw[:, :4])
    modules = (root.backbone_net, pose.backbone_net, pose.pose_net.decoder)
    before = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]
    monkeypatch.setattr(inference, "predict", lambda *a, **kw: (_ for _ in ()).throw(ValueError("failure")))
    with pytest.raises(ValueError, match="failure"):
        differentiable_native_prediction(inference, raw, raw[:, :4], reference)
    assert before == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in modules]


def test_non_eval_and_wrong_publication_contract_are_rejected():
    _, _, _, inference, raw, _ = configured()
    with pytest.raises(ValueError, match="eight-frame"):
        deployed_reference(inference, raw, raw[:, :4], commit_frames=4)
    reference = deployed_reference(inference, raw, raw[:, :4])
    inference._root_model.backbone_net.train()
    with pytest.raises(ValueError, match="deterministic"):
        differentiable_native_prediction(inference, raw, raw[:, :4], reference)


@pytest.mark.parametrize("has_evidence", [False, True])
def test_official_sole_supervision_graph_and_empty_evidence(has_evidence):
    # Synthetic labels test the gradient contract, not real contact classification.
    rep, root, pose, inference, raw, _ = configured()
    trainer = BonesTokenizer(pose_vqvae_network=pose.pose_net, root_vqvae_network=None,
        motion_rep=rep, optimizer=None, scheduler=None, args={"commit_loss_coeff": .02,
        "sole_region_supervision": True, "sole_region_labels_version": "synthetic-gradient-test",
        "sole_region_coeff_contact_height_m": 2., "sole_region_coeff_penetration_m": 2.,
        "sole_region_coeff_slip_mps": 1.}).eval()
    labels = {}
    for side in ("left", "right"):
        width = len(getattr(trainer, f"sole_region_indices_{side}"))
        labels[f"sole_frame_state_{side}"] = torch.zeros(1, 64, dtype=torch.long)
        labels[f"sole_interval_state_{side}"] = torch.zeros(1, 63, dtype=torch.long)
        labels[f"sole_region_measured_{side}"] = torch.full((1, 63), has_evidence)
        labels[f"sole_vertex_measured_{side}"] = torch.full((1, 63, width), has_evidence)
    reference = deployed_reference(inference, raw, raw[:, :4])
    prediction, _, _ = differentiable_native_prediction(inference, raw, raw[:, :4], reference)
    global_target, local = source_target_with_history(raw, raw[:, :4], rep)
    losses = trainer.loss({"local_motions": local, "global_motions": global_target,
        "contact_confidence": torch.ones(1, 64, 4, dtype=torch.bool), **labels},
        {"recon_state": prediction, "l_commit": prediction.sum() * 0., "perplexity": 0.}, None)
    assert torch.isfinite(losses["loss"])
    if not has_evidence:
        assert losses["sole_region_contact_frames_supervised"] == 0
        assert losses["sole_region_measured_stable_intervals_supervised"] == 0
        assert losses["l_measured_slip_mps"] == 0
    else:
        assert losses["sole_region_measured_stable_intervals_supervised"] > 0
        losses["l_measured_slip_mps"].backward()
        for network in (root.backbone_net, pose.backbone_net):
            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            assert sum(float(g.abs().sum()) for g in gradients) > 0
        assert all(p.grad is None for p in pose.pose_net.parameters())


def test_saved_joint_bundle_carries_trained_root_and_preserves_original(tmp_path):
    from scripts.train_soma_pose_rollout import save_bundle
    _, root, pose, _, _, _ = configured()
    original = {"root_state": {"sentinel": torch.tensor(3.)}, "contracts": {"pose": "unchanged"}}
    save_bundle(original, pose, tmp_path / "backend.pt", {"coupled_root_trial": True}, "sampling", 2, root=root)
    adapted = torch.load(tmp_path / "backend.pt", weights_only=False)
    assert list(original["root_state"]) == ["sentinel"]
    assert adapted["contracts"] == original["contracts"]
    assert not adapted["quality_approved"]
    assert adapted["joint_rollout_adaptation"]["completed_updates"] == 2
    for key, value in root.state_dict().items():
        torch.testing.assert_close(adapted["root_state"][key], value.cpu(), atol=0., rtol=0.)
