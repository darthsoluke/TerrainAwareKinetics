import pytest
import torch

from motionbricks.data.bones_control import SomaMotionInference, decode_tokens, state_digest
from motionbricks.data.bones_models import BonesTokenizer
from motionbricks.data.soma_decoder_refinement import source_target_with_history
from motionbricks.data.soma_pose_rollout import (decode_logits, deployed_pose_request,
                                               pose_rollout_loss, straight_through_codes,
                                               require_deployed_tokens_equal)
from test_bones_control import models, source_motion


def fixture():
    rep, root, pose = models()
    inference = SomaMotionInference({"root": root, "pose": pose},
        {"min_tokens": 6, "max_tokens": 16, "cond_root_feature_is_from_motion_rep": "global"}, device="cpu")
    raw = rep.unnormalize(source_motion(rep))
    return rep, root, pose, inference, raw


def test_captured_pose_call_matches_deployed_argmax_and_restores_hook():
    rep, _, pose, inference, raw = fixture()
    before = len(pose.backbone_net._forward_pre_hooks)
    request, decoder, generated, _ = deployed_pose_request(inference, raw, raw[:, :4])
    assert len(pose.backbone_net._forward_pre_hooks) == before
    logits = pose.backbone_net(*request)["pose_logits"]
    decoded, ids = decode_logits(pose.pose_net, logits, decoder)
    assert torch.equal(ids, decoder[2])
    full, mask, tokens, root, active = decoder
    expected = decode_tokens(pose.pose_net, full, mask, tokens, root, active)
    torch.testing.assert_close(decoded, expected, atol=0., rtol=0.)
    assert not generated.requires_grad


def test_straight_through_is_native_hard_code_forward_with_finite_logit_gradient():
    _, _, pose, _, _ = fixture()
    books = pose.pose_net.quantizer.vq.codebook
    logits = torch.randn(2, 16, *books.shape[:2], requires_grad=True)
    latent, ids = straight_through_codes(logits, pose.pose_net.quantizer)
    native = pose.pose_net.quantizer.dequantize(ids, use_overall_indices=False)
    torch.testing.assert_close(latent, native, atol=0., rtol=0.)
    (latent * torch.randn_like(latent)).sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert all(p.grad is None for p in pose.pose_net.quantizer.parameters())
    with pytest.raises(ValueError, match="disagree"):
        straight_through_codes(logits[..., :-1], pose.pose_net.quantizer)


def test_no_future_body_or_height_leaks_into_captured_pose_inputs():
    _, _, _, inference, raw = fixture()
    first, _, _, _ = deployed_pose_request(inference, raw, raw[:, :4])
    changed = raw.clone()
    changed[:, 4:, 5:] += 100
    changed[:, 4:, 1] += 10
    second, _, _, _ = deployed_pose_request(inference, changed, raw[:, :4])
    for a, b in zip(first, second):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b)
        else:
            assert a == b


def test_decoded_objective_changes_pose_gradient_but_not_frozen_tokenizer_or_root():
    rep, root, pose, inference, raw = fixture()
    pose.backbone_net.requires_grad_(True)
    trainer = BonesTokenizer(pose_vqvae_network=pose.pose_net, root_vqvae_network=None,
        motion_rep=rep, optimizer=None, scheduler=None, args={"commit_loss_coeff": .02,
        "joint_vel_loss_coeff": 2., "fk_position_coeff": 5., "sole_position_coeff": 2.})
    frozen = (state_digest(root), state_digest(pose.pose_net))
    global_target, local = source_target_with_history(raw, raw[:, :4], rep)
    request, decoder, _, _ = deployed_pose_request(inference, raw, raw[:, :4])
    confidence = torch.ones(1, 64, 4, dtype=torch.bool)
    gradients = []
    for weight in (0., 1.):
        pose.backbone_net.zero_grad(set_to_none=True)
        total, terms, _ = pose_rollout_loss(pose, trainer, request, decoder, local,
            local, global_target, confidence, {}, weight)
        assert torch.isfinite(total) and terms["decoded_weight"] == weight
        total.backward()
        gradients.append(torch.cat([p.grad.flatten() for p in pose.backbone_net.parameters() if p.grad is not None]))
    assert torch.isfinite(gradients[1]).all()
    assert (gradients[1] - gradients[0]).abs().sum() > 0
    assert frozen == (state_digest(root), state_digest(pose.pose_net))
    assert all(p.grad is None for module in (root, pose.pose_net) for p in module.parameters())


def test_capture_hook_removed_even_when_runtime_fails(monkeypatch):
    _, _, pose, inference, raw = fixture()
    before = len(pose.backbone_net._forward_pre_hooks)
    monkeypatch.setattr(inference, "predict", lambda *a, **kw: (_ for _ in ()).throw(ValueError("test failure")))
    with pytest.raises(ValueError, match="test failure"):
        deployed_pose_request(inference, raw, raw[:, :4])
    assert len(pose.backbone_net._forward_pre_hooks) == before


def test_padding_logits_do_not_affect_masked_decoder_or_active_token_check():
    _, _, pose, inference, raw = fixture()
    request, decoder, _, _ = deployed_pose_request(inference, raw, raw[:, :4])
    decoder = (*decoder[:4], decoder[4].clone())
    decoder[4][:, -2:] = False
    logits = pose.backbone_net(*request)["pose_logits"]
    first, ids = decode_logits(pose.pose_net, logits, decoder)
    altered = logits.clone()
    altered[:, -2:] = -100
    altered[:, -2:, :, 0] = 100
    second, other_ids = decode_logits(pose.pose_net, altered, decoder)
    require_deployed_tokens_equal(other_ids, (*decoder[:2], ids, *decoder[3:]))
    torch.testing.assert_close(first, second, atol=0., rtol=0.)
    other_ids[:, 0] = (ids[:, 0] + 1) % logits.shape[-1]
    with pytest.raises(RuntimeError, match="ACTIVE"):
        require_deployed_tokens_equal(other_ids, (*decoder[:2], ids, *decoder[3:]))


def test_training_surrogate_preserves_actual_deployed_choice_at_numerical_ties():
    _, _, pose, _, _ = fixture()
    books = pose.pose_net.quantizer.vq.codebook
    logits = torch.zeros(1, 16, *books.shape[:2], requires_grad=True)
    runtime_ids = torch.ones(logits.shape[:-1], dtype=torch.long)
    latent, ids = straight_through_codes(logits, pose.pose_net.quantizer, runtime_ids)
    assert torch.equal(ids, runtime_ids) and not torch.equal(ids, logits.argmax(-1))
    torch.testing.assert_close(latent, pose.pose_net.quantizer.dequantize(runtime_ids, False), atol=0., rtol=0.)
    latent.square().sum().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_pose_review_accepts_trained_pose_identity_but_requires_frozen_decoder(tmp_path):
    import json
    from scripts.summarize_soma_decoder_refinement import summarize
    from test_soma_decoder_refinement import summary_fixture
    summary_fixture(tmp_path)
    for arm in ("A", "B", "C"):
        path = tmp_path / arm / "contract.json"
        contract = json.loads(path.read_text())
        contract.update(frozen_state={"root": "root", "tokenizer": "codec"},
                        history_policy="teacher" if arm == "A" else "scheduled_generated",
                        decoded_motion_weight=1. if arm == "C" else 0., commit_frames=8)
        path.write_text(json.dumps(contract))
        path = tmp_path / arm / "result.json"
        result = json.loads(path.read_text())
        result["pose_changed"] = True
        path.write_text(json.dumps(result))
    for arm in ("original", "A", "B", "C"):
        path = tmp_path / ("review_" + arm) / "metrics.json"
        report = json.loads(path.read_text())
        report.update(tokenizer_state="codec", commit_frames=8, frozen_state={"root": "root", "pose": arm,
                      "encoder": "same", "quantizer": "same", "decoder": "same"})
        path.write_text(json.dumps(report))
    assert summarize(tmp_path, "pose")["matched_scoring_verified"]
    path = tmp_path / "review_C/metrics.json"
    report = json.loads(path.read_text())
    report["frozen_state"]["decoder"] = "changed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="frozen model"):
        summarize(tmp_path, "pose")


def test_ue_eight_frame_handoff_uses_last_four_of_published_prefix():
    from motionbricks.data.bones_rollout import SomaLocomotionBackend
    from motionbricks.data.soma_decoder_refinement import deployed_request
    from scripts.soma_ue_live_server import PUBLISH_FRAMES
    assert PUBLISH_FRAMES == 8
    rep, _, _, inference, raw = fixture()
    _, history = deployed_request(inference, raw, raw[:, :4], commit_frames=8)
    _, published = deployed_request(inference, raw, raw[:, :4], commit_frames=8, return_committed=True)
    assert published.shape[1] == 8 and history.shape[1] == 4
    torch.testing.assert_close(history, published[:, -4:])
    backend = SomaLocomotionBackend(inference, raw[:, :4], argmax=True, commit_frames=PUBLISH_FRAMES)
    result = backend.step([[0., 0.]], [0.])
    assert result["motion"].shape[1] == 8 and result["timestamps_s"].shape == (8,)
    torch.testing.assert_close(backend.history, result["motion"][:, -4:])
