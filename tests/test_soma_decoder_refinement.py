import copy

import pytest
import torch

from motionbricks.data.bones_control import SomaMotionInference, state_digest
from motionbricks.data.soma_decoder_refinement import (deployed_request, rollout_probability,
    source_label_window, source_target_with_history, warmup_request)
from test_bones_control import models, source_motion


@pytest.mark.parametrize("policy,expected_gradient", [("provided_velocity", 0.), ("learned_refinement", 1.)])
def test_root_overwrite_reproducer_and_native_root_gradient(policy, expected_gradient):
    rep, _, pose = models()
    net = pose.pose_net
    net.root_condition_policy = policy
    indices = list(net.motion_rep.indices["local_root_rot_vel"]) + list(net.motion_rep.indices["local_root_vel"])
    raw = torch.randn(1, 8, len(net.motion_rep.indices["all"]), requires_grad=True)
    external = torch.zeros(1, 8, 3)
    output = net._apply_condition({"recon_state": raw}, external)
    output["recon_state"][..., indices].sum().backward()
    torch.testing.assert_close(raw.grad[..., indices], torch.full_like(raw.grad[..., indices], expected_gradient))
    if policy == "learned_refinement":
        assert output["recon_state"] is raw
    else:
        assert not output["recon_state"][..., indices].any()


def test_schedule_has_teacher_ramp_and_fully_generated_stages():
    assert [rollout_probability(s) for s in (0, 2499, 2500, 3750, 5000, 9999)] == [0., 0., 0., .5, 1., 1.]


def test_source_future_not_transported_to_generated_drift_or_ground():
    rep, _, _ = models()
    raw = rep.unnormalize(source_motion(rep))
    history = raw[:, :4].clone()
    history[..., 0] += .2
    history[..., 1] += .1
    original = raw.clone()
    global_target, local = source_target_with_history(raw, history, rep)
    target = rep.unnormalize(global_target)
    torch.testing.assert_close(target[:, :4, 1], raw[:, :4, 1])
    torch.testing.assert_close(target[:, 4:, 1], raw[:, 4:-1, 1])
    # A single origin translation affects all frames equally. Future is NOT
    # shifted toward the drifting generated context or rotated to its heading.
    torch.testing.assert_close(target[:, 4:, 0], raw[:, 4:-1, 0] - history[:, :1, 0])
    reconstructed = rep.dual_rep.local_to_global(local, True, False, lengths=torch.tensor([64]))
    torch.testing.assert_close(reconstructed[..., [0, 2]] + target[:, :1, [0, 2]], target[..., [0, 2]], atol=1e-5, rtol=0)
    torch.testing.assert_close(raw, original)


def test_all_label_axes_follow_source_index_and_generated_context_not_supervised():
    labels = {f"sole_{key}_{side}": (torch.arange(80) if key.endswith("state") else
                                    torch.ones(80, 3, dtype=torch.bool) if key == "vertex_measured" else
                                    torch.ones(80, dtype=torch.bool))
              for side in ("left", "right") for key in
              ("frame_state", "interval_state", "region_measured", "vertex_measured")}
    source = copy.deepcopy(labels)
    actual = source_label_window(labels, 11, 64, True)
    assert actual["sole_frame_state_left"].tolist() == list(range(11, 75))
    assert actual["sole_interval_state_left"].tolist() == list(range(11, 74))
    for side in ("left", "right"):
        for key in ("region_measured", "vertex_measured"):
            assert not actual[f"sole_{key}_{side}"][:4].any()
            assert actual[f"sole_{key}_{side}"][4:].all()
    for key in labels:
        torch.testing.assert_close(labels[key], source[key])
    with pytest.raises(ValueError, match="shorter"):
        source_label_window(labels, 30, 64)


def test_captured_request_is_exact_deployment_and_ignores_future_body():
    from motionbricks.data.bones_control import decode_tokens
    rep, root, pose = models()
    inference = SomaMotionInference({"root": root, "pose": pose},
        {"min_tokens": 6, "max_tokens": 16, "cond_root_feature_is_from_motion_rep": "global"}, device="cpu")
    raw = rep.unnormalize(source_motion(rep))
    changed = raw.clone()
    changed[:, 4:, 5:] += 100
    with torch.no_grad():
        request, history = deployed_request(inference, raw, raw[:, :4])
        second, other_history = deployed_request(inference, changed, raw[:, :4])
        full, mask, tokens, root_values, token_mask = request
        output = decode_tokens(pose.pose_net, full, mask, tokens, root_values, token_mask)
        global_output = rep.dual_rep.local_to_global(output, True, False, lengths=torch.tensor([64]))
    for a, b in zip(request, second):
        torch.testing.assert_close(a, b)
        assert not a.requires_grad
    torch.testing.assert_close(history, other_history)
    # Source starts at zero in this fixture; deployed heading restoration does
    # not affect the XYZ/rotation channels used for engine FK.
    torch.testing.assert_close(history[..., :3], global_output[:, 4:8, :3], atol=1e-5, rtol=0)


def test_native_decoder_gradient_with_actual_request_frozen_encoder_and_ema():
    from motionbricks.data.bones_models import BonesTokenizer
    from motionbricks.data.soma_decoder_refinement import refinement_loss
    rep, _, pose = models()
    raw = rep.unnormalize(source_motion(rep))
    global_target, local = source_target_with_history(raw, raw[:, :4], rep)
    pose.pose_net.root_condition_policy = "learned_refinement"
    frozen = (state_digest(pose.pose_net.encoder), state_digest(pose.pose_net.quantizer))
    pose.pose_net.decoder.requires_grad_(True)
    trainer = BonesTokenizer(pose_vqvae_network=pose.pose_net, root_vqvae_network=None, motion_rep=rep,
        optimizer=None, scheduler=None, args={"commit_loss_coeff": .02, "joint_vel_loss_coeff": 2.,
        "fk_position_coeff": 5., "sole_position_coeff": 2.})
    losses, prediction = refinement_loss(trainer, local, global_target,
        torch.ones(1, 64, 4, dtype=torch.bool), {}, warmup_request(pose, local))
    prediction.retain_grad()
    losses["loss"].backward()
    root_indices = list(pose.local_motion_rep.indices["local_root_rot_vel"]) + list(pose.local_motion_rep.indices["local_root_vel"])
    assert torch.isfinite(prediction.grad).all() and prediction.grad[..., root_indices].abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pose.pose_net.decoder.parameters())
    assert all(p.grad is None for module in (pose.pose_net.encoder, pose.pose_net.quantizer) for p in module.parameters())
    assert frozen == (state_digest(pose.pose_net.encoder), state_digest(pose.pose_net.quantizer))


def test_old_bundle_cannot_be_enabled_by_changing_only_root_policy(tmp_path):
    from scripts.portable_bones_control import load_bundle
    path = tmp_path / "untrained.pt"
    torch.save({"schema": "motionbricks.soma_control_bundle.v1", "pose_config": {"model": {
        "pose_vqvae_network": {"root_condition_policy": "learned_refinement"}}}}, path)
    with pytest.raises(ValueError, match="trained decoder"):
        load_bundle(path, "cpu")


def test_command_sequence_covers_expected_game_locomotion():
    from scripts.evaluate_soma_decoder_refinement import command_at
    assert {command_at(i)[0] for i in range(0, 3600, 300)} >= {
        "idle", "walk", "run", "stop", "turn90", "turn180", "backward", "strafe_left", "strafe_right"}


def summary_fixture(folder, no_contact=False):
    import json
    common = {key: "same" for key in ("dataset_sha256", "labels_manifest_sha256", "tokenizer_config_sha256",
        "seed", "steps", "batch", "primitives_per_update", "frames_per_primitive", "learning_rate",
        "loss_arguments", "frozen_state", "source_sha256")}
    common["initial_bundle_sha256"] = "original"
    for arm in ("A", "B", "C"):
        (folder / arm).mkdir()
        (folder / arm / "contract.json").write_text(json.dumps(common))
        (folder / arm / "result.json").write_text(json.dumps({"steps": 10000, "frozen_unchanged": True,
            "sampling_sha256": "same", "bundle_sha256": arm}))
    for arm in ("original", "A", "B", "C"):
        contact = {"region_penetration_m": {"peak": .01}, "valid_samples": {"frames": 10},
                   "contact_height_abs_m": None if no_contact else {"median": .003}}
        score = {"source_contact": {"identity": "source"}, "predicted_contact": contact}
        aggregate = {"native_root": {"mean_window_mpjpe_mm": 20.},
                     "teacher_source_root": {"mean_window_mpjpe_mm": 30.}}
        roll = {"initial_filename": "idle", "initial_frame": 0,
                "boundary_joint_step_mm": {"p95": 30.}, "angular_step_deg": {"max": 20.},
                "root_command_velocity_error_mps": {"mean": .3},
                "per_command": {"idle": {"velocity_error_mps": {"mean": .3}}}}
        report = {"bundle_sha256": arm, "evaluation_sha256": "same", "frozen_state": "same",
                  "test_used": False, "post_processing": None, "aggregate": aggregate,
                  "windows": [{"filename": str(i), "start": 0, "input_sha256": "same",
                               "native_root": score, "teacher_source_root": score} for i in range(57)],
                  "rollouts": {str(i): copy.deepcopy(roll) for i in range(3)}}
        report["rollouts"]["delay_12"] = {"angular_step_deg": {"max": 30.}}
        if arm == "C":
            for index in ("0", "1", "2"):
                report["rollouts"][index]["root_command_velocity_error_mps"]["mean"] = .2
                report["rollouts"][index]["per_command"]["idle"]["velocity_error_mps"]["mean"] = .5
        (folder / ("review_" + arm)).mkdir()
        (folder / ("review_" + arm) / "metrics.json").write_text(json.dumps(report))


def test_average_control_gain_cannot_hide_idle_regression(tmp_path):
    from scripts.summarize_soma_decoder_refinement import summarize
    summary_fixture(tmp_path)
    result = summarize(tmp_path)
    assert result["comparisons"]["C_vs_original"]["long_command_velocity_error_mps"]["no_regression"]
    assert not result["comparisons"]["C_vs_original"]["command_idle_error_mps"]["no_regression"]
    assert not result["candidates"]["C"]["raw_no_regression"] and not result["quality_approved"]


def test_review_rejects_unmatched_sampling(tmp_path):
    import json
    from scripts.summarize_soma_decoder_refinement import summarize
    summary_fixture(tmp_path)
    path = tmp_path / "C/result.json"
    result = json.loads(path.read_text())
    result["sampling_sha256"] = "different"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Sampling"):
        summarize(tmp_path)


def test_absent_contact_evidence_is_null_not_zero(tmp_path):
    from scripts.summarize_soma_decoder_refinement import summarize
    summary_fixture(tmp_path, no_contact=True)
    assert summarize(tmp_path)["arms"]["C"]["mean_window_contact_height_m"] is None
