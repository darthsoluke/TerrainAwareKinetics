"""Read-only CPU integration of real native codec inputs and explicit fixtures."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from motionbricks.data.bones_contract import file_sha256
from motionbricks.scene_control_v2.contracts import (
    NativeWindowFrame, TorchSceneQuery, load_run_config, numpy_sample_to_torch_batch,
)


def fixture_contact_preflight(config):
    from motionbricks.scene_control_v2.data import (
        make_synthetic_fixture, make_synthetic_scene, validate_paired_sample,
    )
    from motionbricks.scene_control_v2.losses import relative_contact_losses
    from motionbricks.scene_control_v2.sampling import (
        QueryBudget, offline_teacher_guidance, sample_scene_queries,
    )

    sample = make_synthetic_fixture(frames=8)
    validate_paired_sample(sample)
    scene = make_synthetic_scene(sample)
    queries = sample_scene_queries(scene, sample, guidance=offline_teacher_guidance(sample),
        mode="offline", as_of_s=float(sample["timestamps_s"][-1]),
        budget=QueryBudget(**config["sampling"]["query_budget"]), seed=2333)
    if queries["policy"] != config["sampling"]["policy_id"]:
        raise ValueError("Configured query policy differs from the actual sampler")
    batch = numpy_sample_to_torch_batch(sample, queries=queries)
    # This is a declared perturbation fixture, not a model prediction or dataset.
    predicted = (batch["body_positions_world_m"] + torch.tensor([.01, .015, -.005])).requires_grad_()
    times = torch.arange(predicted.shape[1])[None, :, None].expand(predicted.shape[:-1])
    bridge = TorchSceneQuery(scene, scene_metadata=sample["scene"], mode="offline",
                            as_of_s=float(sample["timestamps_s"][-1]))
    body_query = bridge(predicted, times, timestamps_s=sample["timestamps_s"])
    contact_indices = batch["contacts"]["body_part_id"][0]
    contact_points = predicted.index_select(2, contact_indices)
    contact_times = times.index_select(2, contact_indices)
    contact_query = bridge(contact_points, contact_times, timestamps_s=sample["timestamps_s"],
        assigned_object_id=batch["contacts"]["object_id"])
    terms = relative_contact_losses(predicted, contact_query["distance_m"], batch["contacts"],
        batch["scene"], contact_indices, contact_query["valid"],
        torch.ones((1, predicted.shape[1] - 1, len(contact_indices)), dtype=torch.bool),
        torch.ones(predicted.shape[:2], dtype=torch.bool), fps=sample["fps"])
    noncontact = body_query["valid"] & batch["body_valid"]
    noncontact[:, :, contact_indices] = False
    terms["penetration_m"] = torch.relu(-body_query["distance_m"])[noncontact].mean()
    gradient, = torch.autograd.grad(sum(terms.values()), predicted)
    if not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
        raise ValueError("Fixture geometry/contact objective lost its point gradients")
    return {"source": "synthetic", "skeleton": sample["skeleton"],
            "purpose": "offline_oracle_perturbation_fixture_not_training_data",
            "guidance_role": queries["guidance_role"], "query_count": len(queries["valid"]),
            "valid_queries": int(queries["valid"].sum()), "policy_id": queries["policy"],
            "distance_semantics": bridge.distance_semantics,
            "contact_loss_terms": {key: float(value.detach()) for key, value in terms.items()},
            "point_gradient_l2": float(gradient.norm()), "quality_approved": False}


def native_codec_preflight(config, bundle, checkpoint):
    from omegaconf import OmegaConf
    from motionbricks.data.native_mb_models import _vqvae_config
    from motionbricks.data.native_mb_vertical_frame import shift_global_motion
    from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    from motionbricks.motionlib.core.utils.stats import Stats
    from motionbricks.scene_control_v2.codec import SceneControlCodec, SomaGeometryAdapter
    from motionbricks.scene_control_v2.quantizer import ExplicitHeadQuantizer, QuantizerConfig

    stats = Stats()
    if any(bundle[key].numel() != config["model"]["statistics_features"] for key in ("stats_mean", "stats_std")):
        raise ValueError("Frozen dual-root statistics width differs from the resolved config")
    stats.register_from_tensors(bundle["stats_mean"], bundle["stats_std"])
    rep = GlobalRootGlobalJoints(30, SOMASkeleton30(), config["model"]["representation"], stats=stats)
    if list(rep.skeleton.bone_order_names) != bundle["bone_names"]:
        raise ValueError("Bundle skeleton differs from native SOMA")
    torch.testing.assert_close(rep.skeleton.neutral_joints, bundle["neutral_joints"], atol=1e-6, rtol=0)
    if (len(rep.indices["all"]) != config["model"]["global_features"] or
            len(rep.dual_rep.indices["all"]) != config["model"]["dual_features"]):
        raise ValueError("Global and dual feature layouts differ from the resolved config")
    native = _vqvae_config(OmegaConf.create(bundle["tokenizer_config"]), rep.dual_rep.local_motion_rep)
    codec_state = {name.removeprefix("codec."): tensor for name, tensor in checkpoint["model_state"].items()
                   if name.startswith("codec.")}
    native.load_state_dict(codec_state, strict=True)
    cfg = config["codec"]
    quantizer = ExplicitHeadQuantizer(QuantizerConfig(entries_per_head=cfg["entries_per_head"],
        num_heads=cfg["num_heads"], code_dim=cfg["code_dim"], decay=cfg["decay"]))
    if cfg["initialization"] == "legacy_exact":
        quantizer.vq.load_state_dict(native.quantizer.vq.state_dict(), strict=True)
    codec = SceneControlCodec(native, quantizer, contact_channels=2).eval().requires_grad_(False)
    before = {key: tensor.clone() for key, tensor in quantizer.state_dict().items()}

    # Keep the captured four-frame seed intact, then rigidly place it away from origin.
    seed = bundle["initial_history"].clone()
    placed = rep.change_first_heading(seed, 1.1, False, False)
    placed = shift_global_motion(placed, rep, placed.new_tensor([.65]))
    placed[..., [0, 2]] += placed.new_tensor([2.4, -1.7])
    frame = NativeWindowFrame.from_global_history(placed, vertical_origin_m=torch.tensor([.65]))
    centered = frame.center_global_motion(placed, rep)
    local = rep.dual_rep.global_to_local(centered, False, True,
        lengths=torch.tensor([placed.shape[1]], dtype=torch.long))
    external = extract_feature_from_motion_rep(local, native.motion_rep,
                                               native.decoder_external_cond_feature_mode)
    with torch.inference_mode():
        output = codec(local, target_cond=local, has_target_cond=torch.ones(local.shape[:2], dtype=torch.bool),
                       external_cond=external)
        restored = frame.restore_local_motion(local, rep)
        torch.testing.assert_close(restored, placed, atol=2e-5, rtol=1e-5)
        skinner = move_skinner(load_official_skinner(rep.skeleton), torch.device("cpu"), torch.float32)
        skinner.skeleton_skin.float()
        geometry = SomaGeometryAdapter(native.motion_rep, skinner,
            torch.tensor([0, 500, 1000, 5000, 12000, 18055]))
        world_from_window = torch.eye(4)[None]
        world_from_window[:, :3, 3] = frame.translation_world_m
        restored_geometry = geometry(local, world_from_window=world_from_window)
        reference = rep.inverse(placed, False, return_quat=False, return_all=False,
                                joint_positions_from="global_rot_data")["posed_joints"]
        torch.testing.assert_close(restored_geometry["joint_positions_m"], reference, atol=2e-5, rtol=1e-5)
        if cfg["initialization"] == "legacy_exact":
            expected = native(local, local, torch.ones(local.shape[:2], dtype=torch.bool), external)["recon_state"]
            torch.testing.assert_close(output["recon_state"], expected, atol=1e-6, rtol=0)
    for key, value in quantizer.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    if not torch.isfinite(output["recon_state"]).all():
        raise ValueError("Nonfinite native codec output")
    return {"source": "existing_bundle_initial_history_real_seed",
            "seed_provenance": bundle["seed_provenance"], "source_frames": placed.shape[1],
            "placement": "rigid_heading_1.1_rad_and_translation_not_new_motion",
            "global_features": placed.shape[-1], "normalized_local_features": local.shape[-1],
            "reconstruction_shape": list(output["recon_state"].shape),
            "indices_shape": list(output["indices"].shape), "index_layout": "B,L,H",
            "entries_per_head": cfg["entries_per_head"], "initialization": cfg["initialization"],
            "contact_head": "untrained_path_only", "normalization_sha256": bundle["stats_tensor_sha256"],
            "seed_fk_world_max_error_m": float((restored_geometry["joint_positions_m"] - reference).abs().max()),
            "window_frame": {"translation_world_m": frame.translation_world_m.tolist(),
                             "initial_heading_rad": frame.initial_heading_rad.tolist(),
                             "geometry_rotation": "heading_retained_no_extra_yaw"},
            "selected_skin_vertices": 6, "real_paired_scene_loaded": False,
            "fullskin_gate_evaluated": False, "quantizer_state_unchanged": True,
            "quality_approved": False}


def run_preflight(config_path):
    config = load_run_config(config_path)
    artifacts = config["artifacts"]
    for path_key, hash_key in (("architecture_bundle", "bundle_sha256"), ("checkpoint", "checkpoint_sha256")):
        if file_sha256(artifacts[path_key]) != artifacts[hash_key]:
            raise ValueError(f"Artifact content differs: {path_key}")
    bundle = torch.load(artifacts["architecture_bundle"], map_location="cpu", mmap=True, weights_only=False)
    checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    if (bundle["schema"] != "motionbricks.mixed_online_soma_bundle.v1" or
            checkpoint["schema"] != "motionbricks.unified.full_corpus.v1" or
            checkpoint["contract"]["bundle_sha256"] != artifacts["bundle_sha256"]):
        raise ValueError("The checkpoint does not bind this architecture bundle")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2333)
        fixture = fixture_contact_preflight(config)
        native = native_codec_preflight(config, bundle, checkpoint)
    return {"schema": "motionbricks.scene_control_v2.cpu_preflight.v1", "resolved_config": config,
            "fixture": fixture, "native": native, "device": "cpu", "read_only": True,
            "actual_adam_updates": 0, "diagnostic_training_exposures": 0, "formal_training_exposures": 0,
            "legacy_completed_exposures": 38400000, "quality_approved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "scene_control_v2_engineering_8x10.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    # Third-party initialization messages stay on stderr; stdout is one JSON receipt.
    with contextlib.redirect_stdout(sys.stderr):
        report = run_preflight(args.config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
