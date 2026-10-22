"""Export trained inference weights and benchmark the real control chain locally."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256, require_same_snapshot
from motionbricks.data.bones_control import SomaMotionInference
from motionbricks.data.bones_rollout import SomaLocomotionBackend, load_control_models
from motionbricks.data.bones_seed import build_representation, GlobalRootGlobalJoints, SOMASkeleton30
from motionbricks.motionlib.core.utils.stats import Stats
from scripts.evaluate_bones_control import command_at, distribution
from scripts.train_bones_control import build_control_model


def load_bundle(path, device):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle["schema"] != "motionbricks.soma_control_bundle.v1":
        raise ValueError("Unknown portable backend contract")
    history_geometry = bundle.get("history_geometry", "predicted_features")
    if history_geometry not in ("predicted_features", "executed_fk"):
        raise ValueError("Unknown history observation contract")
    if history_geometry == "executed_fk":
        trained = bundle.get("joint_rollout_adaptation", {})
        if trained.get("history_geometry") != history_geometry or trained.get("completed_updates", 0) < 1:
            raise ValueError("Observed FK input requires matching training, not a switch on old weights")
    policy = bundle["pose_config"]["model"]["pose_vqvae_network"].get("root_condition_policy", "provided_velocity")
    if policy == "learned_refinement":
        trained = bundle.get("decoder_refinement", {})
        if trained.get("root_condition_policy") != policy or trained.get("completed_updates", 0) < 2500:
            raise ValueError("Learned root output needs a trained decoder, not a policy switch on old weights")
    stats = Stats()
    stats.register_from_tensors(bundle["stats_mean"], bundle["stats_std"])
    rep = GlobalRootGlobalJoints(60, SOMASkeleton30(), "somaskel30_dual_root_global_joints", stats=stats)
    if (list(rep.skeleton.bone_order_names) != bundle["bone_names"]
            or not torch.allclose(rep.skeleton.neutral_joints.cpu(), bundle["neutral_joints"], atol=1e-6, rtol=0)):
        raise ValueError("Local SOMA skeleton differs from the trained backend")
    models = {}
    for stage in ("root", "pose"):
        config = OmegaConf.create(bundle[stage + "_config"])
        tokenizer_state = {name.removeprefix("pose_net."): value for name, value in bundle["pose_state"].items()
                           if name.startswith("pose_net.")} if stage == "pose" else None
        model = build_control_model(stage, config, rep, tokenizer_state=tokenizer_state)
        model.load_state_dict(bundle[stage + "_state"], strict=True)
        model.requires_grad_(False).eval().to(device)
        models[stage] = model
    args = {"min_tokens": models["pose"].args["min_tokens"], "max_tokens": models["pose"].args["max_tokens"],
            "cond_root_feature_is_from_motion_rep": models["pose"].args["cond_root_feature_is_from_motion_rep"]}
    # Hierarchy indices are static CPU topology, not GPU motion state.
    inference = SomaMotionInference(models, args, device=device)
    inference.history_geometry = history_geometry
    rep.skeleton.joint_parents = rep.skeleton.joint_parents.cpu()
    if bundle.get("body_goal_policy") == "single_endpoint":
        from motionbricks.data.soma_goal_conditioning import ReferencePoseLibrary
        inference.goal_library = ReferencePoseLibrary(bundle["reference_pose_library"], rep, device)
    return inference, bundle


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    export = sub.add_parser("export")
    for name in ("dataset", "root-run", "pose-run", "output"):
        export.add_argument("--" + name, required=True, type=Path)
    benchmark = sub.add_parser("benchmark")
    benchmark.add_argument("--bundle", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--iterations", type=int, default=900)
    benchmark.add_argument("--commit-frames", type=int, choices=(4, 8), default=4)
    benchmark.add_argument("--argmax", action="store_true")
    benchmark.add_argument("--decoder-cuda-graph", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.mode == "export":
        if args.output.exists():
            raise ValueError("Refuse to overwrite a portable checkpoint")
        report = json.loads((args.dataset / "quality_report.json").read_text())
        rep = build_representation(args.dataset / report["stats_folder"])
        inference, contracts = load_control_models(args.root_run, args.pose_run, rep, "cpu")
        require_same_snapshot(contracts["pose"]["dataset_snapshot"], dataset_snapshot(args.dataset))
        records = json.loads((args.dataset / "accepted.json").read_text())
        record = next(r for r in records if r["split"] == "validation" and r["behavior"] == "idle")
        with np.load(args.dataset / record["shard"], allow_pickle=False) as data:
            initial = torch.from_numpy(data["global_features"][:4].copy())[None]
        initial = rep.change_first_heading(initial, torch.zeros(1), False, False)
        bundle = {"schema": "motionbricks.soma_control_bundle.v1", "initial_history": initial,
                  "bone_names": list(rep.skeleton.bone_order_names), "neutral_joints": rep.skeleton.neutral_joints.cpu(),
                  "initial_history_source": record["filename"], "contracts": contracts,
                  "stats_mean": rep.dual_rep.stats.mean.cpu(), "stats_std": rep.dual_rep.stats.std.cpu(),
                  "root_state": inference._root_model.cpu().state_dict(),
                  "pose_state": inference._pose_model.cpu().state_dict(),
                  "root_config": OmegaConf.to_container(OmegaConf.load(args.root_run / "upstream_config_overrides.yaml"), resolve=False),
                  "pose_config": OmegaConf.to_container(OmegaConf.load(args.pose_run / "upstream_config_overrides.yaml"), resolve=False),
                  "root_checkpoint_sha256": file_sha256(args.root_run / "checkpoints/final.ckpt"),
                  "pose_checkpoint_sha256": file_sha256(args.pose_run / "checkpoints/final.ckpt"),
                  "quality_approved": False, "post_processing": None}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(bundle, args.output)
        print(json.dumps({"bundle": str(args.output), "sha256": file_sha256(args.output)}), flush=True)
        return
    if not torch.cuda.is_available() or not 1 <= args.iterations <= 900:
        raise ValueError("Benchmark requires CUDA and a bounded 1-900 iteration protocol")
    inference, bundle = load_bundle(args.bundle, "cuda:0")
    if args.decoder_cuda_graph:
        from motionbricks.data.soma_cuda_graph import install_decoder_graph
        install_decoder_graph(inference)
    torch.manual_seed(2333)
    initial = bundle["initial_history"].cuda()
    with torch.inference_mode():
        backend = SomaLocomotionBackend(inference, initial, commit_frames=args.commit_frames, argmax=args.argmax)
        for _ in range(30):
            backend.step([[0, 0]], [0])
        torch.cuda.synchronize()
        backend = SomaLocomotionBackend(inference, initial, commit_frames=args.commit_frames, argmax=args.argmax)
        torch.cuda.reset_peak_memory_stats()
        timings, inference_timings, packets, timestamps = [], [], [], []
        for iteration in range(args.iterations):
            name, velocity, heading = command_at(iteration * args.commit_frames / 60)
            started = time.perf_counter()
            result = backend.step([velocity], [heading])
            pose = inference.motion_rep.inverse(result["motion"], False, return_all=False)
            packet = torch.cat((pose["posed_joints"][..., 0, :],
                                pose["local_joint_rots"].flatten(-3), pose["foot_contacts"]), -1).cpu()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            inference_timings.append(result["generation_seconds"])
            if not torch.isfinite(packet).all():
                raise FloatingPointError("Nonfinite engine pose packet")
            packets.append(packet)
            timestamps.append(result["timestamps_s"].cpu())
        timing = distribution(timings)
        result = {"gpu": torch.cuda.get_device_name(), "iterations": args.iterations,
                  "committed_native_frames": args.iterations * args.commit_frames, "native_pose_fps": 60,
                  "commit_frames": args.commit_frames, "argmax": args.argmax,
                  "decoder_cuda_graph": args.decoder_cuda_graph,
                  "root_pose_decoder_fk_cpu_packet_seconds": timing,
                  "model_generation_seconds": distribution(inference_timings),
                  "chunk_budget_seconds": args.commit_frames / 60,
                  "chunks_exceeding_budget": sum(value > args.commit_frames / 60 for value in timings),
                  "model_calls_per_second_measured": 1 / timing["mean"],
                  "native_pose_throughput_frames_per_second": args.commit_frames / timing["mean"],
                  "peak_allocated_memory_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                  "weights_sha256": file_sha256(args.bundle), "source_checkpoint_sha256": bundle["pose_checkpoint_sha256"],
                  "runtime_code_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path)
                                          for path in (Path(__file__).resolve(),
                                                       Path(__file__).resolve().parents[1] / "motionbricks/data/bones_control.py",
                                                       Path(__file__).resolve().parents[1] / "motionbricks/data/bones_rollout.py",
                                                       Path(__file__).resolve().parents[1] / "motionbricks/data/soma_cuda_graph.py")},
                  "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                  "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                  "execution": ("trained FP32; native CUDA-graph decoder, remaining stages eager" if args.decoder_cuda_graph
                                else "trained FP32 eager; root -> tokens -> decoder -> SOMA FK -> CPU packet"),
                  "post_processing": None, "UE_frame_time_tested": False, "motion_quality_approved": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
        np.savez_compressed(args.output.with_suffix(".npz"), packet=torch.cat(packets, 1).numpy(),
                            timestamps_s=torch.cat(timestamps).numpy(), fps=np.asarray(60))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
