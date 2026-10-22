"""Screen untrained official human networks; not an end-to-end 60 Hz certificate."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

from motionbricks.data.bones_seed import build_representation
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--pose-width", type=int, default=512)
    parser.add_argument("--pose-layers", type=int, default=8)
    parser.add_argument("--frames", type=int, choices=(64, 128), default=128)
    parser.add_argument("--precision", choices=("float32", "float16"), default="float16")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--execution", choices=("eager", "traced", "cuda-graph"), default="eager")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    project = Path(__file__).resolve().parents[1]
    rep = build_representation().cuda()
    # Upstream FK constructs hierarchy levels with Python int(parent[i]). CUDA
    # parents cause one GPU synchronization per joint; topology is static CPU metadata.
    rep.skeleton.joint_parents = rep.skeleton.joint_parents.cpu()
    # Unit statistics serve only shape/timing screening; no motion quality is measured.
    for representation in (rep.dual_rep, rep, rep.dual_rep.global_motion_rep, rep.dual_rep.local_motion_rep):
        from motionbricks.motionlib.core.utils.stats import Stats
        stats = Stats()
        stats.register_from_tensors(torch.zeros(len(representation.indices["all"]), device="cuda"),
                                    torch.ones(len(representation.indices["all"]), device="cuda"))
        representation.stats = stats
    root_config = OmegaConf.load(project / "out/motionbricks_root/version_1/hparams.yaml")
    pose_config = OmegaConf.load(project / "out/motionbricks_pose/version_1/hparams.yaml")
    for config in (root_config, pose_config):
        with open_dict(config):
            config.data = {"text_embeddings": None}
            config.model.args.min_tokens = 6
            config.model.args.max_tokens = args.frames // 4
    with open_dict(pose_config):
        pose_config.model.backbone_network.args.n_embd = args.pose_width
        pose_config.model.backbone_network.args.n_head = args.pose_width // 64
        pose_config.model.backbone_network.args.n_layers = args.pose_layers
    root = instantiate(root_config.model.backbone_network, motion_rep=rep, _recursive_=False).cuda().eval()
    pose = instantiate(pose_config.model.backbone_network, motion_rep=rep, _recursive_=False).cuda().eval()
    tokenizer = instantiate(pose_config.model.pose_vqvae_network,
                            motion_rep=rep.dual_rep.local_motion_rep).cuda().eval()
    pose.init_embedding_from_codebooks(tokenizer.quantizer.vq._codebook.embed, None)
    history_frames, frames, tokens = 4, args.frames, args.frames // 4
    global_root = torch.zeros(1, 8, 5, device="cuda")
    global_root[..., 1], global_root[..., 3] = 1, 1
    global_root[:, 4:, 2] = 1
    local_root = torch.zeros(1, 8, 4, device="cuda")
    local_root[..., -1] = 1
    pose_dimension = len(extract_feature_from_motion_rep(None, rep.dual_rep.local_motion_rep,
                         "joint_positions_and_rotations_and_hip_height", fetch_feat_idx=True))
    observed_pose = torch.zeros(1, 8, pose_dimension, device="cuda")
    root_mask = torch.ones(1, 8, device="cuda", dtype=torch.bool)
    observed_mask = torch.zeros_like(root_mask)
    observed_mask[:, :history_frames] = True
    lengths = torch.full((1, 1), tokens, device="cuda", dtype=torch.long)
    pose_tokens = torch.full((1, tokens, 8), pose.POSE_MASK_ID, device="cuda", dtype=torch.long)
    dense_pose = torch.zeros(1, frames, pose_dimension, device="cuda")
    dense_mask = torch.zeros(1, frames, device="cuda", dtype=torch.bool)
    dense_mask[:, :history_frames] = True
    def update(global_values=global_root, local_values=local_root, pose_values=observed_pose):
        output = root(global_values, root_mask, local_values, root_mask, pose_values, observed_mask, lengths)
        trajectory = output["pred_global_root_values"][:, :frames]
        root_condition = extract_feature_from_motion_rep(trajectory, rep.dual_rep.global_motion_rep,
                                                         pose._args["cond_root_feature"])
        current_pose = torch.cat([pose_values[:, :history_frames], dense_pose[:, history_frames:]], 1)
        output_pose = pose(pose_tokens, root_condition, current_pose, dense_mask, lengths)
        code = output_pose["pose_logits"].argmax(-1)
        local_trajectory = rep.dual_rep.global_to_local(
            trajectory, True, True, lengths=torch.full((1,), frames, device="cuda", dtype=torch.long))
        external = extract_feature_from_motion_rep(local_trajectory, rep.dual_rep.local_motion_rep,
                                                   tokenizer.decoder_external_cond_feature_mode)
        decoded = tokenizer.forward_decoder(code, None, external_cond=external, use_overall_indices=False)
        features = decoded["recon_state"] if isinstance(decoded, dict) else decoded
        return rep.dual_rep.local_motion_rep.inverse(features, True, return_all=False)["posed_joints"]

    cpu_times, gpu_times = [], []
    autocast = args.precision == "float16"
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        for _ in range(50):
            prediction = update()
        torch.cuda.synchronize()
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Nonfinite untrained inference output")
        trace_error = None
        if args.execution != "eager":
            class Chain(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.root, self.pose, self.tokenizer, self.rep = root, pose, tokenizer, rep

                def forward(self, global_values, local_values, pose_values):
                    return update(global_values, local_values, pose_values)

            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast, cache_enabled=False):
                chain = torch.jit.freeze(torch.jit.trace(Chain().eval(),
                    (global_root, local_root, observed_pose), check_trace=False))
            reference = update(global_root, local_root, observed_pose)
            traced_output = chain(global_root, local_root, observed_pose)
            trace_error = (reference - traced_output).abs().max().item()
            if trace_error > 0.002:
                raise RuntimeError(f"Trace changes the reference by {trace_error} m")
            changed = global_root.clone()
            changed[:, 4:, 2] += 0.4
            changed_reference = update(changed, local_root, observed_pose)
            changed_trace = chain(changed, local_root, observed_pose)
            if (changed_reference - changed_trace).abs().max() > 0.002:
                raise RuntimeError("Trace failed changed-input equivalence check")
            if (changed_trace - traced_output).abs().max() < 1e-6:
                raise RuntimeError("Trace ignores changing root input")
            traced_update = lambda: chain(global_root, local_root, observed_pose)
            for _ in range(50):
                traced_update()
            torch.cuda.synchronize()
            if args.execution == "cuda-graph":
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph):
                        graph_prediction = chain(global_root, local_root, observed_pose)
                except RuntimeError as error:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({"execution": "cuda-graph",
                        "capture_passed": False, "acceptance_gate_passed": False,
                        "error": str(error)}, indent=2))
                    raise RuntimeError("CUDA graph capture failed; diagnostic saved to " + str(args.output)) from error
                graph.replay()
                torch.cuda.synchronize()
                if (graph_prediction - traced_output).abs().max() > 0.002:
                    raise RuntimeError("CUDA graph changes the traced reference")
                global_root.copy_(changed)
                graph.replay()
                torch.cuda.synchronize()
                if (graph_prediction - changed_trace).abs().max() > 0.002:
                    raise RuntimeError("CUDA graph does not consume changed input correctly")
                def graph_update():
                    graph.replay()
                    return graph_prediction
        run_update = graph_update if args.execution == "cuda-graph" else traced_update if args.execution == "traced" else update
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profiler:
                run_update()
                torch.cuda.synchronize()
            print(profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
            print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        for _ in range(args.iterations):
            started = time.perf_counter()
            start_event.record()
            prediction = run_update()
            end_event.record()
            torch.cuda.synchronize()
            cpu_times.append((time.perf_counter() - started) * 1000)
            gpu_times.append(start_event.elapsed_time(end_event))
    def summary(values):
        return {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)), "max_ms": float(max(values)),
                "deadline_misses_16_67ms": int((np.array(values) > 1000 / 60).sum())}
    report = {
        "device": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "precision": args.precision, "pose_width": args.pose_width, "pose_layers": args.pose_layers,
        "execution": args.execution, "trace_reference_max_error_m": trace_error,
        "frames": frames, "iterations": args.iterations,
        "scope": "untrained root + single-pass pose + decoder + FK; no transport/UE/current-input loop",
        "acceptance_gate_passed": False, "cpu_wall_clock": summary(cpu_times),
        "gpu_events": summary(gpu_times), "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
