"""Inspect references on CPU; --generate explicitly runs a frozen MB model."""

import argparse
import json
import math
from pathlib import Path

from motionbricks.mb_sonic.reference import (NativeSomaHistory, load_official_reference,
                                            save_soma_reference, save_raw_mb_block)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--time", type=float, default=0.0)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--execution", choices=("native", "trace", "compiled", "full_plan"), default="native")
    parser.add_argument("--conditioning-mode", choices=("native", "B1"), default="native")
    parser.add_argument("--query-origin-soma-xyz", type=float, nargs=3)
    parser.add_argument("--velocity-xz", type=float, nargs=2, default=(0.0, 0.5))
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--rng-seed", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.directory is None and args.artifact is None and args.seed is None:
        parser.error("Provide an official CSV directory, --artifact/--bundle, or --seed")
    if (args.artifact is None) != (args.bundle is None):
        parser.error("--artifact and --bundle must be provided together")
    if args.generate and any(value is None for value in (args.artifact, args.bundle, args.seed, args.manifest, args.output)):
        parser.error("--generate requires --artifact, --bundle, --seed, --manifest and --output")
    if args.execution != "native" and not args.generate:
        parser.error("--execution trace/compiled/full_plan requires --generate and an explicitly allocated CUDA device")
    if args.conditioning_mode == "B1":
        if not args.generate or args.query_origin_soma_xyz is None:
            parser.error("B1 requires --generate and an explicit calibrated --query-origin-soma-xyz")
        if not all(math.isfinite(value) for value in args.query_origin_soma_xyz):
            parser.error("--query-origin-soma-xyz must be finite")
    elif args.query_origin_soma_xyz is not None:
        parser.error("--query-origin-soma-xyz requires explicit --conditioning-mode B1")
    return args


def main():
    args = parse_arguments()
    result = dict(optimizer_steps=0, gpu_seconds=0)
    if args.directory is not None:
        reference = load_official_reference(args.directory, scene_id=args.scene_id)
        window = reference.sonic_window(args.time)
        result["official_reference"] = dict(
            source=reference.provenance, frames=len(reference.timestamps_s),
            sampled_interval_s=reference.timestamps_s[[0, -1]].tolist(),
            latest_complete_window_start_s=reference.latest_window_start_s,
            qpos_shape=reference.qpos.shape, qvel_shape=reference.qvel.shape,
            joint_names_mujoco=reference.joint_names, body_names=reference.body_names,
            window_timestamps_s=window.timestamps_s.tolist(), window_qpos_shape=window.qpos.shape,
        )
    if args.artifact is not None:
        from motionbricks.mb_sonic.mb_reference import inspect_trained_artifacts

        result["trained_mb"] = inspect_trained_artifacts(args.artifact, args.bundle)
    if args.seed is not None:
        import numpy as np
        from motionbricks.data.native_mb_data import build_native_representation
        from motionbricks.mb_sonic.mb_reference import decode_soma_reference

        with np.load(args.seed, allow_pickle=False) as archive:
            features = archive["initial_history"][0].copy()
        timestamps = args.time + (np.arange(4) - 3) / 30
        reference = decode_soma_reference(
            features, timestamps, build_native_representation(), scene_id=args.scene_id,
            command_id="source_seed_inspection", trusted_until_s=float(timestamps[-1]),
            provenance=dict(kind="source_seed", file=str(args.seed.resolve())),
        )
        result["soma_seed"] = dict(
            features_shape=reference.raw_features.shape, bone_names=reference.bone_names,
            timestamp_s=reference.timestamps_s.tolist(), root_positions_m=reference.root_positions_m.tolist(),
            max_root_fk_error_m=float(np.abs(reference.root_positions_m - reference.world_positions_m[:, 0]).max()),
            native_future_coverage_s=0.8, two_real_block_future_coverage_s=1.6,
        )
    if args.generate:
        from motionbricks.mb_sonic.mb_reference import TrainedMBReferenceAdapter
        from motionbricks.mb_sonic.scene import load_scene

        scene = load_scene(args.manifest)
        if scene.manifest.scene_id != args.scene_id:
            raise ValueError("--scene-id differs from the canonical manifest")
        history = NativeSomaHistory(timestamps, features, args.scene_id,
                                    dict(kind="source_seed", file=str(args.seed.resolve())))
        generator = TrainedMBReferenceAdapter.load(
            args.artifact, args.bundle, canonical_scene=scene, device=args.device)
        import random
        import torch
        from motionbricks.data.full_corpus_inference import tensor_digest

        if args.rng_seed is not None:
            random.seed(args.rng_seed)
            np.random.seed(args.rng_seed)
            torch.manual_seed(args.rng_seed)
        model_before = tensor_digest(generator.controller.state_dict())
        if args.execution != "native":
            generator.prepare_compiled(history, velocity_world_xz_mps=args.velocity_xz,
                                       heading_world_rad=args.heading, execution=args.execution,
                                       conditioning_mode=args.conditioning_mode,
                                       query_origin_soma_xyz=args.query_origin_soma_xyz)
        proposal = generator.generate(history, velocity_world_xz_mps=args.velocity_xz,
                                      heading_world_rad=args.heading, command_id="frozen_open_loop_probe",
                                      blocks=args.blocks, conditioning_mode=args.conditioning_mode,
                                      query_origin_soma_xyz=args.query_origin_soma_xyz)
        model_after = tensor_digest(generator.controller.state_dict())
        if model_after != model_before or any(p.grad is not None for p in generator.controller.parameters()):
            raise RuntimeError("Frozen MB generation changed a model tensor or produced parameter gradients")
        args.output.mkdir(parents=True, exist_ok=False)
        save_soma_reference(args.output / "proposal.npz", proposal.source)
        for index, block in enumerate(proposal.raw_blocks):
            if generator.execution == "full_plan":
                save_raw_mb_block(args.output / f"raw_block_{index:02d}_global366.npz", block)
            else:
                save_soma_reference(args.output / f"raw_block_{index:02d}.npz", block)
        result["generation"] = dict(
            experiment_kind="frozen_mb_source_seed_open_loop_reference", device=args.device,
            output=str(args.output.resolve()), timing_ns=proposal.timing_ns,
            optimizer_steps=0, blocks=args.blocks, model_identity=generator.identity,
            trusted_future_coverage_s=float(proposal.source.timestamps_s[-1] - timestamps[-1]),
            geometry_audits=[block.provenance["geometry_audit"] for block in proposal.raw_blocks],
            execution=generator.execution, compilation=generator.compilation,
            conditioning_mode=args.conditioning_mode,
            query_origin_soma_xyz=args.query_origin_soma_xyz,
            rng_seed=args.rng_seed, model_state_sha256_before=model_before,
            model_state_sha256_after=model_after, model_tensors_unchanged=True,
        )
        result.pop("gpu_seconds")
        (args.output / "generation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
