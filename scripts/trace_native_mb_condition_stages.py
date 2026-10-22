"""Frozen runtime ablation of condition masks per module, never an output repair."""

import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_native_mb_smart_locomotion import (
    load_sources, rollout_case, validate_frozen_bundle,
)
from scripts.train_native_mb import select_gpu, sha256, write_json


@contextmanager
def sparse_stage_inputs(infer, stage):
    """Change only a module's input masks; delegate computation and return identity."""
    if stage not in ("root", "pose"):
        raise ValueError("Expected root or pose stage")
    name = "_predict_root_trajectories" if stage == "root" else "_predict_pose_tokens"
    original, on_instance = getattr(infer, name), name in vars(infer)
    observed = []

    def invoke(batch, *args, **kwargs):
        reduced = dict(batch)
        keys = ("has_global_root_values", "has_local_root_values", "has_local_poses") if stage == "root" else ("has_local_poses",)
        for key in keys:
            mask = batch[key].clone()
            start = 2 if key == "has_local_root_values" else 3
            mask[:] = False
            mask[:, [start, 7]] = batch[key][:, [start, 7]]
            reduced[key] = mask
        observed.append({key: reduced[key].detach().cpu().tolist() for key in keys})
        return original(reduced, *args, **kwargs)

    setattr(infer, name, invoke)
    try:
        yield observed
    finally:
        if on_instance:
            setattr(infer, name, original)
        else:
            delattr(infer, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-run", type=Path, required=True)
    parser.add_argument("--pose-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    gpu = select_gpu(args.gpu)
    import torch
    from motionbricks.data.native_mb_evaluation import full_precision_evaluation
    from motionbricks.data.native_mb_locomotion import NativeReferenceClips, NativeSmartLocomotion
    from scripts.evaluate_native_mb_pipeline import load_bundle

    torch.set_num_threads(4)
    with full_precision_evaluation(), torch.inference_mode():
        infer, dataset, bundle = load_bundle(args.root_run, args.pose_run, "cuda")
        validate_frozen_bundle(bundle)
        seed, references, sources = load_sources(dataset, bundle, len(infer.global_motion_rep.indices["all"]))
        seed = infer.global_motion_rep.change_first_heading(torch.as_tensor(seed, device="cuda")[None], 0., False, False)
        bank = NativeReferenceClips(infer.motion_rep, {key: torch.as_tensor(value, device="cuda") for key, value in references.items()})
        args.output.mkdir(parents=True, exist_ok=False)
        report = {"bundle": bundle, "gpu_at_launch": gpu, "sources": sources,
                  "purpose": "input-mask ablation; no training; not a deployment workaround",
                  "source_sha256": sha256(Path(__file__)), "summary": {}}
        for stage, label in (("root", "root1_pose4_decoder4"), ("pose", "root4_pose1_decoder1")):
            torch.manual_seed(2333)
            np.random.seed(2333)
            controller = NativeSmartLocomotion(infer, bank)
            with sparse_stage_inputs(infer, stage) as masks:
                arrays, detail = rollout_case(infer, seed, "native_buffer4", controller)
            detail.update(diagnostic_case=label, module_input_masks=masks,
                          baseline_case="native_buffer4", diagnostic_not_deployment=True)
            folder = args.output / label
            folder.mkdir()
            np.savez_compressed(folder / "motion.npz", **arrays)
            write_json(folder / "report.json", detail)
            report["summary"][label] = detail["summary"]
            write_json(args.output / "report.json", report)
            print(label, detail["summary"]["played_generated_root"], flush=True)


if __name__ == "__main__":
    main()
