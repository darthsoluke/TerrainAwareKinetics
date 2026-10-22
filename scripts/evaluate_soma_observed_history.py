"""Paired no-training counterfactual of the history observation contract."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.bones_contract import file_sha256
from scripts.evaluate_soma_decoder_refinement import long_rollout
from scripts.portable_bones_control import load_bundle


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "bundle", "reference", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    reference = json.loads(args.reference.read_text())
    if reference["bundle_sha256"] != file_sha256(args.bundle):
        raise ValueError("Use the same weights as the reference evaluation")
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    inference, _ = load_bundle(args.bundle, "cuda:0")
    records = {r["filename"]: r for r in json.loads((args.dataset / "accepted.json").read_text())}
    args.output.mkdir(parents=True)
    report = {"bundle_sha256": file_sha256(args.bundle), "test_used": False, "post_processing": None,
              "quality_approved": False, "training": None, "commit_frames": 8,
              "scope": "input geometry only; no changes to generated or published poses; no contact-quality conclusion",
              "reference_sha256": file_sha256(args.reference),
              "code_sha256": {str(p): file_sha256(p) for p in (Path(__file__),
                  Path("motionbricks/data/bones_rollout.py"), Path("scripts/evaluate_soma_decoder_refinement.py"))},
              "rollouts": {}}
    for index in range(3):
        state = reference["rollouts"][str(index)]
        record = records[state["initial_filename"]]
        if record["split"] != "validation":
            raise ValueError("Initial history must be from validation, not test")
        path = args.dataset / record["shard"]
        start = state["initial_frame"]
        with np.load(path, allow_pickle=False) as data:
            initial = torch.from_numpy(data["global_features"][start:start + 4].copy())[None].cuda()
        report["rollouts"][str(index)] = {"filename": record["filename"], "start": start,
                                          "input_sha256": file_sha256(path)}
        for policy in ("predicted_features", "executed_fk"):
            inference.history_geometry = policy
            motion, metrics = long_rollout(inference, initial.clone(), commit_frames=8)
            report["rollouts"][str(index)][policy] = metrics
            np.savez_compressed(args.output / f"{policy}_{index}.npz",
                                global_features=motion[0].cpu().numpy(), fps=np.asarray(60),
                                boundaries=np.arange(8, 3600, 8))
            (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps({"history": index, "policy": policy,
                              "joint_step_p95_mm": metrics["boundary_joint_step_mm"]["p95"],
                              "worst_step_deg": metrics["angular_step_deg"]["max"],
                              "command_error_mps": metrics["root_command_velocity_error_mps"]["mean"]}), flush=True)


if __name__ == "__main__":
    main()
