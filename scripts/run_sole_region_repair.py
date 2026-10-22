"""One bounded, warm-started locomotion reconstruction repair and its control run."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from motionbricks.data.bones_contract import dataset_snapshot, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initial-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--initialization-review", type=Path, help="Reuse an existing content-bound token-decode review")
    parser.add_argument("--gpus", default="2,5")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--comparison", choices=("point-region", "consistency"), default="point-region")
    args = parser.parse_args()
    if not 1000 < args.steps <= 10000:
        parser.error("The bounded repair permits 1001 to 10000 updates, not full-corpus training")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh experiment directory")
    args.output.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    labels = json.loads((args.labels / "manifest.json").read_text())
    records = json.loads((args.dataset / "accepted.json").read_text())
    expected = {r["filename"] for r in records if r["split"] == "train" and r["frames_60hz"] >= 129}
    if {r["filename"] for r in labels["items"]} != expected:
        raise ValueError("Whole training split must be covered without validation or test records")
    if any(r["split"] != "train" for r in labels["items"]):
        raise ValueError("Held-out records must never enter training")
    snapshot = dataset_snapshot(args.dataset)
    checkpoint_name = "final.ckpt" if args.comparison == "consistency" else "best-step=10000.ckpt"
    initial = args.initial_run / "checkpoints" / checkpoint_name
    review_dir = args.output / "initial_review"
    if args.initialization_review:
        review = args.initialization_review
    else:
        subprocess.run([sys.executable, str(project / "scripts/evaluate_bones_tokenizer.py"),
                        "--dataset", str(args.dataset), "--checkpoint", str(initial),
                        "--output", str(review_dir)], check=True)
        review = review_dir / "evaluation.json"
    checked = json.loads(review.read_text())
    if (checked["checkpoint_sha256"] != file_sha256(initial)
            or checked["dataset_snapshot_sha256"] != snapshot["sha256"]
            or not checked["token_decoder_parity"]["passed"]):
        raise ValueError("Initial token-decode review does not bind these exact weights and data")
    common = {"skate_contact_loss_coeff": 0.0, "joint_vel_loss_coeff": 2.0,
              "fk_position_coeff": 5.0, "sole_position_coeff": 2.0}
    objectives = {
        "point": dict(common, trusted_skate_coeff=1.0, penetration_coeff=2.0,
                      stance_anchor_coeff=2.0, support_height_coeff=2.0),
        "region": dict(common, trusted_skate_coeff=0.0, penetration_coeff=0.0,
                       stance_anchor_coeff=0.0, support_height_coeff=0.0,
                       sole_region_coeff_contact_height_m=2.0, sole_region_coeff_penetration_m=2.0,
                       sole_region_coeff_slip_mps=1.0, sole_region_coeff_stance_displacement_m=10.0),
    }
    if args.comparison == "consistency":
        source_contract = json.loads((args.initial_run / "run_contract.json").read_text())
        if not source_contract.get("sole_region_labels_version"):
            raise ValueError("Consistency comparison requires an already trained region checkpoint")
        baseline = dict(source_contract["geometry_coefficients"], kinematic_position_consistency_coeff=0.0,
                        kinematic_velocity_consistency_coeff=0.0)
        objectives = {"control": baseline,
                      "consistent": dict(baseline, kinematic_position_consistency_coeff=5.0,
                                         kinematic_velocity_consistency_coeff=0.5)}
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True)
    (args.output / "gpu_at_launch.txt").write_text(inventory)
    requested = [int(index) for index in args.gpus.split(",")]
    available = {int(row.split(",")[0]) for row in inventory.strip().splitlines()
                 if int(row.split(",")[1]) <= 1024 and int(row.split(",")[2]) <= 5}
    uuids = {int(row.split(",")[0]): row.split(",")[1].strip() for row in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True).strip().splitlines()}
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= available:
        raise RuntimeError("Requested GPUs are not idle; no training launched")
    jobs = []
    try:
        for number, (arm, coefficients) in enumerate(objectives.items()):
            if len(requested) == 1 and jobs:
                if jobs[-1][1].wait() != 0:
                    raise RuntimeError("Control run failed; repair was not launched")
            protocol = {
                "schema": "motionbricks.m1.protocol.v1", "stage": "tokenizer",
                "experiment_kind": "bounded_tokenizer_support_repair", "max_steps": args.steps,
                "batch_size": 16, "scheduler_warmup_steps": 500,
                "validation_interval": 1000, "validation_windows_per_clip": 3,
                "accepted_sha256": snapshot["accepted_sha256"], "stats_folder": snapshot["stats_folder"],
                "initialize_sha256": file_sha256(initial), "tokenizer_profile": "paper-informed",
                "geometry_loss_profile": "support-repair", "geometry_coefficients": coefficients,
                "arm": arm, "reconstruction_only": True,
                "hypothesis": "Continue converged pose weights; compare old point supervision with official "
                              "sole-region slip plus uninterrupted material-point stance displacement.",
                "evaluation": "All 57 fixed held-out windows and the same five showcase windows; no IK or foot locking",
                "stop_rule": "Stop at frozen budget; failed reconstruction/contact screens block deployment, not reporting",
            }
            source_protocol = json.loads((args.initial_run / "experiment_protocol.json").read_text())
            protocol["numerical_screen"] = source_protocol.get("numerical_screen") or json.loads(
                (project / "docs/m1_tokenizer_support_repair_protocol.json").read_text())["numerical_screen"]
            if args.comparison == "consistency":
                protocol["hypothesis"] = "Coupling redundant XYZ/velocity outputs to official FK reduces " \
                    "representation disagreement and source-masked slip without moving supervision masks or changing the network."
                protocol["mechanism_reference"] = "DartControl mld/train_mvae.py calc_loss; SOMA FK/world-axis velocities here"
                protocol["prediction"] = {"fk_feature_disagreement_ratio_max": 0.7,
                                          "measured_slip_ratio_max": 0.8, "mpjpe_ratio_max": 1.05,
                                          "comparison": "consistent/control on identical 57 fixed validation windows"}
                protocol["coefficient_rationale"] = "Position coupling 5/m matches existing FK position scale; " \
                    "velocity coupling 0.5 s/m uses a 0.1-second temporal scale. Fixed before training, no sweep."
            protocol_file = args.output / f"protocol_{arm}.json"
            protocol_file.write_text(json.dumps(protocol, indent=2))
            overrides = dict(coefficients)
            if arm == "region" or args.comparison == "consistency":
                overrides.update(sole_region_supervision=True, sole_region_labels_version="sole_region_full_train_v3")
            loss_file = args.output / f"loss_{arm}.json"
            loss_file.write_text(json.dumps(overrides, indent=2))
            command = [sys.executable, str(project / "scripts/train_bones_locomotion.py"),
                       "--stage", "tokenizer", "--dataset", str(args.dataset), "--output", str(args.output / arm),
                       "--gpus", str(requested[number % len(requested)]), "--max-steps", str(args.steps),
                       "--batch-size", "16", "--workers", "2", "--scheduler-warmup-steps", "500",
                       "--validation-interval", "1000", "--geometry-loss-profile", "support-repair",
                       "--support-repair-pilot", "--protocol", str(protocol_file), "--initialize-from", str(initial),
                       "--initialization-review", str(review), "--geometry-preflight",
                       str(args.initial_run / "geometry_preflight.json"), "--loss-overrides", str(loss_file)]
            if arm == "region" or args.comparison == "consistency":
                command += ["--sole-region-labels", str(args.labels)]
            log = (args.output / f"{arm}.log").open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuids[requested[number % len(requested)]])
            process = subprocess.Popen(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, env=env)
            jobs.append((arm, process, log))
            print(json.dumps({"launched": arm, "pid": process.pid}), flush=True)
        status = {arm: process.wait() for arm, process, log in jobs}
        (args.output / "training_status.json").write_text(json.dumps(status, indent=2))
        if any(status.values()):
            raise RuntimeError(f"Training did not complete: {status}; see per-arm logs")
        contracts = [json.loads((args.output / arm / "run_contract.json").read_text()) for arm in objectives]
        if contracts[0]["initial_weights_sha256"] != contracts[1]["initial_weights_sha256"]:
            raise ValueError("The arms did not start with identical learned/EMA state")
        (args.output / "paired_initialization.json").write_text(json.dumps({
            "same_loaded_state": True, "initial_weights_sha256": contracts[0]["initial_weights_sha256"],
            "same_seed": contracts[0]["seed"] == contracts[1]["seed"],
            "dataset_snapshot_sha256": snapshot["sha256"], "labels_manifest_sha256": file_sha256(args.labels / "manifest.json"),
        }, indent=2))
        print(json.dumps({"completed": status, "paired_initialization": True}), flush=True)
    finally:
        for arm, process, log in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()


if __name__ == "__main__":
    main()
