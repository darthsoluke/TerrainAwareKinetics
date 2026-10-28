"""Evaluate TerrainMotionTransformer checkpoints with and without runtime foot locking.

For every checkpoint and every mode (raw model output; foot locking on the displayed
frame only; foot locking fed back into the model's history) this runs the closed loop on
the trainer's fixed validation batch (validation split, batch 256, shuffle seed 1, first
batch) and reports training.motion_metrics overall and by group, plus the lock activity
and IK residual. An optional longer horizon uses the same split with longer windows.
Nothing is trained; checkpoints are only read.

    python scripts/evaluate_foot_lock.py --dataset runs/bones_env_v1 --output runs/foot_lock_eval \\
        --checkpoint v2=runs/tmt_pilot_v2/latest.pt --checkpoint v5=runs/tmt_pilot_v5/latest.pt
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLAT_KEYS = ("one_step_body_cm", "persistence_body_cm", "closed_loop_root_drift_cm_p50", "closed_loop_root_drift_cm_p90",
             "moving_windows", "closed_loop_progress_ratio_p50", "closed_loop_progress_ratio_p10",
             "closed_loop_joint_penetration_over_2cm_frames", "source_joint_penetration_over_2cm_frames",
             "closed_loop_pelvis_height_error_cm", "closed_loop_planted_sole_speed_mps_p50",
             "source_planted_sole_speed_mps_p50", "closed_loop_grounded_sole_speed_mps_p50",
             "source_grounded_sole_speed_mps_p50", "closed_loop_grounded_sole_fraction",
             "source_grounded_sole_fraction", "closed_loop_joint_accel_mps2_mean", "source_joint_accel_mps2_mean",
             "lock_active_fraction", "lock_reach_residual_m_p95", "lock_reach_residual_m_max")


def fixed_batch(dataset, rows, horizon, size, device):
    import torch
    from torch.utils.data import DataLoader
    from motionbricks.data.bones_env_dataset import BonesEnvWindows
    windows = BonesEnvWindows(dataset, rows, "validation", horizon=horizon)
    loader = DataLoader(windows, batch_size=size, shuffle=True, generator=torch.Generator().manual_seed(1), num_workers=2)
    raw = next(iter(loader))
    digest = hashlib.sha256()
    for key in ("history", "targets", "commands", "box_centers"):
        digest.update(raw[key].numpy().tobytes())
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw.items()}
    return batch, digest.hexdigest(), len(windows)


def write_outputs(output, report):
    """Write report.json, table.tsv and by_group.tsv; called after every run so a later
    failure keeps everything finished before it (undefined statistics are written as null / empty)."""
    import math
    output.mkdir(parents=True, exist_ok=True)

    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    def cell(value):
        return "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else str(value)
    (output / "report.json").write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    with (output / "table.tsv").open("w", encoding="utf-8") as stream:
        stream.write("\t".join(("checkpoint", "horizon", "mode") + FLAT_KEYS) + "\n")
        for run in report["runs"]:
            m = run["metrics"]
            stream.write("\t".join([run["checkpoint"], str(run["horizon"]), run["mode"]]
                                   + [cell(m.get(k)) for k in FLAT_KEYS]) + "\n")
    group_keys = None
    with (output / "by_group.tsv").open("w", encoding="utf-8") as stream:
        for run in report["runs"]:
            for group, values in run["metrics"].get("by_group", {}).items():
                if group_keys is None:
                    group_keys = list(values)
                    stream.write("\t".join(["checkpoint", "horizon", "mode", "group"] + group_keys) + "\n")
                stream.write("\t".join([run["checkpoint"], str(run["horizon"]), run["mode"], group]
                                       + [cell(values[k]) for k in group_keys]) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH; repeat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--long-horizon", type=int, default=64, help="0 disables the long-horizon pass")
    parser.add_argument("--modes", nargs="+", default=["raw", "display", "feedback"])
    parser.add_argument("--lock-on", type=float, default=0.6)
    parser.add_argument("--lock-off", type=float, default=0.4)
    parser.add_argument("--max-onset-speed", type=float, default=0.6)
    parser.add_argument("--max-slip", type=float, default=0.2)
    parser.add_argument("--release-frames", type=int, default=4)
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from motionbricks.controller.closed_loop import ClosedLoop, RolloutConfig, default_sole_offsets
    from motionbricks.controller.foot_lock import FootLock, FootLockConfig, rollout_frames
    from motionbricks.controller.terrain_motion_transformer import TerrainMotionTransformer
    from motionbricks.controller.tokens import PartRayTable, StationProber
    from motionbricks.controller.training import evaluate, motion_metrics
    from motionbricks.data.bones_seed import build_representation
    from motionbricks.data.skeleton_ray_sampler import SkeletonRaySampler

    device = torch.device(args.device)
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines() if line]
    rep_cpu = build_representation()
    rep = build_representation().to(device)
    names = list(rep_cpu.skeleton.bone_order_names)
    sampler = SkeletonRaySampler(names, box_capacity=64)
    table, prober = PartRayTable(sampler.layout), StationProber()
    lock = FootLock(rep, FootLockConfig(on=args.lock_on, off=args.lock_off, max_onset_speed_mps=args.max_onset_speed,
                                        max_slip_m=args.max_slip, release_frames=args.release_frames))

    horizons = [args.horizon] + ([args.long_horizon] if args.long_horizon else [])
    batches = {}
    for horizon in horizons:
        batches[horizon] = fixed_batch(args.dataset, rows, horizon, args.eval_batch, device)

    # FK used by the lock must reproduce the representation's posed joints.
    batch16 = batches[args.horizon][0]
    frames = batch16["history"][:, -1:]
    decoded = rep.inverse(frames, False, return_all=False, joint_positions_from="global_rot_data")
    joints, _ = lock.fk(decoded["local_joint_rots"][:, 0], frames[:, 0, :3])
    fk_error = float((joints - decoded["posed_joints"][:, 0]).norm(dim=-1).max())
    if fk_error > 1e-3:
        raise SystemExit(f"Foot-lock FK disagrees with the representation by {fk_error:.6f} m")

    report = {"dataset": str(args.dataset), "fk_max_error_m": fk_error, "lock_config": vars(lock.config),
              "batches": {str(h): {"sha256": batches[h][1], "split_windows": batches[h][2]} for h in horizons},
              "runs": []}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        checkpoint = torch.load(path, map_location=device)
        model = TerrainMotionTransformer(rep_cpu, names, table.feature_dim, prober.feature_dim).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        file_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for horizon in horizons:
            batch = batches[horizon][0]
            loop = ClosedLoop(model, rep, names, sampler, table, prober, RolloutConfig(horizon=horizon),
                              sole_offsets=default_sole_offsets(rep_cpu))
            base = evaluate(loop, batch, horizon=horizon)
            for mode in args.modes:
                started = time.time()
                frames, info = rollout_frames(loop, batch, horizon, lock=None if mode == "raw" else lock, mode=mode)
                metrics = motion_metrics(loop, batch, frames, horizon)
                metrics.update(one_step_body_cm=base["one_step_body_cm"], persistence_body_cm=base["persistence_body_cm"])
                if mode == "raw":
                    reference = base["closed_loop_root_drift_cm_p50"]
                    metrics["raw_matches_trainer_rollout_cm"] = abs(metrics["closed_loop_root_drift_cm_p50"] - reference)
                if info:
                    residual = info["reach_residual_m"][info["active"]]
                    metrics.update(lock_active_fraction=float(info["active"].float().mean()),
                                   lock_reach_residual_m_p95=float(residual.quantile(0.95)) if residual.numel() else 0.0,
                                   lock_reach_residual_m_max=float(residual.max()) if residual.numel() else 0.0)
                metrics["seconds"] = round(time.time() - started, 2)
                report["runs"].append({"checkpoint": name, "path": path, "sha256": file_hash,
                                       "update": checkpoint.get("state", {}).get("update"),
                                       "horizon": horizon, "mode": mode, "metrics": metrics})
                write_outputs(args.output, report)
                print(json.dumps({"checkpoint": name, "horizon": horizon, "mode": mode,
                                  "drift_p50": metrics["closed_loop_root_drift_cm_p50"],
                                  "progress": metrics["closed_loop_progress_ratio_p50"],
                                  "grounded_speed": metrics["closed_loop_grounded_sole_speed_mps_p50"],
                                  "planted_speed": metrics["closed_loop_planted_sole_speed_mps_p50"]}), flush=True)

    # Runtime cost of the lock for one character (the controller's real use).
    one = {k: (v[:1] if torch.is_tensor(v) else v[:1]) for k, v in batches[args.horizon][0].items()}
    pred = {"local_rot6d": torch.zeros(1, 30, 6, device=device), "root_next": one["history"][:, -1, :3],
            "contact_logits": torch.full((1, 4), 3.0, device=device)}
    pred["local_rot6d"][..., 0] = 1
    pred["local_rot6d"][..., 4] = 1
    state = lock.initial_state(1, device)
    times = []
    for i in range(120):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _, state, _ = lock.apply(pred, state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if i >= 20:
            times.append((time.perf_counter() - start) * 1000)
    report["single_character_lock_ms"] = {"p50": float(np.median(times)), "p99": float(np.percentile(times, 99))}

    write_outputs(args.output, report)
    print(json.dumps({"fk_max_error_m": fk_error, "single_character_lock_ms": report["single_character_lock_ms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
