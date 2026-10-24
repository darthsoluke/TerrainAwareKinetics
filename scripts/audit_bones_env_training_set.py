"""Audit a built BONES motion-scene training set and smoke-test the controller on it.

Samples windows from the dataset, computes the online observations exactly as training
would (body rays from the last history frame, path-station rays along the source's own
future path), and reports what the observations say: e.g. head clearance in crawling vs
upright groups, how often a ray node starts inside a solid, contact rates and command
speeds. Then it runs one untrained forward pass and a short differentiable rollout of
TerrainMotionTransformer on a batch and times both. Nothing is trained or saved besides
report.json.

    python scripts/audit_bones_env_training_set.py --dataset runs/bones_env_v1 \
        --output runs/bones_env_v1_audit --device cuda --samples 4096
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _timed(torch, device, function):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    value = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, (time.perf_counter() - start) * 1000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-horizon", type=int, default=4)
    args = parser.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Subset

    from motionbricks.controller.closed_loop import (ClosedLoop, Critic, RolloutConfig, default_sole_offsets,
                                                     forward_xz, posed_joints)
    from motionbricks.controller.terrain_motion_transformer import TerrainMotionTransformer
    from motionbricks.controller.tokens import BODY_PARTS, PartRayTable, StationProber
    from motionbricks.data.bones_env_dataset import BonesEnvWindows, boxes_of
    from motionbricks.data.bones_seed import build_representation
    from motionbricks.data.skeleton_ray_sampler import SkeletonRaySampler

    device = torch.device(args.device)
    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines() if line]
    dataset = BonesEnvWindows(args.dataset, rows, args.split)
    rng = np.random.default_rng(0)
    chosen = rng.choice(len(dataset), size=min(args.samples, len(dataset)), replace=False)
    loader = DataLoader(Subset(dataset, chosen.tolist()), batch_size=args.batch, num_workers=4)
    rep_cpu = build_representation()
    rep = build_representation().to(device)
    names = list(rep.skeleton.bone_order_names)
    sampler = SkeletonRaySampler(names, box_capacity=64)
    table, prober = PartRayTable(sampler.layout), StationProber()
    labels = sampler.layout.labels
    head_up = labels.index("Head:up")
    groups = [dataset.rows[dataset.items[i][0]]["group"] for i in chosen]

    stats = {"head_up_clearance_m": {}, "ray_node_inside_fraction": [], "ray_hit_fraction": [],
             "contact_rate": [], "command_speed_mps": [], "observation_ms": []}
    offset = 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        boxes = boxes_of(batch)
        history = batch["history"]
        size = len(history)

        def observe():
            joints = posed_joints(rep, history[:, -1:])[:, 0]
            rays = sampler.observe(joints, forward_xz(history), boxes)
            path = batch["targets"][:, [2, 5, 8, 11, 14, 15, 15, 15]][..., [0, 2]]
            stations = prober(path, history[:, -1, :3], forward_xz(history), boxes)
            return rays, stations

        with torch.no_grad():
            (rays, stations), ms = _timed(torch, device, observe)
        stats["observation_ms"].append(ms / size)
        up = (rays["distance"][:, head_up]).cpu().numpy()
        for g, value in zip(groups[offset:offset + size], up):
            stats["head_up_clearance_m"].setdefault(g, []).append(float(value))
        stats["ray_node_inside_fraction"].append(float(rays["inside"].float().mean()))
        stats["ray_hit_fraction"].append(float(rays["hit"].float().mean()))
        stats["contact_rate"].append(batch["contacts"].float().mean(dim=(0, 1)).cpu().numpy().tolist())
        stats["command_speed_mps"].extend(batch["commands"][..., :2].norm(dim=-1).flatten().cpu().numpy().tolist())
        offset += size
    report = {
        "dataset": str(args.dataset), "split": args.split, "windows": len(dataset), "sampled": int(len(chosen)),
        "head_up_clearance_m": {g: {"p10": round(float(np.percentile(v, 10)), 3),
                                    "p50": round(float(np.median(v)), 3), "n": len(v)}
                                for g, v in sorted(stats["head_up_clearance_m"].items())},
        "ray_node_inside_fraction": float(np.mean(stats["ray_node_inside_fraction"])),
        "ray_hit_fraction": float(np.mean(stats["ray_hit_fraction"])),
        "contact_rate": np.mean(stats["contact_rate"], axis=0).round(4).tolist(),
        "command_speed_mps": {q: round(float(np.percentile(stats["command_speed_mps"], int(q[1:]))), 3)
                              for q in ("p10", "p50", "p90")},
        "observation_ms_per_sample": round(float(np.median(stats["observation_ms"])), 4),
    }

    # Untrained model: one forward pass and a short differentiable rollout on one batch.
    torch.manual_seed(0)
    model = TerrainMotionTransformer(rep, names, table.feature_dim, prober.feature_dim).to(device)
    loop = ClosedLoop(model, rep, names, sampler, table, prober, RolloutConfig(horizon=args.rollout_horizon),
                      sole_offsets=default_sole_offsets(rep_cpu))
    critic = Critic(loop.critic_input_dim()).to(device)
    batch = next(iter(DataLoader(Subset(dataset, chosen[:args.batch].tolist()), batch_size=args.batch)))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    boxes = boxes_of(batch)
    horizon = args.rollout_horizon

    def rollout():
        result = loop.rollout(batch["history"], batch["history_mask"], batch["commands"][:, :horizon], boxes,
                              critic=critic, targets=batch["targets"][:, :horizon])
        result["policy_loss"].backward()
        return result

    result, ms = _timed(torch, device, rollout)
    report["smoke_rollout"] = {"batch": args.batch, "horizon": horizon, "milliseconds": round(ms, 1),
                               "policy_loss": float(result["policy_loss"]), "critic_loss": float(result["critic_loss"]),
                               "terms": {k: float(v) for k, v in result["terms"].items()},
                               "parameters": sum(p.numel() for p in model.parameters())}
    model.eval()
    one = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        def step():
            return loop.step(one["history"], one["history_mask"], one["commands"][:, 0], boxes_of(one))
        for _ in range(5):
            step()
        times = [_timed(torch, device, step)[1] for _ in range(50)]
    report["single_character_step_ms"] = {"p50": round(float(np.median(times)), 3),
                                          "p99": round(float(np.percentile(times, 99)), 3),
                                          "note": "eager, includes FK, rays, stations, model and advance"}
    report["body_parts"] = list(BODY_PARTS)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
