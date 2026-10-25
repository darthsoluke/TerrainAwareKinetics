"""Bounded training of TerrainMotionTransformer on the BONES motion-scene set (step 7).

Phases, in order, each with a fixed number of optimiser updates:

    teacher      one-step supervision on real history (+ path)
    loop:H       differentiable closed loop of H frames (e.g. loop:4, loop:8, loop:16)

    torchrun --nproc_per_node 4 scripts/train_terrain_controller.py \
        --dataset runs/bones_env_v1 --output runs/tmt_pilot_v1 \
        --phases teacher:1500 loop:4:500 loop:8:500 loop:16:500 --batch 256 --max-hours 3

The run directory gets config.json, progress.jsonl (one line per logged update, with the
exposure ledger), eval.jsonl (validation diagnostics), and checkpoints saved atomically
(latest.pt and one per phase end). A run never resumes silently: --resume must point at
a checkpoint, and the output directory must not already hold a different config.
Gradients are averaged across ranks by hand (the model is called through plan/act, so a
DistributedDataParallel wrapper would not see the forward pass).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_phases(values):
    phases = []
    for value in values:
        parts = value.split(":")
        if parts[0] == "teacher" and len(parts) == 2:
            phases.append(("teacher", 0, int(parts[1])))
        elif parts[0] == "loop" and len(parts) == 3:
            phases.append(("loop", int(parts[1]), int(parts[2])))
        else:
            raise ValueError(f"Bad phase {value!r}; use teacher:N or loop:H:N")
    return phases


class RolloutDefaults:
    penetration = 100.0


def phase_position(update, phases):
    """(phase index, updates done in it) after `update` total updates of this phase list.

    Resume derives its position from the update counter, not from the saved phase index,
    so a run resumed with a longer phase (or extra phases) continues where it stopped.
    """
    done = 0
    for index, (_, _, updates) in enumerate(phases):
        if update < done + updates:
            return index, update - done
        done += updates
    return len(phases), 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phases", nargs="+", required=True)
    parser.add_argument("--batch", type=int, default=256, help="Per-rank batch size")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--teacher-mix", type=float, default=0.0,
                        help="Weight of the one-step teacher loss added to every closed-loop update (0 = off)")
    parser.add_argument("--penetration-weight", type=float, default=RolloutDefaults.penetration,
                        help="Closed-loop penetration cost weight (cost is weight * depth^2, depth in m)")
    parser.add_argument("--max-hours", type=float, required=True, help="Hard wall-clock budget")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    phases = parse_phases(args.phases)

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    from motionbricks.controller.closed_loop import ClosedLoop, Critic, RolloutConfig, default_sole_offsets
    from motionbricks.controller.terrain_motion_transformer import TerrainMotionTransformer
    from motionbricks.controller.tokens import PartRayTable, StationProber
    from motionbricks.controller.training import closed_loop_step, evaluate, teacher_step
    from motionbricks.data.bones_env_dataset import BonesEnvWindows
    from motionbricks.data.bones_seed import build_representation
    from motionbricks.data.skeleton_ray_sampler import SkeletonRaySampler

    distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)          # identical initialisation on every rank (also broadcast below)
    started = time.time()

    config = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "phases": phases, "world_size": world, "global_batch": args.batch * world}
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        existing = args.output / "config.json"
        if existing.exists() and json.loads(existing.read_text())["phases"] != json.loads(json.dumps(phases)):
            raise SystemExit("Output directory holds a different run; use a new directory")
        existing.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines() if line]
    train = BonesEnvWindows(args.dataset, rows, "train")
    validation = BonesEnvWindows(args.dataset, rows, "validation")
    rep_cpu = build_representation()
    rep = build_representation().to(device)
    names = list(rep_cpu.skeleton.bone_order_names)
    sampler = SkeletonRaySampler(names, box_capacity=64)
    table, prober = PartRayTable(sampler.layout), StationProber()
    model = TerrainMotionTransformer(rep_cpu, names, table.feature_dim, prober.feature_dim).to(device)
    critic = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_updates = sum(n for _, _, n in phases)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(step, total_updates) / total_updates)))
    loops = {}

    def loop_for(horizon):
        if horizon not in loops:
            loops[horizon] = ClosedLoop(model, rep, names, sampler, table, prober,
                                        RolloutConfig(horizon=max(1, horizon), penetration=args.penetration_weight),
                                        sole_offsets=default_sole_offsets(rep_cpu))
        return loops[horizon]

    loop_for(16)
    critic = Critic(loops[16].critic_input_dim()).to(device)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    state = {"update": 0, "exposures": 0, "phase_index": 0, "phase_update": 0}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        critic.load_state_dict(checkpoint["critic"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        schedule.load_state_dict(checkpoint["schedule"])
        state = dict(checkpoint["state"])
        state["phase_index"], state["phase_update"] = phase_position(state["update"], phases)
        if rank == 0:
            print(json.dumps({"resumed_at_update": state["update"], "phase_index": state["phase_index"],
                              "phase_update": state["phase_update"]}), flush=True)

    def synchronize_parameters():
        """Copy rank 0's model and critic weights to every rank."""
        if distributed:
            for module in (model, critic):
                for tensor in list(module.parameters()) + list(module.buffers()):
                    dist.broadcast(tensor.data, src=0)

    def parameters_agree():
        """Largest difference between ranks of a per-tensor checksum (0 when synchronized)."""
        if not distributed:
            return 0.0
        sums = torch.stack([t.detach().double().sum() for m in (model, critic) for t in m.parameters()])
        low, high = sums.clone(), sums.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        return float((high - low).abs().max())

    synchronize_parameters()
    torch.manual_seed(args.seed + 1000 * rank)   # per-rank randomness from here on (data order, noise)
    if parameters_agree() != 0.0:
        raise SystemExit("Model parameters differ between ranks after broadcast")

    def average_gradients(parameters):
        if not distributed:
            return
        for p in parameters:
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= world

    def save(name):
        if rank != 0:
            return
        payload = {"model": model.state_dict(), "critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
                   "critic_optimizer": critic_optimizer.state_dict(), "schedule": schedule.state_dict(),
                   "state": dict(state), "config": config}
        temporary = args.output / f".{name}.tmp"
        torch.save(payload, temporary)
        temporary.replace(args.output / name)

    def log(path, record):
        if rank == 0:
            with (args.output / path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")

    eval_loader = DataLoader(validation, batch_size=args.eval_batch, shuffle=True,
                             generator=torch.Generator().manual_seed(1), num_workers=2)
    eval_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in next(iter(eval_loader)).items()}

    def run_eval():
        model.eval()
        record = {"update": state["update"], "exposures": state["exposures"],
                  "rank_parameter_disagreement": parameters_agree()}
        if rank == 0:
            record.update(evaluate(loop_for(16), eval_batch, horizon=16))
            log("eval.jsonl", record)
        model.train()
        return record

    if state["update"] == 0:
        run_eval()
    epoch = 0
    budget = started + args.max_hours * 3600
    stopped = "complete"
    for phase_index in range(state["phase_index"], len(phases)):
        kind, horizon, updates = phases[phase_index]
        loop = loop_for(horizon if kind == "loop" else 16)
        while state["phase_update"] < updates:
            sampler_ = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=args.seed + epoch)
            loader = DataLoader(train, batch_size=args.batch, sampler=sampler_, num_workers=args.workers,
                                drop_last=True, persistent_workers=False, pin_memory=True)
            for raw in loader:
                if state["phase_update"] >= updates:
                    break
                over = torch.tensor(float(time.time() > budget), device=device)
                if distributed:
                    dist.all_reduce(over, op=dist.ReduceOp.MAX)     # every rank stops at the same update
                if over.item():
                    stopped = "max_hours"
                    break
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in raw.items()}
                optimizer.zero_grad(set_to_none=True)
                if kind == "teacher":
                    loss, terms = teacher_step(loop, batch)
                    loss.backward()
                else:
                    loss, critic_loss, terms = closed_loop_step(loop, batch, critic, horizon)
                    if args.teacher_mix > 0:
                        teacher_loss, teacher_terms = teacher_step(loop_for(16), batch)
                        loss = loss + args.teacher_mix * teacher_loss
                        terms.update({f"teacher_{k}": v for k, v in teacher_terms.items()})
                    loss.backward()
                    critic_optimizer.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    average_gradients(critic.parameters())
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    critic_optimizer.step()
                average_gradients(model.parameters())
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                if not np.isfinite(norm):
                    stopped = "non_finite_gradient"
                    break
                optimizer.step()
                schedule.step()
                state["update"] += 1
                state["phase_update"] += 1
                state["exposures"] += args.batch * world * (max(1, horizon) if kind == "loop" else 1)
                if state["update"] % args.log_every == 0:
                    log("progress.jsonl", {"update": state["update"], "phase": f"{kind}:{horizon}",
                                           "exposures": state["exposures"], "lr": schedule.get_last_lr()[0],
                                           "grad_norm": norm, "elapsed_s": round(time.time() - started, 1), **terms})
                if state["update"] % args.eval_every == 0:
                    run_eval()
                    save("latest.pt")
            if stopped != "complete":
                break
            epoch += 1
        if stopped != "complete":
            break
        save(f"phase{phase_index}_{kind}{horizon or ''}.pt")
        state["phase_index"], state["phase_update"] = phase_index + 1, 0
    run_eval()
    save("latest.pt")
    log("progress.jsonl", {"final": True, "stopped": stopped, **state, "elapsed_s": round(time.time() - started, 1)})
    if distributed:
        dist.destroy_process_group()
    return 0 if stopped == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
