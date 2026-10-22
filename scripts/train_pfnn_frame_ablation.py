"""Bounded A/B/C ablation for the frame actor: one GPU, matched budget.

Arm A: GRU over the flattened motion window, mean-pooled spatial tokens (P1).
Arm B: same GRU, spatial cross-attention residual on the mean (zero at init).
Arm C: causal transformer over per-frame motion tokens, same spatial attention.

Data, losses, command lag, window length and update budget are identical across
arms; only the declared backbone and geometry consumption differ.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.pfnn_3d_scene import BatchedSceneField, CompositeSceneField, path_observation
from motionbricks.data.pfnn_frame_actor import load_environment_encoder, source_one_step_loss
from motionbricks.data.pfnn_frame_actor_v2 import FrameActorV2
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset, FullEpochSampler, sha256
from motionbricks.data.reactive_training_data import CATEGORIES
from motionbricks.data.unified_scene_field import VolumetricEgoSampler

KEYS = ('root_delta', 'body_fk', 'rotation', 'boundary_joint_velocity', 'contact', 'path', 'total')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def model_digest(actor):
    digest = hashlib.sha256()
    for key, value in sorted(actor.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def command_offset(offset, frames):
    """Fixed three-frame command lag; frames=4 reproduces the P1 index exactly."""
    return offset + frames - 4


def source_loss(actor, rep, sampler, worlds, commands, field, start, steps, frames, path_weight):
    """Teacher histories over the declared window; no future pose enters sensing."""
    state = actor.initial_state(len(worlds))
    losses = {}
    length = worlds.shape[1]
    for step, offset in enumerate(range(start, start + steps)):
        if offset + frames >= length:
            raise ValueError('History window escapes the cached clip')
        history = reencode_played_history(worlds[:, offset:offset + frames], rep)
        index = max(0, min(command_offset(offset, frames), commands.shape[1] - 1))
        command = commands[:, index]
        # The motion window is longer than the perception window: the sampler keeps
        # its existing contract of exactly four factual Root frames.
        recent = history[:, -4:, :5]
        coarse = sampler.observe(recent, command[:, :2], field, .8)
        root = actor.root_step(history, command, coarse, state)
        fine = path_observation(sampler, recent, root['path_world_xz'], coarse, field)
        pred = actor.pose_step(history, command, fine, root)
        row = source_one_step_loss(rep, pred, worlds[:, offset + frames], history)
        indices = offset + frames - 1 + torch.arange(1, 9, device=worlds.device) * 3
        known = indices < length
        # A window can sit so late in the clip that every path station is beyond the
        # cache row. An empty reduction is NaN, so those steps simply carry no path term.
        if bool(known.any()):
            row['path'] = F.smooth_l1_loss(root['path_world_xz'][:, known] / .3,
                                          worlds[:, indices[known]][..., [0, 2]] / .3)
            row['total'] = row['total'] + path_weight * row['path']
        else:
            row['path'] = row['total'].new_zeros(())
        for key, value in row.items():
            losses[key] = losses.get(key, 0.) + value / steps
        state = root['state']
    return losses


def evaluate(actor, rep, observer, dataset, selected, plan, device):
    actor.eval()
    rows = torch.zeros(len(CATEGORIES), len(KEYS) + 1, dtype=torch.float64, device=device)
    with torch.no_grad():
        for category_index, category in enumerate(CATEGORIES):
            local = [i for i in selected if dataset.records[i]['category'] == category]
            for begin in range(0, len(local), plan['micro_batch']):
                items = [dataset[i] for i in local[begin:begin + plan['micro_batch']]]
                if not items:
                    continue
                worlds = torch.stack([item['world'] for item in items]).to(device)
                commands = torch.stack([item['command_sequence']['command'] for item in items]).to(device)
                field = BatchedSceneField([CompositeSceneField(item['scene']) for item in items], device=device)
                result = source_loss(actor, rep, observer, worlds, commands, field, 0,
                                     plan['source_steps'], plan['frames'], plan['path_loss_weight'])
                for j, key in enumerate(KEYS):
                    rows[category_index, j] += result[key].double() * len(items)
                rows[category_index, -1] += len(items)
    actor.train()
    if not torch.isfinite(rows).all():
        raise ValueError('Nonfinite validation metric')
    return {category: dict(count=int(row[-1]), **{key: float(row[j] / row[-1])
            for j, key in enumerate(KEYS)}) for category, row in zip(CATEGORIES, rows) if row[-1] > 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', required=True, help='A, B or C; recorded verbatim')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--micro-batch', type=int, default=256)
    parser.add_argument('--source-steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--environment-lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--path-loss-weight', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--validation-per-category', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--pfnn', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    args = parser.parse_args()
    if args.arm not in ('A', 'B', 'C'):
        raise ValueError('arm must be A, B or C')
    backbone = 'transformer' if args.arm == 'C' else 'gru'
    geometry = 'attention' if args.arm in ('B', 'C') else 'mean'
    plan = dict(arm=args.arm, backbone=backbone, geometry=geometry, frames=args.frames,
                updates=args.updates, batch=args.batch, micro_batch=args.micro_batch,
                source_steps=args.source_steps, lr=args.lr, environment_lr=args.environment_lr,
                weight_decay=args.weight_decay, gradient_clip=args.gradient_clip,
                path_loss_weight=args.path_loss_weight, seed=args.seed,
                validation_per_category=args.validation_per_category, device=args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    args.output.mkdir(parents=True, exist_ok=False)
    rep_cpu = build_native_representation(args.stats)
    dataset = PFNNFrameDataset(args.cache, args.pfnn, rep_cpu, 'train')
    validation = PFNNFrameDataset(args.cache, args.pfnn, rep_cpu, 'validation')
    rep = build_native_representation(args.stats).to(device)
    actor = FrameActorV2(rep, backbone=backbone, geometry=geometry, frames=args.frames).to(device)
    parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
    load_environment_encoder(actor, parent['model_state'], freeze=False)
    del parent
    actor.train()
    rng = np.random.default_rng(args.seed + 999)
    selected = []
    for category in CATEGORIES:
        pool = [i for i, row in enumerate(validation.records) if row['category'] == category]
        if len(pool) < args.validation_per_category:
            raise ValueError('Validation category is smaller than the declared quota: ' + category)
        selected.extend(rng.permutation(pool)[:args.validation_per_category].tolist())
    sampler = FullEpochSampler(dataset, seed=args.seed, extras=0)
    observer = VolumetricEgoSampler()
    environment = [p for name, p in actor.named_parameters() if name.startswith('terrain_encoder.')]
    main_params = [p for name, p in actor.named_parameters() if not name.startswith('terrain_encoder.')]
    optimizer = torch.optim.AdamW([dict(params=main_params, lr=args.lr),
                                   dict(params=environment, lr=args.environment_lr)],
                                  weight_decay=args.weight_decay)
    initial = model_digest(actor)
    save(args.output / 'contract.json', dict(plan=plan, initial_sha256=initial,
         parent_sha256=sha256(args.parent), stats_sha256={p.name: sha256(p) for p in args.stats.rglob('*') if p.is_file()},
         trainable_parameters=sum(p.numel() for p in actor.parameters()),
         quality_approved=False, delivery_approved=False))
    before_started = time.monotonic()
    before = evaluate(actor, rep, observer, validation, selected, plan, device)
    evaluation_seconds = time.monotonic() - before_started
    save(args.output / 'validation_initial.json', before)
    order_source = iter(FullEpochSampler(dataset, seed=args.seed, extras=0))
    updates = exposures = 0
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    while updates < args.updates:
        ids = []
        while len(ids) < args.batch:
            try:
                ids.append(next(order_source))
            except StopIteration:
                order_source = iter(FullEpochSampler(dataset, seed=args.seed + updates + 1, extras=0))
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for micro in range(0, len(ids), args.micro_batch):
            chunk = ids[micro:micro + args.micro_batch]
            items = [dataset[i] for i in chunk]
            worlds = torch.stack([item['world'] for item in items]).to(device)
            commands = torch.stack([item['command_sequence']['command'] for item in items]).to(device)
            field = BatchedSceneField([CompositeSceneField(item['scene']) for item in items], device=device)
            # Keep at least one supervised path station inside the 33-frame cache row.
            first = (updates * 7 + micro) % max(1, 33 - args.frames - args.source_steps - 1)
            losses = source_loss(actor, rep, observer, worlds, commands, field, first,
                                 args.source_steps, args.frames, args.path_loss_weight)
            (losses['total'] * len(chunk) / args.batch).backward()
            total += float(losses['total'].detach()) * len(chunk) / args.batch
        grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.gradient_clip,
                                                   error_if_nonfinite=True)
        if not math.isfinite(total) or float(grad_norm) <= 0:
            raise ValueError(f'Invalid training loss or gradient: loss={total} '
                             f'grad_norm={float(grad_norm)} updates={updates}')
        progress = updates / max(1, args.updates)
        for group, initial_lr in zip(optimizer.param_groups, (args.lr, args.environment_lr)):
            group['lr'] = initial_lr * (.01 + .99 * .5 * (1 + math.cos(math.pi * progress)))
        optimizer.step()
        updates += 1
        exposures += len(ids)
        history.append(dict(update=updates, loss=total, grad_norm=float(grad_norm),
                            elapsed_seconds=time.monotonic() - started))
        if updates % 25 == 0 or updates == args.updates:
            save(args.output / 'progress.json', dict(updates=updates, loss=total,
                 grad_norm=float(grad_norm), elapsed_seconds=time.monotonic() - started,
                 peak_memory_mb=torch.cuda.max_memory_allocated(device) / 2 ** 20))
        if updates >= args.updates:
            break
    training_seconds = time.monotonic() - started
    torch.save(dict(model_state=actor.state_dict(), plan=plan, arm=args.arm,
                    initial_sha256=initial), args.output / 'weights.pt')
    after = evaluate(actor, rep, observer, validation, selected, plan, device)
    save(args.output / 'validation_final.json', after)

    def mean_of(table, key):
        weighted = sum(row[key] * row['count'] for row in table.values())
        total_rows = sum(row['count'] for row in table.values())
        return weighted / total_rows

    summary = dict(plan=plan, contract_sha256=model_digest(actor), updates=updates,
                   window_exposures=exposures, training_seconds=training_seconds,
                   evaluation_seconds=evaluation_seconds,
                   peak_memory_mb=torch.cuda.max_memory_allocated(device) / 2 ** 20,
                   trainable_parameters=sum(p.numel() for p in actor.parameters()),
                   loss_first=history[0]['loss'] if history else None,
                   loss_last=history[-1]['loss'] if history else None,
                   initial={key: mean_of(before, key) for key in KEYS},
                   final={key: mean_of(after, key) for key in KEYS})
    save(args.output / 'results.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
