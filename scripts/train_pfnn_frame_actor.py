"""Bounded full-PFNN frame actor training on two explicitly selected GPUs."""

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.pfnn_3d_scene import CompositeSceneField, BatchedSceneField, path_observation
from motionbricks.data.pfnn_frame_actor import FrameActor, load_environment_encoder, source_one_step_loss
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset, FullEpochSampler, sha256
from motionbricks.data.reactive_training_data import CATEGORIES
from motionbricks.data.unified_scene_field import VolumetricEgoSampler


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def model_digest(actor):
    result = hashlib.sha256()
    for key, value in sorted(actor.state_dict().items()):
        result.update(key.encode())
        result.update(value.detach().contiguous().cpu().numpy().tobytes())
    return result.hexdigest()


def save_checkpoint(output, actor, optimizer, plan, binding, updates, exposures, epoch):
    torch.save(dict(model_state=actor.state_dict(), optimizer_state=optimizer.state_dict(),
                    plan=plan, binding=binding, optimizer_updates=updates,
                    window_exposures=exposures, epoch=epoch, quality_approved=False,
                    eligible_for_automatic_export=False), output / 'latest.tmp')
    (output / 'latest.tmp').replace(output / 'latest.pt')


def runtime_identity(local_rank):
    packages = {}
    for package in metadata.distributions():
        name = package.metadata.get('Name', '').lower().replace('_', '-')
        if name.startswith('nvidia-'):
            packages[name] = package.version
    return dict(python=platform.python_version(), python_implementation=platform.python_implementation(),
        torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        nccl=list(torch.cuda.nccl.version()), gpu_name=torch.cuda.get_device_name(local_rank),
        compute_capability=list(torch.cuda.get_device_capability(local_rank)),
        nvidia_packages=dict(sorted(packages.items())))


def batch_indices(order, batch, rank, world_size):
    """Every global row occurs exactly once, including an uneven final batch."""
    for start in range(0, len(order), batch):
        ids = order[start:start + batch]
        yield ids[rank::world_size], len(ids), start


def validate_plan(plan):
    fixed = dict(schema='motionbricks.pfnn_frame_actor_training.v1',
                 fps=30, history_frames=4, primary_rows=48232, extra_rows=12058,
                 automatic_resume=False, eligible_for_automatic_export=False)
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError('PFNN frame actor contract differs')
    if type(plan['world_size']) is not int or plan['world_size'] not in (1, 2):
        raise ValueError('Training supports one or two explicitly selected GPUs')
    for key in ('epochs', 'source_steps', 'global_batch', 'micro_batch', 'validation_per_category', 'seed'):
        if type(plan[key]) is not int or plan[key] < (0 if key == 'seed' else 1):
            raise ValueError('Invalid integer training setting: ' + key)
    for key in ('actor_lr', 'environment_lr', 'gradient_clip', 'path_loss_weight', 'cuda_memory_fraction'):
        if not math.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError('Invalid positive training setting: ' + key)
    if (not 1 <= plan['epochs'] <= 10 or not 1 <= plan['source_steps'] <= 26
            or not 0 < plan['max_p1_wall_seconds'] <= 28800
            or plan['max_window_exposures'] != plan['epochs'] * 60290
            or plan['global_batch'] < 2 or plan['micro_batch'] < 1):
        raise ValueError('Invalid training exposure/time budget')


def source_loss(actor, rep, sampler, worlds, commands, field, start, steps):
    """Teacher histories and recurrent memory; no future pose enters sensing."""
    state = actor.initial_state(len(worlds))
    losses = {}
    times = torch.arange(start, start + steps, device=worlds.device)
    windows = worlds[:, times[:, None] + torch.arange(4, device=worlds.device)]
    histories = reencode_played_history(windows.reshape(-1, 4, 366), rep).reshape(len(worlds), steps, 4, 366)
    for step, offset in enumerate(range(start, start + steps)):
        history = histories[:, step]
        command = commands[:, offset]
        coarse = sampler.observe(history[..., :5], command[:, :2], field, .8)
        root = actor.root_step(history, command, coarse, state)
        fine = path_observation(sampler, history[..., :5], root['path_world_xz'], coarse, field)
        pred = actor.pose_step(history, command, fine, root)
        row = source_one_step_loss(rep, pred, worlds[:, offset + 4], history)
        # Supervise only real in-window future stations. The proposed path above
        # is already queried; these labels never choose observation coordinates.
        indices = offset + 3 + torch.arange(1, 9, device=worlds.device) * 3
        known = indices < worlds.shape[1]
        row['path'] = F.smooth_l1_loss(root['path_world_xz'][:, known] / .3,
                                      worlds[:, indices[known]][..., [0, 2]] / .3)
        row['total'] = row['total'] + actor.path_loss_weight * row['path']
        for key, value in row.items():
            losses[key] = losses.get(key, 0.) + value / steps
        state = root['state']
    return losses


def main(args):
    rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if world_size != plan['world_size']:
        raise ValueError('Worker count differs from the configured GPU count')
    torch.set_num_threads(plan['cpu_threads'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.cuda.set_per_process_memory_fraction(plan['cuda_memory_fraction'], local_rank)
    dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    source_files = [p for root in ('motionbricks', 'scripts', 'out/vendor/kimodo/kimodo')
                    for p in Path(root).rglob('*.py')]
    sources = {str(p).replace('\\', '/'): sha256(p) for p in sorted(source_files)}
    stats_files = sorted(p for p in args.stats.rglob('*') if p.is_file())
    if not stats_files:
        raise ValueError('Statistics directory has no files')
    stats = {p.relative_to(args.stats).as_posix(): sha256(p) for p in stats_files}
    runtime = runtime_identity(local_rank)
    runtimes = [None] * world_size
    dist.all_gather_object(runtimes, runtime)
    if any(value != runtime for value in runtimes):
        raise ValueError('Two-card runtime identity differs')
    binding = dict(plan_sha256=sha256(args.plan), sources=sources,
                   parent_sha256=sha256(args.parent), audit_sha256=sha256(args.audit),
                   stats_sha256=stats, runtime=runtime)
    if binding['parent_sha256'] != plan['parent_sha256']:
        raise ValueError('Parent checkpoint differs')
    audit = json.loads(args.audit.read_text())
    for name in ('manifest_sha256', 'cache_complete_sha256', 'cache_records_sha256'):
        if audit[name] != plan[name]:
            raise ValueError('Full coverage audit identity differs: ' + name)
    if not audit['passed'] or audit['splits']['train']['cache_rows'] != plan['primary_rows']:
        raise ValueError('Full coverage audit failed')
    if args.mode != 'preflight':
        proof = json.loads(args.preflight.read_text())
        if not proof['passed'] or proof['binding'] != binding or proof['optimizer_updates'] != 0:
            raise ValueError('Zero-update proof differs from this execution')
    if args.mode == 'execute':
        smoke = json.loads(args.smoke.read_text())
        realtime = json.loads(args.realtime.read_text())
        if (not smoke['passed'] or smoke['binding'] != binding or smoke['optimizer_updates'] != 2
                or not realtime['passed']):
            raise ValueError('Two-card Adam or local frame budget proof failed')
        runtime_sources = realtime.get('source_sha256', {})
        required = ('motionbricks/data/pfnn_frame_actor.py', 'motionbricks/data/pfnn_3d_scene.py',
                    'motionbricks/data/unified_scene_field.py', 'motionbricks/data/unified_terrain.py',
                    'motionbricks/data/joint_sdf_observation.py')
        if (not all(key in runtime_sources for key in required)
                or any(sources.get(key) != value for key, value in runtime_sources.items())):
            raise ValueError('Local runtime proof is not bound to this source snapshot')
    cpu_rep = build_native_representation(args.stats)
    dataset = PFNNFrameDataset(args.cache, args.pfnn, cpu_rep, 'train')
    validation = PFNNFrameDataset(args.cache, args.pfnn, cpu_rep, 'validation')
    if len(dataset) != plan['primary_rows']:
        raise ValueError('Full training split differs')
    sampler = FullEpochSampler(dataset, seed=plan['seed'], extras=plan['extra_rows'])
    rep = build_native_representation(args.stats).to(device)
    actor = FrameActor(rep).to(device)
    parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
    load_environment_encoder(actor, parent['model_state'], freeze=False)
    del parent
    actor.path_loss_weight = plan['path_loss_weight']
    actor.train()
    initial = model_digest(actor)
    hashes = [None] * world_size
    dist.all_gather_object(hashes, initial)
    if len(set(hashes)) != 1:
        raise ValueError('Two-card initialization differs')
    if args.mode != 'preflight' and proof.get('initial_sha256') != initial:
        raise ValueError('Model initialization differs from preflight')
    if args.mode == 'execute' and smoke.get('initial_sha256') != initial:
        raise ValueError('Model initialization differs from Adam smoke')
    env_params = list(actor.terrain_encoder.parameters())
    main_params = [p for name, p in actor.named_parameters() if not name.startswith('terrain_encoder.')]
    optimizer = torch.optim.AdamW([dict(params=main_params, lr=plan['actor_lr']),
                                  dict(params=env_params, lr=plan['environment_lr'])],
                                 weight_decay=plan['weight_decay'])
    observer = VolumetricEgoSampler()
    updates = exposures = supervised = 0
    selected = []
    rng = np.random.default_rng(plan['seed'] + 999)
    for category in CATEGORIES:
        pool = [i for i, row in enumerate(validation.records) if row['category'] == category]
        if len(pool) < plan['validation_per_category']:
            raise ValueError('Validation category is smaller than the declared fixed quota: ' + category)
        selected.extend(rng.permutation(pool)[:plan['validation_per_category']].tolist())
    if rank == 0:
        save(args.output / 'validation_selection.json', [validation.records[i] for i in selected])

    @torch.no_grad()
    def evaluate(tag):
        actor.eval()
        keys = ('root_delta', 'body_fk', 'rotation', 'boundary_joint_velocity', 'contact', 'path', 'total')
        rows = torch.zeros(len(CATEGORIES), len(keys) + 1, dtype=torch.float64, device=device)
        for category_index, category in enumerate(CATEGORIES):
            local = [i for i in selected if validation.records[i]['category'] == category][rank::world_size]
            for begin in range(0, len(local), plan['micro_batch']):
                items = [validation[i] for i in local[begin:begin + plan['micro_batch']]]
                worlds = torch.stack([item['world'] for item in items]).to(device)
                commands = torch.stack([item['command_sequence']['command'] for item in items]).to(device)
                field = BatchedSceneField([CompositeSceneField(item['scene']) for item in items], device=device)
                result = source_loss(actor, rep, observer, worlds, commands, field, 0, plan['source_steps'])
                for j, key in enumerate(keys):
                    rows[category_index, j] += result[key].double() * len(items)
                rows[category_index, -1] += len(items)
        dist.all_reduce(rows)
        if not torch.isfinite(rows).all():
            raise ValueError('Nonfinite validation metric')
        if rank == 0:
            save(args.output / (tag + '.json'), dict(tag=tag, optimizer_updates=updates,
                window_exposures=exposures, validation_windows=len(selected), rows={
                    category: dict(count=int(row[-1]), **{key: float(row[j] / row[-1])
                        for j, key in enumerate(keys)}) for category, row in zip(CATEGORIES, rows) if row[-1] > 0}))
        actor.train()

    maximum = 2 if args.mode in ('preflight', 'smoke') else math.inf
    epochs = 1 if args.mode != 'execute' else plan['epochs']
    if rank == 0:
        save(args.output / 'contract.json', dict(binding=binding, plan=plan, mode=args.mode,
             initialization='only adaptation.encoder; new recurrent single-frame actor',
             initial_sha256=initial,
             trainable_parameters=sum(p.numel() for p in actor.parameters()),
             quality_approved=False, delivery_approved=False, eligible_for_automatic_export=False))
    evaluate('validation_initial')
    finished = False
    gradient_groups = {}
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        order = list(sampler)
        if len(order) != 60290:
            raise ValueError('Epoch exposure count differs')
        epoch_rows = 0
        for batch_no, (local_ids, global_n, begin) in enumerate(batch_indices(
                order, plan['global_batch'], rank, world_size)):
            stop = torch.tensor(int(time.monotonic() - started >= plan['max_p1_wall_seconds']), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if bool(stop):
                finished = True
                break
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            metric = torch.zeros(2, device=device, dtype=torch.float64)
            loading_seconds = field_seconds = compute_seconds = 0.
            for micro in range(0, len(local_ids), plan['micro_batch']):
                tick_load = time.monotonic()
                ids = local_ids[micro:micro + plan['micro_batch']]
                items = [dataset[i] for i in ids]
                worlds = torch.stack([item['world'] for item in items]).to(device)
                commands = torch.stack([item['command_sequence']['command'] for item in items]).to(device)
                loading_seconds += time.monotonic() - tick_load
                tick_field = time.monotonic()
                field = BatchedSceneField([CompositeSceneField(item['scene']) for item in items], device=device)
                field_seconds += time.monotonic() - tick_field
                tick_compute = time.monotonic()
                first = (epoch * 7 + batch_no) % (27 - plan['source_steps'])
                losses = source_loss(actor, rep, observer, worlds, commands, field, first, plan['source_steps'])
                (losses['total'] * len(ids) / global_n).backward()
                torch.cuda.synchronize()
                compute_seconds += time.monotonic() - tick_compute
                metric[0] += losses['total'].detach().double() * len(ids)
                metric[1] += len(ids)
            gradient_groups = {}
            for name, parameter in actor.named_parameters():
                if parameter.grad is None:
                    if local_ids:
                        raise ValueError('Missing trained parameter gradient: ' + name)
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                group = name.split('.')[0]
                gradient_groups[group] = gradient_groups.get(group, 0.) + parameter.grad.detach().square().sum()
            if args.mode == 'preflight' and any(float(gradient_groups[key]) <= 0 for key in
                    ('terrain_encoder', 'root_head', 'path_head', 'rotation_head', 'contact_head')):
                raise ValueError('A required learned branch has zero gradient')
            grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), plan['gradient_clip'], error_if_nonfinite=True)
            if not torch.isfinite(metric).all() or float(grad_norm) <= 0:
                raise ValueError('Invalid training gradient/loss')
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            if int(metric[1]) != global_n:
                raise ValueError('Distributed partial-batch exposure differs')
            if args.mode != 'preflight':
                progress = exposures / plan['max_window_exposures']
                for group, initial_lr in zip(optimizer.param_groups, (plan['actor_lr'], plan['environment_lr'])):
                    group['lr'] = initial_lr * (.01 + .99 * .5 * (1 + math.cos(math.pi * progress)))
                optimizer.step()
                updates += 1
                exposures += global_n
                supervised += global_n * plan['source_steps']
            torch.cuda.synchronize()
            epoch_rows += global_n
            if rank == 0:
                # Publish an update only after its weights survive process termination.
                if args.mode != 'preflight':
                    save_checkpoint(args.output, actor, optimizer, plan, binding,
                                    updates, exposures, epoch)
                state = dict(mode=args.mode, epoch=epoch, epoch_rows=epoch_rows, batch=batch_no,
                    optimizer_updates=updates, window_exposures=exposures, supervised_next_frames=supervised,
                    validation_training_exposures=0, test_exposures=0, loss=float(metric[0] / metric[1]),
                    gradient_norm=float(grad_norm), batch_seconds=time.monotonic() - tick,
                    loading_seconds=loading_seconds, field_seconds=field_seconds, compute_seconds=compute_seconds,
                    elapsed_seconds=time.monotonic() - started, rank0_pid=os.getpid(),
                    allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                    quality_approved=False, delivery_approved=False)
                save(args.output / 'state.json', state)
                with (args.output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(state) + '\n')
                print(json.dumps(state), flush=True)
            if batch_no + 1 >= maximum:
                finished = True
                break
        if rank == 0:
            receipt = sampler.coverage_receipt()
            receipt.update(executed_epoch_rows=epoch_rows, complete=epoch_rows == len(order))
            save(args.output / f'coverage_{epoch:02d}.json', receipt)
            if args.mode != 'preflight' and updates == 0:
                save_checkpoint(args.output, actor, optimizer, plan, binding,
                                updates, exposures, epoch)
        if args.mode != 'preflight':
            evaluate(f'validation_epoch_{epoch:02d}')
        if finished:
            break
    final = model_digest(actor)
    dist.all_gather_object(hashes, final)
    if len(set(hashes)) != 1 or (args.mode == 'preflight' and final != initial):
        raise ValueError('Rank weights or zero-update integrity failed')
    if args.mode == 'preflight' and epoch_rows != 2 * plan['global_batch']:
        raise ValueError('Preflight did not complete both full batches')
    if args.mode == 'smoke' and updates != 2:
        raise ValueError('Adam smoke did not complete both updates')
    if rank == 0:
        completed_budget = args.mode != 'execute' or exposures == plan['max_window_exposures']
        save(args.output / 'complete.json', dict(passed=completed_budget, binding=binding, mode=args.mode,
            optimizer_updates=updates, window_exposures=exposures, supervised_next_frames=supervised,
            epoch=epoch, completed_epochs=epoch + int(epoch_rows == len(order)),
            stop_reason='epoch_limit' if completed_budget else 'wall_budget',
            elapsed_seconds=time.monotonic() - started, initial_sha256=initial, model_sha256=final,
            gradient_group_squared_norms={key: float(value) for key, value in gradient_groups.items()},
            quality_approved=False, delivery_approved=False, eligible_for_automatic_export=False))
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'execute'), default='preflight')
    for name in ('plan', 'parent', 'cache', 'pfnn', 'stats', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('preflight', 'smoke', 'realtime'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            save(args.output / ('failure_rank' + os.environ.get('RANK', '0') + '.json'),
                 dict(error=repr(error), mode=args.mode, pid=os.getpid()))
        raise
