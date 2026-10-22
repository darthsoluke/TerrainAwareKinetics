"""CPU review probes for the parent runner, without importing model modules.

The runner's real function bodies are compiled from its AST. Only its source
and plan are read; CUDA, data loading and collectives are replaced at the edges.
Regression failures identify edits owned by the parent runner.
"""

import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts/train_pfnn_frame_actor.py'
PLAN = ROOT / 'configs/pfnn_frame_actor_v1.json'
CATEGORIES = ('idle', 'start', 'stop', 'strafe', 'reverse', 'forward')


@pytest.fixture
def code():
    tree = ast.parse(RUNNER.read_text(encoding='utf-8'), filename=str(RUNNER))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(torch=torch, F=F, math=math, json=json, hashlib=hashlib, np=np, CATEGORIES=CATEGORIES,
                     os=os, Path=Path,
                     platform=SimpleNamespace(python_version=lambda: '3.12.4', python_implementation=lambda: 'CPython'),
                     metadata=SimpleNamespace(distributions=lambda: [
                         SimpleNamespace(metadata={'Name': 'nvidia_cublas_cu12'}, version='12.8.4.1'),
                         SimpleNamespace(metadata={'Name': 'nvidia-nccl-cu12'}, version='2.27.3'),
                         SimpleNamespace(metadata={'Name': 'numpy'}, version='2.1.0')]))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(RUNNER), 'exec'), namespace)
    namespace['_tree'] = tree
    return namespace


@pytest.fixture
def plan():
    return json.loads(PLAN.read_text(encoding='utf-8'))


def test_checked_in_plan_and_zero_time_or_exposure_overrun(code, plan):
    code['validate_plan'](plan)
    assert plan['max_window_exposures'] == 10 * (48232 + 12058)
    for change in (dict(max_p1_wall_seconds=0), dict(max_window_exposures=602901),
                   dict(automatic_resume=True), dict(eligible_for_automatic_export=True)):
        with pytest.raises(ValueError):
            code['validate_plan'](dict(plan, **change))


def test_single_gpu3_plan_preserves_data_and_budget(code, plan):
    single = json.loads((ROOT / 'configs/pfnn_frame_actor_pro6000_gpu3_v1.json').read_text())
    code['validate_plan'](single)
    assert single['world_size'] == 1
    assert single['global_batch'] == single['micro_batch'] == 256
    assert {key: value for key, value in single.items() if key not in ('world_size', 'micro_batch')} == {
        key: value for key, value in plan.items() if key not in ('world_size', 'micro_batch')}
    rows = list(code['batch_indices'](list(range(60290)), 256, 0, 1))
    assert rows[-1][1] == 130
    assert [i for ids, _, _ in rows for i in ids] == list(range(60290))


@pytest.mark.parametrize('change', [
    dict(epochs=1.5, max_window_exposures=90435),
    dict(source_steps=True), dict(global_batch=2.5), dict(micro_batch=1.5),
    dict(actor_lr=float('nan')), dict(environment_lr=-1.),
    dict(gradient_clip=-1.), dict(path_loss_weight=float('inf')),
])
def test_plan_rejects_invalid_loop_counts_and_unsafe_optimizer_controls(code, plan, change):
    with pytest.raises(ValueError):
        code['validate_plan'](dict(plan, **change))


@pytest.mark.parametrize('count,batch,ranks', [(60290, 256, 2), (13, 5, 2), (1, 2, 2)])
def test_batch_shards_cover_all_rows_once_without_padding(code, count, batch, ranks):
    batches = [list(code['batch_indices'](list(range(count)), batch, rank, ranks))
               for rank in range(ranks)]
    visited = [i for rank_batches in batches for ids, _, _ in rank_batches for i in ids]
    assert sorted(visited) == list(range(count))
    assert len(visited) == len(set(visited))
    for slot in range(len(batches[0])):
        assert sum(len(rank[slot][0]) for rank in batches) == batches[0][slot][1]


class TinyActor(torch.nn.Module):
    def __init__(self, _rep=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3, dtype=torch.float64))
        self.terrain_encoder = torch.nn.Module()
        self.terrain_encoder.register_parameter('weight', torch.nn.Parameter(torch.tensor(.2, dtype=torch.float64)))


class TinyDataset:
    def __init__(self, split='train', validation_count=32):
        self.split = split
        self.records = [dict(category=category, source_id=f'{category}_{i}')
                        for category in CATEGORIES for i in range(validation_count)]

    def __len__(self):
        return 48232 if self.split == 'train' else len(self.records)

    def __getitem__(self, index):
        return dict(world=torch.full((33, 366), index / 100 + 1., dtype=torch.float64),
                    command_sequence=dict(command=torch.zeros(26, 4)), scene={})


def tiny_loss(actor, _rep, _observer, worlds, _commands, _field, _first, _steps):
    prediction = worlds[:, 0, 0] * actor.weight + actor.terrain_encoder.weight
    loss = prediction.square().mean()
    return dict.fromkeys(('root_delta', 'body_fk', 'rotation', 'boundary_joint_velocity', 'contact', 'path', 'total'), loss)


def accumulation(code, actor, local_ids, global_n, micro_batch):
    """Execute the actual microbatch and gradient-reduction block on CPU."""
    main = next(node for node in code['_tree'].body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    batches = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                   and isinstance(node.target, ast.Tuple) and isinstance(node.target.elts[0], ast.Name)
                   and node.target.elts[0].id == 'batch_no')
    first = next(i for i, node in enumerate(batches.body) if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                 and node.value.func.attr == 'zero_grad')
    last = next(i for i, node in enumerate(batches.body) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple) and isinstance(node.target.elts[0], ast.Name)
                and node.target.elts[0].id == 'name')
    namespace = dict(code)
    cpu_torch = CpuTorch()
    clock = SimpleNamespace(ticks=0)

    def monotonic():
        clock.ticks += 1
        return float(clock.ticks)

    namespace.update(actor=actor, optimizer=torch.optim.SGD(actor.parameters(), lr=.1), device=torch.device('cpu'),
        local_ids=local_ids, global_n=global_n, plan=dict(micro_batch=micro_batch, source_steps=8),
        dataset=TinyDataset(), BatchedSceneField=lambda scenes, **kwargs: scenes, CompositeSceneField=lambda scene: scene,
        epoch=0, batch_no=0, rep=None, observer=None, source_loss=tiny_loss,
        torch=cpu_torch, time=SimpleNamespace(monotonic=monotonic),
        dist=SimpleNamespace(all_reduce=lambda *args, **kwargs: None, ReduceOp=SimpleNamespace(SUM='sum')))
    exec(compile(ast.Module(body=batches.body[first:last + 1], type_ignores=[]), str(RUNNER), 'exec'), namespace)
    micro_count = math.ceil(len(local_ids) / micro_batch)
    assert cpu_torch.synchronizations == micro_count
    assert all(namespace[key] == micro_count for key in ('loading_seconds', 'field_seconds', 'compute_seconds'))
    return namespace['metric']


@pytest.mark.parametrize('count,micro', [(5, 2), (7, 1), (130, 16), (256, 128), (130, 128)])
def test_uneven_rank_and_microbatch_gradients_equal_global_sample_mean(code, count, micro):
    order = list(range(count))
    reference = TinyActor()
    samples = torch.stack([TinyDataset()[i]['world'] for i in order])
    tiny_loss(reference, None, None, samples, None, None, None, None)['total'].backward()
    local_gradients, local_metrics = [], []
    for rank in range(2):
        actor = TinyActor()
        metric = accumulation(code, actor, order[rank::2], count, micro)
        local_gradients.append([p.grad.clone() for p in actor.parameters()])
        local_metrics.append(metric)
    for i, parameter in enumerate(reference.parameters()):
        torch.testing.assert_close(sum(grad[i] for grad in local_gradients), parameter.grad, atol=1e-12, rtol=1e-12)
    assert sum(float(metric[1]) for metric in local_metrics) == count


def test_empty_rank_contributes_zero_gradients_and_still_reaches_collectives(code):
    # E.g. legal global_batch=60289 leaves a singleton tail on the other rank.
    actor = TinyActor()
    accumulation(code, actor, [], global_n=1, micro_batch=16)
    assert all(parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
               for parameter in actor.parameters())


class PathActor:
    path_loss_weight = .25

    def __init__(self):
        self.path = torch.nn.Parameter(torch.arange(16, dtype=torch.float64).reshape(1, 8, 2) / 10)
        self.histories = []
        self.states, self.commands = [], []

    def initial_state(self, count):
        return torch.zeros(count, 1)

    def root_step(self, history, command, coarse, state):
        self.histories.append(history.clone())
        self.states.append(state.clone())
        self.commands.append(command.clone())
        return dict(path_world_xz=self.path.expand(len(history), -1, -1), state=state + 1)

    def pose_step(self, history, command, fine, root):
        return history[:, -1]


def source_loss_probe(code, world, start, steps=1):
    actor, observed = PathActor(), []

    def observe(history, command, field, horizon):
        observed.append(('coarse', history.clone(), command.clone()))
        return dict(features=history)

    def fine(_sampler, history, path, coarse, field):
        observed.append(('fine', history.clone(), path.detach().clone()))
        return coarse

    code.update(reencode_played_history=lambda history, rep: history,
                path_observation=fine,
                source_one_step_loss=lambda rep, pred, target, history: dict(total=pred.sum() * 0))
    losses = code['source_loss'](actor, None, SimpleNamespace(observe=observe), world,
                                torch.zeros(len(world), 26, 4), None, start, steps)
    return actor, observed, losses


@pytest.mark.parametrize('offset', [0, 18, 25])
def test_path_loss_uses_only_real_horizon_labels(code, offset):
    world = torch.arange(33 * 366, dtype=torch.float64).reshape(1, 33, 366) / 1000
    actor, observed, losses = source_loss_probe(code, world, offset)
    stations = offset + 3 + torch.arange(1, 9) * 3
    known = stations < 33
    expected = F.smooth_l1_loss(actor.path[:, known] / .3, world[:, stations[known]][..., [0, 2]] / .3)
    torch.testing.assert_close(losses['path'], expected)
    losses['path'].backward()
    assert torch.count_nonzero(actor.path.grad[:, ~known]) == 0
    assert torch.count_nonzero(actor.path.grad[:, known]) > 0
    torch.testing.assert_close(observed[1][2], actor.path.detach())


def test_future_target_pose_does_not_choose_sensor_or_predicted_path_coordinates(code):
    original = torch.randn(1, 33, 366, dtype=torch.float64)
    changed = original.clone()
    changed[:, 4:] += 1000
    actor1, observed1, losses1 = source_loss_probe(code, original, 0)
    actor2, observed2, losses2 = source_loss_probe(code, changed, 0)
    for first, second in zip(observed1, observed2):
        assert first[0] == second[0]
        torch.testing.assert_close(first[1], second[1], atol=0, rtol=0)
        torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)
    torch.testing.assert_close(actor1.histories[0], actor2.histories[0], atol=0, rtol=0)
    assert not torch.equal(losses1['path'], losses2['path'])


@pytest.mark.parametrize('start,steps', [(0, 8), (18, 8)])
def test_batched_teacher_encoding_preserves_sample_step_clock_and_recurrent_order(code, start, steps):
    world = torch.arange(2 * 33 * 366, dtype=torch.float64).reshape(2, 33, 366) / 1000
    commands = torch.arange(2 * 26 * 4, dtype=torch.float64).reshape(2, 26, 4)
    actor, encoded_inputs = PathActor(), []

    def encoding(windows, rep):
        # Window-dependent centering exposes sample/time flattening mistakes.
        result = windows.clone()
        result[..., :3] -= windows[:, :1, :3]
        result[..., 12] += windows[:, :1, 12]
        return result

    def record_encoding(windows, rep):
        encoded_inputs.append(windows.clone())
        return encoding(windows, rep)

    code.update(reencode_played_history=record_encoding,
                path_observation=lambda sampler, history, path, coarse, field: coarse,
                source_one_step_loss=lambda rep, pred, target, history: dict(total=pred.sum() * 0))
    code['source_loss'](actor, None, SimpleNamespace(observe=lambda *args: {}),
                        world, commands, None, start, steps)
    assert len(encoded_inputs) == 1
    expected_flat = torch.stack([world[b, offset:offset + 4]
                                 for b in range(2) for offset in range(start, start + steps)])
    torch.testing.assert_close(encoded_inputs[0], expected_flat, atol=0, rtol=0)
    for step, offset in enumerate(range(start, start + steps)):
        torch.testing.assert_close(actor.histories[step], encoding(world[:, offset:offset + 4], None), atol=0, rtol=0)
        torch.testing.assert_close(actor.commands[step], commands[:, offset], atol=0, rtol=0)
        torch.testing.assert_close(actor.states[step], torch.full((2, 1), float(step)), atol=0, rtol=0)


class CpuTorch:
    """No real torch.cuda entry point may be called by the main-loop probes."""
    def __init__(self):
        self.synchronizations = 0
        self.__version__ = '2.8.0+cu128'
        self.version = SimpleNamespace(cuda='12.8')
        self.cuda = SimpleNamespace(set_device=lambda *args: None,
            set_per_process_memory_fraction=lambda *args: None, synchronize=self.synchronize,
            max_memory_allocated=lambda: 0,
            get_device_name=lambda index: 'NVIDIA RTX PRO 6000 Blackwell Server Edition',
            get_device_capability=lambda index: (12, 0), nccl=SimpleNamespace(version=lambda: (2, 27, 3)))
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                        cudnn=SimpleNamespace(allow_tf32=False, version=lambda: 91002))

    def synchronize(self):
        self.synchronizations += 1

    def device(self, *_args):
        return torch.device('cpu')

    def set_num_threads(self, _threads):
        pass

    def manual_seed(self, _seed):
        pass

    def load(self, *_args, **_kwargs):
        return dict(model_state={})

    def __getattr__(self, key):
        return getattr(torch, key)


def main_probe(code, plan, tmp_path, monkeypatch, mode, expire_before_first=True, realtime=None,
               validation_count=32, expire_after_first_batch=False, forbid_initialization=False,
               stats_mutation=None, runtime_change=None, initialization_shift=0., smoke_initialization=None,
               peer_runtime_change=None):
    monkeypatch.chdir(tmp_path)
    for key, value in dict(RANK='0', LOCAL_RANK='0', WORLD_SIZE='2').items():
        monkeypatch.setenv(key, value)
    args = SimpleNamespace(mode=mode, **{key: tmp_path / key for key in
        ('plan', 'parent', 'stats', 'audit', 'output', 'cache', 'pfnn', 'preflight', 'smoke', 'realtime')})
    args.plan.write_text(json.dumps(plan))
    args.audit.write_text(json.dumps(dict(passed=True, splits=dict(train=dict(cache_rows=48232)),
        **{key: plan[key] for key in ('manifest_sha256', 'cache_complete_sha256', 'cache_records_sha256')})))
    args.stats.mkdir()
    (args.stats / 'mean.npy').write_bytes(b'CPU mean fixture')
    (args.stats / 'std.npy').write_bytes(b'CPU std fixture')
    cpu_torch = CpuTorch()
    code['torch'] = cpu_torch
    runtime = code['runtime_identity'](0)

    def digest(path):
        if path == args.parent:
            return plan['parent_sha256']
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # Supply a valid bound runtime proof so timeout probes reach their intended
    # branch instead of passing because the runtime gate rejects the fixture.
    main = next(node for node in code['_tree'].body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    required = next(ast.literal_eval(node.value) for node in ast.walk(main) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'required' for target in node.targets))
    runtime_sources = {}
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# CPU proof fixture\n', encoding='utf-8')
        runtime_sources[relative] = digest(path)
    binding = dict(plan_sha256=digest(args.plan), parent_sha256=digest(args.parent),
                   audit_sha256=digest(args.audit), sources=runtime_sources, runtime=runtime,
                   stats_sha256={p.name: digest(p) for p in args.stats.iterdir()})
    initial = code['model_digest'](TinyActor())
    args.preflight.write_text(json.dumps(dict(passed=True, binding=binding, optimizer_updates=0, initial_sha256=initial)))
    args.smoke.write_text(json.dumps(dict(passed=True, binding=binding, optimizer_updates=2,
                                        initial_sha256=smoke_initialization or initial)))
    args.realtime.write_text(json.dumps(realtime if realtime is not None else
                                       dict(passed=True, source_sha256=runtime_sources)))
    if stats_mutation == 'content':
        (args.stats / 'mean.npy').write_bytes(b'changed mean fixture')
    elif stats_mutation == 'missing':
        (args.stats / 'std.npy').unlink()
    elif stats_mutation == 'rename':
        (args.stats / 'std.npy').rename(args.stats / 'different.npy')
    elif stats_mutation == 'relocate':
        relocated = tmp_path / 'relocated_stats'
        args.stats.rename(relocated)
        args.stats = relocated
    if runtime_change is not None:
        code['runtime_identity'] = lambda local_rank: dict(runtime, **runtime_change)
    clock = SimpleNamespace(elapsed=0.)

    def dataset(*args):
        if forbid_initialization:
            pytest.fail('Invalid realtime proof reached dataset/model initialization')
        if expire_before_first:
            clock.elapsed = plan['max_p1_wall_seconds'] + 1.
        return TinyDataset(split=args[-1], validation_count=validation_count)

    class FakeSampler:
        def __init__(self, *_args, **_kwargs):
            pass

        def set_epoch(self, _epoch):
            pass

        def __iter__(self):
            return iter(range(60290))

        def coverage_receipt(self):
            return dict(expected_primary=48232)

    def all_reduce(value, **kwargs):
        # Emulate an identical second rank for the probes that do real updates.
        # Scalar integer stop flags retain their MAX value.
        if value.dtype.is_floating_point:
            value.mul_(2)

    def loss(*args):
        if expire_after_first_batch and args[0].training:
            clock.elapsed = plan['max_p1_wall_seconds'] + 1.
        return tiny_loss(*args)

    def actor(rep):
        result = TinyActor(rep)
        with torch.no_grad():
            result.weight.add_(initialization_shift)
        return result

    def gather(values, item):
        peer = dict(item, **peer_runtime_change) if isinstance(item, dict) and peer_runtime_change else item
        values[:] = [item, peer]

    code.update(torch=cpu_torch, dist=SimpleNamespace(init_process_group=lambda *args, **kwargs: None,
        barrier=lambda: None, all_gather_object=gather,
        all_reduce=all_reduce, ReduceOp=SimpleNamespace(MAX='max', SUM='sum'), destroy_process_group=lambda: None),
        time=SimpleNamespace(monotonic=lambda: clock.elapsed), sha256=digest,
        build_native_representation=lambda _stats: torch.nn.Identity(), PFNNFrameDataset=dataset,
        FullEpochSampler=FakeSampler, FrameActor=actor, load_environment_encoder=lambda *args, **kwargs: None,
        VolumetricEgoSampler=lambda: None, BatchedSceneField=lambda values, **kwargs: values,
        CompositeSceneField=lambda scene: scene, source_loss=loss)
    code['main'](args)
    return json.loads((args.output / 'complete.json').read_text())


@pytest.mark.parametrize('mode', ['preflight', 'smoke', 'execute'])
def test_timeout_before_first_batch_cannot_produce_successful_completion(code, plan, tmp_path, monkeypatch, mode):
    try:
        report = main_probe(code, plan, tmp_path, monkeypatch, mode)
    except (ValueError, RuntimeError, TimeoutError):
        return
    assert report['passed'] is False, report
    assert report.get('timed_out') is True or report.get('stop_reason') in ('timeout', 'wall_budget')


def test_execute_rejects_unrelated_unbound_realtime_proof(code, plan, tmp_path, monkeypatch):
    try:
        main_probe(code, plan, tmp_path, monkeypatch, 'execute',
                   realtime=dict(passed=True, binding=dict(unrelated_project=True)), forbid_initialization=True)
    except (ValueError, RuntimeError):
        return
    pytest.fail('execute accepted a realtime proof with an unrelated binding')


def test_smoke_has_two_updates_and_separate_validation_accounting(code, plan, tmp_path, monkeypatch):
    report = main_probe(code, plan, tmp_path, monkeypatch, 'smoke', expire_before_first=False)
    assert report['passed'] and report['optimizer_updates'] == 2
    assert report['window_exposures'] == 512
    # Initial and post-smoke evaluation each consume the same 192 validation
    # rows. Their actual counts must stay separate from training exposures.
    initial = json.loads((tmp_path / 'output/validation_initial.json').read_text())
    after = json.loads((tmp_path / 'output/validation_epoch_00.json').read_text())
    assert initial['validation_windows'] + after['validation_windows'] == 384
    assert initial['window_exposures'] == 0 and after['window_exposures'] == 512
    progress = [json.loads(line) for line in (tmp_path / 'output/progress.jsonl').read_text().splitlines()]
    assert progress[-1]['validation_training_exposures'] == 0


def test_incomplete_validation_strata_cannot_silently_become_a_192_row_proof(code, plan, tmp_path, monkeypatch):
    try:
        report = main_probe(code, plan, tmp_path, monkeypatch, 'smoke', expire_before_first=False,
                            validation_count=1)
    except (ValueError, RuntimeError):
        return
    assert report['passed'] is False or report.get('validation_windows') == 192


def test_timeout_after_one_execute_batch_cannot_look_like_completed_full_training(code, plan, tmp_path, monkeypatch):
    try:
        report = main_probe(code, plan, tmp_path, monkeypatch, 'execute', expire_before_first=False,
                            expire_after_first_batch=True)
    except (ValueError, RuntimeError, TimeoutError):
        return
    assert report['passed'] is False, report
    assert report['completed_epochs'] < plan['epochs'], report


def test_runtime_identity_is_complete_json_stable_and_card_index_independent(code):
    code['torch'] = CpuTorch()
    identity = code['runtime_identity'](0)
    assert identity == code['runtime_identity'](7)
    assert identity == json.loads(json.dumps(identity))
    assert identity == dict(python='3.12.4', python_implementation='CPython', torch='2.8.0+cu128',
        cuda='12.8', cudnn=91002, nccl=[2, 27, 3],
        gpu_name='NVIDIA RTX PRO 6000 Blackwell Server Edition', compute_capability=[12, 0],
        nvidia_packages={'nvidia-cublas-cu12': '12.8.4.1', 'nvidia-nccl-cu12': '2.27.3'})


@pytest.mark.parametrize('mode', ['smoke', 'execute'])
@pytest.mark.parametrize('change', ['content', 'missing', 'rename'])
def test_changed_statistics_cannot_reuse_preflight(code, plan, tmp_path, monkeypatch, mode, change):
    with pytest.raises(ValueError, match='Zero-update proof differs'):
        main_probe(code, plan, tmp_path, monkeypatch, mode, stats_mutation=change, forbid_initialization=True)


@pytest.mark.parametrize('mode', ['smoke', 'execute'])
@pytest.mark.parametrize('change', [
    dict(python='3.12.5'), dict(torch='2.9.0+cu128'), dict(cuda='13.0'), dict(cudnn=91100),
    dict(nccl=[2, 28, 3]), dict(gpu_name='NVIDIA A100-SXM4-80GB'), dict(compute_capability=[8, 0]),
    dict(nvidia_packages={'nvidia-cublas-cu12': '12.8.5.1', 'nvidia-nccl-cu12': '2.27.3'}),
])
def test_changed_runtime_cannot_reuse_preflight(code, plan, tmp_path, monkeypatch, mode, change):
    with pytest.raises(ValueError, match='Zero-update proof differs'):
        main_probe(code, plan, tmp_path, monkeypatch, mode, runtime_change=change, forbid_initialization=True)


@pytest.mark.parametrize('mode', ['smoke', 'execute'])
def test_equal_rank_weights_still_must_match_preflight_initialization(code, plan, tmp_path, monkeypatch, mode):
    with pytest.raises(ValueError, match='Model initialization differs from preflight'):
        main_probe(code, plan, tmp_path, monkeypatch, mode, initialization_shift=.01)


def test_execute_requires_smoke_to_share_preflight_initialization(code, plan, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='Model initialization differs from Adam smoke'):
        main_probe(code, plan, tmp_path, monkeypatch, 'execute', smoke_initialization='0' * 64)


def test_same_stats_relocated_are_accepted_and_identity_is_recorded(code, plan, tmp_path, monkeypatch):
    report = main_probe(code, plan, tmp_path, monkeypatch, 'smoke', expire_before_first=False,
                        stats_mutation='relocate')
    preflight = json.loads((tmp_path / 'preflight').read_text())
    contract = json.loads((tmp_path / 'output/contract.json').read_text())
    assert report['passed'] and report['optimizer_updates'] == 2
    assert report['binding'] == preflight['binding'] == contract['binding']
    assert report['initial_sha256'] == preflight['initial_sha256'] == contract['initial_sha256']
    assert set(report['binding']['stats_sha256']) == {'mean.npy', 'std.npy'}


def test_preflight_rejects_mixed_card_runtime(code, plan, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='Two-card runtime identity differs'):
        main_probe(code, plan, tmp_path, monkeypatch, 'preflight',
                   peer_runtime_change=dict(gpu_name='NVIDIA A100-SXM4-80GB'), forbid_initialization=True)


def test_kill_after_first_progress_keeps_that_update_checkpoint(code, plan, tmp_path, monkeypatch):
    write = code['save']

    def terminate_after_progress(path, value):
        write(path, value)
        if path.name == 'state.json':
            raise SystemExit('Simulated external termination between batches')

    code['save'] = terminate_after_progress
    with pytest.raises(SystemExit):
        main_probe(code, plan, tmp_path, monkeypatch, 'execute', expire_before_first=False)
    progress = json.loads((tmp_path / 'output/state.json').read_text())
    saved = torch.load(tmp_path / 'output/latest.pt', weights_only=False)
    assert saved['optimizer_updates'] == progress['optimizer_updates'] == 1
    assert saved['window_exposures'] == progress['window_exposures'] == 256
    assert not (tmp_path / 'output/coverage_00.json').exists()


def test_interrupted_checkpoint_write_retains_previous_complete_file(code, tmp_path, monkeypatch):
    actor = TinyActor()
    optimizer = torch.optim.AdamW(actor.parameters())
    code['save_checkpoint'](tmp_path, actor, optimizer, {}, {}, 1, 256, 0)
    previous = (tmp_path / 'latest.pt').read_bytes()

    def interrupted(_payload, path):
        path.write_bytes(b'interrupted write')
        raise OSError('Simulated disk interruption')

    monkeypatch.setattr(torch, 'save', interrupted)
    with pytest.raises(OSError):
        code['save_checkpoint'](tmp_path, actor, optimizer, {}, {}, 2, 512, 0)
    assert (tmp_path / 'latest.pt').read_bytes() == previous
