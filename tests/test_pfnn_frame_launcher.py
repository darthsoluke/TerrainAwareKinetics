"""Launcher contracts with every hardware/process call mocked."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import launch_pfnn_frame_actor as launcher


PRO = 'NVIDIA RTX PRO 6000 Blackwell Server Edition'
A100 = 'NVIDIA A100-SXM4-80GB'


def gpu(index, name=PRO, *, memory=0, usage=0, display='Disabled'):
    return f'{index}, GPU-{index}, {name}, {memory}, {usage}, {display}'


@pytest.fixture
def processes(monkeypatch):
    query = Mock(side_effect=AssertionError('Unexpected hardware query'))
    spawn = Mock(side_effect=AssertionError('Unexpected process launch'))
    monkeypatch.setattr(launcher.subprocess, 'check_output', query)
    monkeypatch.setattr(launcher.subprocess, 'Popen', spawn)
    monkeypatch.setattr(launcher.time, 'time', lambda: 12345.)
    return query, spawn


def snapshots(processes, devices, apps=''):
    query, spawn = processes
    query.side_effect = ['\n'.join(devices) + '\n', apps]
    spawn.side_effect = None
    spawn.return_value = SimpleNamespace(pid=9876)


def arguments(tmp_path, family=None, mode='preflight'):
    argv = ['--output', str(tmp_path / 'run'), '--mode', mode,
            '--plan', 'configs/pfnn_frame_actor_v1.json']
    if family is not None:
        argv += ['--gpu-family', family]
    return launcher.parse_args(argv)


def receipt_for(args):
    return args.output.resolve().with_suffix('.launch.json')


def mixed_devices():
    return [gpu(0, display='Enabled'), gpu(1, A100), gpu(2), gpu(3),
            gpu(4, 'NVIDIA RTX 6000 Ada Generation'), gpu(5, A100), gpu(6)]


@pytest.mark.parametrize('family,expected', [('RTX PRO 6000', [2, 3]), (None, [1, 5])])
def test_selects_exactly_two_requested_family_and_records_all_devices(tmp_path, processes, monkeypatch,
                                                                     family, expected):
    devices = mixed_devices()
    apps = '6543, GPU-4, other_job, 0\n'
    snapshots(processes, devices, apps)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'old-inherited-selection')
    args, extra = arguments(tmp_path, family)
    launcher.main(args, extra)
    query, spawn = processes
    spawn.assert_called_once()
    command = spawn.call_args.args[0]
    env = spawn.call_args.kwargs['env']
    assert env['CUDA_VISIBLE_DEVICES'] == ','.join(f'GPU-{i}' for i in expected)
    assert '--nproc_per_node=2' in command
    assert command[-2:] == ['--plan', 'configs/pfnn_frame_actor_v1.json']
    assert '--gpu-family' not in command
    assert query.call_count == 2
    assert query.call_args_list[0].args[0][1].endswith(',display_active')
    value = json.loads(receipt_for(args).read_text())
    assert value['requested_gpu_family'] == ('A100' if family is None else family)
    assert value['selected'] == [[i, f'GPU-{i}'] for i in expected]
    assert value['all_devices'] == '\n'.join(devices) + '\n'
    assert value['all_compute_apps'] == apps
    assert value['pid'] == 9876 and value['command'] == command


def test_zero_utilization_compute_pid_is_occupied_even_with_zero_memory(tmp_path, processes):
    snapshots(processes, [gpu(0), gpu(1), gpu(2)],
              '4321, GPU-0, "python, owned by another user", 0\n')
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    launcher.main(args, extra)
    assert json.loads(receipt_for(args).read_text())['selected'] == [[1, 'GPU-1'], [2, 'GPU-2']]


@pytest.mark.parametrize('busy', [dict(memory=33), dict(usage=2), dict(display='Enabled'),
                                dict(memory='[N/A]'), dict(usage='[Not Supported]'),
                                dict(display='[N/A]')])
def test_nonidle_or_uncertified_gpu_is_excluded(tmp_path, processes, busy):
    snapshots(processes, [gpu(0, **busy), gpu(1), gpu(2)])
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    launcher.main(args, extra)
    assert json.loads(receipt_for(args).read_text())['selected'] == [[1, 'GPU-1'], [2, 'GPU-2']]


@pytest.mark.parametrize('devices,apps', [
    ([gpu(0, A100), gpu(1, A100)], ''),
    ([gpu(0, 'NVIDIA RTX 6000 Ada Generation'), gpu(1, 'NVIDIA RTX 6000 Ada Generation')], ''),
    ([gpu(0), gpu(1, memory=500), gpu(2, A100), gpu(3, A100)], ''),
    ([gpu(0), gpu(1)], '4321, GPU-1, python, 0\n'),
    ([gpu(0, display='Enabled'), gpu(1)], ''),
    ([], ''),
])
def test_wrong_family_or_insufficient_idle_refuses_without_launching(tmp_path, processes, devices, apps):
    snapshots(processes, devices, apps)
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    with pytest.raises(RuntimeError, match='Two idle RTX PRO 6000'):
        launcher.main(args, extra)
    processes[1].assert_not_called()
    assert not receipt_for(args).exists()
    assert not args.output.exists() and not args.output.with_suffix('.log').exists()


@pytest.mark.parametrize('mode,timeout', [('preflight', 1800), ('smoke', 1800), ('execute', 28800)])
def test_wall_timeout_and_worker_count_are_preserved(tmp_path, processes, mode, timeout):
    snapshots(processes, [gpu(0), gpu(1)])
    args, extra = arguments(tmp_path, 'RTX PRO 6000', mode)
    launcher.main(args, extra)
    command = processes[1].call_args.args[0]
    assert command[:4] == ['timeout', '--signal=TERM', '--kill-after=30', str(timeout)]
    assert '--nproc_per_node=2' in command
    assert command[command.index('--mode') + 1] == mode
    assert json.loads(receipt_for(args).read_text())['wall_timeout_seconds'] == timeout


@pytest.mark.parametrize('existing', ['output', 'receipt', 'log'])
def test_existing_output_receipt_or_log_refuses_before_hardware_calls(tmp_path, processes, existing):
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    if existing == 'output':
        args.output.mkdir()
        marker = args.output / 'keep.txt'
    else:
        marker = receipt_for(args) if existing == 'receipt' else args.output.with_suffix('.log')
    marker.write_text('existing run must remain intact')
    with pytest.raises(FileExistsError, match='already has'):
        launcher.main(args, extra)
    for call in processes:
        call.assert_not_called()
    assert marker.read_text() == 'existing run must remain intact'


def test_second_launch_refused_by_exclusive_receipt(tmp_path, processes):
    snapshots(processes, [gpu(0), gpu(1)])
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    launcher.main(args, extra)
    original = receipt_for(args).read_bytes()
    with pytest.raises(FileExistsError):
        launcher.main(args, extra)
    assert processes[0].call_count == 2
    processes[1].assert_called_once()
    assert receipt_for(args).read_bytes() == original


def test_unknown_family_refused_by_cli_without_hardware_calls(tmp_path, processes):
    with pytest.raises(SystemExit):
        arguments(tmp_path, 'any-free-gpu')
    for call in processes:
        call.assert_not_called()


def test_a100_default_does_not_select_pro_cards(tmp_path, processes):
    snapshots(processes, [gpu(0), gpu(1)])
    args, extra = arguments(tmp_path)
    assert args.gpu_family == 'A100'
    with pytest.raises(RuntimeError, match='Two idle A100'):
        launcher.main(args, extra)
    processes[1].assert_not_called()


def test_unreadable_compute_snapshot_refuses_launch(tmp_path, processes):
    snapshots(processes, [gpu(0), gpu(1)], '[Not Supported]\n')
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    with pytest.raises(ValueError):
        launcher.main(args, extra)
    processes[1].assert_not_called()
    assert not receipt_for(args).exists()


def test_gpu3_only_masks_all_other_cards_even_when_they_are_idle(tmp_path, processes):
    snapshots(processes, [gpu(i) for i in range(4)])
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    args.gpus = [3]
    launcher.main(args, extra)
    call = processes[1].call_args
    assert call.kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-3'
    assert '--nproc_per_node=1' in call.args[0]
    assert json.loads(receipt_for(args).read_text())['selected'] == [[3, 'GPU-3']]


def test_gpu3_only_refuses_busy_gpu3_without_using_another_idle_card(tmp_path, processes):
    snapshots(processes, [gpu(i) for i in range(4)], '1234, GPU-3, another_program, 0\n')
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    args.gpus = [3]
    with pytest.raises(RuntimeError, match='explicitly requested GPU'):
        launcher.main(args, extra)
    processes[1].assert_not_called()
    assert not receipt_for(args).exists()


def test_gpu3_only_works_while_other_cards_are_occupied(tmp_path, processes):
    snapshots(processes, [gpu(i) for i in range(4)],
              ''.join(f'{1200+i}, GPU-{i}, another_program, 9000\n' for i in range(3)))
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    args.gpus = [3]
    launcher.main(args, extra)
    assert processes[1].call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-3'


@pytest.mark.parametrize('indices', [[3, 3], [-1], [0, 1, 2]])
def test_invalid_explicit_gpu_selection_refuses_before_queries(tmp_path, processes, indices):
    args, extra = arguments(tmp_path, 'RTX PRO 6000')
    args.gpus = indices
    with pytest.raises(ValueError):
        launcher.main(args, extra)
    for call in processes:
        call.assert_not_called()
