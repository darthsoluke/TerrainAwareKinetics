"""Continue the same Pose exposure policy with proven frozen-label graph replay."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_pfnn_bias_workflow import digest, process_alive, read, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def successor_command(original, output, handoff, migration):
    command = list(original)
    for flag in ('--resume', '--resume-migration'):
        index = command.index(flag)
        del command[index:index+2]
    if '--await-resume' in command:
        command.remove('--await-resume')
    command[command.index('--output')+1] = str(output/'pose')
    command += ['--graph-frozen-pose-labels', '--resume', str(handoff),
                '--resume-migration', str(migration), '--await-resume']
    return command


def main(a):
    import torch
    from motionbricks.data.pose_graph_migration import validate_pose_graph_migration
    from motionbricks.data.reference_exposure_budget import ExposureBudget, validate_completion_counts
    run = a.base/'pfnn_mixed_20261015'
    code = Path(__file__).resolve().parents[1]
    old = a.previous/'pose'
    producer = read(a.previous/'state.json')
    if (producer['stage'] != 'pose' or producer['child_pid'] != a.pose_pid
            or producer['pid'] != a.previous_supervisor or not process_alive(a.pose_pid)
            or not process_alive(a.previous_supervisor)):
        raise ValueError('Live Pose producer identity differs')
    cmdline = Path(f'/proc/{a.pose_pid}/cmdline').read_bytes().split(b'\0')
    if str(old).encode() not in cmdline or b'pose' not in cmdline:
        raise ValueError('Real Pose command does not identify the authorized producer')
    root = read(a.previous/'root_completion.json')
    codec_directory = Path(read(a.previous/'codec_completion.json')['directory'])
    codec = read(codec_directory/'complete.json')
    codec_contract = read(codec_directory/'contract.json')
    validate_completion_counts(codec_contract, codec['actual_updates'], codec['actual_exposures'])
    if (not codec['protected_hash_passed'] or digest(codec_directory/'weights.pt') != codec['checkpoint_sha256']
            or root['actual_exposures'] != 12800000 or not root['protected_hash_passed']):
        raise ValueError('Completed parent identity or exposure budget differs')
    sources = {path.relative_to(code).as_posix(): digest(path)
               for folder in ('motionbricks', 'scripts') for path in sorted((code/folder).rglob('*.py'))}
    proofs = [dict(role=role, path=str(run/name/'complete.json')) for role, name in (
        ('zero_update', 'pose_frozen_graph_probe_v1'), ('native_reference', 'pose_parallel_exposure_reference_v1'),
        ('graph_continuous', 'pose_graph_optimizer_parallel_v1'), ('graph_cold_resume', 'pose_graph_optimizer_resume_v1'))]
    for proof in proofs:
        proof['sha256'] = digest(Path(proof['path']))
    diagnostic_midpoint = digest(run/'pose_graph_optimizer_parallel_v1/midpoint.pt')

    def receipt(saved, checkpoint_hash):
        return dict(transition='pose-frozen-label-graph-v1', from_checkpoint_sha256=checkpoint_hash,
            from_source_code_sha256=saved['contract']['source_code_sha256'], to_source_code_sha256=sources,
            actual_updates=saved['actual_updates'], actual_exposures=saved['actual_exposures'], proofs=proofs,
            diagnostic_midpoint_sha256=diagnostic_midpoint, no_optimizer_updates_lost=True,
            execution_trajectory_bitwise=True,
            old_failure_is_intentional_checkpoint_barrier=True,
            authorization='user_requested_multigpu_large_memory_reference_50000', optimizer_trajectory_equivalent=False)

    def contract(saved):
        return dict(saved['contract'], frozen_pose_label_graphs=True, source_code_sha256=sources)

    current = read(old/'progress.json')
    old_contract = read(old/'contract.json')
    if old_contract['parent_sha256'] != codec['checkpoint_sha256'] or current['actual_exposures'] >= 12800000:
        raise ValueError('Pose parent or remaining transition budget differs')
    # This validates the proposed policy and proofs. The real saved optimizer
    # and exact checkpoint identity are checked again after the atomic barrier.
    planned = dict(contract=old_contract, actual_updates=current['actual_updates'], actual_exposures=current['actual_exposures'])
    validate_pose_graph_migration(planned, contract(planned), receipt(planned, 'planned-only'))
    if a.validate_only:
        print('Graph policy/source/proofs passed; real handoff checkpoint is not yet available', current['actual_updates'])
        return
    a.output.mkdir(exist_ok=False)
    save(a.output/'root_completion.json', root)
    save(a.output/'codec_completion.json', dict(directory=str(codec_directory), **codec))
    state = dict(pid=os.getpid(), stage='pose', status='waiting_for_two_idle_gpus',
        total_budget_reference_updates=150000,
        total_budget_exposures=38400000, budget_unit='reference_sample_exposures',
        optimizer_trajectory_equivalent=False, quality_approved=False)
    stable = set()
    while True:
        snapshot, available = gpu_snapshot()
        chosen = sorted(stable.intersection(available))
        if len(chosen) >= 2:
            break
        stable = set(available)
        state.update(gpu_snapshot=snapshot, available=available)
        save(a.output/'state.json', state)
        time.sleep(3 if len(available) >= 2 else 15)
    selected = chosen[:2]
    handoff = old/'graph_handoff.pt'
    migration = a.output/'migration.json'
    command = successor_command(producer['command'], a.output, handoff, migration)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                   '--format=csv,noheader'], text=True)
    with (a.output/'pose-resume.log').open('x') as log:
        child = subprocess.Popen(command, cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, selected)), TMPDIR='${PRIVATE_STORAGE}/mbtmp',
                     PYTHONPATH='.:out/vendor/kimodo', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    state.update(status='preloading_pose', child_pid=child.pid, command=command, job_gpus=selected,
                 gpu_snapshot_before=snapshot, compute_processes_before=apps)
    save(a.output/'state.json', state)
    while not (a.output/'pose-await-resume.json').exists():
        if child.poll() is not None:
            raise RuntimeError('Pose preload failed; original producer remains unchanged')
        time.sleep(3)
    if read(a.output/'pose-await-resume.json')['pid'] != child.pid:
        raise ValueError('Unexpected Pose preload acknowledgment')
    if child.poll() is not None or not process_alive(a.pose_pid):
        raise RuntimeError('Producer or preloaded successor exited before handoff')
    save(a.previous/'pose_graph_handoff.json', dict(successor=str(a.output), successor_pid=os.getpid(),
        previous_supervisor_pid=a.previous_supervisor, pose_pid=a.pose_pid, preloaded_child=child.pid,
        proofs=proofs))
    os.kill(a.previous_supervisor, signal.SIGTERM)
    owner = Path(f'/proc/{a.pose_pid}')
    owner_start = owner.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    watchdog = (f'import os,signal,time,pathlib; time.sleep(55); p=pathlib.Path("/proc/{a.pose_pid}/stat"); '
        f'os.kill({a.pose_pid},signal.SIGCONT) if p.exists() and '
        f'p.read_text().rsplit(")",1)[1].split()[19]=={owner_start!r} else None')
    subprocess.Popen([sys.executable, '-c', watchdog], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
    os.kill(a.pose_pid, signal.SIGSTOP)
    try:
        for _ in range(150):
            status = next(line.split()[1] for line in owner.joinpath('status').read_text().splitlines()
                          if line.startswith('State:'))
            if status == 'T':
                break
            time.sleep(.02)
        progress = read(old/'progress.json')
        if status != 'T' or progress['actual_exposures'] >= 12800000:
            raise ValueError('Pose cannot enter a checkpoint handoff')
        minimum_updates = progress['actual_updates']
        (old/'latest.pt').rename(old/'latest_before_graph_handoff.pt')
        (old/'latest.pt').mkdir()
        save(old/'graph_handoff_barrier.json', dict(owner_pid=a.pose_pid, owner_starttime=owner_start,
            published_updates_before=minimum_updates, successor_pid=os.getpid(),
            intentional_stop='Complete torch.save then IsADirectoryError before the next optimizer update'))
    finally:
        os.kill(a.pose_pid, signal.SIGCONT)
    state.update(status='waiting_for_complete_checkpoint', minimum_handoff_updates=minimum_updates)
    save(a.output/'state.json', state)
    while process_alive(a.pose_pid):
        if child.poll() is not None:
            raise RuntimeError('Preloaded Pose successor exited')
        time.sleep(3)
    if 'IsADirectoryError' not in read(old/'failure.json')['error']:
        raise ValueError('Original Pose exited unexpectedly')
    saved = torch.load(old/'latest.tmp.pt', map_location='cpu', mmap=True, weights_only=False)
    progress = read(old/'progress.json')
    prior = saved['contract']['exposure_budget']
    ExposureBudget(prior['initial_updates'], prior['initial_exposures'], 6144,
                   saved['actual_updates'], saved['actual_exposures'])
    if (saved['actual_updates'] < minimum_updates or progress['actual_updates'] != saved['actual_updates']
            or saved['sample_digest'] != progress['sample_digest']
            or {int(value['step']) for value in saved['optimizer_state']['state'].values()} != {saved['actual_updates']}
            or saved['scheduler_state']['last_epoch'] != saved['actual_exposures']//256):
        raise ValueError('Saved Pose optimizer/sample checkpoint differs from the completed update')
    os.link(old/'latest.tmp.pt', handoff)
    (old/'latest.pt').rmdir()
    os.link(handoff, old/'latest.pt')
    record = receipt(saved, digest(handoff))
    validate_pose_graph_migration(saved, contract(saved), record)
    save(migration, record)
    del saved
    state.update(status='running', handoff_updates=record['actual_updates'], handoff_exposures=record['actual_exposures'])
    save(a.output/'state.json', state)
    if child.wait():
        raise RuntimeError('Migrated Pose failed; inspect retained evidence')
    completed = read(a.output/'pose/complete.json')
    validate_completion_counts(read(a.output/'pose/contract.json'), completed['actual_updates'], completed['actual_exposures'])
    if not completed['protected_hash_passed'] or digest(a.output/'pose/weights.pt') != completed['checkpoint_sha256']:
        raise ValueError('Final Pose identity or frozen state differs')
    state.update(status='training_complete_quality_review_required', equivalent_reference_updates=150000,
        actual_updates=root['actual_updates']+codec['actual_updates']+completed['actual_updates'],
        actual_exposures=38400000, final_checkpoint=str(a.output/'pose/weights.pt'))
    save(a.output/'state.json', state)
    save(a.output/'complete.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'previous', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--pose-pid', type=int, required=True)
    parser.add_argument('--previous-supervisor', type=int, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            save(args.output/'failure.json', dict(error=repr(error)))
        raise
