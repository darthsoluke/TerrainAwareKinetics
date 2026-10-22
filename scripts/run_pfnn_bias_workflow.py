"""Continue the same full-corpus budget after a verified checkpoint barrier."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def process_alive(pid):
    p = Path('/proc')/str(pid)
    return p.exists() and next(x.split()[1] for x in (p/'status').read_text().splitlines()
        if x.startswith('State:')) != 'Z'


def main(a):
    run = a.base/'pfnn_mixed_20261015'
    code = Path(__file__).resolve().parents[1]
    old = run/'full_training_v3/root'
    proofs = [dict(role=role, path=str(run/folder/'probe/complete.json')) for role, folder in
        [('full_model', 'owned_full_bias_probe_v1'), ('optimizer_resume', 'owned_bias_resume_probe_v1')]]
    for proof in proofs:
        if not read(Path(proof['path']))['passed']:
            raise ValueError('Numerical/resume proof must pass')
        proof['sha256'] = digest(Path(proof['path']))
    implementation = 'motionbricks/data/dilated_bias_convolution.py'
    validated_hash = digest(run/'code_bias_full_probe_v1'/implementation)
    if digest(code/implementation) != validated_hash:
        raise ValueError('Accelerated implementation differs from validation')
    if a.validate_only:
        print(json.dumps(dict(validated=True, proofs=proofs, implementation_sha256=validated_hash)))
        return
    a.output.mkdir(parents=True, exist_ok=False)
    state = dict(pid=os.getpid(), stage='root', status='waiting_for_checkpoint_barrier',
        previous_root_pid=a.root_pid, actual_budget_updates=150000, quality_approved=False)
    save(a.output/'state.json', state)
    while process_alive(a.root_pid):
        if not (old/'bias_handoff_barrier.json').exists():
            state['status'] = 'waiting_for_barrier_to_be_armed'
        else:
            state['status'] = 'waiting_for_checkpoint_barrier'
        state['root_progress'] = read(old/'progress.json')
        save(a.output/'state.json', state)
        time.sleep(5)
    barrier = read(old/'bias_handoff_barrier.json')
    failure = read(old/'failure.json')
    if 'IsADirectoryError' not in failure['error'] or barrier['owner_pid'] != a.root_pid:
        raise ValueError('Previous Root exited for an unexpected reason')
    import torch
    saved = torch.load(old/'latest.tmp.pt', map_location='cpu', mmap=True, weights_only=False)
    step = saved['actual_updates']
    progress = read(old/'progress.json')
    if (step != progress['actual_updates'] or saved['actual_exposures'] != step*256
            or saved['sample_digest'] != progress['sample_digest']
            or step < barrier['published_updates_before']):
        raise ValueError('Completed checkpoint does not match the last optimizer update')
    if {int(x['step']) for x in saved['optimizer_state']['state'].values()} != {step}:
        raise ValueError('Optimizer state has a different update count')
    handoff = old/f'bias_handoff_step{step}.pt'
    os.link(old/'latest.tmp.pt', handoff)
    if not (old/'latest.pt').is_dir() or list((old/'latest.pt').iterdir()):
        raise ValueError('Checkpoint barrier was modified')
    (old/'latest.pt').rmdir()
    os.link(handoff, old/'latest.pt')
    sources = {p.relative_to(code).as_posix(): digest(p) for folder in ('motionbricks', 'scripts')
        for p in sorted((code/folder).rglob('*.py'))}
    migration = dict(from_checkpoint_sha256=digest(handoff),
        from_source_code_sha256=saved['contract']['source_code_sha256'], to_source_code_sha256=sources,
        actual_updates=step, validated_implementation_sha256=validated_hash, proofs=proofs,
        no_optimizer_updates_lost=True, old_failure_is_intentional_checkpoint_barrier=True)
    save(a.output/'migration.json', migration)
    del saved

    def command(stage, parent):
        result = [sys.executable, '-u', '-m', 'scripts.train_unified_full_corpus', '--stage', stage,
            '--bundle', str(a.base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'),
            '--parent', str(parent), '--preflight', str(run/'preflight_world_v1/complete.json'),
            '--cache', str(run/'cache_mixed_v1'), str(run/'cache_pfnn_world_v2'),
            '--output', str(a.output/stage), '--steps', '50000', '--batch', '256', '--micro-batch', '64',
            '--workers', '4', '--corpus-audit', str(run/'full_training_v3/corpus_audit.json')]
        result += ['--cache-pose-tokens'] if stage == 'pose' else ['--coalesce-dilated-bias']
        return result

    def execute(label, argv):
        while True:
            row = subprocess.check_output(['nvidia-smi', '-i', '0',
                '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            memory, busy = map(int, row.split(','))
            if memory <= 500 and busy <= 5:
                break
            state.update(status='waiting_for_idle_gpu', next_job=label)
            save(a.output/'state.json', state)
            time.sleep(10)
        with (a.output/(label+'.log')).open('x') as stream:
            process = subprocess.Popen(argv, cwd=code, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0',
                    TMPDIR='${PRIVATE_STORAGE}/mbtmp', PYTHONPATH='.:out/vendor/kimodo',
                    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
            state.update(status='running', job=label, child_pid=process.pid, command=argv)
            save(a.output/'state.json', state)
            result = process.wait()
        if result:
            raise RuntimeError(f'{label} exited with code {result}; inspect retained log')

    prototype = a.base/'model_terrain_v3_20261011/parc_rugged32_20261013_v2/unified_native_pose_fullbank_20261015_v2/weights.pt'
    root_command = command('root', prototype)+['--resume', str(handoff), '--resume-migration', str(a.output/'migration.json')]
    execute('root-resume', root_command)
    for stage, parent in [('codec', a.output/'root/weights.pt'), ('pose', a.output/'codec/weights.pt')]:
        state['stage'] = stage
        argv = command(stage, parent)
        execute(stage+'-startup', argv+['--stop-after', '2'])
        if read(a.output/stage/'stopped.json')['actual_updates'] != 2:
            raise ValueError('Startup accounting mismatch')
        execute(stage+'-resume', argv+['--resume', str(a.output/stage/'latest.pt')])
    for stage in ('root', 'codec', 'pose'):
        if read(a.output/stage/'complete.json')['actual_updates'] != 50000:
            raise ValueError('Stage budget incomplete')
    save(a.output/'complete.json', dict(actual_updates=150000, actual_exposures=38400000,
        quality_approved=False, final_checkpoint=str(a.output/'pose/weights.pt')))
    state['status'] = 'complete'
    save(a.output/'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root-pid', type=int, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            save(args.output/'failure.json', dict(error=repr(error)))
        raise
