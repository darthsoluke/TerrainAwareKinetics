"""Launch one identified PFNN preparation job on a freshly verified idle GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess


def main(a):
    base = a.base.resolve()
    root = base/'pfnn_mixed_20261015'
    bundle = base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'
    old = base/'model_terrain_v3_20261011/parc_rugged32_20261013_v2'
    command = [str(base/'env/bin/python'), '-u', '-m']
    if a.job in ('preflight','mechanics'):
        module = 'scripts.check_unified_full_training' if a.job == 'preflight' else 'scripts.check_full_training_resume'
        command += [module, '--bundle', str(bundle),
            '--initial', str(old/'unified_native_pose_fullbank_20261015_v2/weights.pt'),
            '--cases', str(old/'unified_pose_data_20261014_v3/cases.pt')]
        if a.job == 'mechanics':
            command += ['--preflight', str(root/'preflight_v6/complete.json'), '--stats',
                str(base/'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
                '--pfnn', str(root/'dataset_pilot_v1')]
    else:
        command += ['scripts.cache_full_terrain_primitives', '--bundle', str(bundle)]
        if a.job == 'cache-mixed':
            command += ['--mixed', str(base/'long_route_mixed_10k_ab_v1/mixed_dataset')]
        else:
            command += ['--pfnn', str(root/'dataset_full_v1'), '--stats',
                str(base/'long_route_mixed_10k_ab_v1/mixed_dataset/stats')]
    command += ['--output', str(a.output)]
    receipt = a.output.with_suffix('.process.json')
    if receipt.exists() or a.output.exists():
        raise ValueError('Job output or receipt already exists; inspect it instead of launching twice')
    occupancy = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    row = next(line for line in occupancy.splitlines() if int(line.split(',')[0]) == a.gpu)
    _, memory, busy = map(int, row.split(','))
    if memory > 500 or busy > 5:
        raise ValueError('Selected GPU is occupied: '+row)
    code = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONPATH='.:out/vendor/kimodo',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    with a.output.with_suffix('.log').open('xb') as stream:
        child = subprocess.Popen(command, cwd=code, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    value = dict(pid=child.pid, command=command, code=str(code), gpu=a.gpu,
        occupancy_before=occupancy, job=a.job, actual_optimizer_updates=0)
    receipt.write_text(json.dumps(value, indent=2))
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--job', choices=['preflight', 'mechanics', 'cache-mixed', 'cache-pfnn'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
