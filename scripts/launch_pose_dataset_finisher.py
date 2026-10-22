"""Detach one explicitly scoped CPU dataset finisher with an exclusive log."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--base', type=Path, required=True)
p.add_argument('--code', type=Path, required=True)
p.add_argument('--collection', type=Path, required=True)
p.add_argument('--anchor', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--script', type=Path, required=True)
p.add_argument('--workers', type=int, nargs='+', required=True)
a = p.parse_args()
log = a.output.with_suffix('.launch.log')
receipt = a.output.with_suffix('.process.json')
if a.output.exists() or receipt.exists():
    raise SystemExit('Finisher output/receipt already exists')
command = [str(a.base / 'env/bin/python'), str(a.script), '--collection', str(a.collection),
           '--anchor', str(a.anchor), '--output', str(a.output), '--worker-gpus', *map(str, a.workers)]
with log.open('xb') as stream:
    child = subprocess.Popen(command, cwd=a.code, stdin=subprocess.DEVNULL, stdout=stream,
        stderr=subprocess.STDOUT, start_new_session=True,
        env=dict(os.environ, PYTHONPATH='.:out/vendor/kimodo', OMP_NUM_THREADS='4'))
record = dict(pid=child.pid, command=command, cpu_only=True, log=str(log))
receipt.write_text(json.dumps(record, indent=2))
print(json.dumps(record))
