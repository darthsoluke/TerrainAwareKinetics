"""Cancel our own baseline-only job before an invalid mask allocation is trained."""
import argparse
import json
from pathlib import Path
import os
import signal

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True)
p.add_argument('--pid', type=int, required=True)
a = p.parse_args()
command = (Path('/proc') / str(a.pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
state = json.loads((a.run / 'state.json').read_text())
if str(a.run) not in command or 'scripts.pilot_unified_pose_capacity' not in command:
    raise SystemExit('Process identity mismatch')
if state['status'] != 'baseline_review' or any((a.run / name).exists() for name in ('preflight.json', 'candidate.pt')):
    raise SystemExit('Training may have begun; do not cancel through this helper')
os.kill(a.pid, signal.SIGTERM)
(a.run / 'state.json').write_text(json.dumps(dict(status='cancelled_before_optimizer', optimizer_updates=0,
    reason='Rotating batch slot correlated full-mask samples with source parity; use stratified selection.'), indent=2))
print(json.dumps(dict(pid=a.pid, command_verified=True, optimizer_updates=0)))
