"""Retire only an identified idle-wait supervisor, preserving every child artifact."""
import argparse
import json
from pathlib import Path
import os
import signal

p = argparse.ArgumentParser()
p.add_argument('--collection', type=Path, required=True)
p.add_argument('--gpu', type=int, required=True)
p.add_argument('--pid', type=int, required=True)
a = p.parse_args()
path = a.collection / f'worker_{a.gpu}.json'
state = json.loads(path.read_text())
proc = Path('/proc') / str(a.pid)
command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
children = (proc / 'task' / str(a.pid) / 'children').read_text().strip()
if (str(a.collection) not in command or 'scripts.run_parc_rugged_collection' not in command
        or state['pid'] != a.pid or state['status'] != 'waiting_for_idle_gpu' or children):
    raise SystemExit('Supervisor is not the identified childless GPU waiter')
os.kill(a.pid, signal.SIGTERM)
state.update(status='retired_waiting_gpu', reason='GPU now belongs to another user; pending layouts move to a freshly idle card.')
path.write_text(json.dumps(state, indent=2))
print(json.dumps(dict(pid=a.pid, retired_without_active_children=True)))
