"""CPU-only real-process check of normal, timeout and parent-death resumption."""

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def state(pid):
    return next(line.split()[1] for line in (Path('/proc')/str(pid)/'status').read_text().splitlines()
        if line.startswith('State:'))


def main(a):
    reports = []
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        probe = root/'fixture.py'
        for variant in ('error', 'timeout', 'parent_death'):
            owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(90)',
                'scripts.train_unified_full_corpus', '--stage', 'root', 'full_training_v3/root'])
            try:
                probe.write_text('import sys,time; '+('sys.exit(23)' if variant == 'error' else 'time.sleep(20)'))
                output = root/variant
                guard = subprocess.Popen([sys.executable, str(a.guard), '--owner-pid', str(owner.pid),
                    '--probe', str(probe), '--output', str(output), '--probe-timeout', '2', '--maximum-pause', '5'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if variant == 'parent_death':
                    deadline = time.monotonic()+3
                    while time.monotonic() < deadline and state(owner.pid) != 'T':
                        time.sleep(.02)
                    assert state(owner.pid) == 'T'
                    guard.kill()
                stdout, stderr = guard.communicate(timeout=10)
                deadline = time.monotonic()+7
                while time.monotonic() < deadline and state(owner.pid) == 'T':
                    time.sleep(.05)
                assert state(owner.pid) not in ('T', 'Z'), (variant, stdout, stderr)
                if variant != 'parent_death':
                    report = json.loads((output/'complete.json').read_text())
                    assert report['owner_resumed']
                    if variant == 'error':
                        assert report['probe_returncode'] == 23
                    else:
                        assert 'TimeoutExpired' in report['error']
                reports.append(dict(case=variant, owner_resumed=True))
            finally:
                owner.send_signal(signal.SIGCONT)
                owner.terminate()
                owner.wait(timeout=5)
    print(json.dumps(dict(passed=True, cpu_only=True, cases=reports)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard', type=Path, required=True)
    main(parser.parse_args())
