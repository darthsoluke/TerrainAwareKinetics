"""Pause only the verified owned Root briefly; always resume its live state."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def identity(pid):
    root = Path('/proc')/str(pid)
    return (root/'stat').read_text().rsplit(')', 1)[1].split()[19]


def main(a):
    if not 0 < a.probe_timeout < a.maximum_pause <= 55:
        raise ValueError('Probe timeout must be shorter than the bounded pause')
    command = (Path('/proc')/str(a.owner_pid)/'cmdline').read_bytes().replace(bytes([0]), b' ').decode()
    if not all(text in command for text in ('scripts.train_unified_full_corpus', '--stage root', 'full_training_v3/root')):
        raise ValueError('Unexpected owner identity')
    started = identity(a.owner_pid)
    a.output.mkdir(parents=True, exist_ok=False)
    watchdog_code = (
        f'import pathlib,os,signal,time; time.sleep({a.maximum_pause}); '
        f'p=pathlib.Path("/proc/{a.owner_pid}/stat"); '
        f'os.kill({a.owner_pid},signal.SIGCONT) if p.exists() and '
        f'p.read_text().rsplit(")",1)[1].split()[19]=={started!r} else None'
    )
    watchdog = subprocess.Popen([sys.executable, '-c', watchdog_code],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    receipt = dict(owner_pid=a.owner_pid, owner_starttime=started, owner_command=command,
        watchdog_pid=watchdog.pid, maximum_pause_seconds=a.maximum_pause, optimizer_state_remains_in_memory=True)
    (a.output/'receipt.json').write_text(json.dumps(receipt, indent=2))
    beginning = time.monotonic()
    process = None
    try:
        os.kill(a.owner_pid, signal.SIGSTOP)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            state = (Path('/proc')/str(a.owner_pid)/'status').read_text()
            if any(line.startswith('State:') and line.split()[1] == 'T' for line in state.splitlines()):
                break
            time.sleep(.05)
        else:
            raise RuntimeError('Owner did not stop')
        with (a.output/'probe.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(a.probe), '--gpu', '0',
                '--paused-owner-pid', str(a.owner_pid), '--output', str(a.output/'probe')],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            returncode = process.wait(timeout=a.probe_timeout)
        receipt['probe_returncode'] = returncode
    except Exception as error:
        receipt['error'] = repr(error)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    finally:
        if identity(a.owner_pid) != started:
            raise RuntimeError('Owner identity changed; cannot resume a different process')
        os.kill(a.owner_pid, signal.SIGCONT)
        receipt['owner_resumed'] = True
        receipt['pause_seconds'] = time.monotonic()-beginning
        (a.output/'complete.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-pid', type=int, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-timeout', type=float, default=40)
    parser.add_argument('--maximum-pause', type=float, default=55)
    main(parser.parse_args())
