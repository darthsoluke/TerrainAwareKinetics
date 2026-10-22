"""Collect the bounded leg probes without retraining or changing any recording."""

import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    task = Path.cwd().parent
    root = task/'joint_sdf_20261012'
    state = root/'leg_head_sequence_v2/state.json'
    for _ in range(60):
        current = json.loads(state.read_text())
        if current['status'] == 'failed':
            raise ValueError(current)
        if current['status'] == 'complete':
            break
        time.sleep(10)
    else:
        raise TimeoutError('Bounded wait expired; no process was stopped or restarted')
    recordings = []
    for name, run, arm in [('root','render_head_fit_v1/root','root'),('legs','leg_head_v1','legs'),
                           ('sequence','leg_head_sequence_v2','legs')]:
        folder = root/run/'long_review'
        report = json.loads((folder/'complete.json').read_text())
        row = next(row for row in report['rows'] if row['label']=='generated_8030ae7a2a7ad640')
        if row['frames'] == 0:
            raise ValueError('No motion for footfall review; preserve the failure')
        path = folder/arm/f'{row["label"]}-{arm}-{row["frames"]}f.npz'
        recordings.append(name+'='+str(path))
    subprocess.run([sys.executable,'-m','scripts.audit_joint_sdf_stair_footfalls','--recordings',*recordings,
        '--direction','3.026415348','-0.063034154','--output',str(root/'leg_head_footfalls_v2')],check=True)
    subprocess.run([sys.executable,str(task/'audit_joint_sdf_penetration_locations_v1.py'),'--recordings',*recordings,
        '--output',str(root/'leg_head_penetration_v2')],check=True)
    subprocess.run([sys.executable,str(task/'export_joint_sdf_leg_comparison_v2.py'),'--mode','legs'],check=True)


if __name__=='__main__':
    main()
