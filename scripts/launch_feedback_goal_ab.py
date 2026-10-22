"""Launch two isolated, matched diagnostic arms on our two available A100s."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.launch_joint_terrain_training import available_cards
from scripts.train_joint_terrain_v3 import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    args=p.parse_args();base=args.base;task=base/'model_terrain_v3_20261011'
    output=task/'feedback_goal_ab_v1'
    if output.exists():raise ValueError('Inspect the existing experiment; do not duplicate it')
    cards=available_cards([2,3]);output.mkdir()
    code=Path(__file__).resolve().parents[1]
    common=[sys.executable,'-u','-m','scripts.probe_terrain_feedback_repair',
        '--source',str(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'),
        '--dataset',str(base/'long_route_mixed_10k_ab_v1/mixed_dataset'),
        '--checkpoint',str(task/'overfit_v1/feedback-diagnostic.pt'),'--updates','100']
    state={'gpu_snapshot':cards,'arms':{},'quality_approved':False}
    for mode,gpu in (('rolling',2),('requested',3)):
        command=common+['--mode',mode,'--output',str(output/mode)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'))
        with (output/(mode+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=code,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['arms'][mode]={'pid':process.pid,'physical_gpu':gpu,'command':command}
        write_json(output/'launch.json',state)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
