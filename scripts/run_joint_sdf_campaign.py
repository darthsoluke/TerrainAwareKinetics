"""Bounded training, final uncorrected review, and private artifact export."""
import configparser
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def main():
    code=Path.cwd();task=code.parent;base=task.parent
    output=task/'joint_sdf_20261012/train_1600k_v1';review=output/'review';export=output/'export'
    state_file=task/'joint-sdf-campaign-state.json';state={'code':str(code),'output':str(output),'quality_approved':False}
    def update(status,**values):
        state.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),**values);write_json(state_file,state)
        print(json.dumps(state),flush=True)
    def run(command,label,environment=None):
        update(label,command=command)
        with (task/('joint-sdf-'+label+'.log')).open('ab') as stream:
            process=subprocess.Popen(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            update(label,child_pid=process.pid)
            if process.wait()!=0:raise RuntimeError(label+' failed; retain checkpoint and logs, do not blindly restart')
    try:
        if output.exists():raise ValueError('Campaign already exists; inspect and explicitly resume')
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=3','-m','scripts.train_joint_sdf',
            '--checkpoint',str(task/'control_sample_v8_live/weights-005709.pt'),'--output',str(output),
            '--batch','128','--examples','1600000','--checkpoint-every','100']
        run(command+['--stop-at','8'],'production_smoke')
        run(command+['--resume',str(output/'last.pt')],'training')
        completed=json.loads((output/'complete.json').read_text())
        if not completed['budget_complete'] or completed['primitive_examples']<1600000:raise ValueError('Sample budget incomplete')
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
        run([sys.executable,'-u','-m','scripts.evaluate_joint_sdf','--checkpoint',str(output/'last.pt'),'--output',str(review)],'final_review',environment)
        run([sys.executable,'-u','-m','scripts.export_joint_terrain_visuals',
            '--stats',str(base/'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
            '--reviews','sdf='+str(review/'final'),'baseline='+str(review/'disabled_sdf'),
            '--output',str(export/'visuals')],'render_export',environment)
        saved=torch.load(output/'last.pt',map_location='cpu',mmap=True,weights_only=False)
        weights=export/'weights.pt';torch.save({k:saved[k] for k in ('schema','model_state','step','contract','coefficients')},weights)
        receipt={'weights_sha256':sha256(weights),'source_sha256':saved['contract']['source_bundle_sha256'],
            'actual_updates':completed['actual_updates'],'primitive_examples':completed['primitive_examples'],
            'batch32_equivalent_updates':completed['batch32_equivalent_updates'],
            'quality_approved':False,'local_validation_pending':True,'postprocessing':None}
        write_json(export/'receipt.json',receipt)
        write_json(export/'evaluation.json',json.loads((review/'complete.json').read_text()))
        config=base/'drive-auth/private-client.conf';settings=configparser.ConfigParser(interpolation=None);settings.read(config)
        remote='mb-private';expected='GOOGLE_OAUTH_CLIENT_ID'
        if settings[remote].get('client_id')!=expected or not settings[remote].get('client_secret'):
            raise ValueError('Only user-owned private Drive credentials are allowed')
        environment.update(HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
        cloud=remote+':motionbricks/joint_sdf_20261012/train_1600k_v1'
        run(['rclone','--config',str(config),'copy',str(export),cloud,'--immutable','--checksum',
            '--transfers','1','--checkers','1','--tpslimit','1','--drive-chunk-size','64M','--retries','3'],
            'private_upload',environment)
        update('awaiting_local_review',cloud=cloud,receipt=receipt,
            numerical_pass=json.loads((review/'complete.json').read_text())['numerical_pass'])
    except Exception as error:
        update('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
