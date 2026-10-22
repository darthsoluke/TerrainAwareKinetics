"""Export an explicitly requested unreviewed WASD checkpoint via private Drive."""

import argparse
import configparser
import json
import os
from pathlib import Path
import subprocess

import torch

from scripts.train_native_mb import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','output','base'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cloud',required=True)
    p.add_argument('--state-file',type=Path)
    p.add_argument('--direct-drive',action='store_true')
    p.add_argument('--rclone-config',type=Path)
    args=p.parse_args();torch.set_num_threads(2);args.output.mkdir(parents=True,exist_ok=True)
    config_path=args.rclone_config or args.base/'drive-auth/private-client.conf'
    drive=configparser.ConfigParser(interpolation=None);drive.read(config_path)
    remote=args.cloud.split(':',1)[0]
    expected_client='GOOGLE_OAUTH_CLIENT_ID'
    if remote not in drive or drive[remote].get('client_id')!=expected_client or not drive[remote].get('client_secret'):
        raise ValueError('Use the verified user-owned OAuth client; shared Google API credentials are forbidden')
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    plan=saved['contract']['sample_budget'];step=saved['step']
    examples=plan['retained_examples']+(step-plan['anchor_update'])*plan['global_batch']
    path=args.output/f'weights-{step:06d}.pt'
    if not path.exists():
        torch.save({k:saved[k] for k in ('schema','model_state','step','contract','coefficients')},path)
    receipt={'schema':'motionbricks.joint_terrain_export.v1','trained_updates':step+plan['inherited_reference_updates'],
        'primitive_examples':examples,'reference_batch':plan['reference_batch'],
        'reference_batch_equivalent_updates':examples/plan['reference_batch'],
        'weights_file':path.name,'weights_sha256':sha256(path),'source_file':'source.pt',
        'source_sha256':saved['contract']['source_bundle_sha256'],'control_version':saved['contract']['control_version'],
        'model_checks_passed':False,'quality_evaluation':'not run; user explicitly requested direct WASD trial',
        'local_live_validation_pending':True,'quality_approved':False,'postprocessing':None}
    (args.output/'export.json').write_text(json.dumps(receipt,indent=2))
    (args.output/'contract.json').write_text(json.dumps(saved['contract'],indent=2))
    print(json.dumps({'status':'exported','receipt':receipt,'cloud':args.cloud}),flush=True)
    proxy=dict(os.environ,HTTP_PROXY='http://127.0.0.1:27890',HTTPS_PROXY='http://127.0.0.1:27890')
    if args.direct_drive:proxy.update(HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
    subprocess.run(['rclone','--config',str(config_path),'copy',str(args.output),args.cloud,
        '--include','/'+path.name,'--include','/export.json','--include','/contract.json',
        '--immutable','--checksum','--transfers','1','--checkers','1','--tpslimit','1','--tpslimit-burst','1','--drive-chunk-size','64M',
        '--stats','15s','--stats-one-line','--stats-log-level','NOTICE','--log-level','INFO',
        '--timeout','10m','--contimeout','30s','--low-level-retries','10','--retries','5','--retries-sleep','30s'],env=proxy,check=True)
    (args.output/'uploaded.json').write_text(json.dumps({'cloud':args.cloud,'receipt':receipt},indent=2))
    if args.state_file:
        state=json.loads(args.state_file.read_text())
        state.update(status='ready_for_unreviewed_trial',private_cloud=args.cloud,export=receipt,error=None)
        temporary=args.state_file.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(args.state_file)
    print(json.dumps({'status':'uploaded','cloud':args.cloud}),flush=True)


if __name__=='__main__':main()
