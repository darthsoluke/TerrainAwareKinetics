"""Export the completed decoder A/B, via the user's private Drive only."""
import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.train_native_mb import sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment',choices=('decoder','lateral','sourcecodec','backbone','context'),default='decoder')
args=parser.parse_args()
task=Path.cwd().parent;base=task.parent;root=task/'joint_sdf_20261012'
name={'decoder':'decoder_ab_export_v1','lateral':'lateral_export_v1','sourcecodec':'sourcecodec_export_v1','backbone':'backbone_export_v1','context':'context_export_v1'}[args.experiment]
output=root/name;output.mkdir(exist_ok=False)
reviews=[];receipt={'arms':{},'postprocessing':None,'quality_approved':False}
arms={'decoder':('A','B'),'lateral':('control','augmented'),'sourcecodec':('control','auxiliary'),'backbone':('root','pose'),'context':('source','feedback')}[args.experiment]
updates,examples=(300,1200) if args.experiment=='context' else (300,3600) if args.experiment in ('decoder','backbone') else (200,2400)
for arm in arms:
    source=root/('decoder_ab_'+arm if args.experiment=='decoder' else args.experiment+'_'+arm+'_v1')
    report=json.loads((source/'complete.json').read_text())
    if report['actual_updates']!=updates or report['primitive_examples']!=examples:raise ValueError('A/B budget mismatch')
    shutil.copyfile(source/'complete.json',output/(arm+'-complete.json'))
    shutil.copyfile(source/'contract.json',output/(arm+'-contract.json'))
    receipt['arms'][arm]={'checkpoint_sha256':sha256(source/'weights.pt'),'updates':updates,'primitive_examples':examples}
    start_review=source/('official_start_review/complete.json' if args.experiment=='context' else 'start_review/complete.json')
    if start_review.exists():shutil.copyfile(start_review,output/(arm+'-start-review.json'))
    label={'source':'teacher_history','feedback':'generated_history'}.get(arm,arm) if args.experiment=='context' else arm
    reviews.append(label+'='+str(source/'final'))
    if args.experiment=='context':
        shutil.copyfile(source/'preflight.json',output/(arm+'-preflight.json'))
        shutil.copyfile(source/'progress.json',output/(arm+'-exposure.json'))
subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals',
    '--stats',str(base/'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),'--reviews',*reviews,
    '--output',str(output/'visuals')],check=True)
if args.experiment=='context':
    shutil.copyfile(root/'context_pair_summary.json',output/'pair-summary.json')
    subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals',
        '--stats',str(base/'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
        '--reviews','initial='+str(root/'context_source_v1/baseline'),
        '--output',str(output/'baseline_visuals')],check=True)
receipt['files']={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
(output/'receipt.json').write_text(json.dumps(receipt,indent=2))
config=base/'drive-auth/private-client.conf';settings=configparser.ConfigParser(interpolation=None);settings.read(config)
if settings['mb-private'].get('client_id')!='GOOGLE_OAUTH_CLIENT_ID':
    raise ValueError('Private OAuth client differs')
environment=dict(os.environ,HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
subprocess.run(['rclone','--config',str(config),'copy',str(output),
    'private-store:motionbricks/joint_sdf_20261012/'+name,'--immutable','--checksum',
    '--transfers','1','--checkers','1','--tpslimit','1','--retries','2'],env=environment,check=True)
print(json.dumps({'status':'uploaded','directory':str(output),'quality_approved':False}),flush=True)
