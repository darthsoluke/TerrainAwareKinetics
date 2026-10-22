"""Verify matched probes and privately export the predeclared flat failures."""

import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def main(args):
    a=json.loads((args.root/'complete.json').read_text())
    b=json.loads((args.all/'complete.json').read_text())
    if a['schedule']!=b['schedule'] or a['exposure']!=b['exposure']:
        raise ValueError('A/B source/yaw/primitive schedules differ')
    if a['baseline_flat']!=b['baseline_flat'] or a['baseline_stairs']!=b['baseline_stairs']:
        raise ValueError('A/B baselines differ')
    if not (a['frozen_hashes_match'] and b['frozen_hashes_match']):
        raise ValueError('A frozen module changed')
    args.output.mkdir(parents=True,exist_ok=False)
    summaries={}
    for label,rows in [('before',a['baseline_flat']),('root_only',a['final_flat']),('all_stages',b['final_flat'])]:
        critical=next(r for r in rows if abs(r['yaw']-3.141592653589793)<1e-6 and r['request_speed']==1.6 and r['idle_seconds']==16)
        defaults=[r for r in rows if r['request_speed']==1.6 and r['idle_seconds']==16]
        summaries[label]=dict(critical=critical,default_restart_rows=defaults,
            default_stalled_cases=sum(r['speed_last_second']<.2 for r in defaults),
            default_max_side_m=max(abs(r['side_m']) for r in defaults),
            default_max_penetration_m=max(r['body_prism_max_m'] for r in defaults))
    summary=dict(arms=summaries,matched_schedule=True,frozen_hashes_match=True,
        actual_updates_each=a['actual_updates'],actor_primitives_each=a['actor_primitives'],
        source_autoencoder_examples_each=a['source_autoencoder_examples'],
        stairs_pass_counts=dict(before=sum(r['passed'] for r in a['baseline_stairs']),
            root_only=sum(r['passed'] for r in a['final_stairs']),all_stages=sum(r['passed'] for r in b['final_stairs'])),
        quality_approved=False,postprocessing=None)
    write_json(args.output/'summary.json',summary)
    for label,folder in [('root_only',args.root),('all_stages',args.all)]:
        for filename in ('complete.json','preflight.json','contract.json','source_screen.json'):
            shutil.copyfile(folder/filename,args.output/(label+'-'+filename))
    labels={}
    for label,folder in [('before',args.root/'baseline_flat'),('root_only',args.root/'final_flat'),('all_stages',args.all/'final_flat')]:
        target=args.output/'recordings'/label;target.mkdir(parents=True)
        for filename,name in [('idle60-case4.npz','idle16s-forward-yaw180'),
                              ('idle60-case0.npz','idle16s-forward-yaw0'),
                              ('idle0-case0.npz','immediate-forward-yaw0')]:
            shutil.copyfile(folder/filename,target/(name+'.npz'))
        labels[label]=target
    subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals','--stats',str(args.stats),
        '--reviews',*[label+'='+str(folder) for label,folder in labels.items()],
        '--output',str(args.output/'visuals')],check=True)
    path=args.output/'visuals/manifest.json';manifest=json.loads(path.read_text())
    manifest.update(quality_approved=False,postprocessing=None,
        scope='Actual continuous model outputs. Long clips:zero input frames0..479,forward input480..719. No reference pose or padding.',
        selection='Previously reproduced yaw180 dead zone plus fixed yaw0 controls; all16case metrics retained separately.')
    manifest['labels'].update(before='Current live model',root_only='Root command only',all_stages='Root + Pose + decoder command')
    write_json(path,manifest)
    files={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output/'receipt.json',dict(files=files,quality_approved=False,frame_padding=False))
    archive=args.output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as stream:stream.add(args.output,arcname=args.output.name)
    receipt=archive.with_suffix('.receipt.json')
    write_json(receipt,dict(filename=archive.name,sha256=sha256(archive),bytes=archive.stat().st_size,files=len(files)))
    settings=configparser.ConfigParser(interpolation=None);settings.read(args.drive_config)
    if settings['mb-private'].get('client_id')!='GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('User-owned private Drive client required')
    env=dict(os.environ,HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
    for file in (archive,receipt):
        subprocess.run(['rclone','--config',str(args.drive_config),'copyto',str(file),
            'private-store:motionbricks/parc_rugged32_20261013_v2/'+file.name,'--immutable','--checksum',
            '--transfers','1','--checkers','1','--tpslimit','1','--retries','2'],env=env,check=True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','all','stats','output','drive-config'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
