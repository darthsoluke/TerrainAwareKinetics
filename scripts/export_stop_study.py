"""Export all bounded stop studies with independent real clocks and honest labels."""

import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np

from scripts.train_native_mb import sha256


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    studies=dict(before='stop_command_pilot_v1',pose='stop_stage_pose_v1',decoder='stop_stage_decoder_v1',
        feedback='stop_feedback_v2',transition='stop_transition_v1')
    for arm,name in studies.items():
        run=args.root/name;folder=args.output/'recordings'/arm;folder.mkdir(parents=True)
        for path in (run/'final').glob('*.npz'):
            with np.load(path,allow_pickle=False) as archive:frames=len(archive['roots'])
            shutil.copy2(path,folder/(path.stem+f'-{frames}f.npz'))
        for filename in ('complete.json','contract.json','source_screen.json'):
            if (run/filename).exists():shutil.copy2(run/filename,args.output/(arm+'-'+filename))
    subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals','--stats',str(args.stats),
        '--reviews',*[arm+'='+str(args.output/'recordings'/arm) for arm in studies],
        '--output',str(args.output/'visuals')],check=True)
    path=args.output/'visuals/manifest.json';original=json.loads(path.read_text())
    manifest=dict(fps=30,postprocessing=None,quality_approved=False,audit=original['audit'],clips={},
        labels=dict(before='Root-only stop trial: rejected',pose='Pose stop branch: rejected',
            decoder='Decoder stop branch: rejected',feedback='Quality-scaled stop: rejected',
            transition='Stop/restart branch: not approved'),
        scope='Independent actual30Hz recordings. No padding or retiming. Ended recordings are not new generated frames.')
    for name,clip in original['clips'].items():
        base=name.rsplit('-',1)[0]
        merged=manifest['clips'].setdefault(base,dict(frames=0,fps=30,arms={},scene=clip['scene']))
        merged['frames']=max(merged['frames'],clip['frames'])
        for arm,record in clip['arms'].items():
            record.update(frames=clip['frames'],fps=30,duration_seconds=clip['frames']/30)
            merged['arms'][arm]=record
    # Recompute a single scene crop from all actual Root paths without shifting bodies.
    for name,clip in manifest['clips'].items():
        example=next((args.output/'recordings'/arm).glob(name+'-*.npz') for arm in studies)
        file=next(example)
        with np.load(file,allow_pickle=False) as a:
            hf=a['heightfield_y_m'];origin=a['min_point_world_xyz_m'];dx=float(a['dx_m'])
        roots=np.concatenate([record['roots'] for record in clip['arms'].values()])
        lo=roots[:,[0,2]].min(0)-5;hi=roots[:,[0,2]].max(0)+5
        x0,z0=np.maximum(0,np.floor((lo-origin[[0,2]])/dx).astype(int))
        x1,z1=np.minimum([hf.shape[1],hf.shape[0]],np.ceil((hi-origin[[0,2]])/dx).astype(int)+1)
        clip['scene']=dict(heightfield_y_m=[[float(v) if np.isfinite(v) else None for v in row] for row in hf[z0:z1,x0:x1]],
            origin=(origin+np.array([x0*dx,0,z0*dx])).tolist(),dx=dx)
    path.write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False))
    receipt=dict(quality_approved=False,frame_padding=False,playback_retiming=False,
        files={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()})
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    archive=args.output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as stream:stream.add(args.output,arcname=args.output.name)
    archive_receipt=archive.with_suffix('.receipt.json')
    archive_receipt.write_text(json.dumps(dict(filename=archive.name,sha256=sha256(archive),bytes=archive.stat().st_size,
        files=len(receipt['files'])),indent=2))
    settings=configparser.ConfigParser(interpolation=None);settings.read(args.drive_config)
    if settings['mb-private'].get('client_id')!='GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('User-owned client mismatch')
    environment=dict(os.environ,HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
    for file in (archive,archive_receipt):
        subprocess.run(['rclone','--config',str(args.drive_config),'copyto',str(file),
            'private-store:motionbricks/parc_stair64_20261013/'+file.name,'--immutable','--checksum','--transfers','1',
            '--checkers','1','--tpslimit','1','--retries','2'],env=environment,check=True)
    print(json.dumps(dict(status='uploaded',files=len(receipt['files']),quality_approved=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','stats','output','drive-config'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
