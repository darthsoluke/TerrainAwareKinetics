"""Retain the failed control's last checkpoint and audit support before expansion."""

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
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_pure_parc_stairs import review
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    root=args.root
    saved=torch.load(root/'continuity_control_v1/weights.pt',map_location='cpu',mmap=True,weights_only=False)
    model,_=load_model(saved)
    cases=torch.load(root/'pure_parc_data_v2/cases.pt',map_location='cpu',weights_only=False)
    for prior in torch.load(args.regression_cases,map_location='cpu',weights_only=False):
        yaw=torch.tensor(prior['command_heading'])
        request=torch.cat((torch.tensor(prior['command_velocity']),yaw.cos()[None],yaw.sin()[None]))[None]
        cases.append(dict(prior,label='approved-'+prior['entry'],pilot_split='prior_approved_regression',exact_constant_control=request))
    geometry=JointGeometryObjective(model.inference.global_motion_rep,'cuda:0')
    if args.existing_runs:
        runs={name:Path(folder) for name,folder in (item.split('=',1) for item in args.existing_runs)}
        if len(runs)!=len(args.existing_runs) or 'source' not in runs or not all(p.is_dir() for p in runs.values()):
            raise ValueError('Unique existing run directories plus source required')
    else:
        control_output=args.output/'control_review'
        control_rows=review(model,cases,geometry,control_output)
        write_json(args.output/'control-review.json',dict(actual_retained_updates=saved['step'],
            training_failure='unobserved terrain during update3; not restarted or reset',rows=control_rows))
        runs=dict(approved=root/'pure_parc_fit_v1/final',control=control_output,
                  boundaries=root/'continuity_all_boundaries_v1/final',source=root/'pure_parc_fit_v1/baseline')
    audits=[]
    for arm,folder in runs.items():
        destination=args.output/'recordings'/arm;destination.mkdir(parents=True)
        files=folder.glob('*-source.npz') if arm=='source' else folder.glob('*-speed*-*f.npz')
        for path in files:
            with np.load(path,allow_pickle=False) as archive:
                world=torch.from_numpy(archive['world'].copy())[None].cuda()
                scene={key:archive[key].copy() for key in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
            _,points=geometry.geometry(world)
            gap,known=heightfield_gap(points[0,:,:geometry.sole_count],scene)
            lowest=gap.amin(-1);above=(lowest>.05).tolist();longest=current=0
            for flag in above:
                current=current+1 if flag else 0;longest=max(longest,current)
            audits.append(dict(arm=arm,file=path.name,frames=world.shape[1],
                all_soles_above5cm_fraction=float((lowest>.05).float().mean()),
                all_soles_above10cm_fraction=float((lowest>.1).float().mean()),longest_all_soles_above5cm_frames=longest,
                unknown_fraction=float((~known).float().mean()),minimum_sole_gap_m=float(lowest.min()),
                median_minimum_sole_gap_m=float(lowest.median()),maximum_minimum_sole_gap_m=float(lowest.max())))
            if arm=='source':
                for speed in ('1.0','0.7'):
                    shutil.copy2(path,destination/(path.stem.removesuffix('-source')+f'-speed{speed}-{world.shape[1]}f.npz'))
            else:shutil.copy2(path,destination/path.name)
    write_json(args.output/'support-audit.json',dict(rows=audits,note='Geometric support evidence, not force contacts; no motion changed'))
    for arm,folder in runs.items():
        if arm=='source':continue
        for name in ('complete.json','contract.json','gradient_isolation.json'):
            if (folder.parent/name).is_file():shutil.copy2(folder.parent/name,args.output/(arm+'-'+name))
    subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals','--stats',str(args.stats),
        '--reviews',*[arm+'='+str(args.output/'recordings'/arm) for arm in runs],
        '--output',str(args.output/'visuals')],check=True)
    path=args.output/'visuals/manifest.json';original=json.loads(path.read_text())
    manifest=dict(fps=30,postprocessing=None,quality_approved=False,audit=original['audit'],clips={},
        labels=dict(approved='User-approved PARC model',control='Control: retained2 updates',
                    boundaries='Every-boundary training:8 updates',source='Unmodified PARC source',
                    root='Boundaries plus published Root:8 updates',pose='Pose-support test:8 updates'))
    for name,clip in original['clips'].items():
        base=name.rsplit('-',1)[0]
        merged=manifest['clips'].setdefault(base,dict(frames=0,fps=30,arms={},scene=clip['scene']))
        merged['frames']=max(merged['frames'],clip['frames'])
        for arm,record in clip['arms'].items():
            record.update(frames=clip['frames'],fps=30,duration_seconds=clip['frames']/30)
            merged['arms'][arm]=record
    for name,clip in manifest['clips'].items():
        example=next(path for arm in runs for path in (args.output/'recordings'/arm).glob(name+'-*f.npz'))
        with np.load(example,allow_pickle=False) as archive:
            hf,origin,dx=archive['heightfield_y_m'],archive['min_point_world_xyz_m'],float(archive['dx_m'])
        roots=np.concatenate([record['roots'] for record in clip['arms'].values()])
        low,high=roots[:,[0,2]].min(0)-5,roots[:,[0,2]].max(0)+5
        x0,z0=np.maximum(0,np.floor((low-origin[[0,2]])/dx).astype(int))
        x1,z1=np.minimum([hf.shape[1],hf.shape[0]],np.ceil((high-origin[[0,2]])/dx).astype(int)+1)
        clip['scene']=dict(heightfield_y_m=[[float(v) if np.isfinite(v) else None for v in row] for row in hf[z0:z1,x0:x1]],
            origin=(origin+np.array([x0*dx,0,z0*dx])).tolist(),dx=dx)
    path.write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False))
    receipt=dict(files={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()})
    write_json(args.output/'receipt.json',receipt)
    archive=args.output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as stream:stream.add(args.output,arcname=args.output.name)
    receipt_path=archive.with_suffix('.receipt.json')
    write_json(receipt_path,dict(filename=archive.name,bytes=archive.stat().st_size,sha256=sha256(archive),files=len(receipt['files'])))
    config=configparser.ConfigParser(interpolation=None);config.read(args.drive_config)
    if config['mb-private'].get('client_id')!='GOOGLE_OAUTH_CLIENT_ID':raise ValueError('Owned client required')
    env=dict(os.environ,HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
    for file in (archive,receipt_path):
        subprocess.run(['rclone','--config',str(args.drive_config),'copyto',str(file),
            'private-store:motionbricks/parc_stair64_20261013/'+file.name,'--immutable','--checksum','--tpslimit','1'],env=env,check=True)
    print(json.dumps(dict(uploaded=True,files=len(receipt['files']))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','regression-cases','stats','output','drive-config'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--existing-runs',nargs='+')
    main(parser.parse_args())
