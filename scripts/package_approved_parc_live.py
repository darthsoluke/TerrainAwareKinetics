"""Export the accepted network unchanged with factual four-frame live scenes."""

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import torch

from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def state_hash(values):
    digest=hashlib.sha256()
    for name,tensor in sorted(values.items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(args):
    expected='ae558ee88542191b6cbe72659bd96f1482711ea2d8beec6c1641a0ae5b814934'
    if sha256(args.checkpoint)!=expected:raise ValueError('Accepted model identity mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    compact={k:saved[k] for k in ('schema','model_state','step','contract','coefficients','phase','exposure')}
    digest=state_hash(saved['model_state'])
    torch.save(compact,args.output/'weights.pt')
    reloaded=torch.load(args.output/'weights.pt',map_location='cpu',mmap=True,weights_only=False)
    if state_hash(reloaded['model_state'])!=digest:raise ValueError('Export altered a model tensor')
    cases=torch.load(args.cases,map_location='cpu',weights_only=False)
    preparation=json.loads((args.cases.parent/'complete.json').read_text())
    selected=[next(c for c in cases if c['label']==label) for label in (
        'generated_afe4ea1b1d92b874-up','official_dc03d6484b176db0-down')]
    entries=[]
    for case in selected:
        entry=case['entry'];control=case['exact_constant_control'][0].clone()
        if entry=='up':control[:2]*=.7
        heading=float(torch.atan2(control[3],control[2]));travel=float(torch.atan2(control[0],control[1]))
        metadata=dict(schema='parc.live_scene.v1',dataset_manifest_sha256=preparation['dataset_manifest_sha256'],
            history_frames=4,future_motion_included=False,source_id=case['row']['source_id'],
            source_start_frame=case['start_frame'])
        path=args.output/(entry+'-scene.npz')
        np.savez_compressed(path,**{k:torch.as_tensor(case['scene'][k]).numpy() for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')},
            initial_history=case['world'][None,:4].numpy(),metadata_json=np.asarray(json.dumps(metadata)))
        recommendation=dict(velocity_xz_mps=control[:2].tolist(),speed_mps=float(control[:2].norm()),
            heading_rad=heading,travel_heading_rad=travel,heading_offset_rad=float(np.arctan2(np.sin(heading-travel),np.cos(heading-travel))))
        write_json(path.with_suffix('.json'),dict(metadata,scene_sha256=sha256(path),entry_command=recommendation))
        entries.append(dict(entry=entry,label=case['label'],command=recommendation))
    # Offline references remain separate from files accepted by the live loader.
    torch.save(selected,args.output/'offline-cases.pt')
    write_json(args.output/'export.json',dict(schema='motionbricks.approved_parc_live.v1',
        original_checkpoint_sha256=expected,checkpoint_sha256=sha256(args.output/'weights.pt'),
        source_bundle_sha256=saved['contract']['source_bundle_sha256'],model_state_sha256=digest,
        model_tensors_unchanged=True,entries=entries,user_visual_approval=True,live_verified=False,
        files={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}))
    receipt=dict(files={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()})
    write_json(args.output/'receipt.json',receipt)
    archive=args.output.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as stream:stream.add(args.output,arcname=args.output.name)
    archive_receipt=archive.with_suffix('.receipt.json')
    write_json(archive_receipt,dict(filename=archive.name,bytes=archive.stat().st_size,sha256=sha256(archive),files=len(receipt['files'])))
    settings=configparser.ConfigParser(interpolation=None);settings.read(args.drive_config)
    if settings['mb-private'].get('client_id')!='GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Private owned OAuth client required')
    environment=dict(os.environ,HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
    for file in (archive,archive_receipt):
        subprocess.run(['rclone','--config',str(args.drive_config),'copyto',str(file),
            'private-store:motionbricks/parc_stair64_20261013/'+file.name,'--immutable','--checksum',
            '--transfers','1','--checkers','1','--tpslimit','1'],env=environment,check=True)
    print(json.dumps(dict(uploaded=True,model_tensors_unchanged=True,archive_sha256=sha256(archive))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','cases','output','drive-config'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
