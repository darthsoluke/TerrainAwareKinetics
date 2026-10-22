"""Package the existing official PARC runtime and send it through private Drive."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''): value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-upload',action='store_true')
    parser.add_argument('--after-pid',type=int)
    args=parser.parse_args()
    host=Path('${PRIVATE_STORAGE}/projects/motionbricks-parc-physics-20261009')
    nas=Path('${PRIVATE_STORAGE}/motionbricks-parc-physics-20261009')
    if args.after_pid:
        while (Path('/proc')/str(args.after_pid)/'cmdline').exists():
            command=(Path('/proc')/str(args.after_pid)/'cmdline').read_bytes()
            if b'stage_parc_runtime_drive.py' not in command:break
            time.sleep(30)
    if args.resume_upload:
        upload(args.output)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(nas/'pack_tools_20261010'))
    import conda_pack
    archive=args.output/'parc-runtime-env.tar.gz'
    conda_pack.pack(prefix=str(host/'env'),output=str(archive),ignore_editable_packages=True,
                    compress_level=1,n_threads=4)
    assets=args.output/'parc-runtime-assets.tar.gz'
    def keep(info):
        if '__pycache__' in info.name or info.name.endswith('user_config.yaml'): return None
        return info
    with tarfile.open(assets,'w:gz',compresslevel=1) as stream:
        for source,target in ((host/'wide_v1','wide_v1'),(host/'sdk/isaacgym/python','sdk/isaacgym/python'),
            (nas/'inputs_v1/PARC','inputs_v1/PARC'),(nas/'inputs_v1/tracker','inputs_v1/tracker'),
            (host/'run_parc_physics_record.py','run_parc_physics_record.py'),
            (host/'export_parc_physics_recordings.py','export_parc_physics_recordings.py')):
            stream.add(source,arcname=target,filter=keep)
    receipt={'files':[{'name':p.name,'sha256':digest(p),'bytes':p.stat().st_size} for p in (archive,assets)],
             'transport':'private Google Drive','runtime':'official iteration3 generator and tracker; existing physics environment'}
    (args.output/'runtime.json').write_text(json.dumps(receipt,indent=2))
    upload(args.output)


def upload(output):
    receipt=json.loads((output/'runtime.json').read_text())
    files=[output/row['name'] for row in receipt['files']]
    for path,row in zip(files,receipt['files']):
        if path.parent!=output or digest(path)!=row['sha256']:raise ValueError('Runtime package changed')
    env=dict(os.environ)
    for key in ('ALL_PROXY','all_proxy'): env.pop(key,None)
    for key in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): env[key]='http://127.0.0.1:27890'
    config='${PRIVATE_STORAGE}/projects/motionbricks-parc-training-20261009/drive-auth-20261010/rclone.conf'
    for path in [*files,output/'runtime.json']:
        print('uploading',path.name,path.stat().st_size,flush=True)
        while True:
            result=subprocess.run(['rclone','--config',config,'copyto',str(path),
                'mb-transfer:environments/parc-20261010/'+path.name,'--immutable',
                '--drive-chunk-size','64M','--tpslimit','1','--retries','1'],env=env)
            status={'file':path.name,'status':'uploaded' if result.returncode==0 else 'retrying',
                    'last_exit_code':result.returncode,'updated_at':time.time()}
            (output/'upload_status.json').write_text(json.dumps(status,indent=2))
            if result.returncode==0:break
            time.sleep(30)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
