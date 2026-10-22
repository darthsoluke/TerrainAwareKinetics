"""Publish completed A100 B artifact and four-frame trial scenes through private Drive."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from scripts.train_native_mb import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    args=p.parse_args();b=args.base;out=b/'a100_live_export_v1';out.mkdir(exist_ok=True)
    data=torch.load(b/'terrain_decoder_pilot_v1/data/windows.pt',map_location='cpu',weights_only=False)
    corpus=json.loads((b/'terrain_decoder_pilot_v1/data/manifest.json').read_text())
    windows=[w for w in data['windows'] if w['split']=='validation']
    flat=next(w for w in windows if w['origin']=='bones_seed' and w['category']=='idle')
    terrain=next(w for w in windows if w['origin']=='parc_generated' and w['start']==0)
    for name,item in (('flat',flat),('long-terrain',terrain)):
        path=out/(name+'.npz')
        metadata={'schema':'parc.live_scene.v1','dataset_manifest_sha256':corpus['source_dataset_manifest_sha256'],
            'history_frames':4,'future_motion_included':False,
            'window':{key:item[key] for key in ('source_id','start','source_group','split')},
            'origin':item['origin']}
        if not path.exists():
            np.savez_compressed(path,initial_history=item['world'][:,:4].numpy(),
                **{k:torch.as_tensor(v).numpy() for k,v in data['scenes'][item['scene']].items()},
                metadata_json=np.asarray(json.dumps(metadata)))
        metadata['scene_sha256']=sha256(path)
        path.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    remote='mb-transfer:runs/a100-live-20261010'
    prefix=['rclone','--config',str(b/'drive-auth/rclone.conf')]
    subprocess.run(prefix+['copy',str(out),remote,'--include','/*.npz','--include','/*.json','--immutable'],check=True)
    run=b/'long_route_mixed_10k_ab_v2/terrain_reference'
    while not (run/'pipeline.json').exists():
        state=json.loads((run/'state.json').read_text())
        if state['status']=='failed':raise RuntimeError(state.get('error','A100 training failed'))
        time.sleep(20)
    receipt=json.loads((run/'pipeline.json').read_text())
    if sha256(run/'pipeline.pt')!=receipt['sha256']:raise ValueError('Exported model hash mismatch')
    subprocess.run(prefix+['copyto',str(run/'pipeline.pt'),remote+'/a100-terrain.pt','--immutable',
                          '--drive-chunk-size','64M'],check=True)
    subprocess.run(prefix+['copyto',str(run/'pipeline.json'),remote+'/a100-terrain.json','--immutable'],check=True)
    (out/'complete.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'status':'uploaded','sha256':receipt['sha256']}),flush=True)


if __name__=='__main__':main()
