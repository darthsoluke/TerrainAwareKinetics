"""Export recorded, uncorrected V3 rollouts for the existing official-skin viewer."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.soma_skin_reference import load_official_skinner


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stats',type=Path,required=True)
    p.add_argument('--reviews',nargs='+',required=True,help='label=review-directory')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    skin=load_official_skinner(build_native_representation(args.stats).skeleton)
    skin.skeleton_skin.float()
    asset={'positions':skin.bind_vertices.flatten().tolist(),'faces':skin.faces.flatten().tolist(),
        'indices':skin.lbs_indices.flatten().tolist(),'weights':skin.lbs_weights.flatten().tolist(),
        'influences':skin.lbs_indices.shape[-1],'inverse_bind':skin.bind_rig_transform_inv.flatten().tolist()}
    (args.output/'skin.json').write_text(json.dumps(asset,separators=(',',':')))
    manifest={'fps':30,'postprocessing':None,'clips':{},'labels':{},'audit':[]}
    for entry in args.reviews:
        label,directory=entry.split('=',1);directory=Path(directory);manifest['labels'][label]=label
        files=sorted(directory.glob('*.npz'))
        if not directory.is_dir() or not files:raise ValueError(f'Review has no recorded motion: {directory}')
        for file in files:
            with np.load(file,allow_pickle=False) as data:
                local=torch.from_numpy(data['local_rot_mats'].copy()).float()
                roots=torch.from_numpy(data['roots'].copy()).float()
                hf=data['heightfield_y_m'].copy();origin=data['min_point_world_xyz_m'].copy();dx=float(data['dx_m'])
                metadata=json.loads(str(data['metadata_json']))
            source_arm=metadata.get('role')=='unchanged_reference'
            arm='source' if source_arm else label
            name=file.stem[:-7] if source_arm and file.stem.endswith('-source') else file.stem
            if not source_arm and metadata.get('requested_frames',len(roots))>len(roots):
                # An incomplete rollout is its own shorter recording. Never
                # duplicate its last pose to match a completed reference.
                name+=f'-incomplete-{label}'
            manifest['labels'][arm]=arm
            if source_arm and 'source' in manifest['clips'].get(name,{}).get('arms',{}):
                if not np.allclose(manifest['clips'][name]['arms']['source']['roots'],roots.numpy(),atol=1e-6,rtol=0):
                    raise ValueError('Source clocks or references changed between reviews')
                continue
            rigs=[];native_lbs=skin.lbs;qa=None
            for start in range(0,len(roots),32):
                recorded=[]
                def capture(rig):
                    recorded.append(rig.clone());return rig.new_empty(0)
                skin.lbs=capture
                try:skin.skin(local[start:start+32],roots[start:start+32,None].expand(-1,30,-1),rot_is_global=False)
                finally:skin.lbs=native_lbs
                rig=recorded[0];rigs.append(rig.numpy())
                if start==0:
                    vertices=native_lbs(rig[:1])
                    official=skin.skin(local[:1],roots[:1,None].expand(-1,30,-1),rot_is_global=False)
                    torch.testing.assert_close(vertices,official,atol=1e-6,rtol=0)
                    indices=[0,100,1000,5000,10000,18000]
                    qa={'frame':0,'indices':indices,'vertices':vertices[0,indices].tolist()}
            filename=f'{name}-{arm}.rig.json'
            binary=np.concatenate(rigs).astype('<f4').tobytes()
            (args.output/filename).write_text(json.dumps({'base64':base64.b64encode(binary).decode('ascii')}))
            # Crop only display geometry. Metric audits use the original entire scene.
            lo=roots[:,[0,2]].amin(0).numpy()-5;hi=roots[:,[0,2]].amax(0).numpy()+5
            x0,z0=np.maximum(0,np.floor((lo-origin[[0,2]])/dx).astype(int))
            x1,z1=np.minimum(np.array([hf.shape[1],hf.shape[0]]),np.ceil((hi-origin[[0,2]])/dx).astype(int)+1)
            crop=hf[z0:z1,x0:x1]
            if not crop.size:raise ValueError('Recorded motion is outside display scene')
            cropped_origin=origin+np.array([x0*dx,0,z0*dx])
            scene={'heightfield_y_m':[[float(v) if np.isfinite(v) else None for v in row] for row in crop],
                   'origin':cropped_origin.tolist(),'dx':dx}
            clip=manifest['clips'].setdefault(name,{'frames':len(roots),'fps':30,'arms':{},'scene':scene})
            if clip['frames']!=len(roots):raise ValueError('Comparison rollouts have different clocks')
            # Different outputs require the union of their displayed raster crops.
            prior=clip['scene'];p0=np.rint((np.array(prior['origin'])[[0,2]]-origin[[0,2]])/dx).astype(int)
            ph,pw=len(prior['heightfield_y_m']),len(prior['heightfield_y_m'][0])
            ux,uz=np.minimum(p0,[x0,z0]);vx,vz=np.maximum(p0+[pw,ph],[x1,z1])
            union=hf[uz:vz,ux:vx]
            clip['scene']={'heightfield_y_m':[[float(v) if np.isfinite(v) else None for v in row] for row in union],
                'origin':(origin+np.array([ux*dx,0,uz*dx])).tolist(),'dx':dx}
            clip['arms'][arm]={'rig':filename,'roots':roots.tolist(),'qa':qa,'metrics':metadata}
            manifest['audit'].append({'clip':name,'arm':arm,'vertices_per_frame':len(skin.bind_vertices),
                'official_skin_roundtrip_max_error_m':float((vertices-official).abs().max())})
        (args.output/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False))
    print(json.dumps({'clips':len(manifest['clips']),'arms':list(manifest['labels'])}),flush=True)


if __name__=='__main__':main()
