"""Export unmodified pilot motion with official SOMA rig matrices and metric audits."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from motionbricks.data.terrain_decoder_pilot import DecoderExperiment
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.run_mixed_control_workflow import available_devices
from scripts.run_terrain_decoder_pilot import geometry
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.sole_region_contact import sole_region_indices


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():p.error('Use a new visualization export')
    available_devices([0])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True)
    data = torch.load(args.base/'input/data/windows.pt', map_location='cpu', weights_only=False)
    validation = [w for w in data['windows'] if w['split']=='validation']
    manifest = {'fps':30, 'postprocessing':None, 'clips':{}, 'audit':[],
        'labels':{'source':'Original model','baseline':'Baseline finetune','feedback':'Generated history',
                  'terrain':'Terrain decoder','both':'Terrain + history','reference':'Dataset motion'}}
    skin = None
    for arm in ('source','baseline','feedback','terrain','both'):
        inference, bundle = load_terrain_bundle(args.base/'input/source.pt', 'cuda:0')
        pilot = DecoderExperiment(inference, bundle, terrain=arm in ('terrain','both'))
        if arm != 'source':
            checkpoint = torch.load(args.base/'runs'/arm/'decoder.pt', map_location='cuda:0', weights_only=False)
            pilot.codec.decoder.load_state_dict(checkpoint['decoder'],strict=True)
            pilot.adapter.load_state_dict(checkpoint['adapter'],strict=True)
        rep = inference.global_motion_rep
        if skin is None:
            skin = move_skinner(load_official_skinner(rep.skeleton), torch.device('cuda:0'))
            skin.skeleton_skin.float().to('cuda:0')
            asset = {'positions':skin.bind_vertices.cpu().flatten().tolist(),
                'faces':skin.faces.cpu().flatten().tolist(), 'indices':skin.lbs_indices.cpu().flatten().tolist(),
                'weights':skin.lbs_weights.cpu().flatten().tolist(), 'influences':skin.lbs_indices.shape[-1],
                'inverse_bind':skin.bind_rig_transform_inv.cpu().flatten().tolist()}
            (args.output/'skin.json').write_text(json.dumps(asset,separators=(',',':')))
            surfaces, _, _, _ = official_sole_surfaces(rep.skeleton)
            left, right = [sole_region_indices(surfaces, side) for side in ('Left','Right')]
            indices = np.concatenate((left,right))

        def write_motion(name, local, roots, scene, model_name=arm):
            rig_parts=[];gaps=[];qa=[]
            native_lbs = skin.lbs
            for start in range(0,len(roots),64):
                rotations=torch.as_tensor(local[start:start+64],device='cuda:0',dtype=torch.float32)
                root=torch.as_tensor(roots[start:start+64],device='cuda:0',dtype=torch.float32)
                captured=[]
                def capture(rig):
                    captured.append(rig.detach().clone())
                    return rig.new_empty(0)
                skin.lbs=capture
                try:skin.skin(rotations,root[:,None].expand(-1,30,-1),rot_is_global=False)
                finally:skin.lbs=native_lbs
                rig=captured[0];rig_parts.append(rig.cpu().numpy())
                vertices=native_lbs(rig)
                if start==0:
                    direct=skin.skin(rotations[:1],root[:1,None].expand(-1,30,-1),rot_is_global=False)
                    torch.testing.assert_close(vertices[:1],direct,atol=1e-6,rtol=0)
                    sample=sorted(set([0,100,1000,*left[::20].tolist(),*right[::20].tolist()]))
                    qa={'frame':0,'indices':sample,'vertices':vertices[0,sample].cpu().tolist()}
                if name in ('rest','start_stop'):
                    gaps.append(torch.stack([vertices[:,idx,1].amin(-1) for idx in (left,right)],-1).cpu().numpy())
            rig=np.concatenate(rig_parts).astype('<f4')
            filename=f'{name}-{model_name}.rig.json'
            (args.output/filename).write_text(json.dumps({'base64':base64.b64encode(rig.tobytes()).decode('ascii')}))
            clip=manifest['clips'].setdefault(name,{'frames':len(roots),'fps':30,'arms':{},
                'scene':{'heightfield_y_m':torch.as_tensor(scene['heightfield_y_m']).cpu().tolist(),
                         'origin':torch.as_tensor(scene['min_point_world_xyz_m']).cpu().tolist(), 'dx':float(scene['dx_m'])}})
            clip['arms'][model_name]={'rig':filename,'roots':np.asarray(roots).tolist(),'qa':qa}
            return np.concatenate(gaps) if gaps else None

        review=(args.base/'runs/baseline/initial_review' if arm=='source' else args.base/'runs'/arm/'review')
        stats=json.loads((review/'metrics.json').read_text())
        for index,name in ((0,'flat_short'),(27,'terrain_short')):
            with np.load(review/f'window-{index:02d}-played_history.npz') as file:
                scene={'heightfield_y_m':file['hf'],'min_point_world_xyz_m':file['min_point'],'dx_m':file['dx']}
                write_motion(name,file['local_prediction'],file['roots_prediction'],scene)
                if arm=='source':write_motion(name,file['local_source'],file['roots_source'],scene,'reference')
                measured=float(np.linalg.norm(file['prediction']-file['source'],axis=-1).mean())
                row=next(r for r in stats['rows'] if r['window']==index and r['history']=='played_history')
                if abs(measured-row['fk_mpjpe_m'])>1e-5:raise ValueError('FK metric does not match saved motion')
                manifest['audit'].append({'arm':arm,'clip':name,'fk_metric_recompute_error_m':abs(measured-row['fk_mpjpe_m'])})
        for item in validation:
            if item['origin']!='bones_seed' or item['category']!='idle':continue
            world=item['world'].to('cuda:0')
            if float((world[:,1:,:3]-world[:,:-1,:3]).norm(dim=-1).max()*30)>.08:continue
            break
        else:raise ValueError('No matching quiet evaluation seed')
        scene=dict(heightfield_y_m=torch.zeros(401,401),
            min_point_world_xyz_m=torch.tensor([float(world[0,3,0])-40,0.,float(world[0,3,2])-40]),dx_m=.2)
        for name in ('rest','start_stop'):
            history=world[:,:4];parts=[];travel=15 if name=='start_stop' else 0
            for block in range(travel+113):
                heading=world[0,3,3:5];speed=1.6 if block<travel else 0.
                goal=torch.stack((heading[1]*speed,heading[0]*speed,heading[0],heading[1]))[None]
                prediction,_=pilot.request(history,goal,scene);parts.append(prediction[:,4:12])
                history=reencode_played_history(prediction[:,8:12],rep)
            motion=torch.cat(parts,1)
            pose=rep.inverse(motion,False,return_all=True,joint_positions_from='global_rot_data')
            roots=motion[0,:,:3].cpu().numpy();local=pose['local_joint_rots'][0].cpu().numpy()
            with np.load(review/(name+'.npz')) as stored:
                replay_error=float(np.max(np.abs(roots-stored['roots'])))
                if replay_error>2e-5:raise ValueError('Replay differs from measured trajectory')
                previous_gap=stored['sole_gaps_m'].copy()
            gaps=write_motion(name,local,roots,scene)
            start=travel*8+60
            skin_error=float(np.max(np.abs(gaps[start:]-previous_gap)))
            if skin_error>2e-5:raise ValueError('Measured sole region differs from official rendered skin')
            drift=float(np.linalg.norm(roots[start:,[0,2]]-roots[start:start+1,[0,2]],axis=-1).max())
            row=next(r for r in stats['rest'] if r['behavior']==name)
            if abs(drift-row['drift_max_m'])>1e-5:raise ValueError('Drift recomputation mismatch')
            manifest['audit'].append({'arm':arm,'clip':name,'replay_max_error_m':replay_error,
                'official_skin_gap_max_error_m':skin_error,'drift_recomputed_m':drift})
            np.savez_compressed(args.output/f'{name}-{arm}-audit.npz',roots=roots,sole_gaps_m=gaps)
        pilot.close();del pilot,inference,bundle
        torch.cuda.empty_cache()
        (args.output/'manifest.json').write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False))
        print('exported',arm,flush=True)


if __name__=='__main__':main()
