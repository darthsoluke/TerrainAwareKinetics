"""Read-only stage/context decomposition of the completed decoder pilot's drift."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_online_live import predict_planar_terrain
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_decoder_pilot import DecoderExperiment
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.run_mixed_control_workflow import available_devices


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new diagnostic artifact')
    available_devices([1])
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False
    data=torch.load(args.base/'input/data/windows.pt',map_location='cpu',weights_only=False)
    seeds=[w for w in data['windows'] if w['split']=='validation' and w['origin']=='bones_seed'
           and w['category']=='idle' and float((w['world'][:,1:,:3]-w['world'][:,:-1,:3]).norm(dim=-1).max()*30)<.08]
    seed=seeds[0];report={'training_started':False,'output_modified':False,'seed':seed['source_id'],'arms':{}}
    for arm in ('source','both'):
        infer,bundle=load_terrain_bundle(args.base/'input/source.pt','cuda:1')
        pilot=DecoderExperiment(infer,bundle,terrain=arm=='both')
        if arm=='both':
            ckpt=torch.load(args.base/'runs/both/decoder.pt',map_location='cuda:1',weights_only=False)
            pilot.codec.decoder.load_state_dict(ckpt['decoder'],strict=True)
            pilot.adapter.load_state_dict(ckpt['adapter'],strict=True)
        initial=seed['world'][:,:4].to(infer.device);heading=initial[:,-1,3:5]
        goal=torch.cat((torch.zeros_like(heading),heading),-1)
        scene=dict(heightfield_y_m=torch.zeros(401,401,device=infer.device),
            min_point_world_xyz_m=torch.tensor([float(initial[0,3,0])-40,0.,float(initial[0,3,2])-40],device=infer.device),dx_m=.2)
        captures={}
        original=pilot.codec.forward_decoder
        def decoder(*values,**options):
            captures['external']=options['external_cond'].clone()
            result=original(*values,**options);captures['decoded']=result['recon_state'].clone()
            return result
        pilot.codec.forward_decoder=decoder
        cases=[]
        for kind in ('teacher_history','played_history'):
            history=initial.clone();rows=[]
            for block in range(113 if kind=='played_history' else 4):
                if kind=='teacher_history':history=seed['world'][:,block*8:block*8+4].to(infer.device)
                debug={}
                predicted,_=predict_planar_terrain(infer,history,goal,scene,num_tokens=8,
                    config={'num_inference_step':4,'pose_token_sampling_use_argmax':True},diagnostics=debug)
                root=torch.tensor(debug['root_stage_world_xyz_m'][:8],device=infer.device)
                context_error=predicted[0,3,:3]-history[0,3,:3]
                decoded_future_delta=predicted[0,11,:3]-predicted[0,3,:3]
                published_delta=predicted[0,11,:3]-history[0,3,:3]
                torch.testing.assert_close(context_error+decoded_future_delta,published_delta,atol=1e-6,rtol=0)
                extidx=infer.local_motion_rep.indices['local_root_vel']
                external=infer.local_motion_rep.stats.unnormalize(captures['external'],index=extidx)[0]
                rows.append({'block':block,'input_y_m':float(history[0,3,1]),
                    'root_stage_step_xyz_m':(root[-1]-history[0,3,:3]).cpu().tolist(),
                    'context_reconstruction_xyz_m':context_error.cpu().tolist(),
                    'decoded_future_xyz_m':decoded_future_delta.cpu().tolist(),
                    'published_step_xyz_m':published_delta.cpu().tolist(),
                    'decoder_minus_root_step_xyz_m':(predicted[0,11,:3]-root[-1]).cpu().tolist(),
                    'root_condition_prefix_speed_mps':float(external[3:11].norm(dim=-1).mean()),
                    'root_condition_terminal_speed_mps':float(external[31].norm()),
                    'first_emitted_delta_xyz_m':(predicted[0,4,:3]-history[0,3,:3]).cpu().tolist()})
                history=reencode_played_history(predicted[:,8:12],infer.global_motion_rep)
            result={'history':kind,'blocks':rows}
            for key in ('context_reconstruction_xyz_m','decoded_future_xyz_m','published_step_xyz_m',
                        'root_stage_step_xyz_m','decoder_minus_root_step_xyz_m'):
                values=np.asarray([r[key] for r in rows]);result[key+'_sum']=values.sum(0).tolist()
                result[key+'_mean_abs']=np.abs(values).mean(0).tolist()
            result['prefix_speed_mean_mps']=float(np.mean([r['root_condition_prefix_speed_mps'] for r in rows]))
            result['terminal_speed_max_mps']=max(r['root_condition_terminal_speed_mps'] for r in rows)
            cases.append(result)
        interventions=[]
        for dy in (-.1,0.,.1):
            history=shift_global_motion(initial,infer.global_motion_rep,dy)
            debug={};prediction,_=predict_planar_terrain(infer,history,goal,scene,num_tokens=8,
                config={'num_inference_step':4,'pose_token_sampling_use_argmax':True},diagnostics=debug)
            interventions.append({'body_shift_y_m':dy,'root_future_y_mean_m':float(np.asarray(debug['root_stage_world_xyz_m'])[:8,1].mean()),
                'decoded_future_y_mean_m':float(prediction[0,4:12,1].mean()),
                'decoded_one_step_y_m':float(prediction[0,11,1])})
        report['arms'][arm]={'cases':cases,'body_height_intervention':interventions}
        pilot.codec.forward_decoder=original;pilot.close();del pilot,infer,bundle
        torch.cuda.empty_cache()
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({'arm':arm,'cases':[{k:v for k,v in r.items() if k!='blocks'} for r in cases],
                          'body_height_intervention':interventions}),flush=True)


if __name__=='__main__':main()
