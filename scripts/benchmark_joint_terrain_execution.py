"""Bounded real joint-training throughput; diagnostic weights are never deployed."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset,learning_sampling_weights
from motionbricks.data.terrain_learning_episode import TerrainLearningEpisode
from motionbricks.data.terrain_model_v3 import JointTerrainController
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import intents,geometry_precision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','dataset','checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--tokens',type=int,default=8)
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--optimized-geometry',action='store_true')
    p.add_argument('--verify-objective',action='store_true')
    args=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    torch.manual_seed(2333);np.random.seed(2333)
    inference,bundle=load_terrain_bundle(args.source,'cuda:0')
    model=JointTerrainController(inference,bundle);model.set_learning_phase(False)
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(saved['model_state'],strict=True);coeff=saved['coefficients']
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-5)
    optimizer.load_state_dict(saved['optimizer']);del saved
    data=TerrainLearningDataset(args.dataset,build_native_representation(args.dataset/'stats'),'train',max_frames=257,cache_bytes=256*1024**2)
    draws=torch.multinomial(learning_sampling_weights(data),args.batch,replacement=True,generator=torch.Generator().manual_seed(7391))
    items=[data[(int(index),100000+i)] for i,index in enumerate(draws)]
    frames=args.tokens*4
    with torch.no_grad(),geometry_precision():
        episode=TerrainLearningEpisode(items,[0]*len(items),frames,1,inference.global_motion_rep,'cuda:0',yaw=np.random.uniform(0,2*np.pi,len(items)))
    geometry=JointGeometryObjective(inference.global_motion_rep,'cuda:0')
    with geometry_precision():prepared=geometry.prepare_episode(episode) if args.optimized_geometry else None
    measures=[];equivalence=None;torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps+2):
        torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(),geometry_precision():window,physical_window,physical_history,_=episode.window(True)
        task=intents(episode.model_items,window,frames,'cuda:0',episode.history,episode.command_state)
        torch.cuda.synchronize();conditioned=time.perf_counter()
        result=model(episode.history,task,episode.scenes,source=window,num_tokens=args.tokens,straight_through=True,
            scene_frame=episode.frame,packed_scene=episode.packed_scene)
        torch.cuda.synchronize();forward=time.perf_counter()
        with geometry_precision():
            native=model.native_losses(result,window)
            prediction=episode.frame.motion(inference.global_motion_rep,result['world'],inverse=True)
            if prepared is None:
                physical=geometry.losses(prediction,physical_window,physical_history,episode.scenes,episode.identities,volume_source=episode.physical_source)
            else:
                physical=geometry.prepared_losses(prediction,physical_history,prepared,offset=0,diagnostics=True)
        loss=native['native_total']+sum(coeff[key]*physical[key] for key in coeff)
        if step==0 and args.verify_objective:
            with geometry_precision():
                reference=geometry.losses(prediction,physical_window,physical_history,episode.scenes,episode.identities,volume_source=episode.physical_source)
                for key in coeff:torch.testing.assert_close(physical[key],reference[key],atol=2e-6,rtol=2e-5)
                a=torch.autograd.grad(sum(coeff[k]*physical[k] for k in coeff),prediction,retain_graph=True)[0]
                b=torch.autograd.grad(sum(coeff[k]*reference[k] for k in coeff),prediction,retain_graph=True)[0]
                torch.testing.assert_close(a,b,atol=2e-5,rtol=1e-4)
                equivalence={'loss_max_abs':max(float((physical[k]-reference[k]).abs()) for k in coeff),'gradient_max_abs':float((a-b).abs().max())}
        torch.cuda.synchronize();objective=time.perf_counter()
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite benchmark')
        optimizer.step();torch.cuda.synchronize();end=time.perf_counter()
        if step>=2:measures.append({'condition':conditioned-start,'forward':forward-conditioned,'objective':objective-forward,'backward_adam':end-objective,'total':end-start})
    median={key:float(np.median([r[key] for r in measures])) for key in measures[0]}
    report={'batch':args.batch,'tokens':args.tokens,'optimized_geometry':args.optimized_geometry,'timing_seconds':median,
        'samples_per_second':args.batch/median['total'],'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
        'loss':float(loss.detach()),'gradient_norm':float(norm),'objective_equivalence':equivalence,'measured_steps':args.steps,'model_quality_evaluation':False,
        'scope':'same real mixed batch; actual forward, native and physical objectives, backward, clipping and Adam; excludes cold loading and checkpoint I/O'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
