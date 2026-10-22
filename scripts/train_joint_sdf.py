"""Resumable distributed terrain adapters, budgeted in actual primitive examples."""
import argparse
from collections import Counter
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_learning_data import TerrainLearningDataset,learning_sampling_weights,sampling_manifest
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.joint_sdf_controller import JointSDFController,SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import IndependentEpisodes,ExactPenetrationBatch
from motionbricks.data.terrain_training_execution import AsyncCheckpointWriter
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest
from scripts.train_joint_terrain_v3 import write_json


def main(args):
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if min(args.examples,args.batch,args.checkpoint_every)<1:raise ValueError('Positive budgets required')
    if args.resume is None and args.output.exists():raise ValueError('New experiment needs an unused output directory')
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.set_num_threads(4);torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if world>1:torch.distributed.init_process_group('nccl',timeout=timedelta(minutes=20))
    torch.manual_seed(2333);np.random.seed(2333);random.seed(2333)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    source=Path(saved['contract']['source']);dataset=Path(saved['contract']['dataset'])
    if sha256(source)!=saved['contract']['source_bundle_sha256']:raise ValueError('Source bundle hash differs')
    if sha256(dataset/'native_manifest.json')!=saved['contract']['dataset_manifest_sha256']:raise ValueError('Dataset hash differs')
    infer,bundle=load_terrain_bundle(source,device);model=JointSDFController(infer,bundle)
    keys=model.load_state_dict(saved['model_state'],strict=False)
    if keys.unexpected_keys or any(not k.startswith('sdf_') for k in keys.missing_keys):raise ValueError('Unexpected base mismatch')
    model.freeze_base();model.eval()
    frozen={k:state_digest(getattr(model,k)) for k in ('root','pose','codec','control','root_far','pose_far','decoder_near','decoder_attention','decoder_far')}
    data=TerrainLearningDataset(dataset,build_native_representation(dataset/'stats'),'train',max_frames=129,cache_bytes=1024**3)
    weights=learning_sampling_weights(data,0.)
    episodes=IndependentEpisodes(data,weights,args.batch,infer.global_motion_rep,device,2333+rank*100003)
    geometry=JointGeometryObjective(infer.global_motion_rep,device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=5e-5,fused=True)
    steps=math.ceil(args.examples/(args.batch*world));start_step=0
    contract=dict(saved['contract'],joint_sdf={
        'version':SDF_VERSION,'initial_checkpoint':str(args.checkpoint),'initial_sha256':sha256(args.checkpoint),
        'new_primitive_examples':args.examples,'reference_batch':32,'reference_equivalent_updates':args.examples/32,
        'global_batch':args.batch*world,'per_rank_batch':args.batch,'world_size':world,'actual_updates':steps,
        'deterministic_diagnostic':args.deterministic,
        'rounded_examples':steps*args.batch*world,'batch_rounding_excess':steps*args.batch*world-args.examples,
        'optimizer':'fresh Adam 5e-5; no claim of optimization equivalence to batch32',
        'frames':32,'publish_frames':8,'maximum_feedback_depth':13,'independent_episode_rows':True,
        'intent':'rolling SoftCommandState; real straight flat approaches retain weak cruise through terrain; otherwise source velocity/facing; no source endpoint override',
        'geometry':'exact negative prism SDF, official SOMA sparse skin; original loss coefficients',
        'sampling':sampling_manifest(data,weights),'frozen_base':frozen,'postprocessing':None,
        'code_sha256':{str(p.relative_to(Path(__file__).parents[1])):sha256(p) for p in
            [Path(__file__),*[Path(__file__).parents[1]/'motionbricks/data'/n for n in
                ('joint_sdf_controller.py','joint_sdf_observation.py','joint_sdf_training.py')]]}})
    if args.resume:
        resumed=torch.load(args.resume,map_location='cpu',weights_only=False)
        if resumed['contract']!=contract:raise ValueError('Resume contract differs; require explicit migration')
        model.load_state_dict(resumed['model_state'],strict=True);optimizer.load_state_dict(resumed['optimizer'])
        state=resumed['rank_state'][rank];episodes.load_state_dict(state['episodes'])
        torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda']);np.random.set_state(state['numpy']);random.setstate(state['python'])
        start_step=resumed['step']
    else:torch.manual_seed(2333+rank);np.random.seed(2333+rank);random.seed(2333+rank)
    wrapped=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    writer=AsyncCheckpointWriter() if rank==0 else None
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=args.resume is not None);write_json(args.output/'contract.json',contract)
    if world>1:torch.distributed.barrier()
    def checkpoint(step):
        state={'episodes':episodes.state_dict(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate()}
        gathered=[None]*world
        if world>1:torch.distributed.all_gather_object(gathered,state)
        else:gathered=[state]
        if rank==0:
            if frozen!={k:state_digest(getattr(model,k)) for k in frozen}:raise ValueError('Frozen base changed')
            writer.submit(args.output/'last.pt',{'schema':SDF_VERSION,'model_state':model.state_dict(),
                'optimizer':optimizer.state_dict(),'step':step,'contract':contract,'coefficients':saved['coefficients'],
                'rank_state':gathered,'primitive_examples':step*args.batch*world})
        if world>1:torch.distributed.barrier()
    started=time.perf_counter();times=[]
    for step in range(start_step,steps):
        began=time.perf_counter()
        source_window,physical_window,history,physical_history,intent,scenes,frame=episodes.window()
        with torch.no_grad():
            joints,points=geometry.geometry(physical_window)
        result=wrapped(history,intent,scenes,source=source_window,packed_scene=episodes.packed,
            scene_frame=frame,straight_through=True)
        fields=[model._sdf_fields[id(scene['heightfield_y_m'])][1] if id(scene['heightfield_y_m']) in model._sdf_fields
            else HeightfieldPrismSDF(scene) for scene in scenes]
        exact=ExactPenetrationBatch(scenes,device,fields,packed=episodes.packed)
        target={'joints':joints,'points':points,'heights':exact.heights,'volumes':exact}
        native=model.native_losses(result,source_window)
        physical_prediction=frame.motion(infer.global_motion_rep,result['world'],inverse=True)
        physical=geometry.prepared_losses(physical_prediction,physical_history,target,offset=0,diagnostics=False)
        loss=native['native_total']+sum(saved['coefficients'][key]*physical[key] for key in saved['coefficients'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        optimizer.step()
        with torch.no_grad():episodes.advance(reencode_played_history(physical_prediction[:,8:12].detach(),infer.global_motion_rep))
        torch.cuda.synchronize();times.append(time.perf_counter()-began)
        if step%10==0 or step+1==steps or step+1==args.stop_at:
            keys=['native_motion','native_root','token_ce']+list(saved['coefficients'])
            values=torch.tensor([float(loss),float(norm),*[float((native if key in native else physical)[key]) for key in keys]],device=device)
            if world>1:torch.distributed.all_reduce(values);values/=world
            if rank==0:
                speed=(step+1-start_step)*args.batch*world/(time.perf_counter()-started)
                row={'actual_updates':step+1,'primitive_examples':(step+1)*args.batch*world,
                    'batch32_equivalent_updates':(step+1)*args.batch*world/32,'target_examples':args.examples,
                    'loss':float(values[0]),'gradient_norm':float(values[1]),**dict(zip(keys,values[2:].tolist())),
                    'examples_per_second':speed,'seconds_per_update_median':float(np.median(times[-100:])),
                    'remaining_seconds_estimate':(steps-step-1)*args.batch*world/speed,
                    'peak_memory_mb_rank0':torch.cuda.max_memory_allocated()/1024**2,
                    'rank0_exposure':dict(episodes.counts),'sdf':model.sdf_diagnostics,'quality_approved':False}
                write_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
        if (step+1)%args.checkpoint_every==0 or step+1==steps or step+1==args.stop_at:checkpoint(step+1)
        if step+1==args.stop_at:break
    if rank==0:
        writer.close()
        write_json(args.output/'complete.json',{'actual_updates':step+1,'primitive_examples':(step+1)*args.batch*world,
            'batch32_equivalent_updates':(step+1)*args.batch*world/32,'budget_complete':step+1==steps,'quality_approved':False})
    if world>1:torch.distributed.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',type=Path);parser.add_argument('--examples',type=int,default=1600000)
    parser.add_argument('--batch',type=int,default=32);parser.add_argument('--stop-at',type=int)
    parser.add_argument('--checkpoint-every',type=int,default=100)
    parser.add_argument('--deterministic',action='store_true')
    main(parser.parse_args())
