"""Real large-batch Pose Adam continuation with a cold ephemeral label cache."""

import argparse
import os
from pathlib import Path
import time

from scripts.run_pfnn_bias_workflow import save,digest
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(a):
    snapshot,available=gpu_snapshot()
    if a.gpu not in available:raise ValueError('Selected GPU is no longer idle')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import random
    import numpy as np
    import torch
    from motionbricks.data.exposure_codec_training import original_order_indices
    from motionbricks.data.full_terrain_cache import PrimitiveDataset,BalancedPrimitiveBatches,collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage,prepare_native_batch
    from motionbricks.data.dilated_bias_convolution import coalesced_dilated_bias
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch,restore_rng,rng_state,validation
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.9,0);torch.cuda.set_device(0)
    torch.manual_seed(2333);np.random.seed(2333);random.seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    run=a.checkpoint.parent.parent
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    inference,bundle=load_terrain_bundle(run.parent/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt','cuda:0')
    controller=UnifiedTerrainController(inference,bundle)
    controller.load_state_dict(saved['model_state'],strict=True)
    controller.codebook.copy_(controller.codec.get_codebook())
    controller.pose.init_embedding_from_codebooks(controller.codec.get_codebook(),None)
    stage=NativeTrainingStage(controller,'pose',cache_pose_tokens=True,coalesce_dilated_bias=True).train(True)
    parameters=[p for p in controller.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=5e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50000,eta_min=5e-7)
    dataset=PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'],'train')
    heldout=PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'],'validation')
    schedule=BalancedPrimitiveBatches(dataset,256,50000,seed=2333)
    reference_units=a.batch//256
    batches=[collate_primitives([dataset[i] for i in original_order_indices(schedule,dataset,index*reference_units,a.batch)])
             for index in range(4)]

    def cpu(value):
        if torch.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
        return value

    def state(step):
        return dict(model=cpu(controller.state_dict()),optimizer=cpu(optimizer.state_dict()),
            scheduler=cpu(scheduler.state_dict()),rng=cpu(rng_state()),actual_updates=step,
            actual_exposures=step*a.batch,equivalent_reference_updates=step*reference_units)

    def differences(left,right,path=''):
        if torch.is_tensor(left):
            return [] if torch.equal(left,right) else [dict(path=path,max_abs=float((left-right).abs().max()))]
        if isinstance(left,np.ndarray):return [] if np.array_equal(left,right) else [dict(path=path)]
        if isinstance(left,dict):
            if left.keys()!=right.keys():raise ValueError('Checkpoint inventory differs')
            return [item for key in left for item in differences(left[key],right[key],path+'.'+str(key))]
        if isinstance(left,(list,tuple)):
            if len(left)!=len(right):raise ValueError('Checkpoint sequence length differs')
            return [item for index,(x,y) in enumerate(zip(left,right)) for item in differences(x,y,path+'.'+str(index))]
        return [] if left==right else [dict(path=path)]

    initial=state(0)
    frozen={name:value for name,value in initial['model'].items()
            if name not in {name for name,p in controller.named_parameters() if p.requires_grad}}
    first=0
    if a.reference:
        midpoint=torch.load(a.reference/'midpoint.pt',map_location='cpu',weights_only=False)
        controller.load_state_dict(midpoint['model'],strict=True)
        optimizer.load_state_dict(midpoint['optimizer']);scheduler.load_state_dict(midpoint['scheduler'])
        restore_rng(midpoint['rng']);first=midpoint['actual_updates']
        if first!=2 or stage._pose_tokens:raise ValueError('Resume must begin with two real updates and an empty label cache')
    else:
        # Prime the continuous arm, so cold recovery must reproduce warm labels.
        before=rng_state()
        with torch.no_grad(),coalesced_dilated_bias(controller):
            for raw in batches:
                batch=device_batch(raw,'cuda:0')
                prepared=prepare_native_batch(controller,batch['world'],batch['observation'])
                stage.pose_tokens(batch,prepared['codec_target']['local_motions'])
        restore_rng(before)
        if differences(initial['model'],cpu(controller.state_dict())):
            raise ValueError('Cache priming changed model tensors')
        del batch,prepared
    rows=[];computation_changes=[];validation_rows=None
    try:
        for index in range(first,4):
            batch=device_batch(batches[index],'cuda:0');controller.zero_grad(set_to_none=True)
            torch.cuda.synchronize();started=time.perf_counter()
            loss=stage(batch,2333+index*reference_units*103);loss.backward();torch.cuda.synchronize()
            if not torch.isfinite(loss):raise ValueError('Nonfinite Pose loss')
            observed=dict(loss=float(loss),logits=cpu(controller.last_logits),gradients={name:cpu(p.grad) for name,p in controller.named_parameters() if p.requires_grad},
                rng=cpu(rng_state()),sample_ids=batch['sample_ids'].tolist())
            if a.reference:
                computation_changes.append(differences(torch.load(a.reference/f'observed_{index}.pt',weights_only=False),observed))
            else:torch.save(observed,a.output/f'observed_{index}.pt')
            norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
            if not torch.isfinite(norm) or norm<=0:raise ValueError('Nonfinite or zero Pose gradient')
            optimizer.step();a.completed_updates+=1;scheduler.step((index+1)*reference_units)
            torch.cuda.synchronize()
            rows.append(dict(seconds=time.perf_counter()-started,loss=float(loss),gradient_norm=float(norm),
                actual_step=index+1,reference_step=(index+1)*reference_units,
                cache_hits=stage.token_cache_hits,cache_misses=stage.token_cache_misses))
            save(a.output/'state.json',dict(status='running',actual_optimizer_updates=a.completed_updates,
                production_budget_updates=0,last=rows[-1]))
            if index==1 and not a.reference:torch.save(state(2),a.output/'midpoint.pt')
            if index==2:validation_rows=validation(stage,heldout,64)
        final=state(4)
        frozen_changes=differences(frozen,{name:final['model'][name] for name in frozen})
        endpoint_changes=differences(torch.load(a.reference/'endpoint.pt',weights_only=False),final) if a.reference else []
        if not a.reference:torch.save(final,a.output/'endpoint.pt')
        save(a.output/'complete.json',dict(passed=not frozen_changes and not endpoint_changes and not any(computation_changes),
            reference_arm=a.reference is None,cold_process_resume=a.reference is not None,cache_persisted=False,
            rows=rows,computation_changes=computation_changes,endpoint_changes=endpoint_changes,
            validation_between=validation_rows,
            frozen_changes=frozen_changes,global_batch=a.batch,compute_micro=a.batch,memory_fraction=.9,
            actual_optimizer_updates=a.completed_updates,diagnostic_endpoint_updates=4,
            diagnostic_endpoint_exposures=4*a.batch,diagnostic_endpoint_reference_steps=4*reference_units,
            production_budget_updates=0,provisional_codec=True,final_pose_labels_built=False,
            checkpoint_sha256=digest(a.checkpoint),script_sha256=digest(Path(__file__)),
            reference_report_sha256=digest(a.reference/'complete.json') if a.reference else None,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            gpu=a.gpu,gpu_snapshot_before=snapshot,production_approved=False,quality_approved=False))
        save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=a.completed_updates))
    finally:controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,choices=(1024,3072),default=3072)
    parser.add_argument('--reference',type=Path)
    args=parser.parse_args();args.completed_updates=0;args.output.mkdir(parents=True,exist_ok=False)
    try:main(args)
    except Exception as error:
        save(args.output/'failure.json',dict(error=repr(error),actual_optimizer_updates=args.completed_updates,production_budget_updates=0))
        raise
