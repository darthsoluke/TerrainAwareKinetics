"""Measure large-batch frozen-token Pose execution without optimizer updates."""

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
    import torch
    from motionbricks.data.exposure_codec_training import original_order_indices
    from motionbricks.data.full_terrain_cache import PrimitiveDataset,BalancedPrimitiveBatches,collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch,restore_rng,rng_state
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.9,0);torch.cuda.set_device(0)
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
    initial={name:value.detach().cpu().clone() for name,value in controller.state_dict().items()}
    dataset=PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'],'train')
    schedule=BalancedPrimitiveBatches(dataset,256,50000,seed=saved['contract']['seed'])
    ids=original_order_indices(schedule,dataset,saved['actual_updates'],a.micro_batch)
    batch=device_batch(collate_primitives([dataset[i] for i in ids]),'cuda:0')
    rows=[];baseline=None;changes=[];baseline_rng=None
    try:
        for repeat in range(4):
            controller.load_state_dict(initial,strict=True);controller.zero_grad(set_to_none=True)
            restore_rng(saved['rng_state']);torch.cuda.synchronize();started=time.perf_counter()
            loss=stage(batch,2333);loss.backward();torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            gradients={name:p.grad.detach().cpu().clone() for name,p in controller.named_parameters() if p.requires_grad and p.grad is not None}
            current_rng=rng_state()
            if baseline is None:baseline=gradients;baseline_rng=current_rng
            else:
                changes.extend(dict(repeat=repeat,name=name,max_abs=float((value-baseline[name]).abs().max()))
                               for name,value in gradients.items() if not torch.equal(value,baseline[name]))
                if not torch.equal(current_rng['torch'],baseline_rng['torch']) or any(
                        not torch.equal(x,y) for x,y in zip(current_rng['cuda'],baseline_rng['cuda'])):
                    raise ValueError('Cold and warm Pose RNG differs')
            if not torch.isfinite(loss):raise ValueError('Nonfinite Pose loss')
            rows.append(dict(seconds=elapsed,samples_per_second=a.micro_batch/elapsed,loss=float(loss),
                             cache_hits=stage.token_cache_hits,cache_misses=stage.token_cache_misses))
            save(a.output/'state.json',dict(status='running',repeat=repeat+1,actual_optimizer_updates=0))
        unchanged=all(torch.equal(value.cpu(),initial[name]) for name,value in controller.state_dict().items())
        save(a.output/'complete.json',dict(passed=unchanged and not changes and all(r['loss']==rows[0]['loss'] for r in rows),
            rows=rows,gradient_changes=changes,model_unchanged=unchanged,micro_batch=a.micro_batch,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            memory_fraction=.9,gpu=a.gpu,gpu_snapshot_before=snapshot,checkpoint_sha256=digest(a.checkpoint),
            provisional_codec_performance_only=True,final_pose_labels_built=False,actual_optimizer_updates=0,
            production_budget_updates=0,production_approved=False,quality_approved=False))
        save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=0))
    finally:controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--micro-batch',type=int,default=1024)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:main(args)
    except Exception as error:
        save(args.output/'failure.json',dict(error=repr(error),actual_optimizer_updates=0,production_budget_updates=0))
        raise
