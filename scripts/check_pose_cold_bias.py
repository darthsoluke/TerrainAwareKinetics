"""Frozen codec bias coalescing on real cold and warm Pose-token caches."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import time


def save(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False));temporary.replace(path)


def main(a):
    snapshot=subprocess.check_output(['nvidia-smi','-i',str(a.gpu),
        '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    uuid,memory,busy=[x.strip() for x in snapshot.split(',')]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    if int(memory)>500 or int(busy)>5 or uuid in apps.splitlines():
        raise ValueError('Requested GPU is not idle')
    save(a.output/'state.json',dict(status='running',gpu=a.gpu,snapshot=snapshot,actual_optimizer_updates=0))
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from motionbricks.data.full_terrain_cache import PrimitiveDataset,collate_primitives
    from motionbricks.data.dilated_bias_convolution import coalesced_dilated_bias
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch,rng_state,restore_rng
    from scripts.train_native_mb_control import state_digest
    from scripts.train_native_mb import sha256

    torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    base=Path('${PRIVATE_STORAGE}/motionbricks-20261010');run=base/'pfnn_mixed_20261015'
    checkpoint=run/'execution_inputs_v1/root.pt'
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    inference,bundle=load_terrain_bundle(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt','cuda:0')
    controller=UnifiedTerrainController(inference,bundle);controller.load_state_dict(saved['model_state'],strict=True)
    stage=NativeTrainingStage(controller,'pose',cache_pose_tokens=True).train(True)
    initial_hash=state_digest(controller)
    dataset=PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'],'train')
    chooser=np.random.default_rng(2333)
    pools={domain:[i for i,actual in enumerate(dataset.ids) if dataset.records[actual]['domain']==domain]
        for domain in ('pfnn','parc','bones')}
    captured={};original_tokens=stage.pose_tokens
    def tokens(batch,local):
        result=original_tokens(batch,local)
        captured.update(tokens=result.detach().cpu().clone(),stride=tuple(result.stride()))
        return result
    stage.pose_tokens=tokens
    reports=[]
    try:
        for case in range(3):
            ids=[int(i) for domain,count in [('pfnn',32),('parc',16),('bones',16)]
                for i in chooser.choice(pools[domain],count,replace=False)]
            reordered=chooser.permutation(ids).tolist()
            batches=[device_batch(collate_primitives([dataset[i] for i in selected]),'cuda:0')
                for selected in (ids,reordered)]
            torch.manual_seed(2333+case);initial_rng=rng_state();reference=[]
            for accelerated in (False,True):
                stage._pose_tokens.clear();stage._pose_token_layouts.clear()
                stage.token_cache_hits=stage.token_cache_misses=0
                for phase,batch in zip(('cold','warm'),batches):
                    controller.zero_grad(set_to_none=True);restore_rng(initial_rng)
                    with coalesced_dilated_bias(controller) if accelerated else nullcontext():
                        torch.cuda.synchronize();started=time.perf_counter()
                        loss=stage(batch,2333+case);loss.backward();torch.cuda.synchronize()
                        seconds=time.perf_counter()-started
                    actual=dict(loss=float(loss.detach()),tokens=captured['tokens'],stride=captured['stride'],
                        gradients={name:p.grad.detach().cpu().clone() if p.grad is not None else None
                            for name,p in controller.named_parameters() if p.requires_grad},rng=rng_state())
                    if not accelerated:
                        reference.append(actual)
                        reports.append(dict(case=case,phase=phase,variant='native',seconds=seconds,loss=actual['loss']))
                    else:
                        expected=reference[0 if phase=='cold' else 1];errors=[]
                        if actual['loss']!=expected['loss']:errors.append('loss')
                        if actual['stride']!=expected['stride'] or not torch.equal(actual['tokens'],expected['tokens']):errors.append('token values/layout')
                        for name,value in expected['gradients'].items():
                            observed=actual['gradients'][name]
                            if value is None or observed is None:
                                if (value is None)!=(observed is None):errors.append('gradient presence:'+name)
                            elif not torch.equal(value,observed):errors.append('gradient:'+name)
                        if not torch.equal(actual['rng']['torch'],expected['rng']['torch']):errors.append('CPU RNG')
                        if not all(torch.equal(x,y) for x,y in zip(actual['rng']['cuda'],expected['rng']['cuda'])):errors.append('CUDA RNG')
                        reports.append(dict(case=case,phase=phase,variant='coalesced_frozen_codec',seconds=seconds,
                            passed=not errors,errors=errors,loss=actual['loss'],stride=list(actual['stride']),
                            cache_hits=stage.token_cache_hits,cache_misses=stage.token_cache_misses))
                    save(a.output/'progress.json',dict(rows=reports));print(json.dumps(reports[-1]),flush=True)
        if state_digest(controller)!=initial_hash:raise ValueError('Frozen-label diagnostic changed model tensors')
        save(a.output/'complete.json',dict(passed=all(row.get('passed',True) for row in reports),rows=reports,
            actual_optimizer_updates=0,quality_approved=False,production_approved=False,all_model_tensors_unchanged=True,
            checkpoint_sha256=sha256(checkpoint),script_sha256=sha256(Path(__file__)),
            implementation_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/dilated_bias_convolution.py')))
        save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=0))
    finally:controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:main(args)
    except Exception as error:
        save(args.output/'failure.json',dict(error=repr(error),actual_optimizer_updates=0));raise
