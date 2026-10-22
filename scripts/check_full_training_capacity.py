"""Production batch gradient/memory timing without optimizer updates or exports."""

import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from motionbricks.data.full_terrain_cache import PrimitiveDataset, collate_primitives
from motionbricks.data.unified_full_training import NativeTrainingStage
from motionbricks.data.unified_training_goals import source_terminal_goal
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.short_sequence_convolution import short_sequence_convolutions
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_unified_full_corpus import device_batch, split_micro
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest
from scripts.train_joint_terrain_v3 import write_json


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    inference,bundle=load_terrain_bundle(a.bundle,'cuda:0')
    controller=UnifiedTerrainController(inference,bundle)
    saved=torch.load(a.initial,map_location='cpu',weights_only=False)
    controller.load_state_dict(saved['model_state'],strict=True)
    baseline_hash=state_digest(controller)
    cases=torch.load(a.cases,map_location='cpu',weights_only=False)
    parity=[]
    for entry in ('up','down','idle'):
        case=next(row for row in cases if row.get('entry')==entry or row.get('name')==entry)
        world=case['world'][None,:33].cuda()
        command=source_terminal_goal(world[...,:5])
        controller.eval().requires_grad_(False)
        with torch.no_grad():
            torch.use_deterministic_algorithms(False)
            reference=controller(world[:,:4],command,case['scene'])
            reference={key:reference[key].detach().clone() for key in ('world','pose_logits')}
            torch.use_deterministic_algorithms(True,warn_only=True)
            with short_sequence_convolutions(controller) if a.short_conv else nullcontext():
                actual=controller(world[:,:4],command,case['scene'])
        token_changes=int((reference['pose_logits'][:,:8].argmax(-1)!=actual['pose_logits'][:,:8].argmax(-1)).sum())
        difference=float((reference['world']-actual['world']).abs().max())
        torch.testing.assert_close(actual['world'],reference['world'],atol=5e-5,rtol=1e-5)
        if token_changes:
            raise ValueError('Deterministic pooling changed active runtime tokens')
        parity.append(dict(entry=entry,world_feature_max=difference,active_token_changes=token_changes))
    train=PrimitiveDataset(a.mixed/'data','train')
    fixture=PrimitiveDataset(a.fixture/'data','train')
    pfnn=[i for i,actual in enumerate(fixture.ids) if fixture.records[actual]['domain']=='pfnn']
    groups={domain:[i for i,actual in enumerate(train.ids) if train.records[actual]['domain']==domain]
            for domain in ('parc','bones')}
    rng=np.random.default_rng(2333)
    batches=[]
    for repeat in range(3):
        rows=[]
        for micro in range(4):
            rows.extend(fixture[pfnn[i%len(pfnn)]] for i in range(32))
            for domain in ('parc','bones'):
                rows.extend(train[int(i)] for i in rng.choice(groups[domain],16,replace=False))
        batches.append(device_batch(collate_primitives(rows),'cuda:0'))
    results=[]
    for stage_name in ('root','codec','pose'):
        controller.load_state_dict(saved['model_state'],strict=True)
        controller.zero_grad(set_to_none=True)
        torch.use_deterministic_algorithms(True,warn_only=stage_name=='codec')
        stage=NativeTrainingStage(controller,stage_name).train(True)
        parameters=[p for p in controller.parameters() if p.requires_grad]
        times=[]
        torch.cuda.reset_peak_memory_stats()
        for repeat,batch in enumerate(batches):
            controller.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started=time.perf_counter()
            total=0.
            for first in range(0,256,64):
                with short_sequence_convolutions(controller) if a.short_conv else nullcontext():
                    loss=stage(split_micro(batch,first,first+64),2333+repeat*103+first)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite '+stage_name+' production-batch loss')
                (loss/4).backward()
                total+=float(loss.detach())/4
            norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
            if not torch.isfinite(norm) or norm<=0:
                raise ValueError('Invalid production-batch gradient')
            if any(p.grad is not None for p in controller.parameters() if not p.requires_grad):
                raise ValueError('Frozen parameter acquired a gradient')
            torch.cuda.synchronize()
            times.append(time.perf_counter()-started)
        row=dict(stage=stage_name,batch=256,micro_batch=64,
            forward_backward_clip_seconds=times,warm_mean_seconds=float(np.mean(times[1:])),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            estimated_adam_moments_gib=2*sum(p.numel()*p.element_size() for p in parameters)/1024**3,
            final_loss=total,final_gradient_norm=float(norm),actual_optimizer_updates=0)
        results.append(row)
        write_json(a.output/'progress.json',dict(stages=results,runtime_pool_parity=parity))
        controller.zero_grad(set_to_none=True)
        del loss,stage,parameters
        gc.collect()
        torch.cuda.empty_cache()
    controller.load_state_dict(saved['model_state'],strict=True)
    controller.eval().requires_grad_(False)
    if state_digest(controller)!=baseline_hash:
        raise ValueError('Capacity probe failed to restore initial tensors')
    write_json(a.output/'complete.json',dict(passed=True,stages=results,runtime_pool_parity=parity,
        initial_restored=True,actual_optimizer_updates=0,bundle_sha256=sha256(a.bundle),
        initial_sha256=sha256(a.initial),quality_approved=False,
        short_sequence_convolution=a.short_conv,
        limitations='Repeated PFNN fixture, different real PARC/BONES windows; capacity/timing only, no quality claim. '
                    'Timing excludes optimizer, validation, IO and checkpoints; Adam moment size is estimated.'))
    controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','initial','cases','mixed','fixture','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--short-conv',action='store_true')
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
