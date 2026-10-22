"""Profile native production microbatches without optimizer updates."""

import argparse
import cProfile
import pstats
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from motionbricks.data.full_terrain_cache import PrimitiveDataset, collate_primitives
from motionbricks.data.unified_full_training import NativeTrainingStage
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_unified_full_corpus import device_batch
from scripts.train_joint_terrain_v3 import write_json


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    inference,bundle=load_terrain_bundle(a.bundle,'cuda:0')
    controller=UnifiedTerrainController(inference,bundle)
    saved=torch.load(a.initial,map_location='cpu',weights_only=False)
    source=PrimitiveDataset(a.fixture/'data','train')
    ids={source.records[actual]['domain']:i for i,actual in enumerate(source.ids)}
    batch=device_batch(collate_primitives([source[ids[name]] for name,count in
        (('pfnn',32),('parc',16),('bones',16)) for _ in range(count)]),'cuda:0')
    rows=[]
    for name in ('root','codec','pose'):
        controller.load_state_dict(saved['model_state'],strict=True)
        controller.zero_grad(set_to_none=True)
        torch.use_deterministic_algorithms(True,warn_only=name=='codec')
        stage=NativeTrainingStage(controller,name).train(True)
        loss=stage(batch,2333)
        loss.backward()
        controller.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        recording=cProfile.Profile()
        recording.enable()
        started=time.perf_counter()
        loss=stage(batch,2333)
        torch.cuda.synchronize()
        forward_seconds=time.perf_counter()-started
        started=time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds=time.perf_counter()-started
        recording.disable()
        stats=pstats.Stats(recording)
        items=[dict(function=str(key),primitive_calls=value[0],total_calls=value[1],
            self_seconds=value[2],cumulative_seconds=value[3]) for key,value in stats.stats.items()]
        rows.append(dict(stage=name,forward_seconds=forward_seconds,backward_seconds=backward_seconds,
            top_cumulative=sorted(items,key=lambda row:row['cumulative_seconds'],reverse=True)[:20],
            top_self=sorted(items,key=lambda row:row['self_seconds'],reverse=True)[:15]))
        write_json(a.output/'progress.json',dict(rows=rows,actual_optimizer_updates=0))
    controller.close()
    write_json(a.output/'complete.json',dict(rows=rows,actual_optimizer_updates=0,
        fixture_only=True,model_exported=False,
        limitations='Python/C-call wall timing includes synchronization; not per-GPU-kernel timing.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','initial','fixture','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
