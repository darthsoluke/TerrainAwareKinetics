"""Check frozen-checkpoint eager/static parity and per-plan execution latency."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.terrain_model_v3 import JointTerrainController,SCHEMA
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.joint_terrain_execution import JointTerrainExecution
from motionbricks.data.soma_skin_reference import load_official_skinner,move_skinner
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False
    checkpoint=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False)
    if checkpoint['schema']!=SCHEMA or checkpoint['contract']['source_bundle_sha256']!=sha256(args.source):
        raise ValueError('Execution probe source/checkpoint binding differs')
    inference,bundle=load_terrain_bundle(args.source,'cuda:0');model=JointTerrainController(inference,bundle)
    model.load_state_dict(checkpoint['model_state'],strict=True);model.eval().requires_grad_(False)
    skin=move_skinner(load_official_skinner(inference.global_motion_rep.skeleton),torch.device('cuda:0'))
    skin.skeleton_skin.float().to('cuda:0');seed=bundle['initial_history'].to('cuda:0')
    scene=bundle['scene'];began=time.perf_counter()
    execution=JointTerrainExecution(model,scene,skin,seed)
    setup=time.perf_counter()-began
    task=SoftCommandState().intent(seed,[0.,1.6],0.,8)
    durations=[]
    for _ in range(30):
        torch.cuda.synchronize();start=time.perf_counter()
        predicted=execution.predict(seed,task)
        execution.geometry(predicted['world'][:,4:12])
        torch.cuda.synchronize();durations.append((time.perf_counter()-start)*1000)
    result={'schema':'motionbricks.joint_execution_probe.v1','checkpoint_step':checkpoint['step'],
        'device':torch.cuda.get_device_name(),'validation':execution.validation,'compile_seconds':setup,
        'plan_plus_rig_ms_median':float(np.median(durations)),'plan_plus_rig_ms_p95':float(np.quantile(durations,.95)),
        'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1024**2,'quality_approved':False,
        'meaning':'execution parity and latency only; not a learned motion-quality claim'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
