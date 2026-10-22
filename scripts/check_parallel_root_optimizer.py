"""Real Root optimizer and saved-state continuity, without a training budget."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def main(a):
    if len(a.gpus) not in (2,4) or len(set(a.gpus)) != len(a.gpus):
        raise ValueError('Two or four distinct GPUs are required')
    if len(a.gpus)==4 and a.reference_implementation is None:
        raise ValueError('Four-device proof requires the deployed two-device implementation')
    snapshot = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = {int(x.split(',')[0]): [v.strip() for v in x.split(',')[1:]] for x in snapshot.splitlines()}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    occupied = {x.split(',')[1].strip() for x in apps.splitlines()}
    if any(int(rows[g][1]) > 500 or int(rows[g][2]) > 5 or rows[g][0] in occupied for g in a.gpus):
        raise ValueError('All selected GPUs must still be idle')
    save(a.output/'state.json', dict(status='running', gpus=a.gpus, snapshot=snapshot))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,a.gpus))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import numpy as np
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
    from motionbricks.data.parallel_root_training import ParallelRootMicrobatches
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch, split_micro, rng_state, restore_rng
    from scripts.train_native_mb import sha256

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    for device in range(len(a.gpus)): torch.cuda.set_per_process_memory_fraction(.1,device)
    base = Path('${PRIVATE_STORAGE}/motionbricks-20261010'); run = base/'pfnn_mixed_20261015'
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    controllers, stages = [], []
    for index in range(len(a.gpus)):
        inference, bundle = load_terrain_bundle(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt', f'cuda:{index}')
        controller = UnifiedTerrainController(inference, bundle)
        controller.load_state_dict(saved['model_state'], strict=True)
        controllers.append(controller)
        stages.append(NativeTrainingStage(controller, 'root', coalesce_dilated_bias=True).train(True))
    parameters = [p for p in controllers[0].parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000, eta_min=5e-7)
    dataset = PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'], 'train')
    schedule = BalancedPrimitiveBatches(dataset,256,saved['actual_updates']+4,seed=saved['contract']['seed'],start_step=saved['actual_updates'])
    batches = []
    for ids in schedule:
        batch=collate_primitives([dataset[i] for i in ids]);order=[]
        for offset in range(4):
            for domain,quota in (('pfnn',32),('parc',16),('bones',16)):
                chosen=[i for i,x in enumerate(batch['domains']) if x==domain]
                order.extend(chosen[offset*quota:(offset+1)*quota])
        def reorder(x):
            if torch.is_tensor(x):return x[order]
            if isinstance(x,dict):return {k:reorder(v) for k,v in x.items()}
            return [x[i] for i in order]
        batches.append(device_batch(reorder(batch),'cuda:0'))
    def cpu(x):
        if torch.is_tensor(x):return x.detach().cpu().clone()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return type(x)(cpu(v) for v in x)
        return x
    def capture():
        return dict(model=cpu(controllers[0].state_dict()),optimizer=cpu(optimizer.state_dict()),
            scheduler=cpu(scheduler.state_dict()),rng=cpu(rng_state()))
    def restore(s):
        for controller in controllers:controller.load_state_dict(s['model'],strict=True)
        optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler']);restore_rng(s['rng'])
        for c in controllers:c.zero_grad(set_to_none=True)
    optimizer.load_state_dict(saved['optimizer_state']);scheduler.load_state_dict(saved['scheduler_state'])
    restore_rng(saved['rng_state'])
    for device in range(1,len(a.gpus)):
        torch.cuda.set_rng_state(saved['rng_state']['cuda'][0].cpu(),device)
    initial=capture()
    def update(i,executor):
        optimizer.zero_grad(set_to_none=True)
        micros=[split_micro(batches[i],j*64,(j+1)*64) for j in range(4)]
        seeds=[saved['contract']['seed']+(saved['actual_updates']+i)*103+j*64 for j in range(4)]
        started=time.perf_counter()
        if executor is None:
            losses=[]
            for micro,seed in zip(micros,seeds):
                loss=stages[0](micro,seed);(loss*.25).backward();losses.append(float(loss.detach()))
            # Both physical streams represent the same logical completed stream.
            for device in range(1,len(a.gpus)):
                torch.cuda.set_rng_state(torch.cuda.get_rng_state(0),device)
        else:losses=executor.backward(micros,seeds)
        torch.nn.utils.clip_grad_norm_(parameters,.5);optimizer.step();scheduler.step();torch.cuda.synchronize(0)
        return dict(losses=losses,seconds=time.perf_counter()-started)
    def compare(x,y,path='',errors=None):
        errors=[] if errors is None else errors
        if torch.is_tensor(x):
            if not torch.equal(x,y):errors.append(path)
        elif isinstance(x,np.ndarray):
            if not np.array_equal(x,y):errors.append(path)
        elif isinstance(x,dict):
            if x.keys()!=y.keys():errors.append(path+'.keys')
            for k in x:compare(x[k],y[k],path+'.'+str(k),errors)
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):errors.append(path+'.length')
            for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'.'+str(i),errors)
        elif x!=y:errors.append(path)
        return errors
    executor=None;reference_executor=None
    try:
        if a.reference_implementation is not None:
            import importlib.util
            spec=importlib.util.spec_from_file_location('deployed_parallel_root',a.reference_implementation)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            reference_executor=module.ParallelRootMicrobatches(stages[:2])
        serial=[]
        for i in range(4):
            serial.append(update(i,reference_executor))
            if reference_executor is not None:
                for device in range(2,len(a.gpus)):
                    torch.cuda.set_rng_state(torch.cuda.get_rng_state(0),device)
            if i==1 and reference_executor is not None:
                midpoint=capture();midpoint['rng']['cuda']=midpoint['rng']['cuda'][:2]
                torch.save(midpoint,a.output/'reference_midpoint.pt')
        reference=capture()
        if reference_executor:reference_executor.close();reference_executor=None
        restore(initial);executor=ParallelRootMicrobatches(stages)
        parallel=[]
        for i in range(4):
            parallel.append(update(i,executor))
            if i==1:torch.save(capture(),a.output/'midpoint.pt')
        continuous=capture();serial_errors=compare(reference,continuous)
        executor.close();executor=None
        midpoint_name='reference_midpoint.pt' if a.reference_implementation is not None else 'midpoint.pt'
        restore(torch.load(a.output/midpoint_name,map_location='cpu',weights_only=False))
        executor=ParallelRootMicrobatches(stages)
        resumed=[update(i,executor) for i in range(2,4)];resume_errors=compare(continuous,capture())
        if [x['losses'] for x in serial]!=[x['losses'] for x in parallel]:serial_errors.append('losses')
        if [x['losses'] for x in parallel[2:]]!=[x['losses'] for x in resumed]:resume_errors.append('losses')
        save(a.output/'complete.json',dict(passed=not serial_errors and not resume_errors,
            serial_errors=serial_errors,resume_errors=resume_errors,serial=serial,parallel=parallel,resumed=resumed,
            rng_increment=executor.rng_increment,rng_schedule_from_reference=False,source_updates=saved['actual_updates'],
            actual_optimizer_updates=10,mechanics_only=True,production_budget_updates=0,quality_approved=False,
            checkpoint_sha256=sha256(a.checkpoint),implementation_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/parallel_root_training.py')))
        if a.reference_implementation is not None:
            result=json.loads((a.output/'complete.json').read_text())
            result['reference']=result.pop('serial')
            result.update(reference_devices=2,target_devices=len(a.gpus),
                reference_implementation_sha256=sha256(a.reference_implementation),
                resume_from_reference_devices=2,logical_cuda_rng_canonicalized=True)
            save(a.output/'complete.json',result)
        save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=10))
    finally:
        if executor:executor.close()
        if reference_executor:reference_executor.close()
        for controller in controllers:controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    parser.add_argument('--reference-implementation',type=Path)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:main(args)
    except Exception as error:
        save(args.output/'failure.json',dict(error=repr(error)))
        save(args.output/'state.json',dict(status='failed',error=repr(error)))
        raise
