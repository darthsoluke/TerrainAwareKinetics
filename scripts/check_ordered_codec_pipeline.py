"""Verify serial EMA with parallel codec computation on two truly idle GPUs."""

import argparse
import json
import importlib.util
import os
from pathlib import Path
import time

from scripts.run_pfnn_parallel_workflow import gpu_snapshot
from scripts.run_pfnn_bias_workflow import save


def main(a):
    snapshot,available=gpu_snapshot()
    if len(a.gpus) not in (2,4) or len(set(a.gpus))!=len(a.gpus) or any(gpu not in available for gpu in a.gpus):
        raise ValueError('Two or four distinct idle GPUs required')
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,a.gpus));os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from motionbricks.data.ordered_codec_pipeline import OrderedCodecPipeline
    from motionbricks.data.exposure_codec_training import ordered_ema_chunks,original_order_indices
    from motionbricks.data.full_terrain_cache import PrimitiveDataset,BalancedPrimitiveBatches,collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.train_unified_full_corpus import device_batch,split_micro,rng_state,restore_rng
    from scripts.train_native_mb import sha256
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    for device in range(len(a.gpus)):torch.cuda.set_per_process_memory_fraction(a.pipeline_memory_fraction if a.cold_reference else a.reference_memory_fraction,device)
    torch.cuda.set_device(0)
    run=a.checkpoint.parent.parent;base=run.parent
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    reference_origin=saved['actual_exposures']//256
    if any(not torch.equal(state,saved['rng_state']['cuda'][0]) for state in saved['rng_state']['cuda'][1:]):
        raise ValueError('Source checkpoint has no single canonical CUDA RNG state')
    if saved['contract']['stage']!='codec':raise ValueError('Actual codec checkpoint required')
    controllers,stages=[],[]
    for index in range(len(a.gpus)):
        inference,bundle=load_terrain_bundle(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt',f'cuda:{index}')
        controller=UnifiedTerrainController(inference,bundle);controller.load_state_dict(saved['model_state'],strict=True)
        controller.codebook.copy_(controller.codec.get_codebook());controllers.append(controller)
        stages.append(NativeTrainingStage(controller,'codec',coalesce_dilated_bias=True,graph_dilated_convolutions=True).train(True))
    parameters=[p for p in controllers[0].parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=5e-5)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50000,eta_min=5e-7)
    optimizer.load_state_dict(saved['optimizer_state']);scheduler.load_state_dict(saved['scheduler_state'])
    dataset=PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'],'train')
    reference_units=a.compute_micro*a.compute_count//256
    schedule=BalancedPrimitiveBatches(dataset,256,reference_origin+4*reference_units,seed=saved['contract']['seed'],start_step=reference_origin)
    batches=[]
    if a.exposure_mode:
        for index in range(4):
            ids=original_order_indices(schedule,dataset,reference_origin+index*reference_units,a.compute_micro*a.compute_count)
            batches.append(collate_primitives([dataset[i] for i in ids]))
    for ids in ([] if a.exposure_mode else schedule):
        batch=collate_primitives([dataset[i] for i in ids]);order=[]
        for micro in range(4):
            for domain,count in (('pfnn',32),('parc',16),('bones',16)):
                chosen=[i for i,name in enumerate(batch['domains']) if name==domain]
                order+=chosen[micro*count:(micro+1)*count]
        def reorder(value):
            if torch.is_tensor(value):return value[order]
            if isinstance(value,dict):return {k:reorder(v) for k,v in value.items()}
            return [value[i] for i in order]
        batches.append(reorder(batch))
    def cpu(value):
        if torch.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return type(value)(cpu(v) for v in value)
        return value
    def snapshot_state():
        return dict(model=cpu(controllers[0].state_dict()),optimizer=cpu(optimizer.state_dict()),scheduler=cpu(scheduler.state_dict()),rng=cpu(rng_state()))
    def restore(state):
        for controller in controllers:controller.load_state_dict(state['model'],strict=True);controller.zero_grad(set_to_none=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'])
    def differences(left,right,path=''):
        if torch.is_tensor(left):return [] if torch.equal(left,right) else [dict(path=path,max_abs=float((left-right).abs().max()))]
        if isinstance(left,np.ndarray):return [] if np.array_equal(left,right) else [dict(path=path)]
        if isinstance(left,dict):
            if left.keys()!=right.keys():raise ValueError('Snapshot inventory differs')
            return [item for key in left for item in differences(left[key],right[key],path+'.'+str(key))]
        if isinstance(left,(list,tuple)):
            if len(left)!=len(right):raise ValueError('Snapshot length differs')
            return [item for i,(x,y) in enumerate(zip(left,right)) for item in differences(x,y,path+'.'+str(i))]
        return [] if left==right else [dict(path=path)]
    restore_rng(saved['rng_state'])
    for device in range(1,len(a.gpus)):torch.cuda.set_rng_state(saved['rng_state']['cuda'][0].cpu(),device)
    initial=snapshot_state();a.completed_updates=0
    terminal_records=[[] for _ in controllers]
    if a.isolate_first:
        for worker,controller in enumerate(controllers):
            terminal=controller.codec.decoder.model[6]
            def capture(module,args,output,worker=worker):
                row=dict(input=cpu(args[0]),input_stride=list(args[0].stride()),
                    allocated=torch.cuda.memory_allocated(args[0].device),reserved=torch.cuda.memory_reserved(args[0].device))
                terminal_records[worker].append(row)
                def upstream(value):row.update(upstream=cpu(value),upstream_stride=list(value.stride()))
                output.register_hook(upstream)
            terminal.register_forward_hook(capture)
            def weight_gradient(value,worker=worker):terminal_records[worker][-1]['weight_gradient']=cpu(value)
            terminal.weight.register_hook(weight_gradient)
    reference_path=run/'code_full_workflow_v9/motionbricks/data/dilated_graph_convolution.py'
    spec=importlib.util.spec_from_file_location('motionbricks.data._pipeline_reference_graph',reference_path)
    reference_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_module)
    stages[0]._dilated_graph_execution=reference_module.GraphConvolutionExecution()
    def update(index,pipeline,label):
        optimizer.zero_grad(set_to_none=True)
        batch=device_batch(batches[index],'cuda:0')
        micros=[split_micro(batch,i*a.compute_micro,(i+1)*a.compute_micro) for i in range(a.compute_count)]
        seeds=[saved['contract']['seed']+(reference_origin+index*reference_units)*103+i*a.compute_micro for i in range(a.compute_count)]
        torch.cuda.synchronize();started=time.perf_counter()
        if pipeline:
            losses=pipeline.backward(micros,seeds);outputs=pipeline.last_outputs
        else:
            losses,outputs=[],[]
            for micro,seed in zip(micros,seeds):
                with ordered_ema_chunks(controllers[0]):
                    loss=stages[0](micro,seed);(loss/a.compute_count).backward()
                losses.append(float(loss.detach()))
                outputs.append(cpu(controllers[0].last_local))
            for device in range(1,len(a.gpus)):torch.cuda.set_rng_state(torch.cuda.get_rng_state(0),device)
        observed=dict(losses=losses,outputs=outputs,gradients={n:cpu(p.grad) for n,p in controllers[0].named_parameters() if p.requires_grad},
            ema={n:cpu(v) for n,v in controllers[0].codec.quantizer.state_dict().items()},rng=cpu(rng_state()),sample_ids=batch['sample_ids'].tolist())
        if not a.isolate_first and not a.zero_update_comparison:
            torch.nn.utils.clip_grad_norm_(parameters,.5);optimizer.step();a.completed_updates+=1
            if a.exposure_mode:scheduler.step(reference_origin+(index+1)*reference_units)
            else:scheduler.step()
            if a.exposure_mode and a.validation_between and index==2:
                from scripts.train_unified_full_corpus import validation
                heldout=PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'],'validation')
                validation(stages[0],heldout,64)
        torch.cuda.synchronize()
        row=dict(seconds=time.perf_counter()-started,losses=losses,
            ema_order=pipeline.last_ema_order if pipeline else list(range(a.compute_count)))
        save(a.output/'state.json',dict(status='running',phase=label,source_step=saved['actual_updates']+index+1,
            actual_optimizer_updates=a.completed_updates,production_budget_updates=0))
        return observed,row
    pipeline=None
    try:
        if a.cold_reference:
            reference= torch.load(a.cold_reference/'serial_endpoint.pt',weights_only=False)
            if a.cold_start_step==2:
                restore(torch.load(a.resume_state or a.cold_reference/'serial_midpoint.pt',weights_only=False))
            else:restore(initial)
            from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
            stages[0]._dilated_graph_execution=reference_module.GraphConvolutionExecution() if a.serial_cold else GraphConvolutionExecution()
            pipeline=None if a.serial_cold else OrderedCodecPipeline(stages,a.compute_micro,a.compute_count);rows=[];changes=[]
            validation_rows=None
            if a.zero_update_comparison:
                if pipeline is None or a.cold_start_step!=2:raise ValueError('Zero-update comparison requires the retained midpoint and pipeline')
                batch=device_batch(batches[2],'cuda:0')
                for worker in range(2):
                    pipeline.pools[worker].submit(pipeline._warm,worker,split_micro(batch,worker*64,(worker+1)*64),
                        saved['contract']['seed']+(saved['actual_updates']+2)*103+worker*64).result()
                pipeline.ready=True
                del batch
                from scripts.train_unified_full_corpus import validation
                heldout=PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'],'validation')
                validation_rows=pipeline.validate(validation,heldout,64)
                for device in (0,1):torch.cuda.reset_peak_memory_stats(device)
            for index in ([2] if a.zero_update_comparison else range(a.cold_start_step,4)):
                observed,row=update(index,pipeline,'cold_process_resume');rows.append(row)
                torch.save(observed,a.output/f'observed_{index}.pt')
                changes.append(differences(torch.load(a.cold_reference/f'serial_{index}.pt',weights_only=False),observed))
                if index==1:torch.save(snapshot_state(),a.output/'pipeline_midpoint.pt')
                if a.validation_between and index==2 and not a.exposure_mode:
                    from scripts.train_unified_full_corpus import validation
                    heldout=PrimitiveDataset([run/'cache_mixed_v1/data',run/'cache_pfnn_world_v2/data'],'validation')
                    validation_rows=pipeline.validate(validation,heldout,64) if a.validation_worker==1 else validation(stages[0],heldout,64)
            if a.zero_update_comparison:
                save(a.output/'complete.json',dict(passed=not any(changes),zero_update_comparison=True,
                    micro_changes=changes,rows=rows,validation=validation_rows,
                    peak_allocated=[torch.cuda.max_memory_allocated(i) for i in (0,1)],
                    peak_reserved=[torch.cuda.max_memory_reserved(i) for i in (0,1)],
                    implementation_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/ordered_codec_pipeline.py'),
                    candidate_graph_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/dilated_graph_convolution.py'),
                    checkpoint_sha256=sha256(a.checkpoint),reference_midpoint_sha256=sha256(a.cold_reference/'serial_midpoint.pt'),
                    actual_optimizer_updates=a.completed_updates,production_budget_updates=0,production_approved=False))
                save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=0))
                return
            endpoint_changes=differences(reference,snapshot_state())
            torch.save(snapshot_state(),a.output/'pipeline_endpoint.pt')
            paired_changes=differences(torch.load(a.paired_endpoint,weights_only=False),snapshot_state()) if a.paired_endpoint else None
            save(a.output/'complete.json',dict(passed=not any(changes) and not endpoint_changes,cold_process=True,
                micro_changes=changes,endpoint_changes=endpoint_changes,rows=rows,
                warm_memory=pipeline.warm_memory if pipeline else [],pipeline_memory_fraction=a.pipeline_memory_fraction,
                serial_cold=a.serial_cold,paired_endpoint_changes=paired_changes,
                exposure_mode=a.exposure_mode,global_batch=a.compute_micro*a.compute_count,
                execution_devices=len(a.gpus),source_reference_origin=reference_origin,
                compute_micro=a.compute_micro,ema_chunk_size=64,reference_units_per_update=reference_units,
                source_validation_between=a.validation_between,
                paired_endpoint_sha256=sha256(a.paired_endpoint) if a.paired_endpoint else None,
                validation_between=validation_rows,
                validation_worker=a.validation_worker,
                cold_start_step=a.cold_start_step,resume_state_sha256=sha256(a.resume_state) if a.resume_state else None,
                implementation_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/ordered_codec_pipeline.py'),
                candidate_graph_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/dilated_graph_convolution.py'),
                reference_graph_sha256=sha256(reference_path),reference_serial_only=True,
                reference_tensors_sha256={name:sha256(a.cold_reference/name)
                    for name in ['serial_endpoint.pt','serial_midpoint.pt',*[f'serial_{i}.pt' for i in range(4)]]},
                checkpoint_sha256=sha256(a.checkpoint),reference_report_sha256=sha256(a.cold_reference/'complete.json'),
                actual_optimizer_updates=a.completed_updates,production_budget_updates=0,production_approved=False))
            save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=a.completed_updates))
            return
        if a.isolate_first:
            observed,serial_row=update(0,None,'isolation_serial')
            reference_terminal=list(terminal_records[0]);restore(initial)
            from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
            stages[0]._dilated_graph_execution=GraphConvolutionExecution();torch.cuda.empty_cache()
            pipeline=OrderedCodecPipeline(stages)
            candidate,parallel_row=update(0,pipeline,'isolation_pipeline')
            terminal=[terminal_records[index%2][-2+index//2] for index in range(4)]
            fields=('input','input_stride','upstream','upstream_stride','weight_gradient')
            changes=[differences({k:left[k] for k in fields},{k:right[k] for k in fields})
                for left,right in zip(reference_terminal,terminal)]
            torch.save(dict(serial=reference_terminal,pipeline=terminal),a.output/'terminal_tensors.pt')
            save(a.output/'complete.json',dict(isolation=True,passed=not any(changes),terminal_changes=changes,
                full_changes=differences(observed,candidate),serial=serial_row,parallel=parallel_row,
                memory_serial=[{k:r[k] for k in ('allocated','reserved')} for r in reference_terminal],
                memory_pipeline=[{k:r[k] for k in ('allocated','reserved')} for r in terminal],
                warm_memory=pipeline.warm_memory,actual_optimizer_updates=0,production_budget_updates=0))
            return
        serial=[]
        for index in range(4):
            observed,row=update(index,None,'serial');serial.append(row);torch.save(observed,a.output/f'serial_{index}.pt')
            if index==1:torch.save(snapshot_state(),a.output/'serial_midpoint.pt')
        reference=snapshot_state();restore(initial)
        torch.save(reference,a.output/'serial_endpoint.pt')
        if a.serial_only:
            save(a.output/'complete.json',dict(passed=True,serial_reference=True,serial=serial,
                execution_devices=len(a.gpus),source_reference_origin=reference_origin,
                exposure_mode=a.exposure_mode,global_batch=a.compute_micro*a.compute_count,
                compute_micro=a.compute_micro,ema_chunk_size=64,reference_units_per_update=reference_units,
                validation_between=a.validation_between,reference_memory_fraction=a.reference_memory_fraction,
                checkpoint_sha256=sha256(a.checkpoint),reference_graph_sha256=sha256(reference_path),
                actual_optimizer_updates=a.completed_updates,production_budget_updates=0,production_approved=False))
            save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=a.completed_updates))
            return
        for device in range(len(a.gpus)):torch.cuda.set_per_process_memory_fraction(a.pipeline_memory_fraction,device)
        torch.cuda.set_device(0)
        from motionbricks.data.dilated_graph_convolution import GraphConvolutionExecution
        stages[0]._dilated_graph_execution=GraphConvolutionExecution()
        torch.cuda.empty_cache()
        pipeline=OrderedCodecPipeline(stages,a.compute_micro,a.compute_count);parallel=[];micro_changes=[]
        for index in range(4):
            observed,row=update(index,pipeline,'pipeline');parallel.append(row)
            micro_changes.append(differences(torch.load(a.output/f'serial_{index}.pt',weights_only=False),observed))
        continuous=snapshot_state();continuous_changes=differences(reference,continuous)
        torch.save(continuous,a.output/'pipeline_endpoint.pt')
        pipeline.close();pipeline=None;restore(torch.load(a.output/'serial_midpoint.pt',weights_only=False))
        pipeline=OrderedCodecPipeline(stages,a.compute_micro,a.compute_count);resumed=[]
        for index in range(2,4):
            observed,row=update(index,pipeline,'serial_to_pipeline_resume');resumed.append(row)
        resume_changes=differences(reference,snapshot_state())
        save(a.output/'complete.json',dict(passed=not any(micro_changes) and not continuous_changes and not resume_changes,
            micro_changes=micro_changes,continuous_changes=continuous_changes,resume_changes=resume_changes,
            serial=serial,parallel=parallel,resumed=resumed,actual_optimizer_updates=a.completed_updates,
            calibration_microbatches=pipeline.calibration_microbatches,production_budget_updates=0,
            warm_memory=pipeline.warm_memory,
            reference_memory_fraction=a.reference_memory_fraction,pipeline_memory_fraction=a.pipeline_memory_fraction,
            exposure_mode=a.exposure_mode,global_batch=a.compute_micro*a.compute_count,
            compute_micro=a.compute_micro,ema_chunk_size=64,reference_units_per_update=reference_units,
            validation_between=a.validation_between,
            checkpoint_sha256=sha256(a.checkpoint),implementation_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/ordered_codec_pipeline.py'),
            reference_graph_sha256=sha256(reference_path),candidate_graph_sha256=sha256(Path(__file__).parents[1]/'motionbricks/data/dilated_graph_convolution.py'),
            serial_ema=True,independent_rng=True,production_approved=False,quality_approved=False))
        save(a.output/'state.json',dict(status='complete',actual_optimizer_updates=a.completed_updates))
    finally:
        if pipeline:pipeline.close()
        for controller in controllers:controller.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,nargs='+',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--isolate-first',action='store_true')
    parser.add_argument('--pipeline-memory-fraction',type=float,default=.125)
    parser.add_argument('--cold-reference',type=Path)
    parser.add_argument('--cold-start-step',type=int,choices=(0,2),default=2)
    parser.add_argument('--resume-state',type=Path)
    parser.add_argument('--validation-between',action='store_true')
    parser.add_argument('--validation-worker',type=int,choices=(0,1),default=0)
    parser.add_argument('--serial-cold',action='store_true')
    parser.add_argument('--paired-endpoint',type=Path)
    parser.add_argument('--zero-update-comparison',action='store_true')
    parser.add_argument('--exposure-mode',action='store_true')
    parser.add_argument('--compute-micro',type=int,default=64)
    parser.add_argument('--compute-count',type=int,default=4)
    parser.add_argument('--reference-memory-fraction',type=float,default=.125)
    parser.add_argument('--serial-only',action='store_true')
    args=parser.parse_args();args.completed_updates=0;args.output.mkdir(parents=True,exist_ok=False)
    try:main(args)
    except Exception as error:
        save(args.output/'failure.json',dict(error=repr(error),actual_optimizer_updates=args.completed_updates,production_budget_updates=0))
        raise
