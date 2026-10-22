"""Uninstrumented fixed-history latency and isolated event-stage evidence."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import torch
from motionbricks.data.dual_horizon_execution import DualHorizonExecution,DualHorizonBackend,SCHEMA,policy
from motionbricks.data.native_mb_online_live import online_command_goal
from motionbricks.data.soma_lab import SomaLabSession
from motionbricks.data.soma_static_execution import restore_native_transformers,use_dense_transformers
from scripts.dual_horizon_reference import DualHorizonReference
from scripts.dual_horizon_deployment import source_hashes,check_scene
from scripts.review_dual_horizon_candidate import compare_motion,summary,save
from scripts.soma_pfnn_maps_server import PFNNMapRuntime


def new_session(runtime,engine,history,tokens):
    session=SomaLabSession(runtime,'terrain')
    if tokens!=8:
        session.backend=DualHorizonBackend.from_backend(session.backend,engine)
        session.backend.configure_policy(policy(tokens))
    session.backend.history=history.clone()
    return session


@contextmanager
def stage_events(engine,tokens):
    rows=[];old_graphs=engine.graphs;engine.graphs=dict(old_graphs);restored=[]
    def wrap(fn,label):
        def call(*args,**kwargs):
            first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            first.record();result=fn(*args,**kwargs);last.record();rows.append((label,first,last));return result
        return call
    engine.graphs[8]=(wrap(old_graphs[8][0],'root_short'),wrap(old_graphs[8][1],'pose4_decoder'))
    if tokens!=8:engine.graphs[tokens]=(wrap(old_graphs[tokens][0],'root_preview'),old_graphs[tokens][1])
    for owner,name,label in ((engine,'_scene_inputs','coarse_query'),(engine.observer,'sample','path_query'),(engine,'geometry','publication_geometry')):
        original=getattr(owner,name);restored.append((owner,name,original));setattr(owner,name,wrap(original,label))
    try:yield rows
    finally:
        engine.graphs=old_graphs
        for owner,name,original in restored:setattr(owner,name,original)


@torch.inference_mode()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    reference=json.loads((a.candidate/'complete.json').read_text())
    assert reference['completed'] and reference['weights_unchanged']
    report=dict(completed=False,actual_optimizer_updates=0,policy_schema=SCHEMA,
        reference_candidate_sha256=hashlib.sha256((a.candidate/'complete.json').read_bytes()).hexdigest(),
        original_candidate_parity=reference['candidate_parity'],candidate_parity=[],timings=[])
    runtime=None
    try:
        runtime=PFNNMapRuntime(a.bundle,a.catalog,execution='root-path-live',initial_map='007')
        base=runtime.inference._terrain_execution;engine=DualHorizonExecution.from_runner(base)
        engine.validation=reference['candidate_parity']
        native=DualHorizonReference.from_runner(base)
        cold=torch.load(a.candidate/'cold-12-world.pt',map_location=runtime.inference.device,weights_only=True)
        warm=torch.load(a.candidate/'warm-12-world.pt',map_location=runtime.inference.device,weights_only=True)
        histories=[cold[:,:4].contiguous(),warm[:,:4].contiguous(),base.observe(cold[:,128:132].contiguous())]
        report.update(model_sha256=runtime.catalog['checkpoint_sha256'],torch_version=torch.__version__,
            device=torch.cuda.get_device_name(),source_hashes=source_hashes())
        for tokens in (12,16):
            old=torch.load(a.candidate/f'cold-{tokens}-world.pt',map_location=runtime.inference.device,weights_only=True)
            session=new_session(runtime,engine,histories[0],tokens)
            session.plan(dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,3.],heading=0.))
            assert torch.equal(session.pending_playback['motion'],old[:,4:12])
            history=histories[0 if tokens==12 else 1]
            goal=online_command_goal(history[...,:5],[0.,3.],0.,8,base.protocol_version)
            query=history.new_tensor([[-1.,.6,1.,.4]])
            restore_native_transformers(runtime.inference)
            prepared=native.prepare_dual(history,goal,query,tokens,native=True)
            expected=native.native_suffix(history,goal,query,prepared)
            actual=engine.compiled_suffix(history,goal,query,engine.prepare_dual(history,goal,query,tokens))
            report['candidate_parity'].append(dict(lookahead_tokens=tokens,**compare_motion(runtime,expected,actual,8)))
        report['production_matches_candidate_bitwise']=True
        use_dense_transformers(runtime.inference)
        for tokens in (8,12,16):
            raw=[]
            for index in range(12):
                session=new_session(runtime,engine,histories[index%3],tokens)
                command=dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,3.],heading=0.)
                torch.cuda.synchronize();started=time.perf_counter()
                meta,payload=session.plan(command);encoded=json.dumps(meta);torch.cuda.synchronize()
                elapsed=(time.perf_counter()-started)*1000
                if index>=3:raw.append(dict(history_index=index%3,total_ms=elapsed,json_bytes=len(encoded.encode()),binary_bytes=len(payload)))
            stages=[]
            for index in range(3):
                session=new_session(runtime,engine,histories[index],tokens)
                # For base8, use the same original engine but route through its normal eight path.
                if tokens==8:session.backend.execution=engine
                with stage_events(engine,tokens) as events:
                    session.plan(dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,3.],heading=0.))
                    torch.cuda.synchronize();values={};counts={}
                    for label,first,last in events:
                        values[label]=values.get(label,0.)+first.elapsed_time(last);counts[label]=counts.get(label,0)+1
                    assert counts['coarse_query']==2 and counts['root_short']==2 and counts['pose4_decoder']==2
                    if tokens!=8:assert counts['root_preview']==2
                    stages.append(dict(milliseconds=values,calls=counts))
            row=dict(lookahead_tokens=tokens,execute_tokens=8,discarded_warmups=3,raw=raw,
                total=summary([item['total_ms'] for item in raw]),stages=stages,
                stage_summary={k:summary([r['milliseconds'][k] for r in stages]) for k in stages[0]['milliseconds']})
            report['timings'].append(row);save(a.output/'progress.json',report);print(json.dumps(row),flush=True)
        graph_ids=[id(call.graph) for calls in base.graphs.values() for call in calls]
        report['scene_rebinding']=[]
        for map_id in ('005','007'):
            runtime.select_map(map_id)
            current=runtime.inference._terrain_execution
            fresh=DualHorizonExecution.from_runner(current)
            assert fresh.scene is runtime.terrain_scene and fresh.field is current.field
            assert graph_ids==[id(call.graph) for calls in current.graphs.values() for call in calls]
            report['scene_rebinding'].append(dict(map=map_id,**check_scene(current,runtime.terrain_initial_history,
                runtime.acceptance_receipt['scene_sha256'])))
        report.update(completed=True,scope='Uninstrumented session.plan+JSON+CUDA completion; no reference re-encoding/assertion/world snapshot inside timing.',
            stage_scope='Separate event samples; read once after packet; each Root graph includes its encoder.',
            allocated_bytes=torch.cuda.memory_allocated(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert report['source_hashes']==source_hashes()
        save(a.output/'complete.json',report);save(a.output/'proof.json',report)
    except Exception as error:
        report['error']=repr(error);save(a.output/'failure.json',report);raise
    finally:
        if runtime is not None:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
