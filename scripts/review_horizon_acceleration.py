"""Bounded horizon graph, rolling protocol and timing checks on one model."""
import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.anticipatory_speed import DEFAULTS
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.soma_lab import SomaLabSession
from motionbricks.data.soma_static_execution import restore_native_transformers,use_dense_transformers
from scripts.soma_pfnn_maps_server import PFNNMapRuntime


def save(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def summary(values):return dict(n=len(values),p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)))


@contextmanager
def stages(runner,tokens):
    rows=[];originals=[]
    def timed(call,label):
        def run(*a,**kw):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();result=call(*a,**kw);end.record();rows.append((label,start,end))
            return result
        return run
    calls=runner.graphs[tokens]
    runner.graphs[tokens]=tuple(timed(call,label) for call,label in zip(calls,('root_graph','pose_decoder_graph')))
    for name in ('_scene_inputs','_path_inputs','geometry','observe'):
        original=getattr(runner,name);originals.append((name,original));setattr(runner,name,timed(original,name))
    try:yield rows
    finally:
        runner.graphs[tokens]=calls
        for name,original in originals:setattr(runner,name,original)


@torch.inference_mode()
def review(a):
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,actual_optimizer_updates=0,map=a.map,parity=[],timings=[])
    runtime=None
    try:
        runtime=PFNNMapRuntime(a.bundle,a.catalog,execution='root-path-live',initial_map=a.map)
        runner=runtime.inference._terrain_execution
        report.update(startup=runner.validation,constant_storage=runner.constant_storage_report())
        save(a.output/'progress.json',report)
        base=runtime.runtime_contract()['defaults']
        original_decode=runtime.inference._vqvae_pose_model.forward_decoder
        original_prediction=runner._prediction
        for tokens in (8,12,16):
            arms=[]
            for native in (True,False):
                session=SomaLabSession(runtime,'terrain')
                captures=[]
                if native:
                    session.backend.execution=None
                    restore_native_transformers(runtime.inference)
                    def decode(ids,*args,**kw):
                        captures.append(ids[:,:tokens].clone())
                        return original_decode(ids,*args,**kw)
                    runtime.inference._vqvae_pose_model.forward_decoder=decode
                else:
                    use_dense_transformers(runtime.inference)
                    def prediction(*args,**kw):
                        result=original_prediction(*args,**kw);captures.append(result['tokens'][:,:tokens].clone())
                        return result
                    runner._prediction=prediction
                motions=[];history_errors=[]
                try:
                    for index in range(2):
                        session.plan(dict(type='plan',mode='terrain',request_id=index+1,velocity_xz=[0.,3.],heading=0.,
                            runtime_controls=dict(base,num_tokens=tokens,refresh_frames=4)))
                        pending=session.pending_playback
                        motions.append(pending['motion'].clone())
                        factual=reencode_played_history(pending['motion'][:,-4:].contiguous(),runtime.inference.global_motion_rep)
                        error=float((factual-pending['history']).abs().max())
                        assert error<=1e-5
                        history_errors.append(error)
                        expected=pending['history']
                        session.acknowledge_played(dict(request_id=index+1,frame_count=8))
                        assert torch.equal(session.backend.history,expected)
                    arms.append((captures,motions,history_errors))
                finally:
                    runtime.inference._vqvae_pose_model.forward_decoder=original_decode
                    runner._prediction=original_prediction
            assert len(arms[0][0])==len(arms[1][0])==4
            assert all(torch.equal(x,y) for x,y in zip(arms[0][0],arms[1][0]))
            errors=[]
            for first,second in zip(arms[0][1],arms[1][1]):
                joints=[runtime.inference.motion_rep.inverse(m,False,return_all=False)['posed_joints'] for m in (first,second)]
                errors.append(float((joints[0]-joints[1]).norm(dim=-1).max()))
            assert max(errors)<=3e-5
            report['parity'].append(dict(tokens=tokens,active_token_sets=4,tokens_equal=True,fk_max_m=max(errors),
                own_history_error_m=[arm[2] for arm in arms],ack_history_bitwise=True))
            save(a.output/'progress.json',report)
        use_dense_transformers(runtime.inference)
        session=SomaLabSession(runtime,'terrain')
        session.plan(dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,3.],heading=0.,runtime_controls=base))
        transitions=[]
        for source,tokens in enumerate((12,16,8),1):
            backend=session.backend;pending=session.pending_playback
            old_runtime=dict(backend.runtime_controls);old_inference=dict(backend.inference_config);old_state=backend.governor_state
            controls=dict(base,num_tokens=tokens);governor=dict(DEFAULTS,enabled=source%2==1)
            serial=backend._future_backend(pending['history'])
            assert serial.inference_config is not backend.inference_config and serial.runtime_controls is not backend.runtime_controls
            serial.configure_runtime(controls);serial.configure_governor(governor)
            expected=serial.predict_plan([[0.,3.]],[0.])
            meta,_=session.prefetch(dict(type='prefetch',mode='terrain',request_id=source+1,source_request_id=source,
                played_frame_count=4,velocity_xz=[0.,3.],heading=0.,runtime_controls=controls,speed_governor=governor))
            assert meta['type']=='plan' and meta['prefetched']
            candidate=pending['prepared']['candidate']
            assert backend.runtime_controls==old_runtime and backend.inference_config==old_inference and backend.governor_state is old_state
            assert torch.equal(candidate['planned_motion'],expected['planned_motion'])
            assert torch.equal(candidate['published_history'],expected['published_history'])
            assert candidate['governor_state_after']==expected['governor_state_after']
            session.acknowledge_played(dict(request_id=source,frame_count=8))
            assert backend.num_tokens==tokens and backend.runtime_controls==controls and backend.speed_governor==governor
            transitions.append(dict(tokens=tokens,serial_prefetch_bitwise=True,parent_unchanged_until_ack=True))
        session.acknowledge_played(dict(request_id=4,frame_count=8))
        report['transitions']=transitions
        for tokens in (8,12,16):
            session=SomaLabSession(runtime,'terrain');raw=[];event_rows=[]
            for index in range(18):
                command=dict(type='plan',mode='terrain',request_id=index+1,velocity_xz=[0.,3.],heading=0.,
                    runtime_controls=dict(base,num_tokens=tokens,refresh_frames=4))
                if index<15:
                    started=time.perf_counter();meta,payload=session.plan(command);json.dumps(meta);torch.cuda.synchronize()
                    elapsed=(time.perf_counter()-started)*1000
                    if index>=3:raw.append(elapsed)
                else:
                    with stages(runner,tokens) as events:
                        session.plan(command);torch.cuda.synchronize()
                        row={}
                        for name,start,end in events:row[name]=row.get(name,0.)+start.elapsed_time(end)
                        event_rows.append(row)
                session.acknowledge_played(dict(request_id=index+1,frame_count=8))
            report['timings'].append(dict(tokens=tokens,excluded_warmups=3,total=summary(raw),raw_ms=raw,
                stage_samples=event_rows,stage_summary={k:summary([r[k] for r in event_rows]) for k in event_rows[0]},
                allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()))
            save(a.output/'progress.json',report)
            print(json.dumps(report['timings'][-1]),flush=True)
        identities=[id(call.graph) for pair in runner.graphs.values() for call in pair]
        other='007' if a.map!='007' else '005'
        runtime.select_map(other);runtime.select_map(a.map)
        assert identities==[id(call.graph) for pair in runner.graphs.values() for call in pair]
        report.update(completed=True,map_rebind=dict(other=other,restored=a.map,graphs_reused=True),
            final_memory=runner.constant_storage_report())
        save(a.output/'complete.json',report)
    except Exception as error:
        report['error']=repr(error);save(a.output/'failure.json',report);raise
    finally:
        if runtime is not None:runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,required=True);parser.add_argument('--map',default='005')
    parser.add_argument('--output',type=Path,required=True);review(parser.parse_args())
