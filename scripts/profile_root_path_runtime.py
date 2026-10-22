"""Bounded final-model latency evidence; no optimization or training updates."""

import copy
from contextlib import contextmanager
import json
import time

import numpy as np
import torch

from motionbricks.data.native_mb_online_live import online_command_goal
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.pfnn_tuning import tuning_contract
from motionbricks.data.root_path_terrain import RootPathTerrainExecution
from motionbricks.data.soma_lab import SomaLabSession
from motionbricks.data.soma_static_execution import restore_native_transformers, use_dense_transformers


def summary(values):
    return dict(n=len(values),p50_ms=float(np.percentile(values,50)),
                p95_ms=float(np.percentile(values,95)),min_ms=min(values),max_ms=max(values))


@contextmanager
def events_for(execution):
    rows, originals = [], []
    def wrap(owner,name,label):
        original=getattr(owner,name)
        def timed(*args,**kwargs):
            before,after=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            before.record()
            result=original(*args,**kwargs)
            after.record()
            rows.append((label,before,after))
            return result
        originals.append((owner,name,original))
        setattr(owner,name,timed)
    if execution.neural_execution=='native':
        wrap(execution.controller.adaptation['encoder'],'forward','shared_terrain_encoder')
        wrap(execution.controller.sampler,'observe','coarse_volume')
        wrap(execution.controller.root,'forward','root_network')
        wrap(execution.observer,'sample','path_queries')
        wrap(execution.controller.pose,'forward','pose_pass')
        wrap(execution.inference._vqvae_pose_model,'forward_decoder','decoder')
    else:
        wrap(execution,'_scene_inputs','coarse_queries')
        wrap(execution,'_compiled_root','root_graph')
        wrap(execution,'_path_inputs','path_queries')
        wrap(execution,'_compiled_pose','pose_decoder_graph')
    wrap(execution,'geometry','publication_geometry')
    try:yield rows
    finally:
        for owner,name,original in reversed(originals):setattr(owner,name,original)


@torch.inference_mode()
def profile_runtime(runtime,folder,reference=None):
    folder.mkdir(parents=True,exist_ok=False)
    inference=runtime.inference
    compiled=inference._terrain_execution
    native=copy.copy(compiled)
    native.__class__=RootPathTerrainExecution
    native.neural_execution='native'
    native.validation=[dict(neural_execution='native',geometry_verified=True)]
    histories=[runtime.terrain_initial_history.clone()]
    commands=[([0.,3.],0.),([2.12,2.12],.7854),([0.,0.],0.)]
    restore_native_transformers(inference)
    for _ in range(3):
        history=histories[-1]
        goal=online_command_goal(history[...,:5],[0.,3.],0.,8,native.protocol_version)
        with runtime.unified_controller.context(history,runtime.terrain_scene):
            values=native._prediction(history,goal)
        histories.append(native.observe(values['motion'][:,4:8].contiguous()))
    report=dict(map_id=runtime.current_map,device=torch.cuda.get_device_name(),
        checkpoint_sha256=runtime.catalog['checkpoint_sha256'],samples_per_setting=20,
        warmups_per_setting=3,stage_samples_per_setting=6,
        cold_start='model load, traces, graph capture and startup parity excluded',
        actual_optimizer_updates=0,settings=[],parity=compiled.validation,
        metric='session.plan + JSON encoding + final CUDA completion; network measured separately in browser')
    report['stage_scope']='Native encoder is measured separately; each split graph includes its own SharedTerrainEncoder. CUDA events are read only after the complete packet.'
    if reference is not None:
        measured=json.loads(reference.read_text())['report']
        if measured['checkpoint_sha256']!=report['checkpoint_sha256'] or measured['map_id']!=report['map_id']:
            raise ValueError('Measurement reference uses a different model or map')
        report['settings']=measured['settings']
        report['reference_measurements_source']=str(reference.resolve())
    try:
        for refresh in (() if reference is not None else (4,8)):
            controls=dict(tuning_contract()['defaults'],refresh_frames=refresh)
            for label,execution in (('native',native),('split_graph',compiled)):
                if label=='native':restore_native_transformers(inference)
                else:use_dense_transformers(inference)
                object.__setattr__(inference,'_terrain_execution',execution)
                raw=[]
                for index in range(23):
                    session=SomaLabSession(runtime,'terrain')
                    session.backend.execution=execution
                    session.backend.history=histories[index%len(histories)].clone()
                    velocity,heading=commands[index%len(commands)]
                    message=dict(type='plan',mode='terrain',request_id=1,velocity_xz=velocity,
                                 heading=heading,runtime_controls=controls)
                    torch.cuda.synchronize()
                    started=time.perf_counter()
                    metadata,payload=session.plan(message)
                    prepared=time.perf_counter()
                    encoded=json.dumps(metadata)
                    torch.cuda.synchronize()
                    stopped=time.perf_counter()
                    row=dict(total_ms=(stopped-started)*1000,session_ms=(prepared-started)*1000,
                        json_ms=(stopped-prepared)*1000,infer_ms=metadata['inference_ms'],
                        skin_ms=metadata['skin_ms'],json_bytes=len(encoded.encode()),payload_bytes=len(payload),
                        command=velocity,history_index=index%len(histories))
                    if index>=3:raw.append(row)
                stages=[]
                for index in range(6):
                    session=SomaLabSession(runtime,'terrain')
                    session.backend.execution=execution
                    session.backend.history=histories[index%len(histories)].clone()
                    velocity,heading=commands[index%len(commands)]
                    with events_for(execution) as events:
                        session.plan(dict(type='plan',mode='terrain',request_id=1,velocity_xz=velocity,
                            heading=heading,runtime_controls=controls))
                        torch.cuda.synchronize()
                        stage={}
                        for name,before,after in events:stage[name]=stage.get(name,0.)+before.elapsed_time(after)
                        stages.append(stage)
                item=dict(mode=label,refresh_frames=refresh,raw=raw,stages=stages,
                    total=summary([r['total_ms'] for r in raw]),
                    stage_summary={key:summary([r[key] for r in stages]) for key in stages[0]})
                report['settings'].append(item)
                (folder/'progress.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(dict(latency_mode=label,refresh_frames=refresh,total=item['total'])),flush=True)
        # Validate the native/split path at identical generated (not just idle) histories.
        checks=[]
        original_decode=inference._vqvae_pose_model.forward_decoder
        captured={}
        def decode(ids,*args,**kwargs):
            captured['ids']=ids.clone()
            return original_decode(ids,*args,**kwargs)
        inference._vqvae_pose_model.forward_decoder=decode
        try:
            for history in histories[1:]:
                goal=online_command_goal(history[...,:5],[0.,3.],0.,8,native.protocol_version)
                restore_native_transformers(inference)
                with runtime.unified_controller.context(history,runtime.terrain_scene):expected=native._prediction(history,goal)
                ids=captured['ids'].clone()
                actual=compiled._prediction(history,goal)
                poses=[inference.motion_rep.inverse(item['motion'][:,:32],False,return_all=False)['posed_joints']
                       for item in (expected,actual)]
                fk=float((poses[1]-poses[0]).norm(dim=-1).max())
                equal=torch.equal(ids[:,:8],actual['tokens'][:,:8])
                if not equal or fk>3e-5:raise ValueError(f'Generated-history parity: tokens={equal}, FK={fk}')
                checks.append(dict(tokens_equal=equal,fk_max_error_m=fk))
        finally:inference._vqvae_pose_model.forward_decoder=original_decode
        report.update(passed=True,generated_history_parity=checks)
        packet_results=[]
        for label,execution in (('native',native),('split_graph',compiled)):
            if label=='native':restore_native_transformers(inference)
            else:use_dense_transformers(inference)
            object.__setattr__(inference,'_terrain_execution',execution)
            session=SomaLabSession(runtime,'terrain')
            session.backend.execution=execution
            session.backend.history=histories[1].clone()
            captures=[]
            original_prediction=execution._prediction
            original_decode=inference._vqvae_pose_model.forward_decoder
            latest={}
            def decode(ids,*args,**kwargs):
                latest['ids']=ids.clone()
                return original_decode(ids,*args,**kwargs)
            def prediction(history,goal):
                result=original_prediction(history,goal)
                captures.append((result.get('tokens',latest.get('ids'))[:,:8].clone(),result['motion'].clone()))
                return result
            inference._vqvae_pose_model.forward_decoder=decode
            execution._prediction=prediction
            try:
                first,_=session.plan(dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,3.],
                    heading=0.,runtime_controls=dict(tuning_contract()['defaults'],refresh_frames=4)))
                first_motion=session.pending_playback['motion'].clone()
                first_history=session.pending_playback['history'].clone()
                first_factual=reencode_played_history(first_motion[:,-4:].contiguous(),inference.global_motion_rep)
                torch.testing.assert_close(first_history,first_factual,atol=1e-5,rtol=0)
                serial=copy.copy(session.backend)
                serial.history=first_history.clone()
                execution._prediction=original_prediction
                try:serial_result=serial.predict_plan([[2.12,2.12]],[.7854])
                finally:execution._prediction=prediction
                second,_=session.prefetch(dict(type='prefetch',mode='terrain',request_id=2,source_request_id=1,
                    played_frame_count=2,velocity_xz=[2.12,2.12],heading=.7854))
                next_motion=session.pending_playback['prepared']['motion'].clone()
                prepared_history=session.pending_playback['prepared']['history'].clone()
                if not torch.equal(next_motion,serial_result['planned_motion']):raise ValueError('Serial/prefetch published motion differs')
                if not torch.equal(prepared_history,serial_result['published_history']):raise ValueError('Serial/prefetch committed history differs')
                session.acknowledge_played(dict(type='played',request_id=1,frame_count=8))
                torch.testing.assert_close(session.backend.history,first_history,atol=0,rtol=0)
                session.acknowledge_played(dict(type='played',request_id=2,frame_count=8))
                if not torch.equal(session.backend.history,prepared_history):raise ValueError('ACK changed the published history')
                factual=reencode_played_history(next_motion[:,-4:].contiguous(),inference.global_motion_rep)
                torch.testing.assert_close(session.backend.history,factual,atol=1e-5,rtol=0)
                packet_results.append(dict(captures=captures,motions=[first_motion,next_motion],
                    history=session.backend.history.clone(),
                    actual_frame_history_error=float((session.backend.history-factual).abs().max()),
                    first_frame_history_error=float((first_history-first_factual).abs().max()),serial_prefetch_bitwise=True))
            finally:
                execution._prediction=original_prediction
                inference._vqvae_pose_model.forward_decoder=original_decode
        packet_checks=[]
        for index,(reference,actual) in enumerate(zip(packet_results[0]['captures'],packet_results[1]['captures'])):
            if not torch.equal(reference[0],actual[0]):raise ValueError(f'Packet/prefetch token mismatch at internal prediction {index}')
            poses=[inference.motion_rep.inverse(value[:,:32],False,return_all=False)['posed_joints']
                   for value in (reference[1],actual[1])]
            fk=float((poses[1]-poses[0]).norm(dim=-1).max())
            if fk>3e-5:raise ValueError(f'Packet/prefetch FK mismatch: {fk}')
            packet_checks.append(dict(internal_prediction=index,tokens_equal=True,fk_max_error_m=fk))
        drift=(packet_results[0]['history']-packet_results[1]['history']).abs()
        drift_index=tuple(int(value) for value in np.unravel_index(int(drift.argmax()),tuple(drift.shape)))
        feature_names=[name for name,indices in inference.global_motion_rep.indices.items()
                       if name!='all' and drift_index[-1] in np.asarray(indices).reshape(-1).tolist()]
        report['packet_and_prefetch_parity']=dict(passed=True,packet_frames=8,refresh_frames=4,
            predictions=packet_checks,factual_history_commits=2,
            actual_frame_history_errors=[item['actual_frame_history_error'] for item in packet_results],
            first_frame_history_errors=[item['first_frame_history_error'] for item in packet_results],
            serial_prefetch_bitwise=[item['serial_prefetch_bitwise'] for item in packet_results],
            cross_candidate_max_feature=dict(index=drift_index,names=feature_names,
                native=float(packet_results[0]['history'][drift_index]),split=float(packet_results[1]['history'][drift_index])),
            cross_candidate_history_max_error=float((packet_results[0]['history']-packet_results[1]['history']).abs().max()),
            history_gate='Each committed history matches native re-encoding of its own actual published frames at absolute 1e-5; counterfactual candidate features are recorded separately.')
        (folder/'complete.json').write_text(json.dumps(report,indent=2))
    except Exception as error:
        (folder/'failure.json').write_text(json.dumps(dict(error=repr(error),report=report),indent=2))
        raise
    finally:
        object.__setattr__(inference,'_terrain_execution',compiled)
        use_dense_transformers(inference)
