"""One zero-update dual-horizon candidate, not a live deployment entry point."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.dual_horizon_execution import DualHorizonBackend,policy,FAR_FRAMES
from scripts.dual_horizon_reference import DualHorizonReference
from motionbricks.data.full_corpus_inference import tensor_digest
from motionbricks.data.native_mb_online_live import online_command_goal
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.root_path_terrain import RootPathTerrainExecution
from motionbricks.data.soma_lab import SomaLabSession
from motionbricks.data.soma_static_execution import restore_native_transformers,use_dense_transformers
from scripts.collect_horizon_gait import metrics
from scripts.soma_pfnn_maps_server import PFNNMapRuntime


def save(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def summary(values):return dict(n=len(values),p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)))


def session_for(runtime,engine,tokens,seed_backend=None):
    session=SomaLabSession(runtime,'terrain')
    session.backend=DualHorizonBackend.from_backend(seed_backend or session.backend,engine)
    session.backend.configure_policy(policy(tokens))
    return session


def packet(session,index):
    meta,payload=session.plan(dict(type='plan',mode='terrain',request_id=index,velocity_xz=[0.,3.],heading=0.))
    assert meta['generation']['runtime_controls']['num_tokens']==8
    assert meta['generation']['predicted_frames']==32 and meta['generation']['internal_replans']==2
    pending=session.pending_playback
    data=np.frombuffer(payload,dtype='<f4').reshape(8,-1)
    history=pending['history'].clone()
    actual=reencode_played_history(pending['motion'][:,-4:].contiguous(),session.backend.inference.global_motion_rep)
    torch.testing.assert_close(history,actual,atol=1e-5,rtol=0)
    world=pending['motion'].detach().cpu().clone()
    session.acknowledge_played(dict(request_id=index,frame_count=8))
    assert torch.equal(session.backend.history,history)
    return meta,data.copy(),world


def compare_motion(runtime,reference,actual,tokens):
    assert torch.equal(reference['tokens'][:,:tokens],actual['tokens'][:,:tokens])
    poses=[runtime.inference.motion_rep.inverse(row['motion'][:,:tokens*4],False,return_all=False) for row in (reference,actual)]
    fk=float((poses[0]['posed_joints']-poses[1]['posed_joints']).norm(dim=-1).max())
    assert fk<=3e-5
    runner=runtime.inference._terrain_execution
    motion=actual['motion'][:,4:12].contiguous();geometry=runner._geometry(motion)
    pose=runtime.inference.motion_rep.inverse(motion,False,return_all=False)
    vertices=runtime.skinner.skin(pose['global_joint_rots'][0],pose['posed_joints'][0],rot_is_global=True)
    skin=float((runtime.skinner.lbs(geometry['rig'])-vertices).abs().max())
    assert vertices.shape[-2]==18056 and skin<=1e-5
    factual=reencode_played_history(motion[:,-4:].contiguous(),runtime.inference.global_motion_rep)
    history_error=float((geometry['history']-factual).abs().max());assert history_error<=1e-5
    return dict(tokens_equal=True,fk_max_m=fk,skin_max_abs_m=skin,vertices=18056,history_max_abs=history_error)


@torch.inference_mode()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,actual_optimizer_updates=0,policy=policy(16),map='007',speed=3.,
        original_long_parity=[],candidate_parity=[],far_ablation=[],runs=[])
    runtime=None
    try:
        runtime=PFNNMapRuntime(a.bundle,a.catalog,execution='root-path-live',initial_map='007')
        base=runtime.inference._terrain_execution
        engine=DualHorizonReference.from_runner(base)
        before=tensor_digest(runtime.unified_controller.state_dict())
        config=runtime.config();names=config['skeleton']['names']
        # Only reconstruct the64frame prefix needed for exact handoff state; reuse the saved long baseline.
        seed_session=session_for(runtime,engine,8);saved=[];seed_world=[]
        for index in range(8):
            meta,data,world=packet(seed_session,index+1);saved.append((meta,data));seed_world.append(world)
        roots=np.concatenate([data[:,:3] for _,data in saved])
        rig=np.concatenate([data[:,3:].reshape(8,77,4,4) for _,data in saved])
        joints=np.concatenate([np.asarray(meta['debug']['joints_world_xyz_m'],np.float32) for meta,_ in saved])
        with np.load(a.baseline/'frames-8.npz') as reference:
            assert np.array_equal(roots,reference['roots'][:64])
            assert np.array_equal(rig,reference['rig'][:64])
            assert np.array_equal(joints,reference['joints'][:64])
        warm_backend=copy.copy(seed_session.backend)
        warm_backend.history=seed_session.backend.history.clone()
        report['eight_preservation']=dict(frames=64,root_rig_joints_bitwise=True,execution_tokens=8,predicted_frames=32)
        query=runtime.terrain_initial_history.new_tensor([[-1.,.6,1.,.4]])
        histories=[runtime.terrain_initial_history.clone(),warm_backend.history.clone()]
        original_decode=runtime.inference._vqvae_pose_model.forward_decoder
        for hi,history in enumerate(histories):
            goal=online_command_goal(history[...,:5],[0.,3.],0.,16,base.protocol_version)
            captured={}
            def decode(ids,*args,**kwargs):captured['tokens']=ids.clone();return original_decode(ids,*args,**kwargs)
            runtime.inference._vqvae_pose_model.forward_decoder=decode
            try:
                restore_native_transformers(runtime.inference)
                with runtime.unified_controller.context(history,runtime.terrain_scene):
                    native=RootPathTerrainExecution._prediction(base,history,goal,num_tokens=16,query_controls=query)
                native['tokens']=captured['tokens']
            finally:runtime.inference._vqvae_pose_model.forward_decoder=original_decode
            compiled=base._prediction(history,goal,torch.cat((history.new_tensor([[16]]),query),-1))
            report['original_long_parity'].append(dict(history_frame=hi*64,**compare_motion(runtime,native,compiled,16)))
            for tokens in (12,16):
                restore_native_transformers(runtime.inference)
                prepared=engine.prepare_dual(history,goal,query,tokens,native=True)
                native=engine.native_suffix(history,goal,query,prepared)
                assert native['condition_frames']==32 and native['height_features'].shape[1]==32
                torch.testing.assert_close(native['height_features'][0,:,0],(torch.arange(32,device=history.device)-3)/30,atol=0,rtol=0)
                compiled_prepared=engine.prepare_dual(history,goal,query,tokens)
                compiled=engine.compiled_suffix(history,goal,query,compiled_prepared)
                report['candidate_parity'].append(dict(history_frame=hi*64,lookahead_tokens=tokens,
                    **compare_motion(runtime,native,compiled,8)))
                # Same short/long Roots, same history and query. Only far actor observations are withheld.
                removed=engine.native_suffix(history,goal,query,prepared,remove_far=True)
                assert torch.equal(native['root'],removed['root']) and torch.equal(native['local'],removed['local'])
                assert torch.equal(native['height_features'],removed['height_features'])
                memory_delta=float((native['memory']-removed['memory']).abs().max())
                logits_delta=float((native['logits']-removed['logits']).abs().max())
                assert memory_delta>0 and logits_delta>0
                report['far_ablation'].append(dict(history_frame=hi*64,lookahead_tokens=tokens,
                    actor_memory_max_abs_delta=memory_delta,pose_logits_max_abs_delta=logits_delta,
                    active_token_changes=int((native['tokens'][:,:8]!=removed['tokens'][:,:8]).sum()),
                    short_root_local_and_height_time_bitwise=True,condition_frames=32,
                    terminal_frame=31,near_frames=[4,12,20,27],far_frames=list(FAR_FRAMES[tokens])))
        use_dense_transformers(runtime.inference)
        engine.validation=report['candidate_parity']
        save(a.output/'proofs.json',report)
        for tokens in (12,16):
            for mode,count in (('cold',24),('warm',16)):
                session=session_for(runtime,engine,tokens,warm_backend if mode=='warm' else None)
                initial=session.backend.history.clone();raw=[];records=[];packets=[];worlds=[]
                for index in range(count):
                    started=time.perf_counter();meta,data,world=packet(session,index+1)
                    raw.append((time.perf_counter()-started)*1000);records.append((meta,data));worlds.append(world)
                    packets.append(dict(request_id=index+1,generation=meta['generation'],
                        observations=meta['debug']['internal_observations']))
                root=np.concatenate([data[:,:3] for _,data in records])
                rig=np.concatenate([data[:,3:].reshape(8,77,4,4) for _,data in records])
                joint=np.concatenate([np.asarray(meta['debug']['joints_world_xyz_m'],np.float32) for meta,_ in records])
                name=f'{mode}-{tokens}'
                np.savez_compressed(a.output/f'{name}.npz',roots=root,rig=rig,joints=joint,
                    initial_root=initial[0,-1,:3].cpu().numpy())
                torch.save(torch.cat((initial.cpu(),*worlds),1),a.output/f'{name}-world.pt')
                result=dict(mode=mode,lookahead_tokens=tokens,execute_tokens=8,timing_scope='instrumented_with_reference_checks',
                    metrics=metrics(root,joint,initial[0,-1,:3].cpu().numpy(),names),
                    packet_ms=raw,timing_excluding_first3=summary(raw[3:]),packets=packets)
                save(a.output/f'{name}.json',result)
                report['runs'].append({k:v for k,v in result.items() if k not in ('metrics','packets')})
                save(a.output/'progress.json',report)
                print(json.dumps(dict(run=name,mean_speed=result['metrics']['root_speed_mean'],
                    progress=result['metrics']['longitudinal_progress_m'],timing=result['timing_excluding_first3'])),flush=True)
        session=session_for(runtime,engine,8)
        meta,_=session.plan(dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,3.],heading=0.))
        transitions=[]
        for source,tokens in enumerate((12,16,8),1):
            backend=session.backend;pending=session.pending_playback;old_policy=dict(backend.planning_policy);old_view=backend.execution
            serial=backend._future_backend(pending['history']);serial.configure_policy(policy(tokens))
            expected=serial.predict_plan([[0.,3.]],[0.])
            meta,_=session.prefetch(dict(type='prefetch',mode='terrain',request_id=source+1,source_request_id=source,
                played_frame_count=4,velocity_xz=[0.,3.],heading=0.,planning_policy=policy(tokens)))
            assert backend.planning_policy==old_policy and backend.execution is old_view
            candidate=pending['prepared']['candidate']
            assert torch.equal(candidate['planned_motion'],expected['planned_motion'])
            assert torch.equal(candidate['published_history'],expected['published_history'])
            assert candidate['governor_state_after']==expected['governor_state_after']
            session.acknowledge_played(dict(request_id=source,frame_count=8))
            assert backend.planning_policy==policy(tokens) and backend.num_tokens==8
            assert backend.last_generation['predicted_frames']==32
            transitions.append(dict(lookahead_tokens=tokens,serial_prefetch_bitwise=True,count_and_cache_execute8=True))
        session.acknowledge_played(dict(request_id=4,frame_count=8))
        report.update(transitions=transitions,completed=True,weights_unchanged=tensor_digest(runtime.unified_controller.state_dict())==before)
        assert report['weights_unchanged']
        save(a.output/'complete.json',report)
    except Exception as error:
        report['error']=repr(error);save(a.output/'failure.json',report);raise
    finally:
        if runtime is not None:runtime.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
