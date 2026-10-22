"""Read short independent closed-loop sequences from the existing live server."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import aiohttp
import numpy as np


def save(path,data):path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def episodes(mask,minimum=4):
    spans=[];start=None
    for index,value in enumerate(np.r_[mask,False]):
        if value and start is None:start=index
        if not value and start is not None:
            if index-start>=minimum:spans.append(dict(start_frame=start+1,end_frame=index,seconds=(index-start)/30))
            start=None
    return spans


def metrics(roots,joints,initial_root,names):
    velocity=np.diff(np.concatenate((np.asarray(initial_root)[None],roots)),axis=0)*30
    speed=np.linalg.norm(velocity[:,[0,2]],axis=-1)
    ids=[names.index(name) for name in ('LeftLeg','LeftShin','LeftFoot','LeftToeBase','RightLeg','RightShin','RightFoot','RightToeBase')]
    relative=joints[:,ids]-roots[:,None]
    leg_speed=np.r_[np.nan,np.sqrt(np.mean(np.sum(np.diff(relative,axis=0)**2,axis=-1),axis=-1))*30]
    moving=np.arange(len(roots))>=30
    return dict(root_planar_speed_mps=speed.tolist(),root_forward_speed_mps=velocity[:,2].tolist(),
        leg_relative_rms_speed_mps=[None if not np.isfinite(v) else float(v) for v in leg_speed],
        feet_relative_xyz_m=relative[:,[2,6]].tolist(),root_speed_mean=float(speed.mean()),
        root_speed_p05=float(np.percentile(speed,5)),root_speed_max=float(speed.max()),
        root_slow_episodes=episodes(moving&(speed<.35)),
        leg_still_episodes=episodes(moving&(leg_speed<.15)),
        translating_leg_still_episodes=episodes(moving&(speed>.5)&(leg_speed<.15)),
        longitudinal_progress_m=float(roots[-1,2]-initial_root[2]),
        max_lateral_m=float(np.abs(roots[:,0]-initial_root[0]).max()))


async def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,map='007',requested_speed_mps=a.speed,packets_per_arm=a.packets,
        warmup_packets=a.warmup_packets,simulated_seconds=a.packets*8/30,
        wall_delay_seconds=a.delay,actual_optimizer_updates=0,arms=[])
    try:
        async with aiohttp.ClientSession() as client:
            async with client.get(a.url+'/api/config') as response:config=await response.json()
            assert config['map_catalog']['current']=='007'
            report['checkpoint_sha256']=config['live_model'].get('candidate_sha256')
            report['initial_root']=config['initial_root'];report['skeleton']=config['skeleton']
            for tokens in a.tokens:
                async with client.get(a.url+'/api/tuning') as response:current=await response.json()
                assert current['map_id']=='007','User changed maps; do not redirect their scene'
                controls=dict(current['runtime']['defaults'],num_tokens=tokens,refresh_frames=4,inference_steps=4)
                governor=dict(current['runtime']['speed_governor']['defaults'],enabled=False)
                packets=[];roots=[];rigs=[];joints=[]
                async with client.ws_connect(a.url+'/ws',max_msg_size=16*1024*1024) as ws:
                    assert (await ws.receive_json())['type']=='ready'
                    await ws.send_json(dict(type='reset',mode='terrain'))
                    assert (await ws.receive_json())['type']=='ready'
                    for index in range(a.warmup_packets+a.packets):
                        active_controls=dict(controls,num_tokens=8 if index<a.warmup_packets else tokens)
                        started=time.perf_counter()
                        await ws.send_json(dict(type='plan',mode='terrain',request_id=index+1,
                            velocity_xz=[0.,a.speed],heading=0.,runtime_controls=active_controls,speed_governor=governor))
                        meta=await ws.receive_json();assert meta['type']=='plan',meta
                        binary=await ws.receive();assert binary.type==aiohttp.WSMsgType.BINARY
                        data=np.frombuffer(binary.data,dtype='<f4').reshape(8,-1)
                        assert data.shape[1]==3+77*16
                        roots.append(data[:,:3].copy());rigs.append(data[:,3:].reshape(8,77,4,4).copy())
                        joints.append(np.asarray(meta['debug']['joints_world_xyz_m'],dtype=np.float32))
                        generation=meta['generation']
                        assert generation['runtime_controls']==active_controls and generation['internal_replans']==2
                        assert generation['neural_generation'] and generation['neural_execution']=='split-cuda-graphs'
                        observations=meta['debug']['internal_observations'];assert len(observations)==2
                        steps=[]
                        for sub,observation in enumerate(observations):
                            info=observation['speed_governor']
                            assert not info['enabled'] and abs(info['target_speed_mps']-a.speed)<1e-6
                            assert info['source_frame']==index*8+sub*4
                            path=np.asarray(observation['root_stage_world_xyz_m'])
                            origin=np.asarray(observation['observation_root_world_xyz_m'])
                            pv=np.diff(np.concatenate((origin[None],path)),axis=0)*30
                            steps.append(dict(source_frame=info['source_frame'],root_xyz_m=path.tolist(),
                                root_planar_speed_mps=np.linalg.norm(pv[:,[0,2]],axis=-1).tolist(),
                                decoded_xyz_m=observation['decoded_world_xyz_m'],
                                input_intent=observation['input_intent'],governor=info))
                        packets.append(dict(request_id=index+1,start_frame=index*8+1,end_frame=(index+1)*8,
                            generation=generation,steps=steps,request_to_binary_ms=(time.perf_counter()-started)*1000))
                        await ws.send_json(dict(type='played',request_id=index+1,frame_count=8))
                        assert (await ws.receive_json())['type']=='played'
                        await asyncio.sleep(a.delay)
                root=np.concatenate(roots);joint=np.concatenate(joints)
                cut=a.warmup_packets*8
                if cut:
                    with np.load(a.warmup_reference) as reference:
                        assert np.array_equal(root[:cut],reference['roots'][:cut])
                        assert np.array_equal(joint[:cut],reference['joints'][:cut])
                np.savez_compressed(a.output/f'frames-{tokens}.npz',roots=root,joints=joint,rig=np.concatenate(rigs))
                arm=dict(tokens=tokens,controls=controls,governor=governor,packets=packets,
                    target_start_frame=cut+1,warmup_reference_bitwise=bool(cut),
                    metrics=metrics(root[cut:],joint[cut:],root[cut-1] if cut else config['initial_root'],config['skeleton']['names']))
                save(a.output/f'arm-{tokens}.json',arm)
                report['arms'].append({key:value for key,value in arm.items() if key!='packets'})
                save(a.output/'progress.json',report)
                print(json.dumps(dict(tokens=tokens,summary={k:v for k,v in arm['metrics'].items()
                    if k not in ('root_planar_speed_mps','root_forward_speed_mps','leg_relative_rms_speed_mps','feet_relative_xyz_m')})),flush=True)
        report['completed']=True;save(a.output/'complete.json',report)
    except Exception as error:
        report['error']=repr(error);save(a.output/'failure.json',report);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:28944')
    p.add_argument('--speed',type=float,default=3.);p.add_argument('--delay',type=float,default=.5)
    p.add_argument('--tokens',type=int,nargs='+',default=[8,12,16]);p.add_argument('--packets',type=int,default=24)
    p.add_argument('--warmup-packets',type=int,default=0);p.add_argument('--warmup-reference',type=Path)
    p.add_argument('--output',type=Path,required=True);asyncio.run(main(p.parse_args()))
