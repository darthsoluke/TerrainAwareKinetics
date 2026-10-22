"""Actual loopback HTTP/WebSocket and official SOMA rig checks on a bound candidate."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def check_idle(gpu):
    snapshot=subprocess.check_output(['nvidia-smi','-i',str(gpu),
        '--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    uuid,memory,busy=[x.strip() for x in snapshot.split(',')]
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    if int(memory)>500 or int(busy)>5 or uuid in processes.splitlines():raise ValueError('GPU no longer idle')
    return snapshot


async def main(a):
    if a.full_candidate and a.artifact is None:
        raise ValueError('A full candidate requires an explicitly packaged final artifact')
    expected_mechanics = not a.full_candidate
    snapshot=check_idle(a.gpu)
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu);os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    save(a.output/'state.json',dict(status='packaging_fixture',gpu=a.gpu,gpu_snapshot=snapshot,actual_optimizer_updates=0))
    base=Path('${PRIVATE_STORAGE}/motionbricks-20261010');run=base/'pfnn_mixed_20261015'
    bundle=base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'
    artifact=a.artifact if a.artifact is not None else a.output/'artifact'
    if a.artifact is None:
        subprocess.run([sys.executable,'-m','scripts.package_full_corpus_acceptance','--bundle',str(bundle),
            '--checkpoint',str(run/'mechanics_world_v1/pose/weights.pt'),'--output',str(artifact),'--allow-mechanics'],check=True)
    check_idle(a.gpu)
    import numpy as np
    import torch
    import aiohttp
    from aiohttp import web
    from motionbricks.data.soma_lab import SomaLabSession
    from motionbricks.data.full_corpus_inference import tensor_digest
    from motionbricks.data.unified_terrain import PROFILE
    from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
    from scripts.soma_lab_server import create_app,ROOT
    from scripts.train_unified_full_corpus import rng_state,restore_rng
    from scripts.train_native_mb import sha256

    runtime=FullCorpusAcceptanceRuntime(bundle,artifact,'cuda:0',allow_mechanics=expected_mechanics,acceptance_test=True,
        execution=a.execution)
    if a.execution=='verified-cuda' and not getattr(runtime.inference,'_terrain_execution',None):
        raise ValueError('The requested CUDA execution engine was not installed')
    initial_hash=tensor_digest(runtime.unified_controller.state_dict())
    app=create_app(runtime,ROOT/'web/full_corpus_lab');runner=web.AppRunner(app)
    rows=[];rejections=[]
    try:
        await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1];url=f'http://127.0.0.1:{port}'
        save(a.output/'state.json',dict(status='checking_loopback_protocol',port=port,actual_optimizer_updates=0))
        async with aiohttp.ClientSession() as client:
            for route in ('/','/app.js','/styles.css','/vendor/three.module.js'):
                async with client.get(url+route) as response:
                    if response.status!=200:raise ValueError('Viewer asset unavailable: '+route)
                    await response.read()
            async with client.get(url+'/api/config') as response:config=await response.json()
            if (config['runtime_controls']['defaults']!=PROFILE or not config['runtime_controls']['locked']
                    or config['live_model']['quality_approved'] or config['live_model']['mechanics_only']!=expected_mechanics):
                raise ValueError('Acceptance config or fixed runtime profile differs')
            async with client.ws_connect(url+'/ws') as socket:
                ready=await socket.receive_json(timeout=60)
                if ready['type']!='ready' or ready['fps']!=30 or ready['vertex_count']!=18056:
                    raise ValueError('Official SOMA ready packet differs')
                serial=SomaLabSession(runtime,'terrain')
                invalid=dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,1.6],heading=0.,
                    runtime_controls=dict(PROFILE,num_tokens=16))
                await socket.send_json(invalid);error=await socket.receive_json(timeout=60)
                if error['type']!='error':raise ValueError('Runtime drift was not rejected')
                rejections.append('changed_runtime_profile')

                async def compare_plan(message,prefetch=False):
                    before=rng_state()
                    expected,payload=(serial.prefetch(message) if prefetch else serial.plan(message))
                    expected_rng=rng_state();restore_rng(before)
                    await socket.send_json(message)
                    observed=await socket.receive_json(timeout=60)
                    if observed.get('type')!='plan':raise ValueError('Expected a real model packet: '+str(observed))
                    binary=await socket.receive(timeout=60)
                    if binary.type!=aiohttp.WSMsgType.BINARY or binary.data!=payload:
                        raise ValueError('HTTP/WebSocket path differs from serial model geometry')
                    for key in ('fps','frame_count','vertex_count','geometry_format','frame_stride_floats'):
                        if observed[key]!=expected[key]:raise ValueError('Packet contract differs: '+key)
                    after=rng_state()
                    if not torch.equal(after['torch'],expected_rng['torch']) or not all(
                            torch.equal(x,y) for x,y in zip(after['cuda'],expected_rng['cuda'])):
                        raise ValueError('Transport inference consumed a different RNG stream')
                    future=(serial.pending_playback['prepared']['motion'] if prefetch else serial.pending_playback['motion'])
                    packed=np.frombuffer(binary.data,dtype='<f4').reshape(8,3+77*16)
                    rig=torch.from_numpy(packed[:,3:].reshape(8,77,4,4).copy()).to('cuda:0')
                    actual_vertices=runtime.skinner.lbs(rig)
                    _,reference_vertices,_=runtime.skin_features_with_joints(future,runtime.inference)
                    reference_vertices=torch.as_tensor(reference_vertices,device='cuda:0')
                    torch.testing.assert_close(actual_vertices,reference_vertices,atol=1e-5,rtol=1e-5)
                    rows.append(dict(request_id=message['request_id'],prefetched=prefetch,frames=8,
                        bytes=len(binary.data),serial_payload_bitwise_equal=True,rng_equal=True,
                        official_skin_max_abs_m=float((actual_vertices-reference_vertices).abs().max())))

                await compare_plan(dict(type='plan',mode='terrain',request_id=1,
                    velocity_xz=[0.,1.6],heading=0.,runtime_controls=dict(PROFILE)))
                await socket.send_json(dict(type='played',request_id=1,frame_count=7))
                if (await socket.receive_json(timeout=60))['type']!='error':raise ValueError('Partial playback was accepted')
                rejections.append('incomplete_playback_acknowledgment')
                await compare_plan(dict(type='prefetch',mode='terrain',request_id=2,source_request_id=1,
                    played_frame_count=0,velocity_xz=[0.,0.],heading=0.,runtime_controls=dict(PROFILE)),True)
                for request_id in (1,2):
                    message=dict(type='played',request_id=request_id,frame_count=8)
                    expected=serial.acknowledge_played(message);await socket.send_json(message)
                    if await socket.receive_json(timeout=60)!=expected:raise ValueError('Playback commit differs')
                await compare_plan(dict(type='plan',mode='terrain',request_id=3,
                    velocity_xz=[1.6,0.],heading=float(np.pi/2),runtime_controls=dict(PROFILE)))
                message=dict(type='played',request_id=3,frame_count=8)
                expected=serial.acknowledge_played(message);await socket.send_json(message)
                if await socket.receive_json(timeout=60)!=expected:raise ValueError('Final playback commit differs')
        if tensor_digest(runtime.unified_controller.state_dict())!=initial_hash:
            raise ValueError('Acceptance server changed model tensors')
        save(a.output/'complete.json',dict(passed=True,rows=rows,rejections=rejections,
            actual_generated_frames=24,official_vertex_count=18056,actual_optimizer_updates=0,
            mechanics_only=expected_mechanics,weights_unchanged=True,acceptance_receipt_sha256=sha256(artifact/'acceptance.json'),
            model_state_sha256=initial_hash,script_sha256=sha256(Path(__file__)),
            execution=runtime.terrain_quality['execution'],
            execution_validation=runtime.terrain_quality.get('execution_parity'),
            browser_rendering_validated=False,wasd_keyboard_validated=False,realtime_30fps_validated=False,
            quality_approved=False,delivery_approved=False,
            scope='Loopback assets/config/WebSocket plans, committed feedback and official rig parity only.'))
        save(a.output/'state.json',dict(status='complete',server_stopped=True,actual_optimizer_updates=0))
    finally:
        await runner.cleanup();runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--artifact',type=Path)
    parser.add_argument('--full-candidate',action='store_true')
    parser.add_argument('--execution',choices=('eager','verified-cuda'),default='eager')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    try:asyncio.run(main(args))
    except Exception as error:
        save(args.output/'failure.json',dict(error=repr(error),actual_optimizer_updates=0))
        save(args.output/'state.json',dict(status='failed',error=repr(error)));raise
