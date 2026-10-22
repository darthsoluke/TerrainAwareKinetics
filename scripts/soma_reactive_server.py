"""Explicit short-feedback experiment using the frozen native SOMA model."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from aiohttp import web, WSMsgType

from motionbricks.data.reactive_kinematic import (
    SCHEMA, STEP_CHOICES, FUTURE_FRAMES, MAX_PREFIX_FRAMES, ReactivePlayback, ReactiveNativeEngine)
from scripts.soma_pfnn_maps_server import PFNNMapRuntime, ROOT
from scripts.soma_lab_server import create_app


class ReactiveRuntimeView:
    def __init__(self, runtime):
        self.runtime = runtime

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def config(self):
        value = self.runtime.config()
        value['reactive_control'] = dict(schema=SCHEMA,step_frames=4,
            choices=list(STEP_CHOICES),future_frames=FUTURE_FRAMES,fps=30,
            max_committed_prefix_frames=MAX_PREFIX_FRAMES,prefix_policy='measured_latency')
        value['runtime_controls']['locked'] = True
        return value

    def tuning_config(self):
        value = self.runtime.tuning_config()
        value['runtime']['locked'] = True
        return value


class ReactiveConnection:
    def __init__(self, runtime, engine, socket, executor, seed, folder):
        self.runtime, self.engine, self.socket, self.executor = runtime, engine, socket, executor
        self.seed = seed
        self.folder = folder
        self.folder.mkdir(exist_ok=False)
        self.journal = (folder/'events.jsonl').open('x',encoding='utf-8',buffering=1024*1024)
        self.playback = ReactivePlayback(seed,max_speed=runtime.max_speed_mps)
        self.scene_revision = runtime.scene_revision
        self.wake = asyncio.Event()
        self.send_lock = asyncio.Lock()
        self.closed = False
        self.closing = False
        self.failed = False
        self.log(dict(kind='session',schema=SCHEMA,map_id=runtime.current_map,
            checkpoint_sha256=runtime.catalog['checkpoint_sha256'],optimizer_updates=0,
            future_frames=24,history_frames=4,pose_passes=4,tokens=8,
            publication='replaceable_absolute_frame_forecasts',seed=seed.tolist()))
        self.task = asyncio.create_task(self.generate())

    def log(self, event):
        self.journal.write(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n')

    async def ready(self):
        value = await asyncio.get_running_loop().run_in_executor(self.executor,self.runtime.ready,'terrain')
        await self.socket.send_json(dict(value,episode_id=self.playback.episode_id))

    async def receive(self, message):
        if self.scene_revision != self.runtime.scene_revision:
            raise ValueError('Scene changed; reconnect to its current seed')
        if self.failed and message['type'] != 'reset':
            return
        state = self.playback
        kind = message['type']
        if kind == 'intent':
            state.receive_intent(message)
            self.log(dict(kind='intent',episode_id=state.episode_id,frame=state.frame,**state.command))
        elif kind == 'start':
            if state.started:
                raise ValueError('Episode already started')
            state.started = True
        elif kind == 'configure':
            state.configure(message['step_frames'])
            self.log(dict(kind='configure',episode_id=state.episode_id,frame=state.frame,
                step_frames=state.step_frames))
        elif kind == 'played':
            state.acknowledge(message['episode_id'],message['frame'],message['plan_id'])
            self.log(dict(kind='played',episode_id=state.episode_id,frame=state.frame,
                plan_id=message['plan_id'],received_ns=time.perf_counter_ns(),
                client_render_ms=message['client_render_ms']))
        elif kind == 'accept':
            state.accept(message['episode_id'],message['plan_id'],message['first_frame'])
            self.log(dict(kind='accepted',episode_id=state.episode_id,frame=state.frame,
                plan_id=message['plan_id'],first_frame=message['first_frame']))
        elif kind == 'reject':
            state.reject(message['episode_id'],message['plan_id'])
            self.log(dict(kind='rejected',episode_id=state.episode_id,frame=state.frame,
                plan_id=message['plan_id'],reason=message['reason']))
        elif kind == 'reset':
            self.log(dict(kind='episode_closed',episode_id=state.episode_id,frame=state.frame))
            self.playback = ReactivePlayback(self.seed,episode_id=state.episode_id+1,
                step_frames=state.step_frames,max_speed=state.max_speed)
            self.failed = False
            async with self.send_lock:
                await self.ready()
        else:
            raise ValueError('Unknown reactive control command')
        self.wake.set()

    async def generate(self):
        loop = asyncio.get_running_loop()
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()
            while not self.closed and not self.failed and self.playback.due():
                state = self.playback
                request = None
                try:
                    request = state.capture()
                    self.log(dict(kind='generation_started',episode_id=state.episode_id,
                        played_frame=request.played_frame,origin_frame=request.origin_frame,
                        input_seq=request.input_seq,step_frames=request.step_frames,
                        prefix_plan_ids=list(request.prefix_plan_ids),requested_ns=request.requested_ns))
                    result = await loop.run_in_executor(self.executor,self.engine.generate,request)
                    if self.closed:
                        break
                    if state is not self.playback or self.scene_revision != self.runtime.scene_revision:
                        self.log(dict(kind='superseded_episode_result',episode_id=request.episode_id,
                            origin_frame=request.origin_frame))
                        continue
                    request_ms=(time.perf_counter_ns()-request.requested_ns)/1e6
                    state.observe_latency(request_ms)
                    plan_id = state.publish(request,result['motion'])
                    if plan_id is None:
                        self.log(dict(kind='missed_handoff',episode_id=state.episode_id,
                            origin_frame=request.origin_frame,ready_frame=state.frame,timing=result['timing']))
                        continue
                    timing = dict(result['timing'],ready_frame=state.frame,
                        age_frames=state.frame-request.played_frame,
                        request_to_ready_ms=request_ms,
                        measured_lead_frames=state.lead_frames,
                        interval_target_met=(state.frame-request.played_frame<=request.step_frames
                            and request_ms<=request.step_frames*1000/30))
                    metadata = dict(type='forecast',schema=SCHEMA,episode_id=state.episode_id,
                        plan_id=plan_id,origin_frame=request.origin_frame,first_frame=request.origin_frame+1,
                        played_frame=request.played_frame,committed_prefix_frames=request.origin_frame-request.played_frame,
                        frame_count=24,fps=30,vertex_count=self.runtime.vertex_count,
                        frame_stride_floats=3+77*16,encoding='float32-le',
                        geometry_format='soma77-rig-v1',rig_count=77,
                        input_seq=request.input_seq,client_input_ms=request.client_input_ms,
                        step_frames=request.step_frames,inference_ms=timing['history_ms']+timing['prediction_ms'],
                        skin_ms=timing['geometry_ms'],timing=timing)
                    async with self.send_lock:
                        await self.socket.send_json(metadata)
                        await self.socket.send_bytes(result['payload'])
                    path = self.folder/f"forecast_{state.episode_id:04d}_{plan_id:06d}.npz"
                    def save():
                        with path.open('xb') as output:
                            np.savez(output,history=request.history,played_history=request.played_history,motion=result['motion'],
                                rig=result['rig'],joints=result['joints'])
                    await loop.run_in_executor(self.executor,save)
                    self.log(dict(kind='forecast',path=str(path),metadata=metadata,
                        sent_ns=time.perf_counter_ns()))
                except Exception as error:
                    if self.closed:
                        break
                    self.failed = True
                    if request is not None:
                        path=self.folder/f'failed_{state.episode_id:04d}_{request.requested_ns}.npz'
                        with path.open('xb') as output:
                            np.savez(output,history=request.history,played_history=request.played_history,
                                controls=np.array([*request.velocity,request.heading]),
                                origin_frame=request.origin_frame,played_frame=request.played_frame)
                    self.log(dict(kind='failure',episode_id=state.episode_id,frame=state.frame,
                        error=f'{type(error).__name__}: {error}'))
                    if not self.socket.closed:
                        async with self.send_lock:
                            await self.socket.send_json(dict(type='error',message=f'{type(error).__name__}: {error}'))

    async def close(self):
        if self.closing:
            await self.task
            return
        self.closing = True
        self.closed = True
        self.wake.set()
        await self.task
        self.log(dict(kind='closed',episode_id=self.playback.episode_id,frame=self.playback.frame))
        self.journal.close()


def main(args):
    if args.port in (28936,28941,28942,28944):
        raise ValueError('Reactive experiment needs its own port')
    args.output.mkdir(parents=True,exist_ok=False)
    runtime = PFNNMapRuntime(args.bundle,args.catalog,'cuda:0','root-path-live',args.initial_map,
        planning_policy='dual',dual_proof=args.dual_proof)
    query_proofs=json.loads(args.scene_query_proofs.read_text())['scene_query_proofs']
    engine = ReactiveNativeEngine(runtime,query_proofs)
    proof = engine.prepare()
    (args.output/'geometry_proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='reactive-mb')
    app = create_app(ReactiveRuntimeView(runtime),ROOT/'web/full_corpus_lab',executor=executor)
    baseline_runner = None
    connections = set()
    serial = 0

    async def websocket(request):
        nonlocal serial
        socket = web.WebSocketResponse(max_msg_size=8192,heartbeat=20)
        await socket.prepare(request)
        await asyncio.get_running_loop().run_in_executor(executor,engine.prepare)
        seed = await asyncio.get_running_loop().run_in_executor(executor,
            lambda:runtime.terrain_initial_history[0].cpu().numpy().copy())
        serial += 1
        connection = ReactiveConnection(runtime,engine,socket,executor,seed,args.output/f'session_{serial:04d}')
        connections.add(connection)
        try:
            await connection.ready()
            async for event in socket:
                if event.type == WSMsgType.TEXT:
                    try:
                        message = json.loads(event.data)
                        await connection.receive(message)
                    except (ValueError,KeyError,TypeError) as error:
                        async with connection.send_lock:
                            await socket.send_json(dict(type='error',message=str(error)))
        finally:
            await connection.close()
            connections.discard(connection)
        return socket

    async def shutdown(_app):
        for connection in list(connections):
            await connection.socket.close(code=1001,message=b'Service shutdown')
            if not connection.closed:
                await connection.close()
        if baseline_runner is not None:
            await baseline_runner.cleanup()

    async def cleanup(_app):
        runtime.close()

    async def start_baseline(_app):
        nonlocal baseline_runner
        baseline = create_app(runtime,ROOT/'web/full_corpus_lab',executor=executor)
        baseline_runner = web.AppRunner(baseline)
        await baseline_runner.setup()
        await web.TCPSite(baseline_runner,'127.0.0.1',args.baseline_port).start()

    app.router.add_get('/ws/reactive',websocket)
    app.on_shutdown.append(shutdown)
    app.on_cleanup.append(cleanup)
    app.on_startup.append(start_baseline)
    (args.output/'service.json').write_text(json.dumps(dict(schema=SCHEMA,port=args.port,baseline_port=args.baseline_port,
        map_id=runtime.current_map,checkpoint_sha256=runtime.catalog['checkpoint_sha256'],
        optimizer_updates=0,step_choices=list(STEP_CHOICES),geometry_proof=proof),indent=2),encoding='utf-8')
    web.run_app(app,host='127.0.0.1',port=args.port)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=28950)
    parser.add_argument('--baseline-port',type=int,default=28944)
    parser.add_argument('--dual-proof',type=Path,required=True)
    parser.add_argument('--scene-query-proofs',type=Path,required=True)
    parser.add_argument('--initial-map',default='007')
    main(parser.parse_args())
