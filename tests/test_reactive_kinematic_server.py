import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np

from scripts.soma_reactive_server import ReactiveConnection


def test_inflight_compute_does_not_block_or_queue_obsolete_intents(tmp_path):
    async def exercise():
        entered, release = threading.Event(), threading.Event()
        requests = []

        class Engine:
            def generate(self, request):
                requests.append(request)
                if len(requests)==1:
                    entered.set()
                    assert release.wait(3)
                return dict(motion=np.ones((24,6),np.float32),
                    rig=np.zeros((24,77,4,4),np.float32),joints=np.zeros((24,30,3),np.float32),
                    timing=dict(history_ms=0.,prediction_ms=0.,geometry_ms=0.),payload=b'test')

        class Socket:
            closed = False
            def __init__(self): self.metadata=[]
            async def send_json(self,value): self.metadata.append(value)
            async def send_bytes(self,value):
                meta=self.metadata[-1]
                await self.connection.receive(dict(type='accept',episode_id=meta['episode_id'],
                    plan_id=meta['plan_id'],first_frame=meta['first_frame']))
                if len(self.metadata) == 1:
                    for frame in range(1, 5):
                        await self.connection.receive(dict(type='played', episode_id=meta['episode_id'],
                            frame=frame, plan_id=meta['plan_id'], client_render_ms=frame*1000/30))

        runtime=SimpleNamespace(max_speed_mps=12.,scene_revision=1,current_map='fixture',
            vertex_count=18056,catalog={'checkpoint_sha256':'fixture'})
        socket=Socket()
        with ThreadPoolExecutor(max_workers=1) as executor:
            connection=ReactiveConnection(runtime,Engine(),socket,executor,np.zeros((4,6),np.float32),tmp_path/'session')
            socket.connection=connection
            await connection.receive(dict(type='intent',seq=1,velocity_xz=[0.,1.],heading=0.,client_input_ms=1.))
            await connection.receive(dict(type='start'))
            assert await asyncio.to_thread(entered.wait,3)
            for seq in (2,3):
                await connection.receive(dict(type='intent',seq=seq,velocity_xz=[float(seq),0.],heading=0.,client_input_ms=float(seq)))
            release.set()
            async def second():
                while len(socket.metadata)<2:
                    await asyncio.sleep(.001)
            await asyncio.wait_for(second(),3)
            await asyncio.gather(connection.close(),connection.close())
        assert [request.input_seq for request in requests]==[1,3]
        assert [request.played_frame for request in requests] == [0, 4]
        assert np.count_nonzero(requests[0].played_history) == 0
        assert np.all(requests[1].played_history == 1)
        assert requests[0].origin_frame==0
        assert requests[1].origin_frame==4+len(requests[1].prefix_plan_ids)
        assert [value['input_seq'] for value in socket.metadata]==[1,3]
        assert connection.journal.closed

    asyncio.run(exercise())
