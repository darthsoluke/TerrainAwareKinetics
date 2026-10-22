"""Compare real server packets across governor-only mid-packet changes."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time

import aiohttp


async def sequence(client,url,controls,defaults,prefetch):
    rows=[]
    async with client.ws_connect(url+'/ws',max_msg_size=16*1024*1024) as ws:
        assert (await ws.receive_json())['type']=='ready'
        await ws.send_json(dict(type='reset',mode='terrain'))
        assert (await ws.receive_json())['type']=='ready'
        for index,enabled in enumerate((False,True,False,True,False),1):
            message=dict(type='prefetch' if prefetch and index>1 else 'plan',mode='terrain',
                request_id=index,velocity_xz=[0.,3.],heading=0.,runtime_controls=controls,
                speed_governor=dict(defaults,enabled=enabled))
            if message['type']=='prefetch':
                message.update(source_request_id=index-1,played_frame_count=4)
            started=time.perf_counter()
            await ws.send_json(message)
            meta=await ws.receive_json()
            assert meta['type']=='plan',meta
            packet=await ws.receive()
            assert packet.type==aiohttp.WSMsgType.BINARY
            elapsed=(time.perf_counter()-started)*1000
            generation=meta['generation']
            assert generation['speed_governor']['enabled'] is enabled
            assert generation['runtime_controls']==controls
            assert generation['neural_execution']=='split-cuda-graphs'
            steps=meta['debug'].get('internal_observations',[meta['debug']])
            governors=[row['speed_governor'] for row in steps]
            assert all(row['enabled'] is enabled for row in governors)
            assert governors[0]['source_frame']==(index-1)*8
            if enabled:assert any(row['target_speed_mps']<row['requested_speed_mps'] for row in governors)
            rows.append(dict(request_id=index,enabled=enabled,metadata=meta,
                payload_sha256=hashlib.sha256(packet.data).hexdigest(),request_to_binary_ms=elapsed))
            if prefetch and index>1:
                # The candidate already exists while its predecessor is still unacknowledged.
                await ws.send_json(dict(type='played',request_id=index-1,frame_count=8))
                assert (await ws.receive_json())['type']=='played'
            if not prefetch or index==5:
                await ws.send_json(dict(type='played',request_id=index,frame_count=8))
                assert (await ws.receive_json())['type']=='played'
    return rows


async def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    async with aiohttp.ClientSession() as client:
        async with client.get(a.url+'/api/config') as response:config=await response.json()
        assert config['map_catalog']['current']=='007'
        controls=dict(config['runtime_controls']['defaults'])
        assert controls['refresh_frames']==4 and controls['num_tokens']==8 and controls['inference_steps']==4
        defaults=config['runtime_controls']['speed_governor']['defaults']
        assert defaults['enabled'] is False
        serial=await sequence(client,a.url,controls,defaults,False)
        prefetched=await sequence(client,a.url,controls,defaults,True)
        assert [r['payload_sha256'] for r in serial]==[r['payload_sha256'] for r in prefetched]
        result=dict(completed=True,map='007',runtime_controls=controls,governor_defaults=defaults,
            serial=serial,prefetched=prefetched,payload_bitwise=True,toggle_count=4,
            prefetched_before_ack=True,actual_optimizer_updates=0)
        (a.output/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({key:value for key,value in result.items() if key not in ('serial','prefetched')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:28944')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    asyncio.run(main(args))
