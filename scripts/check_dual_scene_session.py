"""One live007->005->007binding and stale-session rejection check."""
import argparse
import asyncio
import json
from pathlib import Path
import aiohttp


async def main(a):
    async with aiohttp.ClientSession() as client:
        async with client.get(a.url+'/api/tuning') as response:initial=await response.json()
        assert initial['map_id']=='007'
        async with client.ws_connect(a.url+'/ws',max_msg_size=16*1024*1024) as old:
            await old.receive_json()
            try:
                async with client.post(a.url+'/api/maps/select?scene_only=1&cached=1',json={'map_id':'005'}) as response:
                    assert response.status==200;changed=await response.json()
                await old.send_json(dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,0.],heading=0.))
                rejected=await old.receive_json();assert rejected['type']=='error' and 'Map changed' in rejected['message']
            finally:
                async with client.post(a.url+'/api/maps/select?scene_only=1&cached=1',json={'map_id':'007'}) as response:
                    assert response.status==200;restored=await response.json()
        async with client.ws_connect(a.url+'/ws',max_msg_size=16*1024*1024) as new:
            await new.receive_json()
            p=dict(initial['runtime']['planning_policy']['defaults'],lookahead_tokens=16)
            await new.send_json(dict(type='plan',mode='terrain',request_id=1,velocity_xz=[0.,0.],heading=0.,planning_policy=p))
            meta=await new.receive_json();binary=await new.receive()
            assert meta['type']=='plan' and binary.type==aiohttp.WSMsgType.BINARY
            assert meta['generation']['planning_policy']==p and meta['generation']['predicted_frames']==32
            assert meta['generation']['runtime_controls']['num_tokens']==8
            origin=meta['debug']['observation_root_world_xyz_m'];assert origin[0]==origin[2]==0.
            await new.send_json(dict(type='played',request_id=1,frame_count=8));assert (await new.receive_json())['type']=='played'
        result=dict(completed=True,sequence=['007','005','007'],old_session_response=rejected,
            new_generation=meta['generation'],new_observation_origin=origin,actual_optimizer_updates=0,
            restored_map=restored['map_catalog']['current'])
        a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:28944');p.add_argument('--output',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
