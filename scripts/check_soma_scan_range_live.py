"""Independent diagnostic sessions test the live compiled range contract."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import aiohttp
import numpy as np

from motionbricks.data.terrain_runtime_controls import RUNTIME_DEFAULTS


async def run(url,blocks):
    records=[]
    async with aiohttp.ClientSession() as client:
        for extent,width in ((3.4,.6),(20.,6.)):
            trajectory=[];hashes=[]
            async with client.ws_connect(url.replace('http','ws',1)+'/ws') as socket:
                await socket.receive_json()
                await socket.send_json({'type':'reset','mode':'terrain','spawn_id':'hub'})
                ready=await socket.receive_json()
                assert ready['type']=='ready'
                settings={**RUNTIME_DEFAULTS,'lookahead_m':extent,'lateral_halfwidth_m':width}
                for step in range(blocks):
                    await socket.send_json({'type':'plan','mode':'terrain','request_id':step+1,
                        'velocity_xz':[0.,0.],'heading':0.,'runtime_controls':settings})
                    meta=await socket.receive_json()
                    assert meta['type']=='plan',meta
                    payload=await socket.receive_bytes()
                    hashes.append(hashlib.sha256(payload).hexdigest())
                    values=np.frombuffer(payload,dtype='<f4').reshape(meta['frame_count'],meta['frame_stride_floats'])
                    trajectory.append(values[:,:3].copy())
                    await socket.send_json({'type':'played','request_id':step+1,'frame_count':8})
                    ack=await socket.receive_json()
                    assert ack['type']=='played',ack
                records.append({'survey_extent_m':extent,'survey_halfwidth_m':width,'rig_block_sha256':hashes,
                    'roots':np.concatenate(trajectory).tolist(),'initial_root':ready['root'],
                    'motion_extent_m':meta['debug']['motion_observation_extent_m'],
                    'motion_halfwidth_m':meta['debug']['motion_observation_halfwidth_m'],
                    'last_survey':meta['debug']['terrain_patches'][-1]})
    left,right=records
    assert left['rig_block_sha256']==right['rig_block_sha256']
    difference=float(np.abs(np.asarray(left['roots'])-np.asarray(right['roots'])).max())
    assert difference==0
    return {'status':'passed','simulated_seconds':blocks*8/30,'root_max_difference_m':difference,
            'rig_bytes_identical':True,'scope':'independent generated-FK diagnostic sessions; not idle approval',
            'records':records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:28931')
    parser.add_argument('--blocks',type=int,default=30)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=asyncio.run(run(args.url,args.blocks))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':main()
