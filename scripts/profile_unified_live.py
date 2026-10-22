"""Read-only localhost timing/response comparisons; no pose modifications."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np

from motionbricks.data.native_preserving_terrain import PROFILE


async def run(url, refresh, blocks):
    rows, roots, joints, observations = [], [], [], []
    async with aiohttp.ClientSession() as client:
        async with client.ws_connect(url+'/ws') as socket:
            ready = await socket.receive_json()
            initial = np.asarray(ready['root'])
            for request_id in range(1, blocks+1):
                start = time.perf_counter()
                await socket.send_json(dict(type='plan', mode='terrain', request_id=request_id,
                    velocity_xz=[0., 1.6], heading=0., runtime_controls=dict(PROFILE, refresh_frames=refresh)))
                metadata = await socket.receive_json()
                if metadata.get('type') != 'plan':
                    raise ValueError(metadata)
                binary = await socket.receive_bytes()
                wall = (time.perf_counter()-start)*1000
                array = np.frombuffer(binary, dtype='<f4').reshape(metadata['frame_count'], metadata['frame_stride_floats'])
                roots.extend(array[:, :3].tolist())
                joints.extend(metadata['debug']['joints_world_xyz_m'])
                rows.append(dict(block=request_id, neural=metadata['generation']['neural_generation'],
                    inference_ms=metadata['inference_ms'], skin_ms=metadata['skin_ms'], wall_ms=wall,
                    remaining=metadata['generation']['remaining_frames']))
                if rows[-1]['neural']:
                    debug = metadata['debug']
                    sdf = debug['unified_sdf']
                    points, distances, valid = np.asarray(sdf['world_xyz_m']), np.asarray(sdf['signed_distance_m']), np.asarray(sdf['valid'])
                    root_path, decoded_path = np.asarray(debug['root_stage_world_xyz_m']), np.asarray(debug['decoded_world_xyz_m'])
                    observations.append(dict(frame=(request_id-1)*8, observed_root=sdf['root_world_xyz_m'],
                        sdf_difference_from_world_y0_plane_m=float(np.max(np.abs(distances[valid]-points[valid, 1]))),
                        max_root_plan_y_m=float(root_path[:, 1].max()), max_decoded_plan_y_m=float(decoded_path[:, 1].max())))
                await socket.send_json(dict(type='played', request_id=request_id, frame_count=8))
                await socket.receive_json()
    root = np.asarray(roots)
    joint = np.asarray(joints)
    step = np.linalg.norm(np.diff(root, axis=0), axis=-1)
    joint_step = np.linalg.norm(np.diff(joint, axis=0), axis=-1).mean(-1)
    boundaries = np.arange(7, len(step), 8)
    internal = np.ones(len(step), dtype=bool)
    internal[boundaries] = False
    rise = np.flatnonzero(root[:, 1] > initial[1]+.1)
    hot = [r['wall_ms'] for r in rows[1:] if r['neural']]
    return dict(refresh_frames=refresh, frames=len(root), blocks=rows, observations=observations,
        root_final=root[-1].tolist(), first_root_rise_10cm_frame=int(rise[0]) if len(rise) else None,
        first_root_rise_10cm_z=float(root[rise[0], 2]) if len(rise) else None,
        root_boundary_step_mean_m=float(step[boundaries].mean()), root_internal_step_mean_m=float(step[internal].mean()),
        joint_boundary_step_mean_m=float(joint_step[boundaries].mean()), joint_internal_step_mean_m=float(joint_step[internal].mean()),
        hot_neural_wall_ms_p50=float(np.median(hot)), hot_neural_wall_ms_p95=float(np.quantile(hot, .95)),
        note='Same fixed W request and initial scene; boundary displacement is not a full gait or collision assessment.')


async def main(args):
    result = dict(cases=[await run(args.url, refresh, args.blocks) for refresh in (24, 8)])
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if k not in ('blocks','observations')} for row in result['cases']], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:28941')
    parser.add_argument('--blocks', type=int, default=24)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
