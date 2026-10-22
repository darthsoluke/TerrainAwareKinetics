"""Record real live-server plans for controlled gait/terrain diagnosis."""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
import numpy as np


async def record(url, output, selected=None):
    output.mkdir(parents=True, exist_ok=False)
    async with aiohttp.ClientSession() as client:
        async with client.get(url + '/api/config') as response:
            config = await response.json()
        names = config['skeleton']['names']
        cases = [
            ('flat_default', 'hub', 1.6, {}),
            ('flat_long', 'hub', 1.6, {'num_tokens': 16}),
            ('flat_buffer24', 'hub', 1.6, {'refresh_frames': 24}),
            ('flat_denoise8', 'hub', 1.6, {'inference_steps': 8}),
            ('stairs_default', 'zone-03', 1.6, {}),
            ('stairs_long', 'zone-03', 1.6, {'num_tokens': 16}),
            ('platform_default', 'zone-15', 1.6, {}),
            ('platform_long', 'zone-15', 1.6, {'num_tokens':16}),
            ('high_platform_default', 'zone-16', 1.6, {}),
        ]
        if selected:
            cases=[case for case in cases if case[0] in selected]
        reports = []
        for name, spawn, speed, changes in cases:
            settings = {**config['runtime_controls']['defaults'], **changes}
            plans, arrays, joints = [], [], []
            async with client.ws_connect(url + '/ws', max_msg_size=16*1024**2) as socket:
                await socket.receive_json()
                await socket.send_json({'type': 'reset', 'mode': 'terrain', 'spawn_id': spawn})
                ready = await socket.receive_json()
                if ready['type'] != 'ready':
                    raise RuntimeError(ready)
                for i in range(45):
                    await socket.send_json({'type': 'plan', 'mode': 'terrain', 'request_id': i+1,
                        'velocity_xz': [0., speed], 'heading': 0., 'runtime_controls': settings})
                    meta = await socket.receive_json()
                    if meta['type'] != 'plan':
                        raise RuntimeError(meta)
                    binary = await socket.receive()
                    array = np.frombuffer(binary.data, dtype='<f4').reshape(meta['frame_count'], -1).copy()
                    if len(array) != 8 or not np.isfinite(array).all():
                        raise ValueError('Invalid live publication')
                    if meta['generation']['runtime_controls'] != settings:
                        raise ValueError('Settings were not applied')
                    arrays.append(array)
                    joints.extend(meta['debug']['joints_world_xyz_m'])
                    plans.append(meta)
                    await socket.send_json({'type': 'played', 'request_id': i+1, 'frame_count': 8})
                    ack = await socket.receive_json()
                    if ack['type'] != 'played':
                        raise RuntimeError(ack)
            frames = np.concatenate(arrays)
            jp = np.asarray(joints)
            root = frames[:, :3]
            report = {'name': name, 'spawn': spawn, 'requested_speed_mps': speed,
                'runtime_controls': settings, 'frames': len(frames),
                'progress_z_m': float(root[-1, 2]-ready['root'][2]),
                'root_y_range_m': [float(root[:, 1].min()), float(root[:, 1].max())],
                'root_speed_last3s_mps': float(np.linalg.norm(np.diff(root[-91:, [0, 2]], axis=0), axis=-1).mean()*30),
                'joint_names': names}
            feet = [next(i for i, n in enumerate(names) if n.lower() == side+'foot') for side in ('left', 'right')]
            relative_z = jp[-180:, feet, 2] - root[-180:, 2, None]
            report['feet_relative_forward_correlation_last6s'] = float(np.corrcoef(relative_z.T)[0, 1])
            report['feet_forward_separation_rms_last6s_m'] = float(np.sqrt(np.mean(np.diff(relative_z, axis=1)**2)))
            report['feet_relative_forward_std_last6s_m'] = relative_z.std(axis=0).tolist()
            report['foot_y_ranges_m'] = [[float(jp[:, i, 1].min()), float(jp[:, i, 1].max())] for i in feet]
            np.savez_compressed(output/(name+'.npz'), frames=frames, joints=jp)
            (output/(name+'.json')).write_text(json.dumps({'ready':ready,'plans':plans,'summary':report}),encoding='utf-8')
            reports.append(report)
            print(json.dumps(report), flush=True)
        (output/'summary.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:28934')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases',nargs='+')
    args = parser.parse_args()
    asyncio.run(record(args.url, args.output,args.cases))
