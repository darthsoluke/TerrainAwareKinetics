"""Select real contiguous idle/start/move sources; no synthetic pose bridges."""

import argparse
from pathlib import Path

import torch

from motionbricks.data.bones_seed_rehearsal import BonesSeedPool
from motionbricks.data.bones_controller_episode import episode_controls
from motionbricks.data.native_mb_data import build_native_representation
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    rep = build_native_representation(args.stats)
    pool = BonesSeedPool(args.dataset, rep, args.stats)
    chosen = []
    for category in ('idle', 'start', 'walk', 'run'):
        indices = sorted(pool.groups[category], key=lambda i:(
            -pool.datasets['train'].records[i]['frames_30hz'],
            pool.datasets['train'].records[i]['source_id']))
        selected = 0
        for index in indices:
            item = pool._item('train', index)
            raw = item['world']; maximum = 513 if category == 'idle' else 97
            if len(raw) < 97: continue
            if category == 'start':
                start = 0
            else:
                start = max(0, (len(raw)-maximum)//2)
            length = min(maximum, len(raw)-start)
            segments = (length-33)//8+1
            length = (segments-1)*8+33
            world = raw[start:start+length].clone()
            velocity = torch.diff(world[:, [0, 2]], dim=0)*30
            speed = velocity.norm(dim=-1)
            displacement = (world[-1, [0, 2]]-world[3, [0, 2]]).norm()
            if category == 'idle' and (float((world[:, [0, 2]]-world[3, [0, 2]]).norm(dim=-1).max()) > .15
                    or float(torch.quantile(speed, .9)) > .15): continue
            if category == 'start' and (float(speed[:3].mean()) > .12 or float(displacement) < .8): continue
            if category in ('walk', 'run'):
                if float(speed[:3].mean()) < .3 or float(displacement) < 1: continue
                efficiency = float(displacement/(velocity[3:].norm(dim=-1).sum()/30).clamp_min(1e-5))
                if efficiency < .85: continue
            controls = episode_controls(world, category, segments, raw)
            chosen.append(dict(world=world, scene=item['scene'], controls=controls,
                category=category, label=item['row']['source_id'], row=item['row'],
                source_start_frame=start, training_segments=segments))
            selected += 1
            if selected == 2: break
        if selected != 2: raise ValueError('Insufficient verified real '+category+' episodes')
    torch.save(chosen, args.output/'cases.pt')
    write_json(args.output/'complete.json', dict(cases_sha256=sha256(args.output/'cases.pt'),
        source_manifest_sha256=pool.contract['manifest_sha256'],
        cases=[{k:c[k] for k in ('label','category','source_start_frame','training_segments')} for c in chosen],
        source_frames=[len(c['world']) for c in chosen], source_split='train',
        selection='two long rest, stationary-start, efficient walk/run sources each; no repeated/padded frames',
        control='weak planar schedule with initial body/travel offset, no future body/height',
        original_held_families=pool.contract['reserved_recording_groups'], optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','stats','output'): parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
