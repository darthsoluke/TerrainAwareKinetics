"""Package the accepted actor and factual stair seeds, with no future live inputs."""

import argparse
import configparser
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def descending_passages(world, scene):
    gap, known = heightfield_gap(world[:,:3], scene)
    level = world[:,1]-gap
    delta = torch.diff(level)
    edges = (torch.nonzero((delta < -.05) & (delta > -.26)).flatten()+1).tolist()
    groups = []
    for edge in edges:
        if not groups or edge-groups[-1][-1] > 18 or (delta[groups[-1][-1]:edge] > .04).any():
            groups.append([])
        groups[-1].append(edge)
    result = []
    for group in groups:
        if len(group) < 4:
            continue
        first, last = group[0], group[-1]
        begin, goal = first-20, last+15
        offset = 8*math.ceil(max(0, goal-begin-11)/8)
        end = begin+offset+33
        if begin < 0 or end > len(world) or not known[begin:end].all():
            continue
        if level[begin:first].max()-level[begin:first].min() > .03 or level[last:goal+1].max()-level[last:goal+1].min() > .03:
            continue
        drops = -delta[torch.tensor(group)-1]
        if drops.max()-drops.min() > .025:
            continue
        travel = world[goal,[0,2]]-world[begin+3,[0,2]]
        if travel.norm() < 1.:
            continue
        axis = travel/travel.norm()
        side = torch.stack((-axis[1],axis[0]))
        lateral = ((world[begin:goal+1,[0,2]]-world[begin+3,[0,2]]) @ side).abs().max()
        if lateral > .25:
            continue
        result.append(dict(start=begin, end=end, goal=goal, edge=first, risers=len(group), lateral_m=float(lateral)))
    return result


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    expected = 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf'
    if sha256(args.checkpoint) != expected:
        raise ValueError('Not the user-approved orange actor')
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    bundle = torch.load(saved['contract']['source'], map_location='cpu', mmap=True, weights_only=False)
    dataset = Path(saved['contract']['dataset'])
    manifest_hash = sha256(dataset/'native_manifest.json')
    if manifest_hash != bundle['metadata']['dataset_manifest_sha256']:
        raise ValueError('Scene seed dataset is not the model dependency')
    rep = build_native_representation(dataset/'stats')
    data = TerrainLearningDataset(dataset, rep, 'train')
    geometry = JointGeometryObjective(rep, 'cpu')
    original = torch.load(args.cases, map_location='cpu', weights_only=False)
    up = next(c for c in original if c['label']=='generated_8030ae7a2a7ad640')
    selected = [dict(up, entry='up')]
    audit = []
    order = sorted(range(len(data.records)), key=lambda i: (data.records[i]['source_id'] != up['label'],
                   data.records[i].get('origin')=='bones_seed',data.records[i]['source_id']))
    for index in order:
        row = data.records[index]
        if row.get('origin') == 'bones_seed':
            continue
        item = data.full_reference_item(index)
        for part in descending_passages(item['world'],item['scene']):
            source = item['world'][part['start']:part['end']].clone()
            metrics = exact_skin_metrics(geometry,source[None],item['scene'])
            audit.append(dict(source_id=row['source_id'], interval=part, **metrics))
            if metrics['body_prism_max_m'] > .02 or metrics['unknown_body_fraction'] or metrics['uncertain_inside_samples']:
                continue
            selected.append(dict(item, world=source, name='stairs', label=row['source_id']+'-down', entry='down',
                start_frame=part['start'], edge_frame=part['edge']-part['start'],
                evaluation_goal_source_frame=part['goal'], training_segments=(len(source)-33)//8+1))
            break
        if len(selected) == 2:
            break
    write_json(args.output/'descent_source_audit.json',dict(candidates=audit, found=len(selected)==2))
    if len(selected) != 2:
        raise ValueError('No clean real descending stair passage; do not reverse an ascent recording')
    reports = []
    for case in selected:
        world, edge = case['world'], case['edge_frame']
        velocity = (world[edge-8,[0,2]]-world[3,[0,2]])/((edge-11)/30)
        yaw = torch.atan2(world[3:edge-8,4].mean(),world[3:edge-8,3].mean())
        case['command_velocity'] = velocity.tolist()
        case['command_heading'] = float(yaw)
        metadata = dict(schema='parc.live_scene.v1', dataset_manifest_sha256=manifest_hash,
            history_frames=4, future_motion_included=False, source_id=case['row']['source_id'],
            source_shard_sha256=case['row']['converted_shard_sha256'], source_start_frame=case['start_frame'])
        path = args.output/(case['entry']+'-scene.npz')
        np.savez_compressed(path, **{k:torch.as_tensor(v).cpu().numpy() for k,v in case['scene'].items()
            if k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')},
            initial_history=world[None,:4].numpy(), metadata_json=np.asarray(json.dumps(metadata)))
        write_json(path.with_suffix('.json'),dict(metadata, scene_sha256=sha256(path)))
        reports.append(dict(entry=case['entry'], source_id=case['row']['source_id'], source_start=case['start_frame'],
            command_velocity_mps=velocity.tolist(), command_heading=float(yaw),
            command_speed_mps=float(velocity.norm()), native_heading_of_travel=float(torch.atan2(velocity[0],velocity[1])),
            source_metrics=exact_skin_metrics(geometry,world[None],case['scene'])))
    torch.save(selected,args.output/'offline-cases.pt')
    receipt = dict(schema='motionbricks.stair_wasd_trial.v1', checkpoint_sha256=expected,
        source_bundle_sha256=saved['contract']['source_bundle_sha256'], entries=reports,
        approved_visual_only='reported-stairs ascent', wasd_verified=False,
        files={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()})
    write_json(args.output/'export.json',receipt)
    print(json.dumps(receipt),flush=True)
    config = Path(args.drive_config)
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Private OAuth identity mismatch')
    environment = dict(os.environ,HTTP_PROXY='',HTTPS_PROXY='',ALL_PROXY='',http_proxy='',https_proxy='',all_proxy='',NO_PROXY='*')
    destination = 'private-store:motionbricks/stair_wasd_20261013/approved_v1/'
    command = ['rclone','--config',str(config),'copyto']
    options = ['--immutable','--checksum','--transfers','1','--checkers','1','--tpslimit','1','--retries','2']
    for path in [*args.output.iterdir(),args.checkpoint]:
        name = 'weights.pt' if path == args.checkpoint else path.name
        subprocess.run([*command,str(path),destination+name,*options],env=environment,check=True)
    write_json(args.output/'upload-complete.json',dict(status='complete',destination=destination))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','cases','output','drive-config'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
