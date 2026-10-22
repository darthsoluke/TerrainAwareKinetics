"""Cache raw causal observations for every admitted full-corpus primitive."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.full_terrain_cache import PrimitiveWriter
from motionbricks.data.local_terrain_observation import extract_local_terrain_observation, trained_query_controls
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_online_conditions import velocity_direction_local
from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.unified_full_training import rotate_volume_observation
from motionbricks.data.unified_scene_field import AnalyticPlaneField, VolumetricEgoSampler, field_from_heightfield
from motionbricks.data.unified_training_goals import source_terminal_goal
from motionbricks.data.pfnn_world_contract import apply_scene_contacts,VERSION as PFNN_WORLD_CONTRACT
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def rotate_scene_world(rep, world, scene, quarter):
    heading = torch.atan2(world[:, 0, 4], world[:, 0, 3])+quarter*np.pi/2
    changed = rep.change_first_heading(world, heading, False, False)
    origin = torch.as_tensor(scene['min_point_world_xyz_m']).clone().float()
    height = torch.as_tensor(scene['heightfield_y_m'])
    spacing = float(scene['dx_m'])
    old = world[0, 0, [0, 2]]
    if quarter == 0:
        transformed = old
        x, z = origin[0], origin[2]
    elif quarter == 1:
        transformed = torch.stack((old[1], -old[0]))
        x, z = origin[2], -origin[0]-(height.shape[1]-1)*spacing
    elif quarter == 2:
        transformed = -old
        x, z = -origin[0]-(height.shape[1]-1)*spacing, -origin[2]-(height.shape[0]-1)*spacing
    else:
        transformed = torch.stack((-old[1], old[0]))
        x, z = -origin[2]-(height.shape[0]-1)*spacing, origin[0]
    shift = changed[0, 0, [0, 2]]-transformed
    origin[0], origin[2] = x+shift[0].cpu(), z+shift[1].cpu()
    result = dict(heightfield_y_m=torch.rot90(height, quarter, (0, 1)).contiguous(),
        min_point_world_xyz_m=origin, dx_m=torch.tensor(spacing))
    if 'heightfield_valid' in scene:
        result['heightfield_valid'] = torch.rot90(torch.as_tensor(scene['heightfield_valid']), quarter, (0, 1)).contiguous()
    return changed, result


@torch.no_grad()
def observations(world, scene, policy, sampler, field):
    history = world[:, :4, :5]
    command = source_terminal_goal(world[..., :5])
    direction = velocity_direction_local(history, command, 28/30)
    query = history.new_tensor([[-1., .6, 1., .4]])
    controls = trained_query_controls(query, direction, policy)
    snapshot = extract_local_terrain_observation(torch.as_tensor(scene['heightfield_y_m'], device=world.device),
        scene['min_point_world_xyz_m'], scene['dx_m'], history[:, -1, :3],
        torch.atan2(history[:, -1, 4], history[:, -1, 3]), direction,
        heightfield_valid=scene.get('heightfield_valid'), config=policy, query_controls=controls)
    volume = sampler.observe(history, command[:, :2], field, 28/30)
    return snapshot.as_model_inputs(), volume


@torch.no_grad()
def plane_minima(rep, geometry, world):
    values, lower, upper = [], [], []
    for start in range(0, len(world), 64):
        part = world[None, start:start+64].to(geometry.skin.bind_vertices.device)
        pose = rep.inverse(part, False, return_all=True, joint_positions_from='global_rot_data')
        vertices = geometry.skin.skin(pose['local_joint_rots'][0], pose['posed_joints'][0], rot_is_global=False)
        values.append(vertices[..., 1].amin(-1).cpu())
        lower.append(vertices[..., [0, 2]].amin(-2).cpu())
        upper.append(vertices[..., [0, 2]].amax(-2).cpu())
    return torch.cat(values), torch.cat(lower), torch.cat(upper)


def sources(args, rep):
    if args.mixed is not None:
        for split in ('train', 'validation', 'test'):
            dataset = TerrainLearningDataset(args.mixed, rep, split)
            for index, row in enumerate(dataset.records):
                item = dataset.full_reference_item(index)
                world, scene = item['world'], item['scene']
                starts = sorted(set(range(0, len(world)-32, 8)) | {len(world)-33})
                yield dict(world=world, scene=scene, starts=starts, prequalified=False,
                    record=dict(domain='bones' if row['origin'] == 'bones_seed' else 'parc',
                        source_id=row['source_id'], source_group=row['source_group'], split=split,
                        source_sha256=row['converted_shard_sha256']))
    if args.pfnn is not None:
        manifest = json.loads((args.pfnn/'manifest.json').read_text())
        if manifest.get('schema')!='motionbricks.pfnn_static_pairs.v2' or manifest.get('world_contract')!=PFNN_WORLD_CONTRACT:
            raise ValueError('PFNN requires verified factual world coordinates and scene-specific contact targets')
        if manifest['pilot'] and not args.pilot:
            raise ValueError('A pilot PFNN dataset cannot become the full corpus')
        for row in manifest['records']:
            if not row['eligible_window_starts']:
                continue
            motion, geometry = args.pfnn/row['motion_shard'], args.pfnn/row['geometry_shard']
            if sha256(motion) != row['motion_sha256'] or sha256(geometry) != row['geometry_sha256']:
                raise ValueError('PFNN paired data identity differs')
            with np.load(motion, allow_pickle=False) as data:
                world = torch.from_numpy(data['global_features'].copy())
                if str(data['world_contract'])!=PFNN_WORLD_CONTRACT:
                    raise ValueError('PFNN motion lacks verified physical-world encoding')
            contact_path=args.pfnn/row['contact_shard']
            if sha256(contact_path)!=row['contact_sha256']:
                raise ValueError('PFNN scene-contact labels changed')
            with np.load(contact_path,allow_pickle=False) as data:
                contacts=torch.from_numpy(data['contacts'].copy())
                offset=int(data['source_frame_offset'])
                if (offset!=row['contact_source_offset'] or len(contacts)!=row['contact_source_frames']
                        or float(data['fps'])!=30 or not np.array_equal(data['source_frame_indices'],np.arange(offset,offset+len(contacts)))):
                    raise ValueError('PFNN scene contacts do not match real source time')
            if any(start<offset or start+33>offset+len(contacts) for start in row['eligible_window_starts']):
                raise ValueError('PFNN supervision escapes its scene-contact interval')
            world=apply_scene_contacts(world,contacts,offset,rep.indices['foot_contacts'])
            with np.load(geometry, allow_pickle=False) as data:
                scene = {key: torch.from_numpy(data[key].copy()) for key in data.files}
            yield dict(world=world, scene=scene, starts=row['eligible_window_starts'], prequalified=True,
                record=dict(domain='pfnn', source_id=row['source_id'], source_group=row['source_group'],
                    split=row['split'], source_sha256=row['source_bvh_sha256']))


@torch.no_grad()
def main(args):
    if args.mixed is None and args.pfnn is None:
        raise ValueError('At least one actual source corpus required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    bundle = torch.load(args.bundle, map_location='cpu', mmap=True, weights_only=False)
    stats = args.mixed/'stats' if args.mixed else args.stats
    rep = build_native_representation(stats)
    torch.testing.assert_close(rep.dual_rep.stats.mean, bundle['stats_mean'], atol=0, rtol=0)
    torch.testing.assert_close(rep.dual_rep.stats.std, bundle['stats_std'], atol=0, rtol=0)
    device_rep = build_native_representation(stats).to(args.device)
    geometry = JointGeometryObjective(device_rep, args.device)
    policy = terrain_observation_config(bundle['root_config']['model']['args'].get('observation_range_policy'))
    sampler = VolumetricEgoSampler()
    writer = PrimitiveWriter(args.output/'data')
    rejected, considered = [], 0
    for number, item in enumerate(sources(args, rep)):
        if args.max_sources and number >= args.max_sources:
            break
        world, scene = item['world'], item['scene']
        base_field = field_from_heightfield(scene)
        plane = isinstance(base_field, AnalyticPlaneField)
        minima = plane_minima(device_rep, geometry, world) if plane and not item['prequalified'] else None
        starts = item['starts'][:args.max_windows] if args.max_windows else item['starts']
        for cursor in starts:
            if cursor < 0 or cursor+33 > len(world):
                raise ValueError('Primitive escaped its real source frames')
            considered += 1
            source = world[None, cursor:cursor+33].to(args.device)
            if not item['prequalified']:
                if plane:
                    heights, lower, upper = [value[cursor:cursor+33] for value in minima]
                    xmin, xmax, zmin, zmax = base_field.bounds
                    passed = (float((base_field.height-heights).clamp_min(0).max()) <= .02
                        and bool((lower >= lower.new_tensor([xmin, zmin])).all())
                        and bool((upper <= upper.new_tensor([xmax, zmax])).all()))
                else:
                    skin = exact_skin_metrics(geometry, source, scene)
                    passed = skin['body_prism_max_m'] <= .02 and skin['unknown_body_fraction'] == 0 and skin['uncertain_inside_samples'] == 0
                if not passed:
                    rejected.append(dict(item['record'], start=cursor, reason='source_skin_geometry_failed'))
                    continue
            observation, volume = observations(source, scene, policy, sampler, base_field)
            for quarter in range(4):
                rotated, rotated_scene = rotate_scene_world(device_rep, source, scene, quarter)
                if plane:
                    changed_volume = rotate_volume_observation(volume, torch.tensor([quarter], device=source.device), field_is_plane=True)
                    changed_observation = observation
                else:
                    changed_observation, changed_volume = observations(rotated, rotated_scene, policy, sampler,
                        field_from_heightfield(rotated_scene))
                envelope = changed_observation.get('query_envelope_m')
                if envelope is None:
                    raise ValueError('Adaptive trained query envelope missing')
                writer.append(dict(world=rotated[0], volume_features=changed_volume['features'][0],
                    volume_distance=changed_volume['distance'][0], volume_valid=changed_volume['valid'][0],
                    observation_points=changed_observation['points'][0], observation_valid=changed_observation['valid'][0],
                    observation_envelope=envelope[0]), dict(item['record'], start=cursor, quarter=quarter))
        write_json(args.output/'progress.json', dict(sources=number+1, considered_base_windows=considered,
            cached_augmented_primitives=len(writer.records), rejected_base_windows=len(rejected)))
    summary = writer.finish()
    write_json(args.output/'rejected.json', rejected)
    write_json(args.output/'complete.json', dict(schema='motionbricks.full_terrain_primitives.v1',
        data=summary, bundle_sha256=sha256(args.bundle), pilot=args.pilot,
        source_manifests={name:sha256(path/'manifest.json' if name == 'pfnn' else path/'native_manifest.json')
            for name,path in (('pfnn',args.pfnn),('mixed',args.mixed)) if path is not None},
        considered_base_windows=considered, rejected_base_windows=len(rejected),
        augmentation='Four joint world/scene quarter rotations; non-plane normals re-queried.',
        stored_features='Raw physical observations, not frozen learned embeddings',
        no_future_height_pose_in_root_inputs=True, source_motion_timestamps_unchanged=True,
        training_started=False, script_sha256=sha256(Path(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--mixed', type=Path)
    parser.add_argument('--pfnn', type=Path)
    parser.add_argument('--stats', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-sources', type=int, default=0)
    parser.add_argument('--max-windows', type=int, default=0)
    parser.add_argument('--pilot', action='store_true')
    main(parser.parse_args())
