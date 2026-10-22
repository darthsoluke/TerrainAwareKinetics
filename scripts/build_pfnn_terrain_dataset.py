"""Build static, quality-screened terrain pairs from all converted PFNN clips."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motionbricks.data.bones_seed import FOOT_CHANNELS, official_sole_surfaces
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_terrain_pairing import fit_terrain, footstep_cycles, static_heightfield, terrain_kind
from motionbricks.data.pfnn_world_contract import encode_pfnn_world,scene_contact_targets,VERSION as WORLD_CONTRACT
from motionbricks.data.lafan_soma_retarget import Y_TO_Z
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def split_group(group):
    value = int(hashlib.sha256(('pfnn:2333:'+group).encode()).hexdigest()[:8], 16) % 10
    return 'test' if value == 0 else 'validation' if value == 1 else 'train'


def window_starts(first, last):
    if last < first:
        return []
    return sorted(set(range(first, last+1, 8)) | {last})


@torch.no_grad()
def source_geometry(data, skeleton, skinner, surfaces, device):
    local = torch.from_numpy(data['local_rot_mats']).to(device)
    positions = torch.from_numpy(data['posed_joints']).to(device)
    minima, supports = [], []
    for start in range(0, len(local), 64):
        vertices = skinner.skin(local[start:start+64], positions[start:start+64], rot_is_global=False)
        minima.append(vertices[..., 1].amin(-1).cpu().numpy())
        channels = []
        for surface in surfaces:
            selected = vertices[:, surface['selected_indices']]
            index = selected[..., 1].argmin(-1)
            channels.append(selected[torch.arange(len(selected), device=device), index])
        supports.append(torch.stack(channels, 1).cpu().numpy())
    return np.concatenate(minima), np.concatenate(supports)


def raw_features(rep, data):
    return encode_pfnn_world(rep,data)[0]


def source_contacts(data):
    names = data['source_names'].astype(str).tolist()
    indices = [names.index(name) for name in FOOT_CHANNELS]
    positions = data['source_positions'][:, indices]
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=-1)*30
    mask = speed < np.sqrt(.02)*60*.01
    return np.concatenate((mask, mask[-1:]), axis=0)


def task_errors(row, data, config_path, names):
    config = json.loads(config_path.read_text())
    source_names = data['source_names'].astype(str).tolist()
    goals, orientations = [], []
    scale = row['calibration']['uniform_scale']
    for name in names:
        entry = config['ik_match_table1'][name]
        source = source_names.index(row['calibration']['mapping'][name])
        offset = Y_TO_Z.T@Rotation.from_quat(entry[4], scalar_first=True).as_matrix()@Y_TO_Z
        rotation = data['source_global_rot_mats'][:, source]@offset
        position = scale*data['source_positions'][:, source]+(rotation@(Y_TO_Z.T@np.asarray(entry[3]))[:, None])[..., 0]
        goals.append(position)
        orientations.append(rotation)
    errors = np.linalg.norm(data['posed_joints']-np.stack(goals, 1), axis=-1)
    relative = data['global_rot_mats'].swapaxes(-1, -2)@np.stack(orientations, 1)
    angles = np.rad2deg(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()).reshape(errors.shape)
    feet = [names.index(name) for name in FOOT_CHANNELS]
    replay = dict(root_task_p95_m=float(np.quantile(errors[:, 0], .95)), root_task_max_m=float(errors[:, 0].max()),
        foot_task_p95_m=float(np.quantile(errors[:, feet], .95)), foot_task_max_m=float(errors[:, feet].max()),
        orientation_task_p95_deg=float(np.quantile(angles, .95)))
    if max(abs(value-row['metrics'][key]) for key, value in replay.items()) > 2e-5:
        raise ValueError('Reconstructed GMR task errors differ from the recorded conversion')
    reliable = (errors[:, 0] <= .05) & (errors[:, feet].max(-1) <= .08)
    return errors, angles, reliable


def valid_retarget_window(errors, angles, feet, start):
    part = errors[start:start+33]
    return np.quantile(part[:, 0], .95) <= .02 and part[:, 0].max() <= .05 \
        and np.quantile(part[:, feet], .95) <= .03 and part[:, feet].max() <= .08 \
        and np.quantile(angles[start:start+33], .95) <= 10


def flat_admission(minima, support, contacts, floor, starts):
    accepted, rejected = [], []
    for start in starts:
        stop = start+33
        peak = float(max(0, floor-minima[start:stop].min()))
        gap = np.abs(support[start:stop, :, 1]-floor)[contacts[start:stop]]
        top_p95 = float(np.quantile(gap, .95)) if len(gap) else None
        passed = peak <= .02 and (top_p95 is None or top_p95 <= .03)
        value = dict(start=start, body_prism_peak_m=peak, stance_top_gap_p95_m=top_p95,
            all_skin_vertices=18056, swept_substeps=8,
            geometry_method='Plane penetration maximum over linear swept skin occurs at original endpoints.')
        (accepted if passed else rejected).append(value)
    return accepted, rejected


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('motions', 'geometry', 'reports', 'contacts'):
        (args.output/name).mkdir()
    torch.set_num_threads(4)
    report = json.loads((args.retarget/'complete.json').read_text())
    if not args.pilot and not report['full_source_conversion']:
        raise ValueError('Full data preparation requires all80converted recordings')
    if args.clips:
        if not args.pilot:
            raise ValueError('Clip selection is allowed only in an explicitly labeled pilot')
        selected = set(args.clips)
        report['rows'] = [row for row in report['rows'] if Path(row['filename']).stem in selected]
        if len(report['rows']) != len(selected):
            raise ValueError('Pilot clip selection did not match source records')
    ready = json.loads((args.source/'patches.receipt.json').read_text())
    if sha256(args.source/'patches.npz') != ready['output_sha256']:
        raise ValueError('Author patch bank identity changed')
    with np.load(args.source/'patches.npz', allow_pickle=False) as archive:
        patches = archive['X'].copy()
    rep = build_native_representation()
    device_rep = build_native_representation().to(args.device)
    skinner = move_skinner(load_official_skinner(device_rep.skeleton), torch.device(args.device))
    skinner.skeleton_skin.float().to(args.device)
    surfaces, _, _, mesh_hash = official_sole_surfaces(rep.skeleton)
    geometry = JointGeometryObjective(device_rep, args.device)
    records, audits = [], []
    for row in report['rows']:
        stem = Path(row['filename']).stem
        shard_path = args.retarget/row['shard']
        if sha256(shard_path) != row['shard_sha256']:
            raise ValueError('Retargeted source changed')
        with np.load(shard_path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        errors, angles, reliable = task_errors(row, data,
            args.retarget/'calibration'/stem/'pfnn_to_soma30.json', list(rep.skeleton.bone_order_names))
        feet = [list(rep.skeleton.bone_order_names).index(name) for name in FOOT_CHANNELS]
        world,encoding_proof = encode_pfnn_world(rep,data)
        minima, supports = source_geometry(data, rep.skeleton, skinner, surfaces, args.device)
        contacts = source_contacts(data)
        motion_path = args.output/'motions'/(stem+'.npz')
        source_times = np.arange(len(world))/30
        phase_path = args.source/Path(row['filename']).with_suffix('.phase')
        phase = np.loadtxt(phase_path)
        phase30 = np.interp(source_times, np.arange(len(phase))/float(data['source_fps']),
                            np.unwrap(phase*2*np.pi))/(2*np.pi) % 1
        np.savez_compressed(motion_path, global_features=world.numpy(), fps=np.float32(30),
            pfnn_phase_audit_only=phase30.astype(np.float32), source_kinematic_stance=contacts,
            world_contract=np.asarray(WORLD_CONTRACT),encoding_proof_json=np.asarray(json.dumps(encoding_proof)))
        kind = terrain_kind(stem)
        episodes = footstep_cycles(args.source/Path(row['filename']).with_name(stem+'_footsteps.txt'),
            float(data['source_fps']), len(world))
        if args.max_episodes:
            episodes = episodes[:args.max_episodes]
        clip_audit = dict(source_id=row['source_id'], kind=kind, total_episodes=len(episodes), episodes=[],
            original_whole_clip_failures=row['failures'], reliable_support_frames=int(reliable.sum()),
            world_encoding=encoding_proof)
        for number, episode in enumerate(episodes):
            start, stop = episode['start'], episode['stop']
            seed = int(hashlib.sha256((stem+':'+str(number)).encode()).hexdigest()[:8], 16)
            try:
                reusable = args.reuse_geometry/'geometry'/(stem+'-'+str(number)+'.npz') if args.reuse_geometry else None
                scene=None
                reuse_error=None
                if reusable is not None and reusable.exists():
                    try:
                        with np.load(reusable,allow_pickle=False) as archive:
                            scene={key:archive[key].copy() for key in archive.files}
                        if set(scene)!={'heightfield_y_m','min_point_world_xyz_m','dx_m'} or not all(np.isfinite(value).all() for value in scene.values()):
                            raise ValueError('Reused geometry schema or finiteness differs')
                    except (OSError,ValueError,EOFError,zipfile.BadZipFile) as error:
                        reuse_error=repr(error)
                        scene=None
                if scene is None:
                    mask = reliable[start:stop]
                    terrain = fit_terrain(patches, supports[start:stop][mask], contacts[start:stop][mask],
                        data['root_positions'][start:stop][mask], kind, float(data['uniform_scale']),
                        args.source/'official_source', seed)
                    scene = static_heightfield(terrain, data['root_positions'][start:stop], args.spacing, 4.)
                    construction=dict(terrain.metadata,reuse_error=reuse_error)
                else:
                    construction=dict(kind=kind,construction='Unchanged already-fitted static field; admission recomputed in factual world.',
                        reused_geometry=str(reusable),reused_sha256=sha256(reusable),seed=seed,
                        original_patch_metadata_recovered=False,measured_scene=False,time_varying_geometry=False)
                target_contacts=scene_contact_targets(rep,world[start:stop],scene,
                    posed_joints=torch.from_numpy(data['posed_joints'][start:stop]))
            except (ValueError, np.linalg.LinAlgError) as error:
                clip_audit['episodes'].append(dict(episode=episode, disposition='fitting_failed', error=repr(error)))
                continue
            contact_path=args.output/'contacts'/(stem+'-'+str(number)+'.npz')
            np.savez_compressed(contact_path,contacts=target_contacts.numpy(),source_frame_offset=np.int64(start),
                source_frame_indices=np.arange(start,stop,dtype=np.int64),fps=np.float32(30))
            starts = window_starts(max(start, episode['core_start']-3), min(stop-33, episode['core_stop']-3))
            accepted, rejected = [], []
            bad_starts = [cursor for cursor in starts if not valid_retarget_window(errors, angles, feet, cursor)]
            rejected.extend(dict(start=cursor, reason='retarget_window_quality_failed') for cursor in bad_starts)
            starts = [cursor for cursor in starts if cursor not in bad_starts]
            if kind == 'flat':
                accepted, bad_geometry = flat_admission(minima, supports, contacts,
                    float(scene['heightfield_y_m'][0, 0]), starts)
                rejected.extend(bad_geometry)
            else:
                for cursor in starts:
                    skin = exact_skin_metrics(geometry, world[None, cursor:cursor+33].to(args.device), scene)
                    points = supports[cursor:cursor+33]
                    contact = contacts[cursor:cursor+33]
                    origin, dx = scene['min_point_world_xyz_m'], float(scene['dx_m'])
                    cells = np.floor((points[..., [0, 2]]-origin[[0, 2]])/dx+.5).astype(int)
                    known = (cells[..., 0] >= 0) & (cells[..., 1] >= 0) \
                        & (cells[..., 0] < scene['heightfield_y_m'].shape[1]) & (cells[..., 1] < scene['heightfield_y_m'].shape[0])
                    gap = points[..., 1]-scene['heightfield_y_m'][
                        cells[..., 1].clip(0, scene['heightfield_y_m'].shape[0]-1),
                        cells[..., 0].clip(0, scene['heightfield_y_m'].shape[1]-1)]
                    selected = np.abs(gap[contact & known])
                    top_p95 = float(np.quantile(selected, .95)) if len(selected) else None
                    passed = skin['body_prism_max_m'] <= .02 and skin['unknown_body_fraction'] == 0 \
                        and skin['uncertain_inside_samples'] == 0 and not (contact & ~known).any() \
                        and (top_p95 is None or top_p95 <= .03)
                    value = dict(start=cursor, body_prism_peak_m=skin['body_prism_max_m'],
                        stance_top_gap_p95_m=top_p95, skin=skin)
                    (accepted if passed else rejected).append(value)
            geometry_path = args.output/'geometry'/(stem+'-'+str(number)+'.npz')
            np.savez_compressed(geometry_path, **scene)
            record = dict(source_id=row['source_id']+':terrain:'+str(number), origin='pfnn',
                source_group='pfnn:'+row['recording_group'], split=split_group(row['recording_group']),
                motion_shard=motion_path.relative_to(args.output).as_posix(), motion_sha256=sha256(motion_path),
                geometry_shard=geometry_path.relative_to(args.output).as_posix(), geometry_sha256=sha256(geometry_path),
                contact_shard=contact_path.relative_to(args.output).as_posix(),contact_sha256=sha256(contact_path),
                contact_source_offset=start,contact_source_frames=stop-start,world_contract=WORLD_CONTRACT,
                source_bvh_sha256=row['source_bvh_sha256'], retarget_shard_sha256=row['shard_sha256'],
                episode=episode, eligible_window_starts=[value['start'] for value in accepted],
                accepted_windows=accepted, rejected_windows=rejected, terrain_construction=construction,
                labels_are_recorded_wasd=False, pfnn_phase_is_runtime_input=False)
            records.append(record)
            clip_audit['episodes'].append(dict(episode=episode, accepted_windows=len(accepted), rejected_windows=len(rejected)))
        audits.append(clip_audit)
        write_json(args.output/'reports'/(stem+'.json'), clip_audit)
        write_json(args.output/'reports'/(stem+'-records.json'),[value for value in records if value['source_bvh_sha256']==row['source_bvh_sha256']])
        write_json(args.output/'progress.json', dict(processed_clips=len(audits), total_clips=len(report['rows']),
            terrain_pairs=len(records), admitted_windows=sum(len(value['eligible_window_starts']) for value in records)))
    result = dict(schema='motionbricks.pfnn_static_pairs.v2', records=records, source_audit=audits,
        retarget_complete_sha256=sha256(args.retarget/'complete.json'), source_inventory_sha256=sha256(args.source/'inventory.json'),
        patch_bank_sha256=sha256(args.source/'patches.npz'), official_mesh_sha256=mesh_hash,
        pilot=args.pilot, full_motion_clips_considered=len(report['rows']),
        admitted_windows=sum(len(row['eligible_window_starts']) for row in records),
        geometry_gate=dict(peak_m=.02, stance_top_p95_m=.03, unknown_fraction=0, uncertain_inside_samples=0),
        no_motion_or_runtime_correction=True, source_contact_labels='PFNN velocity-derived kinematic stance, not measured force',
        terrain_pairing='Static author-style patch/RBF fitting to retargeted SOMA support; not measured capture terrain',
        training_started=False)
    result.update(world_contract=WORLD_CONTRACT,
        contact_target_policy='Existing PARC terrain-relative SOMA foot-joint detector: world speed<0.15m/s and terrain-relative joint height<0.10m.',
        fitted_geometry_reuse=str(args.reuse_geometry) if args.reuse_geometry else None)
    write_json(args.output/'manifest.json', result)
    write_json(args.output/'complete.json', dict(manifest_sha256=sha256(args.output/'manifest.json'),
        admitted_windows=result['admitted_windows'], terrain_pairs=len(records), training_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'retarget', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--spacing', type=float, default=.05)
    parser.add_argument('--max-episodes', type=int, default=0)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--clips', nargs='+')
    parser.add_argument('--reuse-geometry',type=Path)
    main(parser.parse_args())
