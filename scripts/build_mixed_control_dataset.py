"""Pair native BONES flat locomotion with a fixed plane, then retain PARC pairs.

All source frames and split identities are preserved. BONES terrain is an
explicit flat-scene assumption, not measured terrain. The source fixed plane
is not refitted to the predictions or to each frame. Official SOMA sole-region
geometry screens excessive source penetration before admission.
"""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.native_mb_data import build_native_representation, NativeMBMotionDataset
from motionbricks.data.native_mb_terrain_protocol import TERRAIN_PROTOCOL, terrain_observation_config
from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
from motionbricks.data.soma_skin_reference import load_official_skinner, move_skinner


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def validate_selection(original, selected):
    by_shard = {row['shard']: row for row in original}
    if len(by_shard) != len(original) or len({row['shard'] for row in selected}) != len(selected):
        raise ValueError('Duplicate source shard identity')
    for row in selected:
        source = by_shard.get(row['shard'])
        if source is None or any(row.get(key) != value for key, value in source.items()):
            raise ValueError('Selected BONES identity or original split changed')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parc', required=True, type=Path)
    parser.add_argument('--bones', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new dataset directory')
    torch.set_num_threads(4)
    parc_manifest = json.loads((args.parc / 'native_manifest.json').read_text())
    parc_records = json.loads((args.parc / 'accepted.json').read_text())
    selection = json.loads((args.bones / 'selection.json').read_text())
    if sha256(args.bones / 'source_native_manifest.json') != selection['source_manifest_sha256']:
        raise ValueError('BONES source manifest changed')
    bones_manifest = json.loads((args.bones / 'source_native_manifest.json').read_text())
    if sha256(args.bones / 'source_accepted.json') != bones_manifest['accepted_json_sha256']:
        raise ValueError('Original BONES split manifest changed')
    validate_selection(json.loads((args.bones / 'source_accepted.json').read_text()), selection['records'])
    args.output.mkdir(parents=True)
    # The immutable PARC statistics are retained to preserve the warmstart's
    # coordinate scales. Validation/test never contribute new normalization.
    for relative, digest in parc_manifest['artifact_sha256'].items():
        source, target = args.parc / relative, args.output / relative
        if sha256(source) != digest:
            raise ValueError('PARC artifact changed: ' + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if not (args.output / 'stats').exists():
        shutil.copytree(args.parc / 'stats', args.output / 'stats')
    (args.output / 'source_snapshots').mkdir(exist_ok=True)
    for source, name in ((args.parc / 'native_manifest.json', 'parc_manifest.json'),
                         (args.parc / 'accepted.json', 'parc_accepted.json'),
                         (args.bones / 'selection.json', 'bones_selection.json'),
                         (args.bones / 'source_accepted.json', 'bones_accepted.json'),
                         (args.bones / 'source_native_manifest.json', 'bones_native_manifest.json')):
        shutil.copy2(source, args.output / 'source_snapshots' / name)
    rep = build_native_representation(args.parc / 'stats').to(args.device)
    skinner = move_skinner(load_official_skinner(rep.skeleton), torch.device(args.device))
    skinner.skeleton_skin.float().to(args.device)
    surfaces, _, _, mesh_hash = official_sole_surfaces(rep.skeleton)
    feet = []
    for side in ('Left', 'Right'):
        feet.append(torch.tensor(sorted(set(np.concatenate([row['selected_indices'] for row in surfaces
            if row['channel'].startswith(side)]).tolist())), device=args.device, dtype=torch.long))
    selected = torch.unique(torch.cat(feet), sorted=True)
    remap = {value: index for index, value in enumerate(selected.cpu().tolist())}
    feet = [torch.tensor([remap[value] for value in indices.cpu().tolist()], device=args.device)
            for indices in feet]
    full_skinner = skinner
    # Official LBS is independent per vertex. Retain each selected vertex's
    # complete eight influences, including toe articulation and inverse bind.
    from copy import copy
    skinner = copy(full_skinner)
    for name in ('bind_vertices', 'lbs_indices', 'lbs_weights'):
        setattr(skinner, name, getattr(full_skinner, name).index_select(0, selected))
    subset_verified = False
    records, rejected, geometry_rows = list(parc_records), [], []
    (args.output / 'bones_shards').mkdir()
    (args.output / 'bones_terrain').mkdir()
    for number, row in enumerate(selection['records']):
        source = args.bones / row['shard']
        source.resolve().relative_to(args.bones.resolve())
        if sha256(source) != row['converted_shard_sha256']:
            raise ValueError('BONES source shard changed')
        with np.load(source, allow_pickle=False) as shard:
            arrays = {key: shard[key].copy() for key in ('global_features', 'fps', 'feature_fk_roundtrip_error_m')}
        raw = torch.from_numpy(arrays['global_features']).to(args.device)
        if (raw.shape[1] != len(rep.indices['all']) or float(arrays['fps']) != 30.
                or not torch.isfinite(raw).all()
                or not 0 <= float(arrays['feature_fk_roundtrip_error_m']) < 2e-5):
            raise ValueError('BONES native feature/FPS contract differs')
        gaps = []
        for start in range(0, len(raw), 64):
            pose = rep.inverse(raw[None, start:start+64], False, return_all=False,
                               joint_positions_from='global_rot_data')
            vertices = skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
            if not subset_verified:
                full = full_skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
                torch.testing.assert_close(vertices, full.index_select(1, selected), atol=1e-6, rtol=1e-6)
                subset_verified = True
            gaps.append(torch.stack([vertices[:, indices, 1].amin(-1) for indices in feet], -1).cpu())
        gaps = torch.cat(gaps).numpy()
        penetration = np.maximum(-gaps, 0)
        check = {'filename': row['filename'], 'split': row['split'], 'fixed_plane_y_m': 0.,
                 'sole_penetration_p99_m': float(np.percentile(penetration, 99)),
                 'sole_penetration_max_m': float(penetration.max()),
                 'median_lower_foot_gap_m': float(np.median(gaps.min(-1)))}
        # A conservative source screen; this is not an approval of generated motion.
        if check['sole_penetration_p99_m'] > .03 or check['sole_penetration_max_m'] > .08:
            rejected.append(dict(row, reason='source_sole_geometry', geometry=check))
            continue
        geometry_rows.append(check)
        root = raw[:, :5]
        low = root[:, [0, 2]].amin(0).cpu().numpy() - 4.
        high = root[:, [0, 2]].amax(0).cpu().numpy() + 4.
        spacing = .2
        size = np.ceil((high-low)/spacing).astype(int) + 1
        plane = torch.zeros(int(size[1]), int(size[0]), device=args.device)
        origin = torch.tensor([float(low[0]), 0., float(low[1])], device=args.device)
        shard_name = f'bones_shards/{source.name}'
        geometry_name = f'bones_terrain/{source.stem}.npz'
        # Only the current command query is used online. Keep source motion
        # bytes unchanged; do not manufacture unused per-frame observations.
        shutil.copy2(source, args.output / shard_name)
        np.savez_compressed(args.output / geometry_name, heightfield_y_m=plane.cpu().numpy(),
            min_point_world_xyz_m=origin.cpu().numpy(), dx_m=np.float32(spacing),
            canonical_init_heading_quat=np.array([1., 0., 0., 0.], dtype=np.float32),
            canonical_root_pos_init_xz=np.zeros(2, dtype=np.float32))
        record = dict(row, shard=shard_name, source_id='bones:' + source.stem,
                      origin='bones_seed', source_group='bones:' + row['recording_group'],
                      terrain_family='flat', terrain_pairing='declared_planar_source_fixed_plane',
                      terrain_storage='static_scene_online_query',
                      converted_shard_sha256=sha256(args.output / shard_name),
                      terrain_geometry_snapshot=geometry_name,
                      terrain_geometry_sha256=sha256(args.output / geometry_name))
        records.append(record)
        if number % 20 == 0:
            print(json.dumps({'processed_bones': number+1, 'admitted_bones': len(geometry_rows),
                              'rejected_bones': len(rejected)}), flush=True)
    groups = {}
    for row in records:
        group = row.get('source_group', row.get('recording_group'))
        if group in groups and groups[group] != row['split']:
            raise ValueError('Source family crosses splits: ' + group)
        groups[group] = row['split']
    if not any(row['origin'] == 'bones_seed' and row['split'] == 'train'
               and row['control_category'] == 'idle' for row in records):
        raise ValueError('No genuine BONES idle training clips passed')
    save(args.output / 'accepted.json', records)
    save(args.output / 'bones_geometry_review.json', {'mesh_sha256': mesh_hash,
        'plane_source': 'unchanged source fixed plane at Y=0; no new fitting',
        'screen_thresholds_m': {'p99': .03, 'max': .08}, 'admitted': geometry_rows, 'rejected': rejected})
    manifest = {key: deepcopy(parc_manifest[key]) for key in ('contract', 'terrain_conditioning')}
    save(args.output / 'mixed_acceptance.json', {'status': 'data_ready_not_motion_approved',
        'parc': 'existing bound motion/terrain admissions retained',
        'bones': 'native SOMA30, original split, declared flat plane, official sole-region screen',
        'retarget_status': 'passed', 'motion_terrain_pairing_status': 'passed',
        'bones_plane_is_measured': False, 'bones_geometry_report': 'bones_geometry_review.json'})
    manifest['terrain_conditioning']['acceptance']['report'] = 'mixed_acceptance.json'
    manifest['source_policy'] = {'version': 'parc-bones-soma-v1',
        'allowed_origins': ['parc_official', 'parc_generated', 'bones_seed'],
        'parc_manifest_sha256': sha256(args.parc / 'native_manifest.json'),
        'bones_source_manifest_sha256': selection['source_manifest_sha256'],
        'bones_selection_sha256': sha256(args.bones / 'selection.json'),
        'normalization': 'unchanged PARC train-only scales for weights-only warmstart',
        'new_flat_scene_geometry': 'weak paired constant source plane; geometry screened, not measured force truth'}
    manifest['online_terrain_storage'] = 'static scenes; no stored observations are consumed by online training'
    manifest['artifact_sha256'] = {str(path.relative_to(args.output)).replace('\\', '/'): sha256(path)
        for path in sorted(args.output.rglob('*')) if path.is_file()}
    save(args.output / 'native_manifest.json', manifest)
    for split in ('train', 'validation', 'test'):
        dataset = NativeMBMotionDataset(args.output, split, max_frames=65, min_frames=65,
                                        require_terrain=True, require_online_terrain=True, online_only=True)
        for index in range(len(dataset)):
            item = dataset[index]
            if not torch.isfinite(item['motion']).all():
                raise ValueError('Nonfinite mixed training features')
    receipt = {'manifest_sha256': sha256(args.output / 'native_manifest.json'),
               'counts': dict(Counter(row['split'] + '/' + row['origin'] for row in records)),
               'rejected_bones': len(rejected), 'status': 'data_ready_not_motion_approved'}
    save(args.output / 'receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
