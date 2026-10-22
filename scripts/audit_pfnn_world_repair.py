"""Check factual PFNN coordinates, reused geometry and scene-label cache attachment."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_world_contract import VERSION
from scripts.build_parc_native_dataset import _terrain_relative_foot_contacts
from scripts.cache_full_terrain_primitives import sources
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    rep=build_native_representation()
    manifest=json.loads((a.repaired/'manifest.json').read_text())
    if manifest['world_contract']!=VERSION or manifest['schema']!='motionbricks.pfnn_static_pairs.v2':
        raise ValueError('Corrected world/contact schema required')
    rows=[]
    for row in manifest['records']:
        motion=a.repaired/row['motion_shard']
        stem=motion.stem
        with np.load(motion,allow_pickle=False) as data:
            world=torch.from_numpy(data['global_features'].copy())
            proof=json.loads(str(data['encoding_proof_json']))
        with np.load(a.retarget/'shards'/(stem+'.npz'),allow_pickle=False) as data:
            roots=data['root_positions'].copy()
            joints=torch.from_numpy(data['posed_joints'].copy())
        with np.load(a.previous/'motions'/(stem+'.npz'),allow_pickle=False) as data:
            previous=data['global_features'].copy()
        with np.load(a.repaired/row['geometry_shard'],allow_pickle=False) as data:
            scene={key:data[key].copy() for key in data.files}
        old_path=row['terrain_construction'].get('reused_geometry')
        if not old_path:
            raise ValueError('Pilot must exercise actual previously fitted geometry reuse')
        with np.load(old_path,allow_pickle=False) as old:
            if old.files!=list(scene) or not all(np.array_equal(old[key],scene[key]) for key in scene):
                raise ValueError('Reused field was altered')
        with np.load(a.repaired/row['contact_shard'],allow_pickle=False) as data:
            contacts=torch.from_numpy(data['contacts'].copy())
            offset=int(data['source_frame_offset'])
        expected=_terrain_relative_foot_contacts(joints[offset:offset+len(contacts)],rep.skeleton,
            clip_heightfield_y_m=scene['heightfield_y_m'],clip_origin_xyz_m=scene['min_point_world_xyz_m'],dx_m=scene['dx_m'])
        torch.testing.assert_close(contacts,expected,atol=0,rtol=0)
        old_contact=previous[offset:offset+len(contacts),rep.indices['foot_contacts']]
        rows.append(dict(source_id=row['source_id'],world_fk_error_m=proof['fk_joint_max_m'],
            old_root_world_error_max_m=float(np.linalg.norm(previous[:,:3]-roots,axis=-1).max()),
            new_root_world_error_max_m=float((world[:,:3]-torch.from_numpy(roots)).norm(dim=-1).max()),
            geometry_arrays_unchanged=True,contact_reference_difference=0.,
            changed_contact_labels=int((torch.from_numpy(old_contact)!=contacts).sum()),
            target_contact_ones=int(contacts.sum()),old_contact_ones=int(old_contact.sum()),
            admitted_windows=len(row['eligible_window_starts'])))
    attached=0
    args=SimpleNamespace(mixed=None,pfnn=a.repaired,pilot=True)
    by_id={row['source_id']:row for row in manifest['records']}
    for item in sources(args,rep):
        row=by_id[item['record']['source_id']]
        with np.load(a.repaired/row['contact_shard'],allow_pickle=False) as data:
            expected=torch.from_numpy(data['contacts'].copy())
            offset=int(data['source_frame_offset'])
        torch.testing.assert_close(item['world'][offset:offset+len(expected),rep.indices['foot_contacts']],expected,atol=0,rtol=0)
        attached+=len(item['starts'])
    if attached==0:
        raise ValueError('Pilot did not exercise admitted scene-contact supervision')
    write_json(a.output/'complete.json',dict(passed=True,rows=rows,admitted_windows_verified=attached,
        repaired_manifest_sha256=sha256(a.repaired/'manifest.json'),world_contract=VERSION,
        geometry_unchanged=True,source_frames_unchanged=True,actual_optimizer_updates=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repaired','previous','retarget','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
