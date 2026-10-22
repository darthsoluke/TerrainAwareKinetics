"""Audit completed PFNN clip records while preparation continues independently."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_world_contract import VERSION
from scripts.build_parc_native_dataset import _terrain_relative_foot_contacts
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coverage(values):
    if not values:
        return dict(unique_source_windows=0)
    values = np.asarray(values)
    prefix, terminal, rise, heading_cosine = values.T
    edges = [0, .05, .25, .5, 1, 2, 3, 4, 6, 100]
    return dict(unique_source_windows=len(values),
        terminal_speed_quantiles_mps=np.quantile(terminal, [0, .05, .5, .95, 1]).tolist(),
        speed_bin_edges_mps=edges, speed_bin_counts=np.histogram(terminal, edges)[0].tolist(),
        speed_overflow_count=int((terminal >= edges[-1]).sum()),
        source_idle_goal_windows=int((terminal < .05).sum()),
        source_start_windows=int(((prefix < .1) & (terminal > .3)).sum()),
        source_stop_windows=int(((prefix > .3) & (terminal < .1)).sum()),
        root_height_excursion_over10cm=int((rise > .1).sum()),
        sideways_or_backward_terminal_windows=int(((heading_cosine < .5) & (terminal > .3)).sum()))


@torch.no_grad()
def main(a):
    require(not a.output.exists(), 'Audit output already exists; preserve the old snapshot')
    # Only the records committed at entry are included; a live writer may add more.
    paths = sorted((a.dataset / 'reports').glob('*-records.json'))
    require(bool(paths), 'No completed per-clip records available')
    a.output.mkdir(parents=True)
    torch.set_num_threads(4)
    rep = build_native_representation()
    retarget_receipt = a.retarget / 'complete.json'
    source_rows = {Path(row['filename']).stem: row
        for row in json.loads(retarget_receipt.read_text())['rows']}
    families, windows, counts = defaultdict(set), defaultdict(dict), defaultdict(int)
    groups = defaultdict(set)
    results, snapshot = [], []
    for path in paths:
        stem = path.name.removesuffix('-records.json')
        records = json.loads(path.read_text())
        source = source_rows[stem]
        original_path = a.retarget / source['shard']
        original_hash = sha256(original_path)
        require(original_hash == source['shard_sha256'], 'Retarget source identity changed')
        motion_path = a.dataset / 'motions' / (stem + '.npz')
        motion_hash = sha256(motion_path)
        with np.load(original_path, allow_pickle=False) as data:
            roots = data['root_positions'].copy()
            joints = torch.from_numpy(data['posed_joints'].copy())
            require(float(data['fps']) == 30, 'Retarget frame clock differs')
        with np.load(motion_path, allow_pickle=False) as data:
            require(str(data['world_contract']) == VERSION and float(data['fps']) == 30,
                'Motion world contract or frame clock differs')
            world = torch.from_numpy(data['global_features'].copy())
        require(len(world) == len(roots) and torch.isfinite(world).all(), 'Motion frames differ')
        root_error = float(np.linalg.norm(world[:, :3].numpy() - roots, axis=-1).max())
        require(root_error <= 5e-5, 'Root no longer agrees with factual source world')
        fk_error = 0.
        for start in range(0, len(world), 256):
            decoded = rep.inverse(world[None, start:start+256], False, return_all=False,
                joint_positions_from='global_rot_data')['posed_joints'][0]
            fk_error = max(fk_error, float((decoded - joints[start:start+256]).norm(dim=-1).max()))
        require(fk_error <= 5e-5, 'Decoded world FK no longer agrees with source')
        checked_labels, positive_labels, admitted = 0, 0, 0
        for row in records:
            require(row['world_contract'] == VERSION and row['motion_sha256'] == motion_hash
                and row['retarget_shard_sha256'] == original_hash
                and row['source_bvh_sha256'] == source['source_bvh_sha256'], 'Pair provenance differs')
            require(row['motion_shard'] == motion_path.relative_to(a.dataset).as_posix(),
                'Pair uses a different source motion')
            family, split = row['source_group'], row['split']
            families[family].add(split)
            require(split in ('train', 'validation', 'test') and len(families[family]) == 1,
                'Recording family split mismatch')
            geometry_path, contact_path = a.dataset / row['geometry_shard'], a.dataset / row['contact_shard']
            require(sha256(geometry_path) == row['geometry_sha256']
                and sha256(contact_path) == row['contact_sha256'], 'Geometry/contact hash mismatch')
            with np.load(geometry_path, allow_pickle=False) as data:
                scene = {key: data[key].copy() for key in data.files}
            with np.load(contact_path, allow_pickle=False) as data:
                contacts = torch.from_numpy(data['contacts'].copy())
                offset = int(data['source_frame_offset'])
                require(offset == row['contact_source_offset'] and len(contacts) == row['contact_source_frames']
                    and float(data['fps']) == 30 and np.array_equal(data['source_frame_indices'],
                        np.arange(offset, offset+len(contacts))), 'Contact source clock mismatch')
            require(0 <= offset and offset+len(contacts) <= len(world), 'Contacts escape source frames')
            expected = _terrain_relative_foot_contacts(joints[offset:offset+len(contacts)], rep.skeleton,
                clip_heightfield_y_m=scene['heightfield_y_m'],
                clip_origin_xyz_m=scene['min_point_world_xyz_m'], dx_m=scene['dx_m'])
            require(torch.equal(contacts, expected), 'Scene contacts disagree with factual source joints')
            checked_labels += contacts.numel()
            positive_labels += int(contacts.sum())
            starts = row['eligible_window_starts']
            require(starts == [value['start'] for value in row['accepted_windows']], 'Admission rows differ')
            kind = row['terrain_construction']['kind']
            key = (split, kind)
            counts[key] += len(starts)
            if starts:
                groups[key].add(family)
            for cursor in starts:
                require(offset <= cursor and cursor+33 <= offset+len(contacts), 'Window escapes contact episode')
                part = world[cursor:cursor+33].numpy()
                v = (part[32, [0, 2]] - part[31, [0, 2]]) * 30
                terminal = float(np.linalg.norm(v))
                prefix = float(np.linalg.norm((part[3, [0, 2]] - part[2, [0, 2]]) * 30))
                forward = part[32, [4, 3]]
                windows[key][(stem, cursor)] = (prefix, terminal, float(np.ptp(part[:, 1])),
                    float(np.dot(v, forward) / max(terminal, 1e-8)))
            admitted += len(starts)
        results.append(dict(clip=stem, frames=len(world), pairs=len(records), admitted_pair_windows=admitted,
            root_max_error_m=root_error, fk_max_error_m=fk_error,
            checked_contact_labels=checked_labels, positive_contact_labels=positive_labels))
        snapshot.append(dict(records_path=str(path), records_sha256=sha256(path),
            motion_sha256=motion_hash, retarget_sha256=original_hash))
        write_json(a.output / 'progress.json', dict(checked_clips=len(results), snapshot_clips=len(paths)))
    write_json(a.output / 'complete.json', dict(passed=True, snapshot_time_unix=time.time(),
        dataset=str(a.dataset), retarget_receipt_sha256=sha256(retarget_receipt),
        script_sha256=sha256(Path(__file__)),
        world_contract=VERSION, snapshot=snapshot, clips=results,
        coverage=[dict(split=key[0], kind=key[1], admitted_pair_windows=counts[key],
            admitted_recording_families=len(groups[key]), **coverage(list(windows[key].values())))
            for key in sorted(counts)],
        full_dataset_complete_at_end=(a.dataset / 'complete.json').exists(),
        independent_full_skin_recomputed=False, geometry_admission_unchanged=True,
        actual_optimizer_updates=0, quality_approved=False,
        note='Partial immutable clip snapshot. Mirrors and overlapping windows remain correlated. '
             'Source terminal velocities are hindsight labels, not recorded WASD. '
             'Contact labels are compared to the existing policy on original retarget joints.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'retarget', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Audit output already exists; preserve the old snapshot')
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir() and not (args.output / 'complete.json').exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
