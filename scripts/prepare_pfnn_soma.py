"""Retarget all verified PFNN recordings offline, retaining failures and clocks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from motionbricks.data.pfnn_soma_retarget import PFNNSomaRetargeter, load_source
from motionbricks.data.lafan_soma_retarget import Y_TO_Z
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def convert(task):
    row, options = task
    torch.set_num_threads(1)
    root, output = Path(options['source']), Path(options['output'])
    source_path = root/row['filename']
    if sha256(source_path) != row['sha256']:
        raise ValueError('PFNN raw BVH identity changed')
    official = root/'official_source/motion'
    bind = load_source(root/'data/animations/rest.bvh', official)
    source = load_source(source_path, official)
    skeleton = SOMASkeleton30()
    stem = source_path.stem
    retarget = PFNNSomaRetargeter(bind, skeleton, options['gmr'], output/'calibration'/stem)
    local_ref = np.asarray(retarget.calibration['reference_local_rot_mats'])
    reference_root = retarget.calibration['uniform_scale']*bind['root_positions'][0]
    retarget.engine.configuration.update(retarget.qpos_from_motion(local_ref, reference_root))
    for _ in range(10):
        ref_local, ref_root = retarget.solve(bind['posed_joints'][0], bind['global_rot_mats'][0])
    _, ref_positions, _ = skeleton.fk(torch.from_numpy(ref_local[None]).float(), torch.from_numpy(ref_root[None]).float())
    desired_ref = np.asarray(retarget.calibration['reference_positions_relative_root_m'])+reference_root
    reference_error = float(np.abs(ref_positions[0].numpy()-desired_ref).max())
    if reference_error > .001:
        raise ValueError('PFNN bind conversion failed: '+str(reference_error))
    rng = np.random.default_rng(17121)
    fk_error = retarget.validate_fk(Rotation.from_rotvec(rng.normal(0, .2, (30, 3))).as_matrix(), np.array([.1, .9, -.2]))
    count = len(source['root_positions'])
    if options['max_frames']:
        count = min(count, options['max_frames'])
    retarget.engine.configuration.update(retarget.qpos_from_motion(local_ref,
        source['root_positions'][0]*retarget.calibration['uniform_scale']))
    local, roots, desired, rotations = [], [], [], []
    started = time.monotonic()
    names = list(skeleton.bone_order_names)
    for frame in range(count):
        lr, root_pos = retarget.solve(source['posed_joints'][frame], source['global_rot_mats'][frame])
        if frame == 0:
            for _ in range(10):
                lr, root_pos = retarget.solve(source['posed_joints'][frame], source['global_rot_mats'][frame])
        local.append(lr)
        roots.append(root_pos)
        desired.append([retarget.engine.scaled_human_data[name][0]@Y_TO_Z for name in names])
        rotations.append([Y_TO_Z.T@Rotation.from_quat(retarget.engine.scaled_human_data[name][1], scalar_first=True).as_matrix()@Y_TO_Z for name in names])
    local, roots = np.asarray(local, np.float32), np.asarray(roots, np.float32)
    with torch.no_grad():
        global_rot, positions, _ = skeleton.fk(torch.from_numpy(local), torch.from_numpy(roots))
    errors = np.linalg.norm(positions.numpy()-np.asarray(desired), axis=-1)
    feet = [names.index(side+suffix) for side in ('Left', 'Right') for suffix in ('Foot', 'ToeBase')]
    relative_rot = global_rot.numpy().swapaxes(-1, -2)@np.asarray(rotations)
    angular_error = np.rad2deg(Rotation.from_matrix(relative_rot.reshape(-1, 3, 3)).magnitude())
    metrics = dict(reference_pose_max_m=reference_error, fk_parity_max=fk_error,
        root_task_p95_m=float(np.quantile(errors[:, 0], .95)), root_task_max_m=float(errors[:, 0].max()),
        foot_task_p95_m=float(np.quantile(errors[:, feet], .95)), foot_task_max_m=float(errors[:, feet].max()),
        orientation_task_p95_deg=float(np.quantile(angular_error, .95)))
    failures = [key for key, bound in dict(root_task_p95_m=.02, root_task_max_m=.05,
        foot_task_p95_m=.03, foot_task_max_m=.08, orientation_task_p95_deg=10.).items() if metrics[key] > bound]
    folder = output/'shards'
    folder.mkdir(exist_ok=True)
    shard = folder/(stem+'.npz')
    np.savez_compressed(shard, local_rot_mats=local, root_positions=roots,
        posed_joints=positions.numpy(), global_rot_mats=global_rot.numpy(),
        source_positions=source['posed_joints'][:count].astype(np.float32),
        source_global_rot_mats=source['global_rot_mats'][:count].astype(np.float32),
        source_names=np.asarray(source['source_skeleton']['names']),
        fps=np.float32(30), source_fps=np.float64(source['source_fps']),
        uniform_scale=np.float64(retarget.calibration['uniform_scale']), source_frame_times=np.arange(count)/30)
    result = dict(filename=row['filename'], source_id='pfnn:'+stem,
        recording_group=row['source_recording_group'], mirrored=row['mirrored'],
        original_frames=source['original_frames'], resampled_total_frames=len(source['root_positions']),
        actual_converted_frames=count, source_bvh_sha256=row['sha256'],
        shard=shard.relative_to(output).as_posix(), shard_sha256=sha256(shard),
        metrics=metrics, failures=failures, calibration=retarget.calibration,
        geometry_pairing_pending=True, training_approved=False, elapsed_seconds=time.monotonic()-started)
    write_json(output/'reports'/(stem+'.json'), result)
    print(json.dumps(dict(clip=stem, frames=count, metrics=metrics, failures=failures)), flush=True)
    return result


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'reports').mkdir()
    inventory = json.loads((args.source/'inventory.json').read_text())
    rows = [row for row in inventory['clips'] if row['role'] == 'motion']
    if args.clips:
        selected = set(args.clips)
        rows = [row for row in rows if Path(row['filename']).stem in selected]
        if len(rows) != len(selected):
            raise ValueError('Requested pilot clip was not present')
    if not rows or any(not row['official_crc_match'] for row in rows):
        raise ValueError('All raw motions must match the verified release')
    options = dict(source=str(args.source.resolve()), output=str(args.output.resolve()),
        gmr=str(args.gmr.resolve()), max_frames=args.max_frames)
    results = []
    tasks = [(row, options) for row in rows]
    if args.workers == 1:
        iterator = map(convert, tasks)
        for result in iterator:
            results.append(result)
            write_json(args.output/'progress.json', dict(completed=len(results), total=len(rows), rows=results))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(convert, tasks):
                results.append(result)
                write_json(args.output/'progress.json', dict(completed=len(results), total=len(rows), rows=results))
    write_json(args.output/'complete.json', dict(rows=results, source_inventory_sha256=sha256(args.source/'inventory.json'),
        full_source_conversion=args.max_frames == 0 and args.clips is None,
        converted_frames=sum(row['actual_converted_frames'] for row in results),
        source_motion_modified=False, runtime_postprocessing=None, geometry_pairing_pending=True,
        training_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'gmr'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--clips', nargs='+')
    main(parser.parse_args())
