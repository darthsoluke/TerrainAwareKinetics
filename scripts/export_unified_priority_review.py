"""Export every live-stair comparison arm at its original length via private Drive."""

import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import torch

from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    inference, _ = load_terrain_bundle(args.bundle, 'cpu')
    rep = inference.global_motion_rep
    with np.load(args.scene, allow_pickle=False) as archive:
        hf = archive['heightfield_y_m'].copy()
        origin = archive['min_point_world_xyz_m'].copy()
        dx = float(archive['dx_m'])
    arms = dict(before=('unified_goal_legacy_20261014_v1', 'baseline'),
        legacy=('unified_goal_legacy_20261014_v1', 'final'),
        terminal=('unified_goal_terminal_20261014_v1', 'final'),
        coverage=('unified_goal_coverage_20261014_v1', 'final'),
        root_stage=('unified_root_stage_20261014_v1', 'final'),
        root_height=('unified_root_height_20261014_v1', 'final'))
    if args.include_approach:
        arms['approach'] = ('unified_approach_fit_20261014_v1', 'final')
    records, review_paths = [], {}
    for arm, (run, stage) in arms.items():
        report = json.loads((args.runs / run / 'complete.json').read_text())
        if not report['frozen_hash_unchanged'] or report['deployed']:
            raise ValueError('Expected frozen-body, non-deployed research candidates')
        folder = args.output / 'recordings' / arm
        folder.mkdir(parents=True)
        review_paths[arm] = folder
        shutil.copyfile(args.runs / run / 'complete.json', args.output / (arm + '-report.json'))
        for row in report[stage]['live_stairs']:
            speed = row['speed_request_mps']
            source_file = args.runs / run / stage / 'live_stairs' / f'forward-{speed:g}.npz'
            with np.load(source_file, allow_pickle=False) as archive:
                world = torch.from_numpy(archive['global_features'].copy())[None]
            count = world.shape[1]
            if count != row['actual_frames']:
                raise ValueError('Recorded frame count differs from evaluation')
            pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
            clip = f'live-stairs-speed{speed:g}'
            stem = f'{clip}-{arm}-{count}f'
            metadata = dict(row, role='generated', requested_frames=count,
                source_recording_sha256=sha256(source_file), initial_history_frames=4,
                completion='goal_reached' if row['reached_after_stair_goal'] else 'bounded_review_ended_without_goal',
                postprocessing=None, quality_approved=False, frame_padding=False)
            np.savez_compressed(folder / (stem + '.npz'),
                local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(), roots=world[0, :, :3].numpy(),
                heightfield_y_m=hf, min_point_world_xyz_m=origin, dx_m=np.float32(dx),
                metadata_json=np.array(json.dumps(metadata)))
            records.append(dict(clip=clip, stem=stem, arm=arm, frames=count, source_file=str(source_file)))
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(args.stats),
        '--reviews', *[name + '=' + str(path) for name, path in review_paths.items()],
        '--output', str(args.output / 'visuals')], check=True)
    path = args.output / 'visuals/manifest.json'
    original = json.loads(path.read_text())
    manifest = dict(fps=30, clips={}, postprocessing=None, quality_approved=False, audit=original['audit'],
        labels=dict(before='Current 28941', legacy='Legacy-label control', terminal='Aligned terminal labels',
                    coverage='Dense source windows', root_stage='Native Root-stage candidate',
                    root_height='Root height-readout candidate', approach='Real earlier approach training'),
        scope='Uncorrected model outputs at original30Hz. Each arm keeps its actual frame count; ended panels do not extend the recording. '
              'All live-stair speeds and all failed candidates are retained. No matching real-source motion exists for this authored test world.')
    for record in records:
        arm = record['arm']
        entry = dict(original['clips'][record['stem']]['arms'][arm], frames=record['frames'], fps=30,
                     duration_seconds=record['frames'] / 30)
        if len(entry['roots']) != record['frames']:
            raise ValueError('Export changed a recording length')
        clip = manifest['clips'].setdefault(record['clip'], dict(frames=0, fps=30, arms={}))
        clip['frames'] = max(clip['frames'], record['frames'])
        clip['arms'][arm] = entry
    for clip in manifest['clips'].values():
        roots = np.concatenate([arm['roots'] for arm in clip['arms'].values()])
        lo, hi = roots[:, [0, 2]].min(0) - 5, roots[:, [0, 2]].max(0) + 5
        x0, z0 = np.maximum(0, np.floor((lo - origin[[0, 2]]) / dx).astype(int))
        x1, z1 = np.minimum([hf.shape[1], hf.shape[0]], np.ceil((hi - origin[[0, 2]]) / dx).astype(int) + 1)
        clip['scene'] = dict(heightfield_y_m=hf[z0:z1, x0:x1].tolist(),
            origin=(origin + np.array([x0 * dx, 0., z0 * dx])).tolist(), dx=dx)
    write_json(path, manifest)
    files = {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output / 'receipt.json', dict(quality_approved=False, frame_padding=False, playback_retiming=False,
        files=files))
    archive = args.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(args.output, arcname=args.output.name)
    receipt = archive.with_suffix('.receipt.json')
    write_json(receipt, dict(filename=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size, files=len(files)))
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(args.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('The user-owned private OAuth client is required')
    environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='',
                       http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for file in (archive, receipt):
        subprocess.run(['rclone', '--config', str(args.drive_config), 'copyto', str(file),
            'private-store:motionbricks/parc_rugged32_20261013_v2/' + file.name,
            '--immutable', '--checksum', '--transfers', '1', '--checkers', '1', '--tpslimit', '1',
            '--retries', '2'], env=environment, check=True)
    print(json.dumps(dict(status='uploaded', clips=len(manifest['clips']), arms=len(arms),
                         recordings=len(records), bytes=archive.stat().st_size, quality_approved=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'scene', 'stats', 'runs', 'output', 'drive-config'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--include-approach', action='store_true')
    main(parser.parse_args())
