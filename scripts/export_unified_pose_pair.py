"""Private, hash-bound official-skin source/before/Pose-A/Pose-B recordings."""

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
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    cases += [c for c in torch.load(args.flat_cases, map_location='cpu', weights_only=False) if c['name'] in ('idle', 'walk')]
    by_label = {c['label']: c for c in cases}
    with np.load(args.scene, allow_pickle=False) as data:
        scene = {k: data[k].copy() for k in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
    report_a = json.loads((args.adapter / 'complete.json').read_text())
    report_b = json.loads((args.capacity / 'complete.json').read_text())
    coverage = args.comparison == 'coverage'
    if coverage:
        for key in ('initial_sha256', 'bundle_sha256', 'data_sha256', 'profile', 'updates',
                    'trainable_parameters', 'full_mask_fraction', 'pose_condition_stage', 'zero_pose_residual'):
            if report_a['contract'][key] != report_b['contract'][key]:
                raise ValueError('Coverage comparison differs beyond source-window order: ' + key)
        if (report_a['contract'].get('window_schedule', 'global-step-phase-v1') != 'global-step-phase-v1'
                or report_b['contract']['window_schedule'] != 'per-source-phase-v2'):
            raise ValueError('Expected archived global-step versus corrected per-source schedule')
        schedules = [json.loads((p / 'schedule.json').read_text()) for p in (args.adapter, args.capacity)]
        orders = [[(d['case'], d['mask_seed']) for batch in schedule for d in batch] for schedule in schedules]
        if orders[0] != orders[1]:
            raise ValueError('Source identity order or masking RNG changed')
    else:
        for key in ('schedule_sha256', 'corruption_sha256'):
            if report_a[key] != report_b[key]:
                raise ValueError('Matched Pose arms have different exposures or corruption')
    if report_a['baseline'] != report_b['baseline']:
        raise ValueError('Baseline comparisons differ')
    placement = args.comparison == 'placement'
    sparse_history = args.comparison == 'sparse-history'
    joint = args.comparison == 'joint'
    if joint:
        for key in ('initial_sha256', 'bundle_sha256', 'data_sha256', 'profile', 'updates',
                    'full_mask_fraction', 'pose_condition_stage', 'zero_pose_residual', 'window_schedule'):
            if report_a['contract'][key] != report_b['contract'][key]:
                raise ValueError('Joint comparison changed another contract field: ' + key)
        if (report_a['contract']['arm'] != 'adapter' or report_b['contract']['arm'] != 'root_pose'
                or not report_b['contract']['joint_preflight_sha256']):
            raise ValueError('Expected Pose-only control and gated joint Root/Pose adaptation')
    if sparse_history:
        for key in ('initial_sha256', 'bundle_sha256', 'data_sha256', 'profile', 'updates',
                    'trainable_parameters', 'full_mask_fraction', 'pose_condition_stage', 'zero_pose_residual', 'window_schedule'):
            if report_a['contract'][key] != report_b['contract'][key]:
                raise ValueError('Sparse-history comparison changed another factor: ' + key)
        if (report_a['contract'].get('teacher_history', 'full') != 'full'
                or report_b['contract']['teacher_history'] != 'native-sparse-start-v1'):
            raise ValueError('Require full versus native sparse teacher history')
    if placement:
        for report, stage in ((report_a, 'input'), (report_b, 'output')):
            if (report['contract']['pose_condition_stage'] != stage
                    or not report['contract']['zero_pose_residual'] or report['contract']['arm'] != 'adapter'):
                raise ValueError('Placement export requires matched zero-initialized adapter arms')
        if args.mask or args.downhill or args.additional_downhill:
            raise ValueError('Do not relabel an older capacity/downhill result as a placement result')
    manifest = dict(fps=30, clips={}, labels=dict(source='Real source replay', before='Before: current28941',
        adapter='Pose A: SDF adapter', capacity='Pose B: added native capacity'), quality_approved=False,
        postprocessing=None, scope='Original frame counts and failed outputs retained. Source is recorded expert motion; '
                                    'both Pose candidates are unapproved model outputs.')
    if placement:
        manifest['labels'] = dict(source='Real source replay', zero='Matched zero Pose residual',
            input='Pose SDF at transformer input', output='Pose SDF before token projection')
    if coverage:
        manifest['labels'] = dict(source='Real source replay', zero='Matched zero Pose residual',
            control='Previous source-window schedule', coverage='Corrected source-window coverage')
    if sparse_history:
        manifest['labels'] = dict(source='Real source replay', zero='Matched zero Pose residual',
            control='Training with full Pose history', sparse='Training with sparse Pose history')
    if joint:
        manifest['labels'] = dict(source='Real source replay', zero='Matched zero Pose residual',
            control='Pose-only token supervision', joint='Joint Root and Pose, decoded-motion supervision')
    folders, records = {}, []

    def save(clip, arm, world, terrain, metadata):
        folder = args.output / 'recordings' / arm
        folder.mkdir(parents=True, exist_ok=True); folders[arm] = folder
        stem = clip + '-' + arm + '-' + str(len(world)) + 'f'
        pose = rep.inverse(world[None], False, return_all=True, joint_positions_from='global_rot_data')
        np.savez_compressed(folder / (stem + '.npz'), local_rot_mats=pose['local_joint_rots'][0].numpy(),
            roots=world[:, :3].numpy(), heightfield_y_m=np.asarray(terrain['heightfield_y_m']),
            min_point_world_xyz_m=np.asarray(terrain['min_point_world_xyz_m']), dx_m=np.float32(terrain['dx_m']),
            metadata_json=np.array(json.dumps(dict(metadata, role='unchanged_reference' if arm == 'source' else 'generated',
                actual_frames=len(world), requested_frames=len(world), no_padding=True, postprocessing=None))))
        records.append(dict(clip=clip, arm=arm, stem=stem, frames=len(world)))

    for case in cases:
        save(case['label'], 'source', case['world'], case['scene'], dict(source_id=case['row']['source_id'], quality_approved=False))
    arms = [('before', args.adapter, report_a, 'baseline'), ('adapter', args.adapter, report_a, 'final'),
            ('capacity', args.capacity, report_b, 'final')]
    if placement:
        arms = [('zero', args.adapter, report_a, 'baseline'), ('input', args.adapter, report_a, 'final'),
                ('output', args.capacity, report_b, 'final')]
    if coverage:
        if args.mask or args.downhill or args.additional_downhill:
            raise ValueError('Do not relabel older results as a source-coverage comparison')
        arms = [('zero', args.adapter, report_a, 'baseline'), ('control', args.adapter, report_a, 'final'),
                ('coverage', args.capacity, report_b, 'final')]
    if sparse_history:
        if args.mask or args.downhill or args.additional_downhill:
            raise ValueError('Do not mix other experiments into the history comparison')
        arms = [('zero', args.adapter, report_a, 'baseline'), ('control', args.adapter, report_a, 'final'),
                ('sparse', args.capacity, report_b, 'final')]
    if joint:
        if args.mask or args.downhill or args.additional_downhill:
            raise ValueError('Do not relabel older results as a joint-adaptation comparison')
        arms = [('zero', args.adapter, report_a, 'baseline'), ('control', args.adapter, report_a, 'final'),
                ('joint', args.capacity, report_b, 'final')]
    if args.reference_run:
        reference = json.loads((args.reference_run / 'complete.json').read_text())
        if reference['contract']['initial_sha256'] != report_a['contract']['initial_sha256']:
            raise ValueError('Original live reference uses a different initial model')
        arms.insert(0, ('before', args.reference_run, reference, 'baseline'))
        manifest['labels']['before'] = 'Original current28941'
    if args.mask:
        mask_report = json.loads((args.mask / 'complete.json').read_text())
        if mask_report['schedule_sha256'] != report_a['schedule_sha256'] or mask_report['baseline'] != report_a['baseline']:
            raise ValueError('Mask comparison must retain the source schedule and initial baseline')
        arms.append(('mask', args.mask, mask_report, 'final'))
        manifest['labels']['mask'] = 'Pose C: 25 percent full mask'
    for arm, run, report, stage in arms:
        if not report['protected_hash_unchanged'] or report['deployed']:
            raise ValueError('Expected protected, undeployed research checkpoint')
        shutil.copyfile(run / 'complete.json', args.output / (arm + '-report.json'))
        for row in report[stage]['sources']:
            case = by_label[row['label']]
            with np.load(run / stage / 'sources' / (row['label'] + '.npz'), allow_pickle=False) as data:
                world = torch.from_numpy(data['global_features'].copy())
            if len(world) != row['actual_frames']:
                raise ValueError('Frame count mismatch')
            save(row['label'], arm, world, case['scene'], row)
        for row in report[stage]['live_stairs']:
            speed = row['speed_request_mps']
            with np.load(run / stage / 'live_stairs' / f'forward-{speed:g}.npz', allow_pickle=False) as data:
                world = torch.from_numpy(data['global_features'].copy())
            save(f'live-stairs-speed{speed:g}', arm, world, scene, row)
    for index, folder in enumerate([args.downhill, *args.additional_downhill]):
        if folder is None:
            continue
        downhill = json.loads((folder / 'complete.json').read_text())
        shutil.copyfile(folder / 'complete.json', args.output / f'downhill-{index}-report.json')
        for row in downhill['rows']:
            with np.load(folder / row['recording'], allow_pickle=False) as data:
                world = torch.from_numpy(data['global_features'].copy())
            save(f'live-downstairs-speed{row["speed_request_mps"]:g}', row['arm'], world, scene, row)
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(args.stats),
        '--reviews', *[k + '=' + str(v) for k, v in folders.items()], '--output', str(args.output / 'visuals')], check=True)
    original = json.loads((args.output / 'visuals/manifest.json').read_text())
    manifest['audit'] = original['audit']
    for record in records:
        old = original['clips'][record['stem']]
        clip = manifest['clips'].setdefault(record['clip'], dict(fps=30, frames=0, arms={}, scene=old['scene']))
        entry = dict(old['arms'][record['arm']], frames=record['frames'], duration_seconds=record['frames'] / 30)
        if len(entry['roots']) != record['frames']:
            raise ValueError('Export padded a recording')
        clip['arms'][record['arm']] = entry
        clip['frames'] = max(clip['frames'], record['frames'])
        # Generated live travel can exceed the source crop; retain the widest original crop.
        if len(old['scene']['heightfield_y_m']) * len(old['scene']['heightfield_y_m'][0]) > len(clip['scene']['heightfield_y_m']) * len(clip['scene']['heightfield_y_m'][0]):
            clip['scene'] = old['scene']
    write_json(args.output / 'visuals/manifest.json', manifest)
    files = {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output / 'receipt.json', dict(files=files, frame_padding=False, quality_approved=False))
    archive = args.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(args.output, arcname=args.output.name)
    receipt = archive.with_suffix('.receipt.json')
    write_json(receipt, dict(filename=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size, files=len(files)))
    settings = configparser.ConfigParser(interpolation=None); settings.read(args.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Private user-owned OAuth client required')
    environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for path in (archive, receipt):
        subprocess.run(['rclone', '--config', str(args.drive_config), 'copyto', str(path),
            'private-store:motionbricks/parc_rugged32_20261013_v2/' + path.name, '--immutable', '--checksum',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=environment, check=True)
    print(json.dumps(dict(status='uploaded', recordings=len(records), clips=len(manifest['clips']), archive_sha256=sha256(archive))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'cases', 'flat-cases', 'scene', 'stats', 'adapter', 'capacity', 'output', 'drive-config'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--downhill', type=Path)
    p.add_argument('--mask', type=Path)
    p.add_argument('--additional-downhill', type=Path, nargs='*', default=[])
    p.add_argument('--comparison', choices=('capacity', 'placement', 'coverage', 'sparse-history', 'joint'), default='capacity')
    p.add_argument('--reference-run', type=Path)
    main(p.parse_args())
