"""Export every matched route result and clearly labeled command-timing diagnostics."""
import argparse
import configparser
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    rep = build_native_representation(a.stats)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    by_label = {case['label']: case for case in cases}
    pose_report = json.loads((a.pose_run / 'complete.json').read_text())
    if not pose_report['training_bank_gate_pass']:
        raise ValueError('Expected the completed native source-window bank')
    anchors = {w['label']: w['start'] for w in pose_report['contract']['capacity_anchors']}
    folders, records, scenes = {}, [], {}

    def read(path):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}

    def add(clip, arm, value, scene, **metadata):
        folder = a.output / 'recordings' / arm; folder.mkdir(parents=True, exist_ok=True)
        folders[arm] = folder; scenes[clip] = scene
        count = len(value['roots']); stem = f'{clip}-{arm}-{count}f'
        if any(c in stem for c in ':\\/'):
            raise ValueError('Portable filenames required')
        meta = json.loads(str(value['metadata_json']))
        meta.update(metadata, actual_frames=count, quality_approved=False, postprocessing=None)
        value['metadata_json'] = np.array(json.dumps(meta))
        np.savez_compressed(folder / (stem + '.npz'), **value)
        records.append(dict(clip=clip, arm=arm, stem=stem, frames=count))

    def source_value(world, scene, role):
        result = rep.inverse(world[None], False, return_all=True, joint_positions_from='global_rot_data')
        return dict(local_rot_mats=result['local_joint_rots'][0].numpy(), roots=world[:, :3].numpy(),
            heightfield_y_m=np.asarray(scene['heightfield_y_m']), min_point_world_xyz_m=np.asarray(scene['min_point_world_xyz_m']),
            dx_m=np.float32(scene['dx_m']), metadata_json=np.array(json.dumps(dict(role=role))))

    reports = {}
    for arm, folder in [('before', a.before_run), ('native', a.feedback_run),
                        ('refresh8', a.refresh_run), ('source_commands', a.schedule_run)]:
        report = json.loads((folder / 'complete.json').read_text()); reports[arm] = report
        if arm == 'before' and report.get('initial_baseline_sha256') != 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979':
            raise ValueError('Baseline must be preserved28941')
        if arm != 'before' and report['pose_sha256'] != pose_report['checkpoint_sha256']:
            raise ValueError('Diagnostic actor weights differ')
        for row in report['rows']:
            case = by_label[row['case']]; clip = case['entry'] + '-' + row['mode']
            add(clip, arm, read(folder / 'recordings' / row['mode'] / (row['recording'] + '.npz')), case['scene'],
                offline_command_schedule=arm == 'source_commands', refresh_diagnostic=arm == 'refresh8')
            if arm == 'before':
                start = anchors[row['case']] if row['mode'] == 'trained_entry' else 0
                reference = row['mode'].startswith('idle_')
                source_arm = 'reference' if reference else 'source'
                value = source_value(case['world'][start:], case['scene'],
                    'route_reference_different_initial_state' if reference else 'unchanged_reference')
                add(clip, source_arm, value, case['scene'], not_time_aligned=reference)
    for window in pose_report['contract']['capacity_anchors']:
        case = by_label[window['label']]
        clip = case.get('entry', case['name']) + '-source-window'
        filename = window['label'] + f'-f{window["start"]}.npz'
        for arm, folder in [('source', Path(pose_report['final_recordings']) / 'source'),
                            ('before_pose', a.pose_run / 'baseline/actor'),
                            ('native_window', Path(pose_report['final_recordings']) / 'actor')]:
            add(clip, arm, read(folder / filename), case['scene'], source_history_window_only=True)
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(a.stats),
        '--reviews', *[label + '=' + str(folder) for label, folder in folders.items()],
        '--output', str(a.output / 'visuals')], check=True)
    old = json.loads((a.output / 'visuals/manifest.json').read_text())
    manifest = dict(fps=30, clips={}, audit=old['audit'], postprocessing=None, quality_approved=False,
        labels=dict(before='Preserved28941', native='Native staged model: constant command',
            refresh8='Diagnostic: same model, refresh8', source_commands='OFFLINE source-derived command schedule',
            source='Real source replay', reference='Route reference: different initial pose',
            before_pose='Same codec/Root: before Pose training', native_window='Trained actor: source history window'),
        scope='All continuous route failures retained. Source-command schedule is offline evidence, not autonomous WASD. '
              'Source-history windows are capacity tests. Different real lengths remain independent recordings.')
    for record in records:
        entry = old['clips'][record['stem']]
        arm = record['arm']
        value = dict(entry['arms'][arm], frames=record['frames'], duration_seconds=record['frames'] / 30)
        clip = manifest['clips'].setdefault(record['clip'], dict(frames=0, fps=30, arms={}, scene=entry['scene']))
        clip['arms'][arm] = value; clip['frames'] = max(clip['frames'], record['frames'])
        full = scenes[record['clip']]; origin = np.asarray(full['min_point_world_xyz_m']); dx = float(full['dx_m'])
        rectangles = [clip['scene'], entry['scene']]
        lo = np.min([np.rint((np.asarray(s['origin'])[[0, 2]] - origin[[0, 2]]) / dx).astype(int) for s in rectangles], axis=0)
        hi = np.max([np.rint((np.asarray(s['origin'])[[0, 2]] - origin[[0, 2]]) / dx).astype(int)
                     + [len(s['heightfield_y_m'][0]), len(s['heightfield_y_m'])] for s in rectangles], axis=0)
        crop = np.asarray(full['heightfield_y_m'])[lo[1]:hi[1], lo[0]:hi[0]]
        clip['scene'] = dict(origin=(origin + [lo[0] * dx, 0, lo[1] * dx]).tolist(), dx=dx,
            heightfield_y_m=[[float(v) if np.isfinite(v) else None for v in row] for row in crop])
    write_json(a.output / 'visuals/manifest.json', manifest)
    reports['pose'] = pose_report
    for label, report in reports.items():
        write_json(a.output / (label + '-report.json'), report)
    files = {str(path.relative_to(a.output)): sha256(path) for path in a.output.rglob('*') if path.is_file()}
    write_json(a.output / 'receipt.json', dict(files=files, frame_padding=False, quality_approved=False))
    archive = a.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(a.output, arcname=a.output.name)
    receipt = archive.with_suffix('.receipt.json')
    write_json(receipt, dict(filename=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive), files=len(files)))
    settings = configparser.ConfigParser(interpolation=None); settings.read(a.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Owned private OAuth required')
    env = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for path in (archive, receipt):
        subprocess.run(['rclone', '--config', str(a.drive_config), 'copyto', str(path),
            'private-store:motionbricks/parc_rugged32_20261013_v2/' + path.name,
            '--immutable', '--checksum', '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=env, check=True)
    print(json.dumps(dict(recordings=len(records), clips=len(manifest['clips']), sha256=sha256(archive))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('pose-run', 'feedback-run', 'before-run', 'refresh-run', 'schedule-run', 'cases', 'stats', 'output', 'drive-config'):
        parser.add_argument('--' + key, type=Path, required=True)
    main(parser.parse_args())
