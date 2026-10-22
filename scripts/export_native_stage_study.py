"""Official-SOMA capacity and feedback evidence, explicitly not a deployed controller."""
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

from motionbricks.data.native_mb_data import build_native_representation
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    rep = build_native_representation(a.stats)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    by_label = {c['label']: c for c in cases}
    pose_report = json.loads((a.pose_run / 'complete.json').read_text())
    codec_report = json.loads((a.codec_run / 'complete.json').read_text())
    feedback = json.loads((a.feedback_run / 'complete.json').read_text())
    if not pose_report['capacity_gate_pass'] or not codec_report['capacity_gate_pass']:
        raise ValueError('Export must accurately identify the passed four-window study')
    folders, records, scenes = {}, [], {}

    def add(clip, arm, values, scene):
        folder = a.output / 'recordings' / arm; folder.mkdir(parents=True, exist_ok=True)
        folders[arm] = folder; scenes[clip] = scene
        count = len(values['roots'])
        stem = f'{clip}-{arm}-{count}f'
        if ':' in stem or '\\' in stem or '/' in stem:
            raise ValueError('Unsafe portable recording name')
        np.savez_compressed(folder / (stem + '.npz'), **values)
        records.append(dict(clip=clip, arm=arm, stem=stem, frames=count))

    def read(path):
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}

    def source_values(world, scene, role='unchanged_reference'):
        pose = rep.inverse(world[None], False, return_all=True, joint_positions_from='global_rot_data')
        return dict(local_rot_mats=pose['local_joint_rots'][0].numpy(), roots=world[:, :3].numpy(),
            heightfield_y_m=np.asarray(scene['heightfield_y_m']),
            min_point_world_xyz_m=np.asarray(scene['min_point_world_xyz_m']), dx_m=np.float32(scene['dx_m']),
            metadata_json=np.array(json.dumps(dict(role=role, actual_frames=len(world), postprocessing=None))))

    anchors = {w['label']: w['start'] for w in pose_report['contract']['source_windows']}
    for label, start in anchors.items():
        case = by_label[label]
        name = case.get('entry', case['name']) + '-window'
        source = read(a.pose_run / 'final/source' / (label + '.npz'))
        add(name, 'source', source, case['scene'])
        for arm, path in [('before_pose', a.pose_run / 'baseline/actor'), ('trained_actor', a.pose_run / 'final/actor'),
                          ('codec_before', a.codec_run / 'baseline/reconstruction'), ('codec_after', a.codec_run / 'final/reconstruction')]:
            value = read(path / (label + '.npz'))
            meta = json.loads(str(value['metadata_json']))
            if arm in ('before_pose', 'trained_actor'):
                for key in ('local_rot_mats', 'roots'):
                    future = value[key][4:].copy(); value[key][:4] = source[key][:4]
                    np.testing.assert_array_equal(value[key][4:], future)
                meta.update(history_display='four actual observed frames; generated future unchanged', capacity_test_only=True)
            else:
                meta['offline_source_root_tokens'] = True
            value['metadata_json'] = np.array(json.dumps(meta))
            add(name, arm, value, case['scene'])
    for row in feedback['rows']:
        case = by_label[row['case']]
        name = case['entry'] + '-' + row['mode']
        value = read(a.feedback_run / 'recordings' / row['mode'] / (row['recording'] + '.npz'))
        add(name, 'feedback', value, case['scene'])
        start = anchors[row['case']] if row['mode'] == 'trained_entry' else 0
        source = source_values(case['world'][start:], case['scene'])
        reference_arm = 'reference' if row['mode'].startswith('idle_') else 'source'
        if reference_arm == 'reference':
            metadata = json.loads(str(source['metadata_json']))
            metadata.update(role='route_reference_different_initial_state', not_time_aligned=True)
            source['metadata_json'] = np.array(json.dumps(metadata))
        add(name, reference_arm, source, case['scene'])
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(a.stats),
        '--reviews', *[key + '=' + str(value) for key, value in folders.items()], '--output', str(a.output / 'visuals')], check=True)
    old = json.loads((a.output / 'visuals/manifest.json').read_text())
    manifest = dict(fps=30, clips={}, audit=old['audit'], quality_approved=False, postprocessing=None,
        labels=dict(source='Real source replay', before_pose='Codec and Root ready; Pose before training',
            trained_actor='Trained actor: learned window', codec_before='OFFLINE source codec: before',
            codec_after='OFFLINE source codec: after', feedback='Generated feedback: capacity model',
            reference='Route reference: different starting pose'),
        scope='Four-window learning proof plus all ten generated-history reviews, including failures. '
              'No WASD/generalization approval. Codec arms use source Root/tokens offline only. '
              'Actor arms predict their own Root/tokens. All real lengths retained.')
    for record in records:
        prior = old['clips'][record['stem']]
        arm = 'source' if record['arm'] == 'source' else record['arm']
        entry = dict(prior['arms'][arm], frames=record['frames'], duration_seconds=record['frames'] / 30)
        clip = manifest['clips'].setdefault(record['clip'], dict(frames=0, fps=30, arms={}, scene=prior['scene']))
        clip['frames'] = max(clip['frames'], record['frames']); clip['arms'][record['arm']] = entry
        # Render the union of all actual recordings against their unchanged original heightfield.
        full = scenes[record['clip']]; origin = np.asarray(full['min_point_world_xyz_m']); dx = float(full['dx_m'])
        rectangles = [clip['scene'], prior['scene']]
        lo = np.min([np.rint((np.asarray(s['origin'])[[0, 2]] - origin[[0, 2]]) / dx).astype(int) for s in rectangles], axis=0)
        hi = np.max([np.rint((np.asarray(s['origin'])[[0, 2]] - origin[[0, 2]]) / dx).astype(int)
            + [len(s['heightfield_y_m'][0]), len(s['heightfield_y_m'])] for s in rectangles], axis=0)
        crop = np.asarray(full['heightfield_y_m'])[lo[1]:hi[1], lo[0]:hi[0]]
        clip['scene'] = dict(origin=(origin + [lo[0] * dx, 0, lo[1] * dx]).tolist(), dx=dx,
            heightfield_y_m=[[float(v) if np.isfinite(v) else None for v in row] for row in crop])
    write_json(a.output / 'visuals/manifest.json', manifest)
    for label, report in [('codec', codec_report), ('pose', pose_report), ('feedback', feedback)]:
        write_json(a.output / (label + '-report.json'), report)
    files = {str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob('*') if p.is_file()}
    write_json(a.output / 'receipt.json', dict(files=files, frame_padding=False, quality_approved=False))
    archive = a.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(a.output, arcname=a.output.name)
    receipt = archive.with_suffix('.receipt.json')
    write_json(receipt, dict(filename=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size, files=len(files)))
    settings = configparser.ConfigParser(interpolation=None); settings.read(a.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Private user-owned OAuth required')
    env = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for path in (archive, receipt):
        subprocess.run(['rclone', '--config', str(a.drive_config), 'copyto', str(path),
            'private-store:motionbricks/parc_rugged32_20261013_v2/' + path.name, '--immutable', '--checksum',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=env, check=True)
    print(json.dumps(dict(recordings=len(records), clips=len(manifest['clips']), sha256=sha256(archive))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('codec-run', 'pose-run', 'feedback-run', 'cases', 'stats', 'output', 'drive-config'):
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())
