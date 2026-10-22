"""Export the first new physical PARC/SOMA source group, including its failure."""
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
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    audit = json.loads(a.audit.read_text())
    evidence = {r['source_id']: r for r in audit['rows']}
    rep = build_native_representation(a.dataset / 'stats')
    data = TerrainLearningDataset(a.dataset, rep, 'train')
    folder = a.output / 'recordings'; folder.mkdir()
    records = []
    for index, record in enumerate(data.records):
        if record['source_id'] not in evidence:
            continue
        item = data.full_reference_item(index)
        world, scene = item['world'], item['scene']
        pose = rep.inverse(world[None], False, return_all=True, joint_positions_from='global_rot_data')
        label = record['source_id']
        metadata = dict(evidence[label], role='unchanged_reference', requested_frames=len(world),
                        source_kind='new physically tracked PARC retargeted to SOMA; not trained MB output')
        np.savez_compressed(folder / (label + '.npz'), local_rot_mats=pose['local_joint_rots'][0].numpy(),
            roots=world[:, :3].numpy(), heightfield_y_m=scene['heightfield_y_m'].numpy(),
            min_point_world_xyz_m=scene['min_point_world_xyz_m'].numpy(), dx_m=np.float32(scene['dx_m']),
            metadata_json=np.array(json.dumps(metadata)))
        records.append(dict(label=label, frames=len(world), admitted=evidence[label]['ready_for_source_window_selection']))
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(a.dataset / 'stats'),
                    '--reviews', 'source=' + str(folder), '--output', str(a.output / 'visuals')], check=True)
    original = json.loads((a.output / 'visuals/manifest.json').read_text())
    clip = dict(frames=max(r['frames'] for r in records), fps=30, arms={})
    manifest = dict(fps=30, clips={'new-real-stair-sources': clip}, labels={}, audit=original['audit'],
        scope='All four are physically tracked PARC/SOMA source recordings, not MotionBricks predictions. '
              'The source failing the strict2cm screen remains visible; original frames and speeds retained.',
        trained_model_output=False, postprocessing=None)
    for i, row in enumerate(records):
        arm = f'source_{i}'
        old = original['clips'][row['label']]
        clip['arms'][arm] = dict(old['arms']['source'], frames=row['frames'], duration_seconds=row['frames'] / 30)
        clip.setdefault('scene', old['scene'])
        manifest['labels'][arm] = f'PARC source {i + 1}: ' + ('source screen passed' if row['admitted'] else 'source screen failed')
    write_json(a.output / 'visuals/manifest.json', manifest)
    write_json(a.output / 'source-audit.json', audit)
    files = {str(p.relative_to(a.output)): sha256(p) for p in a.output.rglob('*') if p.is_file()}
    write_json(a.output / 'receipt.json', dict(files=files, original_frames=True, trained_model_output=False))
    archive = a.output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(a.output, arcname=a.output.name)
    receipt = archive.with_suffix('.receipt.json')
    write_json(receipt, dict(filename=archive.name, sha256=sha256(archive), bytes=archive.stat().st_size, files=len(files)))
    settings = configparser.ConfigParser(interpolation=None); settings.read(a.drive_config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('Private OAuth required')
    env = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='', http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    for path in (archive, receipt):
        subprocess.run(['rclone', '--config', str(a.drive_config), 'copyto', str(path),
            'private-store:motionbricks/parc_rugged32_20261013_v2/' + path.name, '--immutable', '--checksum',
            '--transfers', '1', '--checkers', '1', '--tpslimit', '1', '--retries', '2'], env=env, check=True)
    print(json.dumps(dict(recordings=len(records), archive_sha256=sha256(archive), uploaded=True)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'audit', 'output', 'drive-config'):
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())
