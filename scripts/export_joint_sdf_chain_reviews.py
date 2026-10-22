"""Package read-only chain diagnostics as uncorrected official SOMA comparisons."""

import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from motionbricks.data.native_mb_data import build_native_representation
from scripts.evaluate_joint_sdf import save_motion
from scripts.train_native_mb import sha256


def export_context_data(task):
    root = task / 'joint_sdf_20261012'
    source = root / 'context_pilot_data_v3'
    exact = root / 'exact_surface_v1'
    if not (source / 'complete.json').exists() or not (exact / 'complete.json').exists():
        raise ValueError('Context data and exact surface audits must complete')
    output = root / 'context_data_export_v1'
    output.mkdir(exist_ok=False)
    for file, name in ((source / 'complete.json', 'context-data-complete.json'),
                       (source / 'cases.pt', 'cases.pt'),
                       (exact / 'complete.json', 'exact-surface-complete.json'),
                       (root / 'source_context_goals.json', 'source-context-goals.json')):
        shutil.copyfile(file, output / name)
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals',
                    '--stats', str(task.parent / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),
                    '--reviews', 'source=' + str(source), '--output', str(output / 'visuals')], check=True)
    return output


def export_chain(task):
    base = task.parent
    root = task / 'joint_sdf_20261012'
    output = root / 'chain_context_export_v1'
    output.mkdir(exist_ok=False)
    trace = root / 'chain_trace_v2'
    review = root / 'source_context_v2'
    if not (trace / 'complete.json').exists() or not (review / 'complete.json').exists():
        raise ValueError('Both read-only diagnostics must complete before export')
    stats = base / 'long_route_mixed_10k_ab_v1/mixed_dataset/stats'
    rep = build_native_representation(stats)
    case = torch.load(root / 'decoder_ab_A/cases.pt', weights_only=False, map_location='cpu')[0]
    labels = []
    for refinement in (1, 2):
        directory = output / 'recordings' / ('refine' + str(refinement))
        directory.mkdir(parents=True)
        for label in ('initial', 'adapters', 'root', 'pose'):
            shutil.copyfile(trace / f'{label}-refine{refinement}.npz', directory / (label + '.npz'))
            if refinement == 1:
                save_motion(directory / (label + '-source.npz'), case['world'][None, 4:84],
                            case['scene'], rep, {'role': 'unchanged_reference', 'quality_approved': False})
        labels.append('refine' + str(refinement) + '=' + str(directory))
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals',
                    '--stats', str(stats), '--reviews', *labels, '--output', str(output / 'chain_visuals')], check=True)
    subprocess.run([sys.executable, '-m', 'scripts.export_joint_terrain_visuals', '--stats', str(stats),
                    '--reviews', *[label + '=' + str(review / label)
                                   for label in ('source', 'constant_approach', 'source_velocity_schedule')],
                    '--output', str(output / 'source_context_visuals')], check=True)
    for path, name in ((trace / 'complete.json', 'chain-complete.json'),
                       (review / 'complete.json', 'source-context-complete.json'),
                       (root / 'pilot_feedback_coverage.json', 'feedback-coverage.json'),
                       (root / 'stair_source_context_audit.json', 'source-screen.json')):
        shutil.copyfile(path, output / name)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('chain', 'context-data'), default='chain')
    args = parser.parse_args()
    task = Path.cwd().parent
    base = task.parent
    output = export_context_data(task) if args.stage == 'context-data' else export_chain(task)
    receipt = {'quality_approved': False, 'postprocessing': None,
               'files': {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    config = base / 'drive-auth/private-client.conf'
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(config)
    if settings['mb-private'].get('client_id') != 'GOOGLE_OAUTH_CLIENT_ID':
        raise ValueError('User-owned OAuth client mismatch')
    environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='',
                       http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    subprocess.run(['rclone', '--config', str(config), 'copy', str(output),
                    'private-store:motionbricks/joint_sdf_20261012/' + output.name,
                    '--immutable', '--checksum', '--transfers', '1', '--checkers', '1',
                    '--tpslimit', '1', '--retries', '2'], env=environment, check=True)
    print(json.dumps({'status': 'uploaded', 'output': str(output), 'quality_approved': False}), flush=True)


if __name__ == '__main__':
    main()
