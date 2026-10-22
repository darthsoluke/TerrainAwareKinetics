"""Package a user-requested experimental actor without altering model tensors."""

import argparse
import configparser
import json
import os
from pathlib import Path
import subprocess

import torch

from scripts.package_approved_parc_live import state_hash
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def main(args):
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(args.drive_config)
    owned = 'GOOGLE_OAUTH_CLIENT_ID'
    if settings['mb-private'].get('client_id') != owned:
        raise ValueError('User-owned private Drive client required')
    torch.set_num_threads(2)
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    stage = saved['contract']['stair_training_stage']
    if stage['name'] != 'rugged_bones50_v1' or saved['step'] != 200:
        raise ValueError('Expected the completed fixed-SDF BONES pilot')
    if sum(saved['exposure'].values()) != stage['actor_primitives']:
        raise ValueError('Training exposure differs from stage contract')
    args.output.mkdir(parents=True, exist_ok=False)
    compact = {k: saved[k] for k in (
        'schema', 'model_state', 'step', 'contract', 'coefficients', 'exposure')}
    weights = args.output/'weights.pt'
    digest = state_hash(saved['model_state'])
    torch.save(compact, weights)
    restored = torch.load(weights, map_location='cpu', mmap=True, weights_only=False)
    if state_hash(restored['model_state']) != digest:
        raise ValueError('Portable export changed a model tensor')
    receipt = dict(schema='motionbricks.experimental_bones_live.v1',
        original_checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_sha256=sha256(weights),
        source_bundle_sha256=saved['contract']['source_bundle_sha256'],
        model_state_sha256=digest, model_tensors_unchanged=True,
        training_stage=stage, quality_approved=False, user_requested_trial=True,
        known_regression='Accepted stair skin peak 0.69cm -> 8.50cm',
        postprocessing=None, future_reference_inputs=False)
    write_json(args.output/'export.json', receipt)
    write_json(args.output/'receipt.json', dict(files={
        p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}))
    environment = dict(os.environ, HTTP_PROXY='', HTTPS_PROXY='', ALL_PROXY='',
        http_proxy='', https_proxy='', all_proxy='', NO_PROXY='*')
    subprocess.run(['rclone', '--config', str(args.drive_config), 'copy',
        str(args.output), args.cloud, '--immutable', '--checksum', '--transfers', '1',
        '--checkers', '1', '--tpslimit', '1', '--retries', '2',
        '--stats', '15s', '--stats-one-line', '--stats-log-level', 'NOTICE'],
        env=environment, check=True)
    print(json.dumps(dict(uploaded=True, **receipt)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'output', 'drive-config'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cloud', required=True)
    main(parser.parse_args())
