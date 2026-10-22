"""Actual real-data optimizer/resume check; fixture is never eligible full data."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from motionbricks.data.full_terrain_cache import PrimitiveWriter
from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.unified_scene_field import VolumetricEgoSampler, field_from_heightfield
from scripts.cache_full_terrain_primitives import observations,sources
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


def tensor_digest(state):
    import hashlib
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def fixture(a):
    target = a.output/'fixture'
    target.mkdir()
    bundle = torch.load(a.bundle, map_location='cpu', mmap=True, weights_only=False)
    policy = terrain_observation_config(bundle['root_config']['model']['args'].get('observation_range_policy'))
    sampler = VolumetricEgoSampler()
    rep = build_native_representation(a.stats).to('cuda:0')
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    rows = [(next(case for case in cases if case.get('entry') == 'up'), 'parc'),
            (next(case for case in cases if case.get('name') == 'idle'), 'bones')]
    item=next(sources(SimpleNamespace(mixed=None,pfnn=a.pfnn,pilot=True),rep))
    start=item['starts'][0]
    rows.append((dict(world=item['world'][start:start+33],scene=item['scene'],label=item['record']['source_id']),'pfnn'))
    writer = PrimitiveWriter(target/'data')
    for case, domain in rows:
        world = case['world'][None, :33].to('cuda:0')
        observation, volume = observations(world, case['scene'], policy, sampler, field_from_heightfield(case['scene']))
        tensors = dict(world=world[0], volume_features=volume['features'][0], volume_distance=volume['distance'][0],
            volume_valid=volume['valid'][0], observation_points=observation['points'][0],
            observation_valid=observation['valid'][0], observation_envelope=observation['query_envelope_m'][0])
        for split in ('train','validation'):
            writer.append(tensors, dict(domain=domain, split=split, source_group=case['label'],
                source_id=case['label'], role='mechanics fixture, repeated source is not held-out quality evidence'))
    write_json(target/'complete.json', dict(data=writer.finish(), pilot=True,
        bundle_sha256=sha256(a.bundle), independent_validation=False))
    return target


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cache = fixture(a)
    common = [sys.executable, '-u', '-m', 'scripts.train_unified_full_corpus', '--bundle', str(a.bundle),
        '--preflight', str(a.preflight), '--cache', str(cache), '--steps', '2', '--batch', '8',
        '--micro-batch', '4', '--workers', '0', '--mechanics-only', '--checkpoint-every', '1',
        '--validation-every', '2']
    calls = []
    def run(name, stage, parent, extra=()):
        calls.append(name)
        command = common+['--stage',stage,'--parent',str(parent),'--output',str(a.output/name),*extra]
        with (a.output/(name+'-'+str(len(calls))+'.log')).open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return a.output/name/'weights.pt'
    uninterrupted = run('root_continuous', 'root', a.initial)
    run('root_resume', 'root', a.initial, ['--stop-after','1'])
    resumed = run('root_resume', 'root', a.initial, ['--resume',str(a.output/'root_resume/latest.pt')])
    first, second = [torch.load(path, map_location='cpu', weights_only=False) for path in (uninterrupted,resumed)]
    if tensor_digest(first['model_state']) != tensor_digest(second['model_state']) \
            or first['sample_digest'] != second['sample_digest']:
        raise ValueError('Checkpoint-resume changed model tensors or sample schedule')
    del first, second
    codec = run('codec', 'codec', resumed)
    pose = run('pose', 'pose', codec)
    write_json(a.output/'complete.json', dict(passed=True, actual_optimizer_updates=8,
        root_resume_model_bitwise_equal=True, root_resume_samples_equal=True,
        full_stage_order_executed=['root','codec','pose'], final_checkpoint_sha256=sha256(pose),
        mechanics_only=True, quality_approved=False, full_training_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','initial','cases','preflight','stats','pfnn','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    options = parser.parse_args()
    try:
        main(options)
    except Exception as error:
        if options.output.is_dir():
            write_json(options.output/'failure.json', dict(error=repr(error)))
        raise
