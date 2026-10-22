"""Compare serial and committed eager playback using a strictly loaded full model."""

import argparse
import json
import math
from pathlib import Path

import torch

from motionbricks.data.full_corpus_inference import inference_receipt, load_candidate, tensor_digest
from motionbricks.data.unified_committed_playback import CommittedUnifiedBackend
from motionbricks.data.unified_terrain import UnifiedTerrainBackend, UnifiedTerrainController, PROFILE
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def random_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_random(state):
    torch.set_rng_state(state[0])
    torch.cuda.set_rng_state_all(state[1])


def exact(left, right):
    torch.testing.assert_close(left, right, atol=0, rtol=0)
    return float((left - right).abs().max()) if left.numel() else 0.


def commands(yaw):
    for label, count, speed, direction in (
            ('forward', 4, .8, yaw), ('stop', 2, 0., yaw),
            ('turn90', 2, .8, yaw + math.pi/2), ('run', 2, 1.6, yaw + math.pi/2)):
        for _ in range(count):
            yield label, [speed*math.sin(direction), speed*math.cos(direction)], direction


@torch.inference_mode()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    mechanics = saved['contract']['mechanics_only']
    if mechanics != a.mechanics:
        raise ValueError('Mechanics fixture requires explicit diagnostic opt-in')
    checkpoint_hash, bundle_hash = sha256(a.checkpoint), sha256(a.bundle)
    receipt = inference_receipt(saved, checkpoint_hash, checkpoint_hash, bundle_hash,
        allow_mechanics=a.mechanics)
    inference, bundle = load_terrain_bundle(a.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    load_candidate(controller, saved, receipt, bundle_hash, allow_mechanics=a.mechanics)
    before = tensor_digest(controller.state_dict())
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    selected = [next(case for case in cases if case.get('entry') == name or case.get('name') == name)
        for name in ('up', 'down', 'idle')]
    results = []
    try:
        for case in selected:
            history = case['world'][None, :4].to(inference.device)
            yaw = float(torch.atan2(history[0, -1, 4], history[0, -1, 3]))
            serial = UnifiedTerrainBackend(controller, history.clone(), case['scene'])
            committed = CommittedUnifiedBackend(controller, history.clone(), case['scene'])
            for backend in (serial, committed):
                backend.configure_runtime(dict(PROFILE))
            rows, previous, rejected = [], None, 0
            for block, (label, velocity, heading) in enumerate(commands(yaw)):
                if previous is not None:
                    serial.history = previous['published_history'].clone()
                rng = random_state()
                expected = serial.predict_plan(velocity, heading)
                restore_random(rng)
                if previous is None:
                    actual = committed.predict_plan(velocity, heading)
                else:
                    # Preparing ahead may read the already committed model block,
                    # but must not advance the history or consume the live buffer.
                    observed = committed.history.clone()
                    buffer = committed._future_buffer.clone()
                    published = committed._last_published_history.clone()
                    command = committed._last_command
                    actual = committed.prepare_next_plan(velocity, heading, observed,
                        previous['published_history'])
                    exact(committed.history, observed)
                    exact(committed._future_buffer, buffer)
                    exact(committed._last_published_history, published)
                    if committed._last_command != command:
                        raise ValueError('Preparation mutated the live command')
                    factual = previous['published_history'].clone()
                    committed.history = factual.clone()
                    committed.history[0, -1, 0] += .001
                    try:
                        committed.commit_prepared_plan(actual)
                    except ValueError:
                        rejected += 1
                    else:
                        raise AssertionError('Incorrect played history accepted a prepared plan')
                    committed.history = factual
                    committed.commit_prepared_plan(actual)
                differences = {key: exact(actual[key], expected[key])
                    for key in ('planned_motion', 'published_history')}
                differences['remaining_buffer'] = exact(committed._future_buffer, serial._future_buffer)
                if (committed._last_command != serial._last_command
                        or committed.last_generation['neural_generation'] != serial.last_generation['neural_generation']):
                    raise ValueError('Committed command or generation schedule differs')
                rows.append(dict(block=block, command=label, differences=differences,
                    neural_generation=serial.last_generation['neural_generation'],
                    remaining_frames=serial._future_buffer.shape[1]))
                previous = actual
            result = dict(label=case['label'], packets=rows, generated_frames=len(rows)*8,
                mismatched_history_rejections=rejected)
            results.append(result)
            write_json(a.output / ('case-' + str(len(results)) + '.json'), result)
        after = tensor_digest(controller.state_dict())
        if before != after:
            raise ValueError('Playback comparison changed model tensors')
        write_json(a.output / 'complete.json', dict(passed=True, mechanics_only=mechanics,
            checkpoint_sha256=checkpoint_hash, bundle_sha256=bundle_hash,
            cases_sha256=sha256(a.cases), script_sha256=sha256(Path(__file__)),
            model_state_sha256=before, weights_unchanged=True, runtime_profile=dict(PROFILE),
            rows=results, actual_optimizer_updates=0, interactive_server_started=False,
            quality_approved=False, execution='eager', rng_matched_per_call=True,
            scope='Backend equivalence and immutable playback guards only. No async browser timing, '
                  '30fps, collision quality, or full-route acceptance.'))
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'checkpoint', 'cases', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mechanics', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; preserve previous evidence')
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
