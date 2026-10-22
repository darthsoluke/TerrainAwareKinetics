"""Bounded codec capacity A/B: source reconstruction plus factual played-prefix loss."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.decoded_factual_history import FACTUAL_PREFIX_VERSION, decoded_factual_history_loss
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from motionbricks.data.joint_sdf_controller import JointSDFController, SDF_VERSION
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_full_context import command, evaluate, load_cases
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def frozen_hashes(model):
    values = {name: state_digest(getattr(model, name)) for name in
        ('root', 'pose', 'control', 'root_far', 'pose_far', 'decoder_near', 'decoder_attention',
         'decoder_far', 'sdf_encoder', 'sdf_joint_embedding', 'sdf_root', 'sdf_pose', 'sdf_decoder')}
    values.update(encoder=state_digest(model.codec.encoder), quantizer=state_digest(model.codec.quantizer))
    return values


@torch.no_grad()
def collect_histories(model, cases):
    bank = []
    digest = hashlib.sha256()
    for case in cases:
        history = case['world'][None, :4].cuda()
        state = SoftCommandState()
        entries = []
        for position in range(case['training_segments']):
            control = command(case, position * 8)
            intent = state.from_controls(history, control, 8)
            entries.append(dict(history=history[0].cpu().clone(), intent=intent[0].cpu().clone()))
            digest.update(entries[-1]['history'].numpy().tobytes())
            digest.update(entries[-1]['intent'].numpy().tobytes())
            result = model(history, intent, [case['scene']])
            history = reencode_played_history(result['world'][:, 8:12], model.inference.global_motion_rep)
        bank.append(entries)
    return bank, digest.hexdigest()


def main(args):
    if not 1 <= args.updates <= 100:
        raise ValueError('This diagnostic is limited to100 actual updates')
    args.output.mkdir(parents=True, exist_ok=False)
    state_path = args.output / 'state.json'
    write_json(state_path, dict(status='initializing', quality_approved=False))
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    np.random.seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    source = Path(saved['contract']['source'])
    if sha256(source) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle identity mismatch')
    infer, bundle = load_terrain_bundle(source, 'cuda:0')
    model = JointSDFController(infer, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    cases = load_cases(args.cases, Path(saved['contract']['dataset']))
    geometry = JointGeometryObjective(infer.global_motion_rep, 'cuda:0')
    frozen = frozen_hashes(model)
    bank, bank_hash = collect_histories(model, cases)
    torch.save(bank, args.output / 'factual_history_bank.pt')
    contract = dict(arm=args.arm, actual_updates=args.updates, batch=4,
        actor_prefix_exposures=args.updates * 4, source_reconstruction_exposures=args.updates * 4,
        initial_checkpoint=str(args.checkpoint), initial_sha256=sha256(args.checkpoint),
        cases_sha256=sha256(args.cases), factual_history_bank_sha256=bank_hash,
        trainable='codec.decoder only', frozen_hashes=frozen, optimizer='fresh Adam5e-5, clip0.5',
        actor_targets='only actual played root0..2, height0..3, RIC and rotations0..3; no actor future target',
        auxiliary='unchanged native source codec reconstruction with its own source history and own observed terrain',
        source_future_use='training-only source autoencoder branch, never actor conditions or actual played history',
        prefix_version=FACTUAL_PREFIX_VERSION, prefix_coefficient=1 if args.arm == 'prefix' else 0,
        diagnostic_scope='bounded decoder recovery capacity, not continuation of previous joint training; '
                         'no replacement of impossible actor futures by fabricated labels',
        training_distribution='three stairs plus alternating idle/walk;400 actor and400 source-codec examples per100updates',
        quality_approved=False, postprocessing=None)
    write_json(args.output / 'contract.json', contract)
    write_json(state_path, dict(status='baseline', quality_approved=False))
    baseline = evaluate(model, bundle, cases, geometry, args.output / 'baseline')
    model.codec.decoder.requires_grad_(True)
    parameters = list(model.codec.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    positions = [0] * 4
    indices = [0, 1, 2, 3]
    schedule = hashlib.sha256()
    started = time.perf_counter()
    logs = []
    write_json(state_path, dict(status='training', quality_approved=False))
    for step in range(args.updates):
        for row, index in enumerate(indices):
            if positions[row] == len(bank[index]):
                positions[row] = 0
                if row == 3:
                    indices[row] = 4 if index == 3 else 3
        selected = [cases[index] for index in indices]
        windows = torch.stack([case['world'][position * 8:position * 8 + 33]
                               for case, position in zip(selected, positions)]).cuda()
        history = torch.stack([bank[index][position]['history'] for index, position in zip(indices, positions)]).cuda()
        intent = torch.stack([bank[index][position]['intent'] for index, position in zip(indices, positions)]).cuda()
        scenes = [case['scene'] for case in selected]
        schedule.update(json.dumps([indices, positions]).encode())
        if step == 0:
            with torch.no_grad():
                plain = model(history, intent, scenes)['world'].clone()
                changed = windows.clone()
                changed[:, 4:, :3] += 3
                with_target = model(history, intent, scenes, source=changed)['world'].clone()
                torch.testing.assert_close(plain, with_target, atol=0, rtol=0)
            with torch.inference_mode():
                runtime = model(history, intent, scenes)['world'].clone()
            torch.testing.assert_close(plain, runtime, atol=3e-5, rtol=1e-5)
            parity = dict(actor_future_target_difference=float((plain - with_target).abs().max()),
                          runtime_difference=float((plain - runtime).abs().max()))
        actor = model(history, intent, scenes)
        prefix_loss = decoded_factual_history_loss(infer, model.last_local[:, :32], history, actor['observation'])
        actor_before_auxiliary = actor['world'].detach().clone()
        with torch.no_grad():
            source_intents = torch.cat([SoftCommandState().from_controls(windows[row:row + 1, :4],
                command(case, position * 8), 8) for row, (case, position) in enumerate(zip(selected, positions))])
            source_result = model(windows[:, :4], source_intents, scenes, source=windows)
        source_loss = native_codec_reconstruction(model, source_result)['loss']
        torch.testing.assert_close(actor['world'], actor_before_auxiliary, atol=0, rtol=0)
        loss = source_loss + contract['prefix_coefficient'] * prefix_loss
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite capacity loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or float(norm) == 0:
            raise ValueError('No finite decoder gradient')
        if any(p.grad is not None for name, p in model.named_parameters() if not name.startswith('codec.decoder.')):
            raise ValueError('Gradient entered a frozen component')
        optimizer.step()
        positions = [position + 1 for position in positions]
        row = dict(step=step + 1, source_loss=float(source_loss.detach()), prefix_loss=float(prefix_loss.detach()),
                   total_loss=float(loss.detach()), gradient_norm=float(norm), elapsed_seconds=time.perf_counter() - started)
        logs.append(row)
        if step == 7 or step + 1 == args.updates:
            if frozen_hashes(model) != frozen:
                raise ValueError('Frozen modules changed')
            write_json(args.output / 'preflight.json', dict(**parity, checked_updates=step + 1,
                frozen_hashes_match=True, gradients_decoder_only=True, finite=True))
        if (step + 1) % 10 == 0 or step == 0:
            write_json(args.output / 'progress.json', dict(**row, schedule_sha256=schedule.hexdigest(),
                actor_prefix_exposures=(step + 1) * 4, source_reconstruction_exposures=(step + 1) * 4,
                quality_approved=False))
            print(json.dumps(row), flush=True)
        if step == 7 or (step + 1) % 50 == 0 or step + 1 == args.updates:
            pending = args.output / 'weights.pending.pt'
            torch.save(dict(schema=SDF_VERSION, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
                step=step + 1, coefficients=saved['coefficients'],
                contract=dict(saved['contract'], factual_prefix_capacity=contract),
                positions=positions, indices=indices, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(), schedule_sha256=schedule.hexdigest()), pending)
            pending.replace(args.output / 'weights.pt')
    model.requires_grad_(False)
    write_json(args.output / 'training.json', dict(logs=logs, schedule_sha256=schedule.hexdigest(), contract=contract))
    write_json(state_path, dict(status='final_review', quality_approved=False))
    final = evaluate(model, bundle, cases, geometry, args.output / 'final')
    write_json(args.output / 'complete.json', dict(arm=args.arm, actual_updates=args.updates,
        actor_prefix_exposures=args.updates * 4, source_reconstruction_exposures=args.updates * 4,
        factual_history_bank_sha256=bank_hash, schedule_sha256=schedule.hexdigest(),
        baseline=baseline, final=final, quality_approved=False))
    write_json(state_path, dict(status='complete', quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('control', 'prefix'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'state.json', dict(status='failed', error=str(error), quality_approved=False))
        raise
