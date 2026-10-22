"""Matched bounded Pose/decoder stop-intent experiments with all prior weights frozen."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.stop_command_adapter import StopCommandAdapter
from motionbricks.data.stop_stage_adapter import install_stop_stage, STOP_STAGE_VERSION
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_stair_stop_command import prepare, review, write
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def main(args):
    if not 8 <= args.updates <= 50:
        raise ValueError('Hard50-update limit for this stage-specific probe')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.parent) != 'ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf':
        raise ValueError('Approved parent changed')
    if sha256(args.checkpoint) != '8bd03d3d069c3b160f22f652fc7dc51a2bda1a73e86c8a10f692cc346fcb9571':
        raise ValueError('Expected matched Root-only stop initialization')
    parent = torch.load(args.parent, map_location='cpu', mmap=True, weights_only=False)
    model, _ = load_model(parent)
    model.control = StopCommandAdapter(model.control)
    initial = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    model.load_state_dict(initial['model_state'], strict=True)
    model.requires_grad_(False)
    module, handles = install_stop_stage(model, args.stage)
    model.eval()
    fixed = [k for k in model.state_dict() if not k.startswith('stop_'+args.stage+'_condition.')]
    frozen = state_digest(model, fixed)
    cases = prepare(model, args)
    baseline = review(model, args, args.output/'baseline')
    for filename in ('up-cruise.npz','up-release_resume.npz','down-cruise.npz','down-release_resume.npz'):
        with (np.load(args.checkpoint.parent/'final'/filename, allow_pickle=False) as prior,
              np.load(args.output/'baseline'/filename, allow_pickle=False) as current):
            np.testing.assert_array_equal(prior['world'], current['world'])
    torch.manual_seed(2333)
    contract = dict(stage=args.stage, version=STOP_STAGE_VERSION, initial_sha256=sha256(args.checkpoint),
        approved_parent_sha256=sha256(args.parent), actual_updates=args.updates, batch=4,
        actor_examples=args.updates*4, losses='Existing native objectives plus existing factual-history objective at original Root coefficient',
        stop_feature_fields='desired VX/VZ, desired heading and horizon; not stale release-position delta',
        frozen='All approved and Root-only stop parameters, vocabulary and existing SDF modules',
        teacher_history=True, generated_history_in_review=True,
        source_cases=[c['metadata'] for c in cases], future_truth_in_actor=False,
        postprocessing=None, quality_approved=False)
    write(args.output/'contract.json', contract)
    optimizer = torch.optim.Adam(module.parameters(), lr=5e-5, weight_decay=0.)
    logs = []
    for step in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        details = []
        for offset in range(4):
            case = cases[(step*4+offset) % len(cases)]
            source, history = case['world'], case['world'][:, :4]
            if step == 0 and offset == 0:
                with torch.no_grad():
                    plain = model(history, case['intent'], [case['scene']])['world'].clone()
                    altered = source.clone()
                    altered[:,4:] += 3.
                    changed = model(history, case['intent'], [case['scene']], source=altered)['world']
                torch.testing.assert_close(plain, changed, atol=0, rtol=0)
            result = model(history, case['intent'], [case['scene']], source=source, straight_through=True)
            local = model.last_local[:, :32]
            if step == 0 and offset == 0:
                torch.testing.assert_close(result['world'], plain, atol=3e-5, rtol=1e-5)
                write(args.output/'preflight.json', dict(future_target_actor_error=float((plain-changed).abs().max()),
                    training_forward_error=float((plain-result['world']).abs().max())))
            prefix = decoded_factual_history_loss(model.inference, local, history, result['observation'],
                root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff']))
            native = model.native_losses(result, source)['native_total']
            loss = native+prefix
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            (loss/4).backward()
            details.append(dict(**case['metadata'], native=float(native.detach()), prefix=float(prefix.detach())))
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('New conditioning stage has no finite learning gradient')
        optimizer.step()
        if state_digest(model, fixed) != frozen:
            raise ValueError('Prior weights changed')
        row = dict(step=step+1, gradient=float(norm), samples=details)
        logs.append(row)
        write(args.output/'progress.json', row)
        if step % 10 == 0:
            print(json.dumps(dict(step=step+1, stage=args.stage, native=sum(r['native'] for r in details)/4,
                prefix=sum(r['prefix'] for r in details)/4)), flush=True)
    torch.save(dict(schema='motionbricks.stop_stage_pilot.v1', version=STOP_STAGE_VERSION,
        stage=args.stage, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
        contract=parent['contract'], actual_updates=args.updates, actor_examples=args.updates*4,
        initial_sha256=sha256(args.checkpoint), quality_approved=False), args.output/'weights.pt')
    final = review(model, args, args.output/'final', reference=args.output/'baseline')
    write(args.output/'complete.json', dict(baseline=baseline, final=final, logs=logs,
        actual_updates=args.updates, actor_examples=args.updates*4, prior_parameters_unchanged=True,
        candidate_sha256=sha256(args.output/'weights.pt'), quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','checkpoint','dataset','audit','artifact','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--stage', choices=('pose','decoder'), required=True)
    parser.add_argument('--updates', type=int, default=50)
    main(parser.parse_args())
