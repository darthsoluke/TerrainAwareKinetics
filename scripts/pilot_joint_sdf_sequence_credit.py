"""Matched whole-passage imitation with/without gradients through played history."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from motionbricks.data.joint_sdf_controller import SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_contact_preserve import load_model, prepare_cases, control, exact_reviews
from scripts.pilot_joint_sdf_full_context import evaluate
from scripts.pilot_joint_sdf_reciprocal_source import configure_trainable, frozen_state
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def main(args):
    if not 1 <= args.updates <= 20:
        raise ValueError('This sequence-credit probe is capped at20 episode updates')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    parent = saved['contract'].get('reciprocal_source_pilot')
    if not parent or parent['arm'] != 'coordinated' or saved['step'] != 200:
        raise ValueError('Expected the finished200-step coordinated source checkpoint')
    model, bundle = load_model(saved)
    configure_trainable(model, 'coordinated')
    rep = model.inference.global_motion_rep
    cases = prepare_cases(args.cases, Path(saved['contract']['dataset']))
    geometry = JointGeometryObjective(rep, 'cuda:0')
    fields, prepared = [], []
    with torch.no_grad():
        for case in cases:
            prepared.append(geometry.geometry(case['world'][None].cuda()))
            fields.append(HeightfieldPrismSDF(case['scene']))
    frozen = frozen_state(model, 'coordinated')
    attached = args.arm == 'through_history'
    contract = dict(arm=args.arm, episode_updates=args.updates, source_sha256=sha256(args.checkpoint),
        inherited_source_updates=200, fresh_optimizer=True, gradient_accumulation='four complete cases per update',
        trainable='Root/Pose/decoder; vocabulary and terrain adapters frozen', frozen_hashes=frozen,
        histories='same true initial4 frames, then own published skeletal history without resets',
        objective='same-start complete real trajectory FK/contact/material velocity; native first-window loss; one real source AE per case; factual prefix',
        history_gradients=attached, inference='unchanged defaults; gradient flag only alters training graph',
        objective_coefficients=saved['coefficients'], future_truth_in_actor=False, postprocessing=None,
        expected_actor_primitives=args.updates*43, expected_source_ae_examples=args.updates*4,
        optimizer='fresh Adam5e-5, clip0.5, one update after four case gradients', quality_approved=False)
    write_json(args.output/'contract.json', contract)
    write_json(args.output/'state.json', dict(status='baseline', quality_approved=False))
    baseline = exact_reviews(model, cases, geometry, args.output/'baseline')
    torch.manual_seed(2333)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    logs, exposure = [], Counter()
    began = time.perf_counter()
    write_json(args.output/'state.json', dict(status='training', quality_approved=False))
    for update in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        case_rows = []
        for index in (0, 1, 2, 3 if update % 2 == 0 else 4):
            case = cases[index]
            source = case['world'][None].cuda()
            initial = source[:, :4]
            history, parts, prefixes = initial, [], []
            driver = SoftCommandState()
            first_prediction = native_first = None
            for block in range(case['training_segments']):
                command = control(case, block)
                intent = driver.from_controls(history.detach(), command, 8)
                if attached:
                    # A cruise goal translates with the current body, rather
                    # than becoming an accidental fixed target in the gradient.
                    shift = history[:, 3, [0, 2]]-history[:, 3, [0, 2]].detach()
                    intent = torch.cat((intent[:, :2]+shift, intent[:, 2:]), -1)
                if update == 0 and index == 0 and block == 0:
                    with torch.no_grad():
                        plain = model(history, intent, [case['scene']])['world'].clone()
                        changed = source[:, :33].clone()
                        changed[:, 4:, :3] += 3.
                        altered = model(history, intent, [case['scene']], source=changed)['world']
                    torch.testing.assert_close(plain, altered, atol=0, rtol=0)
                    parity = dict(future_target_error=float((plain-altered).abs().max()))
                result = model(history, intent, [case['scene']], straight_through=True,
                    history_gradients=attached, source=source[:, :33] if block == 0 else None)
                world = result['world']
                local = model.last_local[:, :32]
                if update == 0 and index == 0 and block == 0:
                    torch.testing.assert_close(plain, world, atol=3e-5, rtol=1e-5)
                    parity['training_forward_error'] = float((plain-world).abs().max())
                prefixes.append(decoded_factual_history_loss(model.inference, local, history,
                    result['observation'], root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff'])))
                if block == 0:
                    first_prediction = world
                    native_first = model.native_losses(result, source[:, :33])['native_total']
                parts.append(world[:, 4:12])
                history = reencode_played_history(world[:, 8:12], rep, differentiable=attached)
                exposure[case['label']] += 1
            if update == 0 and index == 0:
                gradient = torch.autograd.grad(world[:, 4:12].square().mean(), first_prediction,
                    allow_unused=True, retain_graph=True)[0]
                temporal = 0. if gradient is None else float(gradient.norm())
                if (attached and (gradient is None or not torch.isfinite(gradient).all() or temporal == 0)) or (not attached and temporal != 0):
                    raise ValueError('History-credit experiment does not have its declared gradient path')
                parity['late_to_first_prediction_gradient_norm'] = temporal
            # The final4 forecast frames are genuine model output excluded by
            # the existing physical objective, not appended playback frames.
            episode = torch.cat((initial, *parts, world[:, 12:16]), 1)
            exact = ExactPenetrationBatch([case['scene']], source.device, [fields[index]])
            target = dict(joints=prepared[index][0], points=prepared[index][1], heights=exact.heights, volumes=exact)
            physical = geometry.prepared_losses(episode, initial, target, offset=0, diagnostics=False)
            auxiliary_position = update % case['training_segments']
            source_window = source[:, auxiliary_position*8:auxiliary_position*8+33]
            with torch.no_grad():
                aux_intent = SoftCommandState().from_controls(source_window[:, :4], control(case, auxiliary_position), 8)
                aux_result = model(source_window[:, :4], aux_intent, [case['scene']], source=source_window)
            auxiliary = native_codec_reconstruction(model, aux_result)['loss']
            loss = native_first + auxiliary + torch.stack(prefixes).mean() + sum(saved['coefficients'][key]*physical[key] for key in saved['coefficients'])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite whole-sequence loss')
            (loss/4).backward()
            case_rows.append(dict(label=case['label'], frames=len(parts)*8, loss=float(loss.detach()),
                physical={k:float(physical[k].detach()) for k in saved['coefficients']}))
            del loss, episode, prefixes, parts, result, world, local, history, first_prediction, native_first, auxiliary, aux_result
        if any(p.grad is not None for name in frozen for p in model.get_submodule(name).parameters()):
            raise ValueError('Frozen module received a gradient')
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid accumulated gradient')
        optimizer.step()
        row = dict(update=update+1, cases=case_rows, gradient_norm=float(norm), actor_primitives=sum(exposure.values()),
            source_ae_examples=(update+1)*4, elapsed_seconds=time.perf_counter()-began,
            peak_memory_bytes=torch.cuda.max_memory_allocated())
        logs.append(row)
        write_json(args.output/'progress.json', dict(row, exposure=dict(exposure)))
        print(json.dumps(row), flush=True)
        if update == 0 or (update+1)%5 == 0 or update+1 == args.updates:
            if frozen_state(model, 'coordinated') != frozen:
                raise ValueError('Frozen state changed')
            write_json(args.output/'preflight.json', dict(parity, checked_updates=update+1, frozen_hashes_match=True, finite=True))
            pending = args.output/'weights.pending.pt'
            torch.save(dict(schema=SDF_VERSION, model_state=model.state_dict(), optimizer=optimizer.state_dict(), step=update+1,
                contract=dict(saved['contract'], sequence_credit_pilot=contract), coefficients=saved['coefficients'],
                exposure=dict(exposure), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), pending)
            pending.replace(args.output/'weights.pt')
    model.requires_grad_(False)
    write_json(args.output/'training.json', dict(logs=logs, contract=contract))
    write_json(args.output/'state.json', dict(status='review', quality_approved=False))
    final = exact_reviews(model, cases, geometry, args.output/'final')
    varied = evaluate(model, bundle, cases, geometry, args.output/'varied')
    write_json(args.output/'complete.json', dict(arm=args.arm, actual_episode_updates=args.updates,
        actor_primitives=sum(exposure.values()), source_ae_examples=args.updates*4, exposure=dict(exposure),
        baseline=baseline, final=final, varied=varied, quality_approved=False))
    write_json(args.output/'state.json', dict(status='complete', quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('detached', 'through_history'), required=True)
    parser.add_argument('--updates', type=int, default=20)
    for name in ('checkpoint', 'cases', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/'state.json', dict(status='failed', error=str(error), quality_approved=False))
        raise
