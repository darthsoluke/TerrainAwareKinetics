"""Bounded leg-output fit with the learned Root head and all other weights frozen."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from motionbricks.data.joint_sdf_controller import SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.source_sole_position_objective import source_sole_position_loss
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_contact_preserve import load_model, prepare_cases, control, exact_reviews
from scripts.pilot_joint_sdf_full_context import evaluate
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def leg_rotation_rows(rep):
    names = [side+part for side in ('Left', 'Right') for part in ('Leg', 'Shin', 'Foot', 'ToeBase')]
    indices = torch.as_tensor(rep.indices['global_rot_data']).reshape(30, 6)
    return indices[[rep.skeleton.bone_order_names.index(name) for name in names]].flatten().tolist(), names


def restrict_head_gradients(head, rows):
    mask = head.weight.new_zeros(head.out_channels)
    mask[rows] = 1
    return [head.weight.register_hook(lambda grad: grad*mask[:, None, None]),
            head.bias.register_hook(lambda grad: grad*mask)]


def whole_passage_loss(model, case, index, update, geometry, prepared, field, coefficients):
    source = case['world'][None].cuda()
    initial = history = source[:, :4]
    parts, prefixes = [], []
    driver = SoftCommandState()
    rep = model.inference.global_motion_rep
    parity = None
    for block in range(case['training_segments']):
        intent = driver.from_controls(history.detach(), control(case, block), 8)
        shift = history[:,3,[0,2]]-history[:,3,[0,2]].detach()
        intent = torch.cat((intent[:,:2]+shift, intent[:,2:]), -1)
        if update == 0 and index == 0 and block == 0:
            with torch.no_grad():
                plain = model(history, intent, [case['scene']])['world'].clone()
                changed = source[:,:33].clone()
                changed[:,4:] += 3.
                altered = model(history, intent, [case['scene']], source=changed)['world']
            torch.testing.assert_close(plain, altered, atol=0, rtol=0)
            with torch.inference_mode():
                runtime = model(history, intent, [case['scene']])['world']
            torch.testing.assert_close(plain, runtime, atol=3e-5, rtol=1e-5)
            parity = dict(future_target_error=float((plain-altered).abs().max()), runtime_error=float((plain-runtime).abs().max()))
        result = model(history, intent, [case['scene']], straight_through=True, history_gradients=True,
                       source=source[:,:33] if block == 0 else None)
        world = result['world']
        if update == 0 and index == 0 and block == 0:
            torch.testing.assert_close(plain, world, atol=3e-5, rtol=1e-5)
            parity['training_forward_error'] = float((plain-world).abs().max())
        prefixes.append(decoded_factual_history_loss(model.inference, model.last_local[:,:32], history,
            result['observation'], root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff'])))
        if block == 0:
            first = world
            native = model.native_losses(result, source[:,:33])['native_total']
        parts.append(world[:,4:12])
        history = reencode_played_history(world[:,8:12], rep, differentiable=True)
    temporal = None
    if update == 0 and index == 0:
        gradient = torch.autograd.grad(world[:,4:12].square().mean(), first, retain_graph=True)[0]
        temporal = float(gradient.norm())
        if not torch.isfinite(gradient).all() or temporal == 0:
            raise ValueError('Missing temporal gradient')
    episode = torch.cat((initial, *parts, world[:,12:16]), 1)
    exact = ExactPenetrationBatch([case['scene']], source.device, [field])
    target = dict(joints=prepared[0][None], points=prepared[1][None], heights=exact.heights, volumes=exact)
    physical = geometry.prepared_losses(episode, initial, target, offset=0, diagnostics=False)
    _, points = geometry.geometry(episode)
    end = episode.shape[1]-4
    xyz, active = source_sole_position_loss(points[:,4:end,:geometry.sole_count], target['points'][:,4:end,:geometry.sole_count], exact.heights)
    if int(active.sum()) != physical['support_vertices']:
        raise ValueError('Whole-passage source-mask mismatch')
    position = update % case['training_segments']
    window = source[:,position*8:position*8+33]
    with torch.no_grad():
        intent = SoftCommandState().from_controls(window[:,:4], control(case, position), 8)
        aux_result = model(window[:,:4], intent, [case['scene']], source=window)
    auxiliary = native_codec_reconstruction(model, aux_result)['loss']
    loss = native+auxiliary+torch.stack(prefixes).mean()+coefficients['fk_m']*xyz+sum(coefficients[k]*physical[k] for k in coefficients)
    details = dict(label=case['label'], primitives=len(parts), loss=float(loss.detach()),
        sole_xyz_m=float(xyz.detach()), physical={k:float(physical[k].detach()) for k in coefficients},
        temporal_gradient=temporal, parity=parity)
    return loss, details


def main(args):
    if not 8 <= args.updates <= (20 if args.whole_passage else 100):
        raise ValueError('This probe has a hard100-update limit')
    if sha256(args.checkpoint) != 'a2b168d5d7428c25ae836ebb1bb7d04d9ff42182ee9e270801581bb15fdf68d8':
        raise ValueError('Expected the verified Root-only head candidate')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if sha256(Path(saved['contract']['source'])) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle mismatch')
    model, bundle = load_model(saved)
    model.requires_grad_(False)
    model.eval()
    head = model.codec.decoder.model[-1]
    head.requires_grad_(True)
    rows, names = leg_rotation_rows(model.inference.local_motion_rep)
    handles = restrict_head_gradients(head, rows)
    frozen_rows = sorted(set(range(head.out_channels))-set(rows))
    original_weight, original_bias = head.weight.detach().clone(), head.bias.detach().clone()
    head_addresses = {head.weight.data_ptr(), head.bias.data_ptr()}
    immutable_keys = [key for key, value in model.state_dict().items() if value.data_ptr() not in head_addresses]
    immutable = state_digest(model, immutable_keys)
    cases = prepare_cases(args.cases, Path(saved['contract']['dataset']))
    geometry = JointGeometryObjective(model.inference.global_motion_rep, 'cuda:0')
    prepared, fields = [], []
    with torch.no_grad():
        for case in cases:
            joints, points = geometry.geometry(case['world'][None].cuda())
            prepared.append((joints[0], points[0]))
            fields.append(HeightfieldPrismSDF(case['scene']))
    coefficients = saved['coefficients']
    contract = dict(parent_sha256=sha256(args.checkpoint), actual_updates=args.updates, batch=4,
        actor_examples=4*args.updates, source_autoencoder_examples=4*args.updates,
        inherited_optimizer_updates=saved['inherited_optimizer_updates'], inherited_fit_operations=saved['fit_operations'],
        selected_joints=names, trainable_output_rows=rows, frozen_output_rows=frozen_rows,
        non_head_sha256=immutable, optimizer='fresh Adam5e-5, no weight decay, clip0.5; exact row gradient mask',
        losses=dict(native=1., source_autoencoder=1., factual_prefix=1., source_sole_xyz=coefficients['fk_m'], **coefficients),
        history='real source first4 frames in training; continuous actual generated history in review',
        targets='unmodified source motion and measured sole supports/velocities; no weak phase labels',
        root_policy='Root head WEIGHTS immutable; no trajectory clamp. Changed rotations can change feedback/queries and therefore later Root outputs.',
        future_truth_in_actor=False, postprocessing=None, quality_approved=False)
    if args.whole_passage:
        contract.update(actor_examples=args.updates*43, source_autoencoder_examples=args.updates*4,
            batch='four complete cases accumulated per optimizer update',
            history='true initial4 frames, then continuous differentiable generated history; no resets',
            objective_scope='physical objectives on complete published trajectories; native first window; one rotating sourceAE per case',
            compute_comparison='20 episode updates/860 primitives, not compute-matched to100 source-window updates/400 primitives')
    write_json(args.output/'contract.json', contract)
    write_json(args.output/'state.json', dict(status='baseline', quality_approved=False))
    baseline = exact_reviews(model, cases, geometry, args.output/'baseline')
    baseline_parity = {}
    for case in cases:
        name = case['label']+'.npz'
        with np.load(args.checkpoint.parent/'final'/name, allow_pickle=False) as old, np.load(args.output/'baseline'/name, allow_pickle=False) as new:
            np.testing.assert_allclose(new['world'], old['world'], atol=3e-5, rtol=1e-5)
            baseline_parity[case['label']] = float(np.abs(new['world']-old['world']).max())
    write_json(args.output/'baseline_parity.json', baseline_parity)
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-5, weight_decay=0.)
    positions, indices = [0]*4, [0, 1, 2, 3]
    logs, schedule = [], []
    start = time.perf_counter()
    for step in range(args.updates):
        if args.whole_passage:
            optimizer.zero_grad(set_to_none=True)
            details = []
            for index in (0,1,2,3 if step%2 == 0 else 4):
                loss, detail = whole_passage_loss(model, cases[index], index, step, geometry, prepared[index], fields[index], coefficients)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite whole-passage loss')
                (loss/4).backward()
                details.append(detail)
                del loss
            if torch.count_nonzero(head.weight.grad[frozen_rows]) or torch.count_nonzero(head.bias.grad[frozen_rows]):
                raise ValueError('Frozen rows received temporal gradients')
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), .5)
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError('Invalid whole-passage gradient')
            optimizer.step()
            if not torch.equal(head.weight[frozen_rows], original_weight[frozen_rows]) or not torch.equal(head.bias[frozen_rows], original_bias[frozen_rows]):
                raise ValueError('Frozen head rows changed')
            if state_digest(model, immutable_keys) != immutable:
                raise ValueError('Non-head state changed')
            row = dict(step=step+1, cases=details, actor_primitives=(step+1)*43,
                gradient_norm=float(norm), elapsed=time.perf_counter()-start)
            logs.append(row)
            schedule.append(dict(case_indices=[0,1,2,3 if step%2==0 else 4]))
            write_json(args.output/'progress.json', row)
            write_json(args.output/'preflight.json', dict(checked_updates=step+1, frozen_rows_exact=True, non_head_hash_match=True,
                temporal_gradient=logs[0]['cases'][0]['temporal_gradient'],
                **logs[0]['cases'][0]['parity']))
            print(json.dumps(row), flush=True)
            continue
        for row, index in enumerate(indices):
            if positions[row] == cases[index]['training_segments']:
                positions[row] = 0
                if row == 3:
                    indices[row] = 4 if index == 3 else 3
        selected = [cases[i] for i in indices]
        windows = torch.stack([case['world'][p*8:p*8+33] for case, p in zip(selected, positions)]).cuda()
        history = windows[:, :4]
        intent = torch.cat([SoftCommandState().from_controls(history[r:r+1], control(case, p), 8)
                            for r, (case, p) in enumerate(zip(selected, positions))])
        scenes = [case['scene'] for case in selected]
        if step == 0:
            with torch.no_grad():
                plain = model(history, intent, scenes)['world']
                changed = windows.clone()
                changed[:, 4:] += 3.
                altered = model(history, intent, scenes, source=changed)['world']
                torch.testing.assert_close(plain, altered, atol=0, rtol=0)
            with torch.inference_mode():
                runtime = model(history, intent, scenes)['world']
            torch.testing.assert_close(plain, runtime, atol=3e-5, rtol=1e-5)
            parity = dict(future_target_error=float((plain-altered).abs().max()), runtime_error=float((plain-runtime).abs().max()))
        result = model(history, intent, scenes, source=windows, straight_through=True)
        if step == 0:
            torch.testing.assert_close(plain, result['world'], atol=3e-5, rtol=1e-5)
            parity['straight_through_error'] = float((plain-result['world']).abs().max())
        exact = ExactPenetrationBatch(scenes, history.device, [fields[i] for i in indices])
        target = dict(joints=torch.stack([prepared[i][0][p*8:p*8+33] for i,p in zip(indices,positions)]),
            points=torch.stack([prepared[i][1][p*8:p*8+33] for i,p in zip(indices,positions)]), heights=exact.heights, volumes=exact)
        native = model.native_losses(result, windows)
        physical = geometry.prepared_losses(result['world'], history, target, offset=0, diagnostics=False)
        _, points = geometry.geometry(result['world'])
        end = result['world'].shape[1]-4
        xyz, active = source_sole_position_loss(points[:,4:end,:geometry.sole_count], target['points'][:,4:end,:geometry.sole_count], exact.heights)
        if int(active.sum()) != physical['support_vertices']:
            raise ValueError('Source contact-mask mismatch')
        prefix = decoded_factual_history_loss(model.inference, result['local_prediction'], history, result['observation'],
            root_coefficient=float(model.inference._root_model.args['local_root_loss_coeff']))
        auxiliary = native_codec_reconstruction(model, result)['loss']
        loss = native['native_total']+auxiliary+prefix+coefficients['fk_m']*xyz+sum(coefficients[k]*physical[k] for k in coefficients)
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        if torch.count_nonzero(head.weight.grad[frozen_rows]) or torch.count_nonzero(head.bias.grad[frozen_rows]):
            raise ValueError('Frozen output row received gradient')
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid leg-head gradient')
        optimizer.step()
        if not torch.equal(head.weight[frozen_rows], original_weight[frozen_rows]) or not torch.equal(head.bias[frozen_rows], original_bias[frozen_rows]):
            raise ValueError('Root or other frozen output rows changed')
        schedule.append(dict(indices=list(indices), positions=list(positions)))
        positions = [p+1 for p in positions]
        row = dict(step=step+1, loss=float(loss.detach()), sole_xyz_m=float(xyz.detach()),
            physical={k:float(physical[k].detach()) for k in coefficients}, gradient_norm=float(norm), elapsed=time.perf_counter()-start)
        logs.append(row)
        if step == 7 or step+1 == args.updates:
            if state_digest(model, immutable_keys) != immutable:
                raise ValueError('Frozen non-head tensors changed')
            write_json(args.output/'preflight.json', dict(parity, checked_updates=step+1, frozen_rows_exact=True, non_head_hash_match=True))
        if step == 0 or (step+1)%10 == 0:
            write_json(args.output/'progress.json', row)
            print(json.dumps(row), flush=True)
    for handle in handles:
        handle.remove()
    model.requires_grad_(False)
    torch.save(dict(schema=SDF_VERSION, model_state=model.state_dict(), optimizer=optimizer.state_dict(), step=args.updates,
        contract=dict(saved['contract'], leg_head_fit=contract), coefficients=coefficients,
        inherited_optimizer_updates=saved['inherited_optimizer_updates'], fit_operations=saved['fit_operations']), args.output/'weights.pt')
    write_json(args.output/'training.json', dict(logs=logs, schedule=schedule))
    write_json(args.output/'state.json', dict(status='review', quality_approved=False))
    final = exact_reviews(model, cases, geometry, args.output/'final')
    varied = evaluate(model, bundle, cases, geometry, args.output/'varied')
    write_json(args.output/'complete.json', dict(baseline=baseline, final=final, varied=varied, actual_updates=args.updates, quality_approved=False))
    subprocess.run([sys.executable, '-m', 'scripts.review_joint_sdf_natural_stair_time', '--runs', 'legs='+str(args.output/'weights.pt'),
        '--cases', str(args.cases), '--output', str(args.output/'long_review')], check=True)
    write_json(args.output/'state.json', dict(status='complete', quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'cases', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--whole-passage', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/'state.json', dict(status='failed', error=str(error), quality_approved=False))
        raise
