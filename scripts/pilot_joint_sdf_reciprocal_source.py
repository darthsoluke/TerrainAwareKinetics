"""Bounded matched source-history imitation of verified reciprocal stair motion."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.decoded_factual_history import decoded_factual_history_loss
from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
from motionbricks.data.joint_sdf_controller import SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import source_support_evidence
from motionbricks.data.source_sole_position_objective import source_sole_position_loss
from scripts.pilot_joint_sdf_contact_preserve import load_model, prepare_cases, control, exact_reviews
from scripts.pilot_joint_sdf_full_context import evaluate
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def configure_trainable(model, arm):
    if arm not in ('decoder', 'coordinated'):
        raise ValueError('Unknown matched training arm')
    model.requires_grad_(False)
    model.codec.decoder.requires_grad_(True)
    if arm == 'coordinated':
        model.root.requires_grad_(True)
        model.pose.requires_grad_(True)
    model.codec.encoder.requires_grad_(False)
    model.codec.quantizer.requires_grad_(False)
    model.eval()


def frozen_state(model, arm):
    names = ['codec.encoder', 'codec.quantizer', 'control', 'root_far', 'pose_far',
             'decoder_near', 'decoder_attention', 'decoder_far', 'sdf_encoder',
             'sdf_joint_embedding', 'sdf_root', 'sdf_pose', 'sdf_decoder']
    if arm == 'decoder':
        names += ['root', 'pose']
    return {name: state_digest(model.get_submodule(name)) for name in names}


def main(args):
    maximum = 800 if args.resume_from else 200
    if not 8 <= args.updates <= maximum:
        raise ValueError('Bounded source fit:200 fresh updates or at most800 continuation updates')
    args.output.mkdir(parents=True, exist_ok=False)
    state = args.output / 'state.json'
    write_json(state, dict(status='initializing', quality_approved=False))
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.resume_from or args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if sha256(args.checkpoint) != 'b4faa6a64defb4d7f76320f5a4003ade7a88038a2a527a61f9ed72d064df0f35':
        raise ValueError('Preserve the user-accepted gait initialization')
    if sha256(Path(saved['contract']['source'])) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle mismatch')
    inherited = 0
    if args.resume_from:
        prior = saved['contract'].get('reciprocal_source_pilot', {})
        allowed = {200: 800, 1000: 200}
        if args.arm != 'coordinated' or prior.get('arm') != args.arm or saved['step'] not in allowed or args.updates > allowed[saved['step']]:
            raise ValueError('Only bounded200->1000 or1000->1200 source probes may continue')
        inherited = saved['step']
    model, bundle = load_model(saved)
    configure_trainable(model, args.arm)
    rep = model.inference.global_motion_rep
    cases = prepare_cases(args.cases, Path(saved['contract']['dataset']))
    geometry = JointGeometryObjective(rep, 'cuda:0')
    prepared, fields, support_audit = [], [], []
    with torch.no_grad():
        for case in cases:
            joints, points = geometry.geometry(case['world'][None].cuda())
            prepared.append((joints[0], points[0]))
            fields.append(HeightfieldPrismSDF(case['scene']))
            support, _ = source_support_evidence(points[0, :, :geometry.sole_count], case['scene'])
            support_audit.append(dict(label=case['label'], source_frames=len(case['world']),
                source_support_vertex_frames=int(support.sum()), source_support_frames=int(support.any(-1).sum())))
    frozen = frozen_state(model, args.arm)
    contract = dict(arm=args.arm, actual_updates=args.updates, batch=4,
        source_checkpoint=str(args.checkpoint), source_sha256=sha256(args.checkpoint),
        inherited_updates=inherited, cumulative_updates=inherited+args.updates,
        resume_checkpoint=str(args.resume_from) if args.resume_from else None,
        resume_sha256=sha256(args.resume_from) if args.resume_from else None,
        optimizer_state_preserved=bool(args.resume_from),
        source_support_xyz=args.support_position,
        source_support_xyz_coefficient=float(saved['coefficients']['fk_m']) if args.support_position else 0.,
        parent_schedule_sha256=saved.get('schedule_sha256') if args.resume_from else None,
        history='factual source first4 frames for every real33-frame window; teacher-forced training only',
        inference_history='unmodified actual generated history, never reset in reviews',
        labels='real source Root/Pose tokens/FK/sole contact and measured source material velocity; no weak teacher phase mask',
        trainable='codec.decoder' if args.arm == 'decoder' else 'native Root, native Pose, codec.decoder',
        frozen_hashes=frozen, future_truth_in_actor=False, postprocessing=None,
        losses=dict(native_root_pose_motion=1., native_source_autoencoder=1., factual_prefix=1.,
                    factual_root=float(model.inference._root_model.args['local_root_loss_coeff']), **saved['coefficients']),
        optimizer='restored Adam5e-5, clip0.5' if args.resume_from else 'fresh Adam5e-5, clip0.5', source_support_audit=support_audit,
        caveat='Source-history fit is not generated-state recovery and does not certify closed-loop quality.',
        quality_approved=False)
    write_json(args.output / 'contract.json', contract)
    write_json(state, dict(status='baseline', quality_approved=False))
    baseline = exact_reviews(model, cases, geometry, args.output / 'baseline')
    with np.load(args.accepted_recording, allow_pickle=False) as expected, np.load(args.output / 'baseline/generated_8030ae7a2a7ad640.npz', allow_pickle=False) as actual:
        np.testing.assert_allclose(expected['world'], actual['world'], atol=3e-5, rtol=1e-5)
        recording_error = float(np.abs(expected['world']-actual['world']).max())
    torch.manual_seed(2333)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    positions, indices = [0]*4, [0, 1, 2, 3]
    exposure, schedule, logs = Counter(), hashlib.sha256(), []
    if args.resume_from:
        optimizer.load_state_dict(saved['optimizer'])
        positions, indices = list(saved['positions']), list(saved['indices'])
        exposure.update(saved['exposure'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        if any(group['lr'] != 5e-5 for group in optimizer.param_groups):
            raise ValueError('Resume changed the learning rate')
        if any(int(value['step']) != inherited for value in optimizer.state.values() if 'step' in value):
            raise ValueError('Adam counters do not match the inherited checkpoint')
        write_json(args.output/'resume_receipt.json', dict(model_sha256=sha256(args.resume_from),
            inherited_updates=inherited, adam_counters=inherited, positions=positions, indices=indices,
            retained_actor_examples=sum(exposure.values()), rng_restored=True,
            schedule_policy='new-stage digest plus immutable parent digest; not a recomputed concatenated hash'))
    start = time.perf_counter()
    write_json(state, dict(status='training', quality_approved=False))
    for step in range(args.updates):
        for row, index in enumerate(indices):
            if positions[row] == cases[index]['training_segments']:
                positions[row] = 0
                if row == 3:
                    indices[row] = 4 if index == 3 else 3
        selected = [cases[index] for index in indices]
        windows = torch.stack([case['world'][position*8:position*8+33] for case, position in zip(selected, positions)]).cuda()
        history = windows[:, :4]
        controls = [control(case, position) for case, position in zip(selected, positions)]
        intents = torch.cat([SoftCommandState().from_controls(history[row:row+1], value, 8)
                             for row, value in enumerate(controls)])
        scenes = [case['scene'] for case in selected]
        schedule.update(json.dumps([indices, positions, [value.cpu().tolist() for value in controls]]).encode())
        if step == 0:
            with torch.no_grad():
                plain = model(history, intents, scenes)['world'].clone()
                changed = windows.clone()
                changed[:, 4:, :3] += 3.
                altered = model(history, intents, scenes, source=changed)['world']
                torch.testing.assert_close(plain, altered, atol=0, rtol=0)
            with torch.inference_mode():
                runtime = model(history, intents, scenes)['world']
            torch.testing.assert_close(plain, runtime, atol=3e-5, rtol=1e-5)
            parity = dict(recording_error=recording_error, future_target_error=float((plain-altered).abs().max()),
                          runtime_error=float((plain-runtime).abs().max()))
        result = model(history, intents, scenes, source=windows, straight_through=True)
        if step == 0:
            torch.testing.assert_close(plain, result['world'], atol=3e-5, rtol=1e-5)
            parity['straight_through_forward_error'] = float((plain-result['world']).abs().max())
        exact = ExactPenetrationBatch(scenes, history.device, [fields[i] for i in indices])
        target = dict(joints=torch.stack([prepared[i][0][p*8:p*8+33] for i,p in zip(indices,positions)]),
                      points=torch.stack([prepared[i][1][p*8:p*8+33] for i,p in zip(indices,positions)]),
                      heights=exact.heights, volumes=exact)
        native = model.native_losses(result, windows)
        physical = geometry.prepared_losses(result['world'], history, target, offset=0, diagnostics=False)
        support_xyz = physical['fk_m'].new_zeros(())
        support_xyz_count = 0
        if args.support_position:
            _, predicted_points = geometry.geometry(result['world'])
            end = result['world'].shape[1]-4
            support_xyz, support_xyz_mask = source_sole_position_loss(
                predicted_points[:, 4:end, :geometry.sole_count],
                target['points'][:, 4:end, :geometry.sole_count], exact.heights)
            support_xyz_count = int(support_xyz_mask.sum())
            if support_xyz_count != physical['support_vertices']:
                raise ValueError('XYZ objective changed the verified source-contact mask')
        prefix = decoded_factual_history_loss(model.inference, result['local_prediction'], history,
            result['observation'], root_coefficient=contract['losses']['factual_root'])
        auxiliary = native_codec_reconstruction(model, result)['loss']
        loss = native['native_total'] + auxiliary + prefix + sum(saved['coefficients'][key]*physical[key] for key in saved['coefficients'])
        loss = loss + contract['source_support_xyz_coefficient']*support_xyz
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite supervised loss')
        loss.backward()
        gradient_groups = {name: sum(float(p.grad.detach().square().sum()) for p in model.get_submodule(name).parameters()
                            if p.grad is not None)**.5 for name in ('root', 'pose', 'codec.decoder')}
        if args.arm == 'coordinated' and min(gradient_groups.values()) <= 0:
            raise ValueError('Coordinated branch failed to train every declared predictor')
        if any(p.grad is not None for name in frozen for p in model.get_submodule(name).parameters()):
            raise ValueError('Gradient entered frozen vocabulary/conditioning')
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid gradient')
        optimizer.step()
        row = dict(step=inherited+step+1, stage_step=step+1, native_root=float(native['native_root'].detach()),
            native_pose=float(native['token_ce'].detach()), native_motion=float(native['native_motion'].detach()),
            source_aux=float(auxiliary.detach()), prefix=float(prefix.detach()),
            source_support_xyz_m=float(support_xyz.detach()), source_support_xyz_vertices=support_xyz_count,
            physical={key: float(physical[key].detach()) for key in saved['coefficients']},
            loss=float(loss.detach()), gradient_groups=gradient_groups, elapsed_seconds=time.perf_counter()-start)
        logs.append(row)
        for slot,(index,position) in enumerate(zip(indices,positions)):
            exposure[cases[index]['label']+':'+str(position)] += 1
            positions[slot] += 1
        if step == 7 or step+1 == args.updates:
            if frozen_state(model, args.arm) != frozen:
                raise ValueError('Frozen module hash mismatch')
            write_json(args.output / 'preflight.json', dict(parity, checked_updates=step+1,
                cumulative_updates=inherited+step+1,
                frozen_hashes_match=True, gradient_groups=gradient_groups, finite=True))
        if step == 0 or (step+1)%10 == 0:
            write_json(args.output / 'progress.json', dict(row, exposure=dict(exposure), schedule_sha256=schedule.hexdigest()))
            print(json.dumps(row), flush=True)
        if step == 7 or (step+1)%50 == 0 or step+1 == args.updates:
            pending = args.output / 'weights.pending.pt'
            torch.save(dict(schema=SDF_VERSION, model_state=model.state_dict(), optimizer=optimizer.state_dict(),
                step=inherited+step+1, contract=dict(saved['contract'], reciprocal_source_pilot=contract),
                coefficients=saved['coefficients'], positions=positions, indices=indices,
                exposure=dict(exposure), schedule_sha256=schedule.hexdigest(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), pending)
            pending.replace(args.output / 'weights.pt')
    model.requires_grad_(False)
    write_json(args.output / 'training.json', dict(logs=logs, contract=contract))
    write_json(state, dict(status='review', quality_approved=False))
    final = exact_reviews(model, cases, geometry, args.output / 'final')
    varied = evaluate(model, bundle, cases, geometry, args.output / 'varied')
    write_json(args.output / 'complete.json', dict(arm=args.arm, actual_updates=args.updates,
        inherited_updates=inherited, cumulative_updates=inherited+args.updates,
        cumulative_actor_examples=sum(exposure.values()),
        actor_examples=args.updates*4, source_autoencoder_examples=args.updates*4,
        exposure=dict(exposure), schedule_sha256=schedule.hexdigest(), baseline=baseline, final=final,
        varied=varied, quality_approved=False))
    write_json(state, dict(status='complete', quality_approved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('decoder', 'coordinated'), required=True)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--support-position', action='store_true')
    for name in ('checkpoint', 'cases', 'accepted-recording', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/'state.json', dict(status='failed', error=str(error), quality_approved=False))
        raise
