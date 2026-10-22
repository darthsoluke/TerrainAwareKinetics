"""Measure native-loss output gradients without changing weights or motion."""

import argparse
import os
from pathlib import Path
import subprocess

from scripts.run_pfnn_bias_workflow import digest, read, save
from scripts.run_pfnn_parallel_workflow import gpu_snapshot


def main(args):
    snapshot, available = gpu_snapshot()
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv'], text=True)
    if args.gpu not in available:
        raise ValueError('Selected GPU is no longer idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from motionbricks.data.full_corpus_inference import inference_receipt, load_candidate, tensor_digest
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, collate_primitives
    from motionbricks.data.unified_full_training import NativeTrainingStage
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from scripts.portable_parc_online_soma import load_terrain_bundle

    torch.set_num_threads(4)
    torch.manual_seed(281016)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.9)
    state = dict(pid=os.getpid(), gpu=args.gpu, domain=args.domain,
        gpu_snapshot_before=snapshot, compute_processes_before=processes,
        actual_optimizer_updates=0, quality_approved=False, delivery_approved=False)
    save(args.output/'state.json', dict(state, status='loading_frozen_final_model'))
    previous = read(args.previous/args.domain/'complete.json')
    checkpoint_hash, bundle_hash = digest(args.checkpoint), digest(args.bundle)
    if previous['checkpoint_sha256'] != checkpoint_hash or not previous['weights_unchanged']:
        raise ValueError('Prior diagnostic does not bind this frozen candidate')
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    inference, bundle = load_terrain_bundle(args.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    load_candidate(controller, saved,
        inference_receipt(saved, checkpoint_hash, checkpoint_hash, bundle_hash), bundle_hash)
    stage = NativeTrainingStage(controller, 'codec').eval()
    controller.requires_grad_(False)
    before = tensor_digest(controller.state_dict())
    coefficients = dict(reconstruction=1., commitment=controller.loss_model.args['commit_loss_coeff'],
        skate=controller.loss_model.args['skate_contact_loss_coeff'],
        joint_velocity=controller.loss_model.args['joint_vel_loss_coeff'])
    rep = inference.local_motion_rep
    names = rep.skeleton.bone_order_names
    leg_ids = [names.index(side+joint) for side in ('Left', 'Right') for joint in ('Leg', 'Shin', 'Foot')]
    foot_ids = [names.index(side+'Foot') for side in ('Left', 'Right')]

    def move(value):
        if torch.is_tensor(value):
            return value.to('cuda:0')
        return {key: move(item) for key, item in value.items()} if isinstance(value, dict) else value

    def cosine(left, right):
        norm = left.double().norm()*right.double().norm()
        return float((left.double()*right.double()).sum()/norm) if norm > 0 else None

    rows = []
    try:
        for prior in previous['rows']:
            dataset = PrimitiveDataset([path/'data' for path in args.cache], prior['split'])
            lookup = {actual: i for i, actual in enumerate(dataset.ids)}
            ids = [window['sample_id'] for window in prior['windows']]
            for window in prior['windows']:
                if dataset.records[window['sample_id']] != window['record']:
                    raise ValueError('Previously selected window identity changed')
            batch = move(collate_primitives([dataset[lookup[index]] for index in ids]))
            captured = {}
            original_loss = controller.loss_model.loss

            def capture(target, output, root):
                captured.update(target=target, output=output)
                return original_loss(target, output, root)

            controller.loss_model.loss = capture
            try:
                with torch.no_grad():
                    original_value = stage(batch)
            finally:
                controller.loss_model.loss = original_loss
            if abs(float(original_value)-prior['native_losses']['codec']) > 5e-6:
                raise ValueError('Replay native loss differs from prior frozen diagnostic')
            target, output = captured['target'], captured['output']
            reconstruction = output['recon_state'].detach().clone().requires_grad_(True)
            losses = original_loss(target, dict(output, recon_state=reconstruction), None)
            pred = rep.inverse(reconstruction, True, return_all=False,
                joint_positions_from='global_rot_data')['posed_joints']
            with torch.no_grad():
                truth = rep.inverse(target['local_motions'], True, return_all=False,
                    joint_positions_from='global_rot_data')['posed_joints']
            pred_relative, truth_relative = pred-pred[:, :, :1], truth-truth[:, :, :1]
            delta = pred_relative[:, 4:28]-truth_relative[:, 4:28]
            leg_error = delta[:, :, leg_ids].norm(dim=-1).mean()
            foot_y_error = delta[:, :, foot_ids, 1].abs().mean()
            expected_leg = sum(w['source_codec_errors']['leg_relative_mpjpe_m']
                               for w in prior['windows'])/len(ids)
            if abs(float(leg_error.detach())-expected_leg) > 1e-5:
                raise ValueError('Physical reconstruction metric differs from prior diagnostic')
            values = dict(native=losses['loss'], reconstruction=losses['l_recons_pose'],
                skate=coefficients['skate']*losses['l_skate_contact'],
                joint_velocity=coefficients['joint_velocity']*losses['l_joint_vel'],
                leg_mpjpe=leg_error, foot_y_mae=foot_y_error)
            gradients = {name: torch.autograd.grad(value, reconstruction, retain_graph=True)[0]
                         for name, value in values.items()}
            if any(not torch.isfinite(value).all() for value in gradients.values()):
                raise ValueError('Nonfinite output gradient')
            component_sum = gradients['reconstruction']+gradients['skate']+gradients['joint_velocity']
            torch.testing.assert_close(component_sum, gradients['native'], atol=1e-7, rtol=1e-5)
            terms = {}
            for name in ('native', 'reconstruction', 'skate', 'joint_velocity'):
                grad = gradients[name]
                terms[name] = dict(value=float(values[name].detach()), gradient_norm=float(grad.double().norm()),
                    cosine_to_leg_mpjpe=cosine(grad, gradients['leg_mpjpe']),
                    cosine_to_foot_y_mae=cosine(grad, gradients['foot_y_mae']),
                    squared_gradient_by_feature={key: float(grad[..., rep.indices[key]].double().square().sum())
                        for key in ('root', 'ric_data', 'global_rot_data', 'local_vel', 'foot_contacts')})
            rows.append(dict(split=prior['split'], sample_ids=ids, examples=len(ids),
                native_loss_replay_error=abs(float(original_value)-prior['native_losses']['codec']),
                physical_metric_replay_error=abs(float(leg_error.detach())-expected_leg),
                leg_mpjpe_m=float(leg_error.detach()), foot_y_mae_m=float(foot_y_error.detach()),
                commitment_value=float(losses['l_commit_pose']), terms=terms))
            save(args.output/'progress.json', dict(state, rows=rows))
        if any(p.grad is not None for p in controller.parameters()):
            raise ValueError('Frozen model unexpectedly acquired a parameter gradient')
        if tensor_digest(controller.state_dict()) != before:
            raise ValueError('Frozen model tensors changed')
        save(args.output/'complete.json', dict(state, completed=True, rows=rows,
            coefficients=coefficients, checkpoint_sha256=checkpoint_hash,
            previous_report_sha256=digest(args.previous/args.domain/'complete.json'),
            script_sha256=digest(Path(__file__)), weights_unchanged=True,
            parameter_gradients_created=False, output_values_changed=False,
            actor_inputs_changed=False, geometry_evaluated=False,
            scope='Gradients only with respect to a detached reconstruction tensor at unchanged output. '
                  'Cosines are local output-space directions, not parameter/Adam directions or improvement proof. '
                  'Commitment is constant with respect to decoder output; encoder/codebook gradients are not measured.'))
    finally:
        controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'checkpoint', 'previous', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', type=Path, nargs='+', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--domain', choices=('pfnn', 'parc', 'bones'), required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    try:
        main(args)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        raise
