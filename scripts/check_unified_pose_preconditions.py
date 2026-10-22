"""Codec ceiling, native Pose masking and gradient checks; no optimizer."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_pose_training import configure_pose_scope, pose_example, pose_teacher
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.audit_unified_stair_gait import difference, summarize
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def scores(logits, tokens):
    logprob = logits.log_softmax(-1)
    expected = logprob.gather(-1, tokens[..., None]).squeeze(-1)
    other = logits.clone().scatter(-1, tokens[..., None], -torch.inf).amax(-1)
    margin = logits.gather(-1, tokens[..., None]).squeeze(-1) - other
    return dict(ce=float(-expected.mean()), accuracy=float((logits.argmax(-1) == tokens).float().mean()),
                head_ce=(-expected.mean((0, 1))).tolist(), true_logit_margin_mean=float(margin.mean()))


def restore(inference, local, source, observation):
    world = inference.motion_rep.dual_rep.local_to_global(local, True, False,
        lengths=torch.tensor([32], device=local.device))
    offsets, heading, _ = inference._extract_initial_root_info(source[:, :4, :5])
    world = inference._reapply_initial_root_info(dict(pred_global_poses=world,
        reference_start_root_global_offsets=offsets, reference_start_root_global_heading=heading))
    return shift_global_motion(world, inference.global_motion_rep, vertical_origin(inference, source[:, :4], observation))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    controller = UnifiedTerrainController(inference, bundle)
    saved = torch.load(args.initial, map_location='cpu', weights_only=False)
    if saved['contract']['bundle_sha256'] != sha256(args.bundle):
        raise ValueError('Initial baseline mismatch')
    controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
    before = state_digest(controller, list(controller.state_dict()))
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    flat = torch.load(args.flat_cases, map_location='cpu', weights_only=False)
    cases += [dict(c, native_split='train') for c in flat if c['name'] in ('idle', 'walk')]
    geometry = JointGeometryObjective(inference.global_motion_rep, args.device)
    rows, grads = [], []
    for case in cases:
        source = case['world'][None, :33].to(args.device)
        controller.eval().requires_grad_(False)
        pass_logits = []
        handle = controller.pose.register_forward_hook(lambda _m, _a, output: pass_logits.append(output['pose_logits'].detach().clone()))
        try:
            example = pose_example(controller, source, case['scene'])
        finally:
            handle.remove()
        if len(pass_logits) != 4:
            raise ValueError('Expected exactly4native Pose passes')
        with torch.no_grad():
            teacher, logits, batch, corruption = pose_teacher(controller, example, 15100)
            _, all_logits, _, _ = pose_teacher(controller, example, 15100, all_mask=True)
            torch.testing.assert_close(all_logits, pass_logits[0][:, :8], atol=3e-5, rtol=1e-5)
            tokens = example['tokens']
            target, result = example['target'], example['result']
            local = target['local_motions']
            root = extract_feature_from_motion_rep(local, controller.codec.motion_rep, controller.codec.decoder_external_cond_feature_mode)
            poses = extract_feature_from_motion_rep(local, controller.codec.motion_rep, controller.codec.decoder_target_cond_feature_mode)
            mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=args.device)
            mask[:, :4] = True
            ceiling = {}
            previous_root = controller.last_root
            for enabled in (False, True):
                controller._adapting = enabled
                controller.last_root = dict(previous_root, pred_global_root_values=target['global_motions'][..., :5])
                try:
                    prediction = controller.codec.forward_decoder(tokens, external_cond=root, target_cond=poses,
                        has_target_cond=mask, token_mask=torch.ones(tokens.shape[:2], device=args.device, dtype=torch.bool),
                        use_overall_indices=False)['recon_state']
                finally:
                    controller._adapting = False
                    controller.last_root = previous_root
                world = restore(inference, prediction, source, result['observation'])
                ceiling['adapter_on' if enabled else 'native_codec'] = dict(
                    difference(inference.global_motion_rep, world[:, 4:28], source[:, 4:28]),
                    skin=exact_skin_metrics(geometry, world[:, :28], case['scene']))
            replay = controller(source[:, :4], example['command'], case['scene'])
            torch.testing.assert_close(replay['world'], result['world'], atol=0, rtol=0)
            altered = source.clone()
            altered[:, 4:, 1] += 3
            altered[:, 4:, 5:] += .25
            again = controller(altered[:, :4], example['command'], case['scene'])
            torch.testing.assert_close(again['world'], result['world'], atol=0, rtol=0)
        if case['native_split'] == 'train':
            for arm in ('adapter', 'capacity'):
                names = configure_pose_scope(controller, arm)
                controller.zero_grad(set_to_none=True)
                ce, _, _, _ = pose_teacher(controller, example, 15100)
                ce.backward()
                groups = {}
                for name, parameter in controller.named_parameters():
                    if not parameter.requires_grad and parameter.grad is not None:
                        raise ValueError('Frozen parameter acquired gradient')
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise ValueError('Nonfinite Pose gradient')
                        key = '.'.join(name.split('.')[:3])
                        groups[key] = groups.get(key, 0.) + float(parameter.grad.double().square().sum())
                if not groups or sum(groups.values()) == 0:
                    raise ValueError('Missing Pose gradients')
                grads.append(dict(label=case['label'], arm=arm, parameter_count=sum(p.numel() for p in controller.parameters() if p.requires_grad),
                                  norms={k: v ** .5 for k, v in groups.items()}, ce=float(ce.detach())))
                controller.zero_grad(set_to_none=True)
        row = dict(label=case['label'], split=case['native_split'],
            teacher=scores(logits, tokens), teacher_focused_ce=float(teacher),
            passes=[scores(x[:, :8], tokens) for x in pass_logits],
            all_mask_replay_error=float((all_logits - pass_logits[0][:, :8]).abs().max()),
            codec=ceiling, replay_error=0., future_actor_error=0.)
        rows.append(row)
        write_json(args.output / 'progress.json', dict(rows=rows, gradients=grads))
    if state_digest(controller, list(controller.state_dict())) != before:
        raise ValueError('Read-only preconditions changed weights')
    write_json(args.output / 'complete.json', dict(rows=rows, gradients=grads, optimizer_updates=0,
        weights_unchanged=True, future_actor_error=0., initial_sha256=sha256(args.initial),
        limitations='Source codec reconstruction uses future truth offline only. '
                    'Codec ceiling failures remain independent deployment blockers, not a reason to call CE a geometry guarantee.'))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'cases', 'flat-cases', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
