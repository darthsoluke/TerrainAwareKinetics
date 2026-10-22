"""Verify optional factual-past Root targets on the actual checkpoint, without updates."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_root_supervision import ROOT_HISTORY_SUPERVISION, factual_history_root_targets
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.pilot_joint_sdf_full_context import command, load_cases
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def check_root_losses(model, result, history):
    target = result['target']
    original = {'global_motions': target['global_motions'], 'local_motions': target['local_motions'],
                'groundtruth_num_tokens': torch.full((len(history), 1), 8, dtype=torch.long, device=history.device)}
    revised = factual_history_root_targets(model.inference, target, history, result['observation'])
    rep = model.inference.global_motion_rep
    torch.testing.assert_close(revised['global_motions'][:, 4:],
                               extract_feature_from_motion_rep(target['global_motions'], rep, 'root')[:, 4:], atol=0, rtol=0)
    torch.testing.assert_close(revised['local_motions'][:, 3:],
                               extract_feature_from_motion_rep(target['local_motions'], model.inference.local_motion_rep, 'root')[:, 3:], atol=0, rtol=0)
    output = dict(result['root_output'])
    predicted = output['pred_global_root_values'].detach().clone().requires_grad_()
    output['pred_global_root_values'] = predicted
    original_loss = model.inference._root_model.loss(original, output)['loss']
    revised_loss = model.inference._root_model.loss(revised, output)['loss']
    original_gradient = torch.autograd.grad(original_loss, predicted, retain_graph=True)[0]
    revised_gradient = torch.autograd.grad(revised_loss, predicted)[0]
    return {'original_loss': float(original_loss), 'factual_history_loss': float(revised_loss),
            'loss_absolute_difference': float((original_loss - revised_loss).abs()),
            'root_output_gradient_max_difference': float((original_gradient - revised_gradient).abs().max()),
            'future_global_and_boundary_onward_local_targets_unchanged': True}


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    path = Path(saved['contract']['source'])
    if sha256(path) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source identity mismatch')
    inference, bundle = load_terrain_bundle(path, 'cuda:0')
    model = JointSDFController(inference, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.requires_grad_(False)
    model.eval()
    cases = load_cases(args.cases, Path(saved['contract']['dataset']))
    rows = []
    for case in cases[:3]:
        source = case['world'][None].cuda()
        history = source[:, :4]
        state = SoftCommandState()
        for position in range(case['training_segments']):
            offset = position * 8
            window = source[:, offset:offset + 33]
            controls = command(case, offset)
            task = state.from_controls(history, controls, 8)
            with torch.no_grad():
                rollout = model(history, task, [case['scene']], source=window)
                source_task = SoftCommandState().from_controls(window[:, :4], controls, 8)
                source_result = model(window[:, :4], source_task, [case['scene']], source=window)
            reference_check = check_root_losses(model, source_result, window[:, :4])
            if reference_check['loss_absolute_difference'] > 1e-7 or reference_check['root_output_gradient_max_difference'] > 1e-7:
                raise ValueError('Factual-source Root loss or gradient changed')
            generated_check = check_root_losses(model, rollout, history)
            with torch.no_grad():
                repeated = model(history, task, [case['scene']])['world']
                torch.testing.assert_close(rollout['world'], repeated, atol=0, rtol=0)
                history = reencode_played_history(rollout['world'][:, 8:12], inference.global_motion_rep)
            rows.append({'source_id': case['label'], 'position': position,
                         'source_history': reference_check, 'generated_history': generated_check,
                         'actor_replay_difference': 0.})
        write_json(args.output / 'progress.json', {'rows': rows, 'quality_approved': False})
    summary = {'source_windows': len(rows),
               'source_history_loss_max_difference': max(row['source_history']['loss_absolute_difference'] for row in rows),
               'source_history_root_gradient_max_difference': max(row['source_history']['root_output_gradient_max_difference'] for row in rows),
               'actor_replay_max_difference': 0., 'optimizer_updates': 0,
               'future_target_preservation': True, 'version': ROOT_HISTORY_SUPERVISION, 'quality_approved': False}
    write_json(args.output / 'complete.json', {'summary': summary, 'rows': rows,
               'scope': 'Optional loss-side correction only, not adopted into existing runners. '
                        'No runtime output changes and no evidence of improved motion.'})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
