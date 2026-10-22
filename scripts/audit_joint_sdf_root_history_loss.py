"""Read-only attribution of source-past Root targets under generated history."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.pilot_joint_sdf_full_context import command, load_cases
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def flatten_gradient(loss, parameters, retain_graph=True):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
                      for parameter, gradient in zip(parameters, gradients)])


def cosine(a, b):
    denominator = a.norm() * b.norm()
    return float(a.dot(b) / denominator) if denominator > 1e-15 else None


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, weights_only=False, map_location='cpu', mmap=True)
    source = Path(saved['contract']['source'])
    if sha256(source) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source identity differs')
    inference, bundle = load_terrain_bundle(source, 'cuda:0')
    model = JointSDFController(inference, bundle)
    model.load_state_dict(saved['model_state'], strict=True)
    model.freeze_base()
    model.eval()
    rep = inference.global_motion_rep
    cases = load_cases(args.cases, Path(saved['contract']['dataset']))
    parameters = list(model.sdf_root.parameters())
    rows = []
    for case in cases[:3]:
        world = case['world'][None].cuda()
        history = world[:, :4]
        state = SoftCommandState()
        for position in range(case['training_segments']):
            offset = position * 8
            window = world[:, offset:offset + 33]
            task = state.from_controls(history, command(case, offset), 8)
            selected = position in (0, 3, case['training_segments'] - 1)
            with torch.set_grad_enabled(selected):
                result = model(history, task, [case['scene']], source=window, straight_through=True)
                if selected:
                    predicted = result['root_output']['pred_global_root_values'][:, :32]
                    target = extract_feature_from_motion_rep(result['target']['global_motions'], rep, 'root')
                    observed = shift_global_motion(history, rep, -vertical_origin(inference, history, result['observation']))
                    observed[..., [0, 2]] -= history[:, :1, [0, 2]]
                    observed = extract_feature_from_motion_rep(rep.normalize(observed), rep, 'root')
                    past = F.smooth_l1_loss(predicted[:, :4], target[:, :4])
                    actual_past = F.smooth_l1_loss(predicted[:, :4], observed)
                    future = F.smooth_l1_loss(predicted[:, 4:], target[:, 4:])
                    historical_gradient = flatten_gradient(past * (4 / 32), parameters)
                    factual_gradient = flatten_gradient(actual_past * (4 / 32), parameters)
                    future_gradient = flatten_gradient(future * (28 / 32), parameters)
                    difference = history[..., :3] - window[:, :4, :3]
                    direction = task[0, 2:4] / task[0, 2:4].norm()
                    next_target = window[0, 4, [0, 2]] - history[0, 3, [0, 2]]
                    block_target = window[0, 11, [0, 2]] - history[0, 3, [0, 2]]
                    predicted_raw = rep.unnormalize(predicted[:, :4])
                    target_raw = rep.unnormalize(target[:, :4])
                    factual_raw = rep.unnormalize(observed)
                    row = {'source_id': case['label'], 'feedback_position': position,
                           'observed_vs_source_history_root_rms_m': float(difference.square().sum(-1).mean().sqrt()),
                           'predicted_past_vs_source_root_rms_m': float((predicted_raw[..., :3] - target_raw[..., :3]).square().sum(-1).mean().sqrt()),
                           'predicted_past_vs_observed_root_rms_m': float((predicted_raw[..., :3] - factual_raw[..., :3]).square().sum(-1).mean().sqrt()),
                           'source_past_global_loss': float(past), 'factual_past_global_loss': float(actual_past),
                           'future_global_loss': float(future),
                           'source_next_frame_progress_from_observed_m': float(next_target.dot(direction)),
                           'source_next_block_end_progress_from_observed_m': float(block_target.dot(direction)),
                           'source_native_next_step_progress_m': float((window[0, 4, [0, 2]] - window[0, 3, [0, 2]]).dot(direction)),
                           'source_next_height_delta_from_observed_m': float(window[0, 4, 1] - history[0, 3, 1]),
                           'source_past_gradient_norm': float(historical_gradient.norm()),
                           'factual_past_gradient_norm': float(factual_gradient.norm()),
                           'future_gradient_norm': float(future_gradient.norm()),
                           'source_past_vs_future_gradient_cosine': cosine(historical_gradient, future_gradient),
                           'factual_past_vs_future_gradient_cosine': cosine(factual_gradient, future_gradient),
                           'source_past_vs_factual_past_gradient_cosine': cosine(historical_gradient, factual_gradient)}
                    rows.append(row)
                    write_json(args.output / 'progress.json', {'rows': rows, 'quality_approved': False})
                    print(json.dumps(row), flush=True)
            with torch.no_grad():
                history = reencode_played_history(result['world'][:, 8:12].detach(), rep)
    write_json(args.output / 'complete.json', {'rows': rows, 'quality_approved': False,
               'checkpoint_sha256': sha256(args.checkpoint),
               'scope': 'No optimizer or output mutation. Gradients compare only native global Root history/future '
                        'terms in sdf_root parameters; not a causal proof about the complete loss. '
                        'Root targets include all32 frames; decoded-motion mask begins at4.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
