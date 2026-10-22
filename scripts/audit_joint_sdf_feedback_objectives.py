"""Measure competing geometry objectives on unchanged generated-history states."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_full_context import command, load_cases
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def cos(a, b):
    product = a.norm() * b.norm()
    return float(a.dot(b) / product) if product > 1e-15 else None


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
    rep = inference.global_motion_rep
    geometry = JointGeometryObjective(rep, 'cuda:0')
    cases = load_cases(args.cases, Path(saved['contract']['dataset']))
    rows = []
    for case in cases[:3]:
        source = case['world'][None].cuda()
        history = source[:, :4]
        state = SoftCommandState()
        field = HeightfieldPrismSDF(case['scene'])
        exact = ExactPenetrationBatch([case['scene']], source.device, [field])
        with torch.no_grad():
            source_joints, source_skin = geometry.geometry(source)
        for position in range(case['training_segments']):
            offset = position * 8
            task = state.from_controls(history, command(case, offset), 8)
            with torch.no_grad():
                result = model(history, task, [case['scene']])
            if position in (0, 3, case['training_segments'] - 1):
                predicted = result['world'].detach().clone().requires_grad_(True)
                target = {'joints': source_joints[:, offset:offset + 33],
                          'points': source_skin[:, offset:offset + 33],
                          'heights': exact.heights, 'volumes': exact}
                terms = geometry.prepared_losses(predicted, history, target, offset=0, diagnostics=False)
                gradients = {}
                first = {}
                with torch.no_grad():
                    observed_joints = rep.inverse(history, False, return_all=False,
                                                  joint_positions_from='global_rot_data')['posed_joints']
                # Diagnose the vector identity in float64: world coordinates
                # near50m lose micrometres when subtracting float32 endpoints.
                absolute_next = target['joints'][:, 4].double()
                velocity_next = observed_joints[:, 3].double() + absolute_next - target['joints'][:, 3].double()
                torch.testing.assert_close(velocity_next - absolute_next,
                                           observed_joints[:, 3].double() - target['joints'][:, 3].double(), atol=1e-12, rtol=0)
                for name, weight in saved['coefficients'].items():
                    gradient = torch.autograd.grad(weight * terms[name], predicted, retain_graph=True)[0]
                    gradients[name] = gradient.flatten()
                    first[name] = gradient[:, 4].flatten()
                row = {'source_id': case['label'], 'position': position,
                       'next_joint_zero_loss_target_separation_mean_m': float((absolute_next - velocity_next).norm(dim=-1).mean()),
                       'next_joint_zero_loss_target_separation_max_m': float((absolute_next - velocity_next).norm(dim=-1).max()),
                       'terms': {key: {'value': float(terms[key]), 'weight': weight,
                                      'output_gradient_norm': float(gradients[key].norm()),
                                      'first_frame_output_gradient_norm': float(first[key].norm())}
                                 for key, weight in saved['coefficients'].items()},
                       'all_output_gradient_cosines': {}, 'first_frame_gradient_cosines': {}}
                for a, b in (('fk_m', 'boundary_mps'), ('collision_m2', 'boundary_mps'),
                             ('collision_m2', 'fk_m'), ('support_gap_m', 'boundary_mps')):
                    row['all_output_gradient_cosines'][a + ':' + b] = cos(gradients[a], gradients[b])
                    row['first_frame_gradient_cosines'][a + ':' + b] = cos(first[a], first[b])
                rows.append(row)
                write_json(args.output / 'progress.json', {'rows': rows, 'quality_approved': False})
                print(json.dumps(row), flush=True)
            with torch.no_grad():
                history = reencode_played_history(result['world'][:, 8:12], rep)
    write_json(args.output / 'complete.json', {'rows': rows, 'quality_approved': False, 'optimizer_updates': 0,
               'scope': 'Original geometry weights, output-space gradients through actual FK, not model-parameter '
                        'gradient magnitudes. Absolute position and source-velocity boundary have incompatible '
                        'zero-loss endpoints whenever played and source histories differ. No proposed target '
                        'is supplied to actor or substituted into rendered motion.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
