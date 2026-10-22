"""Real-model gates for the always-conditioned single-chain terrain prototype."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE, SCHEMA
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@torch.no_grad()
def rollout(inference, controller, seed, scene, quick=False):
    result = {}

    def backend(history):
        value = (NativeTerrainOnlineBackend(inference, history, scene) if controller is None
                 else UnifiedTerrainBackend(controller, history, scene))
        value.configure_runtime(dict(PROFILE))
        return value

    for yaw in (0., np.pi/2, np.pi, -np.pi/2):
        for speed in (1.6, 5.):
            for idle in ((0,) if quick else (0, 60)):
                history = inference.global_motion_rep.change_first_heading(seed.clone(), yaw, False, False)
                actor = backend(history)
                pieces = []
                for block in range(idle+(3 if quick else 30)):
                    velocity = [np.sin(yaw)*speed, np.cos(yaw)*speed] if block >= idle else [0., 0.]
                    value = actor.predict_plan(velocity, yaw)
                    pieces.append(value['planned_motion'].clone())
                    actor.history = value['published_history'].clone()
                result[f'{yaw:.6f}-{speed:g}-{idle}'] = torch.cat(pieces, 1).cpu()
    actor = backend(seed)
    pieces = []
    for yaw, speed in [(0., 1.6), (0., 0.), (np.pi/2, 1.6), (-np.pi/2, 1.6), (np.pi, 1.6), (0., 1.6)]:
        for _ in range(1 if quick else 6):
            value = actor.predict_plan([np.sin(yaw)*speed, np.cos(yaw)*speed], yaw)
            pieces.append(value['planned_motion'].clone())
            actor.history = value['published_history'].clone()
    result['continuous-wasd-release'] = torch.cat(pieces, 1).cpu()
    return result


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(6143)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sha256(args.bundle) != '0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d':
        raise ValueError('Approved bundle identity differs')
    inference, bundle = load_terrain_bundle(args.bundle, args.device)
    keys = list(inference.state_dict())
    frozen = state_digest(inference, keys)
    root_args, pose_args = deepcopy(inference._root_model.args), deepcopy(inference._pose_model.args)
    codec = inference._vqvae_pose_model
    with np.load(args.seed_scene, allow_pickle=False) as archive:
        seed = torch.from_numpy(archive['initial_history'].copy()).to(args.device)
    seed = inference.global_motion_rep.change_first_heading(seed, 0., False, False)
    seed[..., [0, 2]] -= seed[:, -1:, [0, 2]].clone()
    scene = dict(heightfield_y_m=torch.zeros(201, 201), min_point_world_xyz_m=torch.tensor([-50., 0., -50.]), dx_m=torch.tensor(.5))
    write_json(args.output/'state.json', dict(status='baseline_rollouts', quick=args.quick))
    reference = rollout(inference, None, seed, scene, args.quick)
    controller = UnifiedTerrainController(inference, bundle)
    if inference._vqvae_pose_model is not codec or root_args != inference._root_model.args or pose_args != inference._pose_model.args:
        raise ValueError('Original model or protocol changed')
    initial = {k: v.detach().clone() for k, v in controller.adaptation.state_dict().items()}
    write_json(args.output/'state.json', dict(status='zero_adapter_rollouts'))
    current = rollout(inference, controller, seed, scene, args.quick)
    parity = []
    for key in reference:
        torch.testing.assert_close(current[key], reference[key], atol=0, rtol=0)
        parity.append(dict(case=key, frames=current[key].shape[1], max_feature_error=float((current[key]-reference[key]).abs().max())))
    if controller.native_calls != controller.environment_calls or controller.native_calls < 1:
        raise ValueError('Expected one environment encoding and one native chain per replan')
    write_json(args.output/'flat_gate.json', dict(passed=True, cases=parity,
        native_calls=controller.native_calls, environment_encodings=controller.environment_calls,
        trained_flat_bitwise_equality_claimed=False))
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    selected = [case for case in cases if case['native_split'] == 'train']
    if len(selected) != 2:
        raise ValueError('Need the prepared one-up/one-down source pair')
    optimizer = torch.optim.Adam(controller.adaptation.parameters(), lr=5e-5)
    gradients = []
    write_json(args.output/'state.json', dict(status='source_history_gradient_probes'))
    for case in selected:
        source = case['world'][None, 24:57].to(args.device)
        command = source_terminal_goal(inference.global_motion_rep.extract_root(source))
        with torch.no_grad():
            if not gradients:
                controller.enabled = False
                before = controller(source[:, :4], command, case['scene'])['world'].clone()
                controller.enabled = True
            expected = controller(source[:, :4], command, case['scene'])['world'].clone()
            if not gradients:
                torch.testing.assert_close(before, expected, atol=0, rtol=0)
        optimizer.zero_grad(set_to_none=True)
        output = controller(source[:, :4], command, case['scene'], training=True, straight_through=True)
        torch.testing.assert_close(output['world'].detach(), expected, atol=3e-5, rtol=1e-5)
        loss = controller.native_losses(output, source, source[:, :4])['native_total']
        loss.backward()
        norms = {name: sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**.5
                 for name, module in controller.adaptation.items()}
        if not all(np.isfinite(value) for value in norms.values()) or not norms['decoder'] > 0:
            raise ValueError('Invalid or absent adaptation gradients')
        if gradients and not norms['encoder'] > 0:
            raise ValueError('The environment encoder did not receive second-step gradients')
        if any(p.grad is not None for p in inference.parameters()):
            raise ValueError('Frozen native parameters acquired gradients')
        norm = torch.nn.utils.clip_grad_norm_(controller.adaptation.parameters(), .5)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite diagnostic update')
        optimizer.step()
        gradients.append(dict(label=case['label'], native_loss=float(loss.detach()), groups=norms,
                              straight_through_forward_max_error=float((output['world'].detach()-expected).abs().max())))
    # Unlike the old gate, a trained flat residual is allowed to change the output.
    played = seed.detach().clone().requires_grad_(True)
    command = played.new_tensor([[0., 1.6, 1., 0.]])
    optimizer.zero_grad(set_to_none=True)
    output = controller(played, command, scene, training=True, straight_through=True)
    history_gradient = torch.autograd.grad(output['world'][:, 4:28].square().mean(), played)[0]
    if not torch.isfinite(history_gradient).all() or history_gradient.norm() <= 0:
        raise ValueError('Factual-history gradient disconnected')
    if state_digest(inference, keys) != frozen:
        raise ValueError('Frozen tensors changed')
    controller.adaptation.load_state_dict(initial, strict=True)
    result = dict(schema=SCHEMA, status='complete', device=args.device, quick=args.quick,
                  baseline_sha256=sha256(args.bundle), zero_adapter_flat_parity=parity,
                  gradients=gradients, factual_history_gradient_norm=float(history_gradient.norm()),
                  native_calls=controller.native_calls, environment_encodings=controller.environment_calls,
                  sampling_points=3*controller.sampler.config.grid_size**3, environment_tokens=24,
                  native_hash_unchanged=True, original_protocol_unchanged=True, diagnostic_updates=2,
                  diagnostic_updates_retained=False, fitting_updates=0, quality_approved=False,
                  source_future_in_actor=False, postprocessing=None)
    write_json(args.output/'complete.json', result)
    write_json(args.output/'state.json', dict(status='complete', quality_approved=False))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'seed-scene', 'cases', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output/'state.json', dict(status='failed', error=repr(error)))
        raise
