"""Zero-update gate for causal Root/Pose native decoded-motion supervision."""
import argparse
from pathlib import Path

import torch

from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_pose_training import configure_pose_scope, pose_example, pose_teacher, decoded_native_losses
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(15114)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    saved = torch.load(a.initial, map_location='cpu', weights_only=False)
    if saved['contract']['bundle_sha256'] != sha256(a.bundle):
        raise ValueError('Initial bundle differs')
    controller.adaptation.load_state_dict(saved['adaptation'], strict=True)
    with torch.no_grad():
        controller.adaptation['pose'].output.weight.zero_()
    configure_pose_scope(controller, 'root_pose')
    keys = list(controller.state_dict())
    frozen = state_digest(controller, keys)
    parameters = [(name, p) for name, p in controller.named_parameters() if p.requires_grad]
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    selected = [next(c for c in cases if c.get('entry') == entry) for entry in ('up', 'down')]
    selected += [next(c for c in cases if c.get('name') == name) for name in ('idle', 'walk')]
    rows = []
    for index, case in enumerate(selected):
        source = case['world'][None, :33].to(a.device)
        normal = pose_example(controller, source, case['scene'])
        example = pose_example(controller, source, case['scene'], actor_gradient=True)
        forward_error = float((normal['result']['world'] - example['result']['world']).abs().max())
        torch.testing.assert_close(normal['result']['world'], example['result']['world'], atol=3e-5, rtol=0)
        ce, _, _, _ = pose_teacher(controller, example, 17114 + index)
        parts = dict(ce=ce, **decoded_native_losses(controller, example, source))
        gradients = {}
        for name, loss in parts.items():
            grads = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
            values = {}
            for (parameter_name, _), gradient in zip(parameters, grads):
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise ValueError('Nonfinite component gradient')
                group = parameter_name.split('.')[1]
                values[group] = values.get(group, 0.) + float(gradient.double().square().sum())
            gradients[name] = {k: v ** .5 for k, v in values.items()}
        for component in ('root_stage1', 'root_stage2', 'pose'):
            if sum(g.get(component, 0.) for g in gradients.values()) <= 0:
                raise ValueError('Missing joint-scope gradient: ' + component)
        controller.zero_grad(set_to_none=True)
        sum(parts.values()).backward()
        if any(p.grad is not None for p in controller.parameters() if not p.requires_grad):
            raise ValueError('Frozen module acquired gradient')
        controller.zero_grad(set_to_none=True)
        with torch.no_grad():
            modified = source.clone(); modified[:, 4:, 1] += 5; modified[:, 4:, 5:] += .1
            replay = controller(modified[:, :4], normal['command'], case['scene'])['world']
            torch.testing.assert_close(replay, normal['result']['world'], atol=0, rtol=0)
            native = controller.native_losses(normal['result'], source, source[:, :4])
            for key in ('native_motion', 'native_root'):
                torch.testing.assert_close(native[key], parts[key], atol=3e-5, rtol=1e-5)
        rows.append(dict(label=case['label'], losses={k: float(v.detach()) for k, v in parts.items()},
            gradient_norms=gradients, actor_st_forward_difference=forward_error,
            future_target_actor_difference=0., native_loss_equivalence=True))
        write_json(a.output / 'progress.json', dict(rows=rows))
    if state_digest(controller, keys) != frozen:
        raise ValueError('Preflight changed weights')
    write_json(a.output / 'complete.json', dict(rows=rows, all_checks_passed=True, optimizer_updates=0,
        weights_unchanged=True, initial_sha256=sha256(a.initial), cases_sha256=sha256(a.cases),
        trainable_count=sum(p.numel() for _, p in parameters),
        scope='Root/Pose adapters only; native CE + existing native motion/Root loss coefficients. '
        'ST forwards agree within stated tolerance; ST is a surrogate gradient, not exact argmax differentiation. '
        'Pose height remains detached; no claim of a Pose-height gradient into Root.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
