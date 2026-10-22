"""Real-model identity, gradient and no-output-mutation gate for source fidelity."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_codec_fidelity import source_fidelity_losses, VERSION
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_codec import source_decode
from scripts.probe_unified_native_stages import prepare
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.parent, map_location='cpu', weights_only=False)
    if saved['contract']['cases_sha256'] != sha256(a.cases) or saved['contract']['bundle_sha256'] != sha256(a.bundle):
        raise ValueError('Parent identity differs')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True)
    controller.eval().requires_grad_(False)
    frozen = state_digest(controller, list(controller.state_dict()))
    report = json.loads(a.parent.with_name('complete.json').read_text())
    windows = {f'{w["label"]}-f{w["start"]}': w for w in saved['contract']['source_windows']}
    chosen, seen = [], set()
    for row in sorted(report['final'], key=lambda x: -x['skin']['body_prism_max_m']):
        window = windows[row['label']]
        if window['label'] not in seen:
            chosen.append(window); seen.add(window['label'])
    prepared = prepare(controller, torch.load(a.cases, map_location='cpu', weights_only=False), chosen, a.device)
    controller.codec.decoder.requires_grad_(True)
    parameters = list(controller.codec.decoder.parameters())
    rows = []
    for item in prepared:
        example = item['example']
        example['target'] = {k: v for k, v in example['target'].items() if k not in ('motion_frame_mask', 'motion_interval_mask')}
        local = example['target']['local_motions']
        identity = source_fidelity_losses(inference, local, example['target'], item['source'][:, :4], example['result']['observation'])
        if float(identity['total']) > 1e-6:
            raise ValueError('Source identity is not a zero of the proposed objective')
        prediction = source_decode(controller, example)['recon_state']
        before = prediction.detach().clone()
        native = controller.loss_model.loss(example['target'], dict(recon_state=prediction,
            l_commit=prediction.sum() * 0, perplexity=0.), None)['loss']
        fidelity = source_fidelity_losses(inference, prediction, example['target'], item['source'][:, :4], example['result']['observation'])
        norms = {}
        for name, loss in dict(native=native, root=fidelity['decoded_root_native'], fk=fidelity['rendered_fk']).items():
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            active = [g for g in gradients if g is not None]
            if not active or any(not torch.isfinite(g).all() for g in active):
                raise ValueError('Invalid codec fidelity gradient')
            norms[name] = sum(float(g.double().square().sum()) for g in active) ** .5
        controller.zero_grad(set_to_none=True)
        (native + fidelity['total']).backward()
        if any(p.grad is not None for p in controller.parameters() if not p.requires_grad):
            raise ValueError('Frozen module acquired gradients')
        controller.zero_grad(set_to_none=True)
        torch.testing.assert_close(prediction, before, atol=0, rtol=0)
        with torch.no_grad():
            replay = controller(item['source'][:, :4], example['command'], item['case']['scene'])['world']
            torch.testing.assert_close(replay, example['result']['world'], atol=0, rtol=0)
        rows.append(dict(**item['window'], identity_loss=float(identity['total']),
            native_loss=float(native.detach()), losses={k: float(v.detach()) for k, v in fidelity.items()},
            gradient_norms=norms, actor_difference=0., prediction_mutation=0.))
    if state_digest(controller, list(controller.state_dict())) != frozen:
        raise ValueError('Read-only fidelity gate changed weights')
    write_json(a.output / 'complete.json', dict(version=VERSION, rows=rows, optimizer_updates=0,
        all_checks_passed=True, weights_unchanged=True, parent_sha256=sha256(a.parent), cases_sha256=sha256(a.cases),
        coefficients=dict(global_root=float(inference._root_model.args['global_root_loss_coeff']),
            local_root=float(inference._root_model.args['local_root_loss_coeff']), rendered_fk=1.),
        source='Native Root loss coefficients and existing RIC position scales; explicit fixed FK group weight1, no coefficient search',
        runtime_correction=False))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'parent', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
