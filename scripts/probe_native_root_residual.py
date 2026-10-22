"""Isolate the worst remaining causal Root window without changing its loss."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.unified_pose_training import pose_memory
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_stages import fresh_kwargs, prepare, root_review
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(17122)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.parent, map_location='cpu', weights_only=False)
    report = json.loads(a.parent.with_name('complete.json').read_text())
    if (saved['schema'] != 'motionbricks.unified.native_route_curriculum.v1'
            or saved['contract']['stage'] != 'root' or report['training_bank_gate_pass']
            or saved['contract']['cases_sha256'] != sha256(a.cases)
            or saved['contract']['bundle_sha256'] != sha256(a.bundle)):
        raise ValueError('Use the matched incomplete Root bank')
    selected = max(report['final'], key=lambda row: row['root_xyz_rms_m'])
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True); controller.eval().requires_grad_(False)
    prepared = prepare(controller, torch.load(a.cases, map_location='cpu', weights_only=False),
                       saved['contract']['source_windows'], a.device)
    item = next(i for i in prepared if i['window']['label'] == selected['label'] and i['window']['start'] == selected['start'])
    baseline = root_review(controller, prepared)
    controller.root.requires_grad_(True)
    for key in ('root_stage1', 'root_stage2'):
        controller.adaptation[key].requires_grad_(True)
    names = [n for n, p in controller.named_parameters() if p.requires_grad]
    protected_keys = [n for n in controller.state_dict() if n not in names]
    protected = state_digest(controller, protected_keys)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-7)
    curve = []
    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        with pose_memory(controller, item['example']):
            output = controller.root(*item['root_args'], **fresh_kwargs(item['root_kwargs']))
        loss = inference._root_model.loss(item['root_target'], output)['loss']
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite native residual loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, .5, error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        if (step + 1) % 10 == 0:
            row = root_review(controller, [item])[0]
            curve.append(dict(step=step + 1, loss=float(loss.detach()), **row))
            write_json(a.output / 'progress.json', dict(selected=selected, curve=curve))
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected modules changed')
    final = root_review(controller, prepared)
    selected_final = next(row for row in final if row['label'] == selected['label'] and row['start'] == selected['start'])
    write_json(a.output / 'complete.json', dict(schema='motionbricks.native_root_residual_diagnostic.v1',
        parent_sha256=sha256(a.parent), selected=selected, selected_final=selected_final,
        baseline=baseline, final=final, curve=curve, optimizer_updates=100, primitive_exposures=100,
        native_loss_unchanged=True, actor_future_truth=False, protected_hash_unchanged=True,
        quality_approved=False, not_deployable=True, checkpoint_saved=False))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'parent', 'cases', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
