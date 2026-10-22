"""Fit only existing decoder output rows using causal actor features and source labels."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from motionbricks.data.joint_sdf_controller import SDF_VERSION
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import training_targets
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_contact_preserve import load_model, prepare_cases, control, exact_reviews
from scripts.pilot_joint_sdf_full_context import evaluate
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


def convolution_design(x, head):
    if head.kernel_size != (3,) or head.stride != (1,) or head.padding != (1,) or head.dilation != (1,) or head.groups != 1:
        raise ValueError('Only the existing final3-tap decoder projection is supported')
    patches = F.pad(x, (1, 1)).unfold(-1, 3, 1).permute(0, 2, 1, 3).flatten(-2)
    return torch.cat((patches, torch.ones_like(patches[..., :1])), -1)


def ridge_delta(x, y, prior, held_out):
    x, y, prior = x.double(), y.double(), prior.double()
    train_x, train_y = x[~held_out], y[~held_out]
    residual = train_y-train_x @ prior.T
    gram = train_x.T @ train_x/len(train_x)
    rhs = train_x.T @ residual/len(train_x)
    scale = gram.diag()[:-1].mean().clamp_min(1e-9)
    trials = []
    best = None
    for multiplier in (1e-4, 1e-3, 1e-2, 1e-1):
        regularization = scale*multiplier
        delta = torch.linalg.solve(gram+torch.eye(len(gram), device=x.device, dtype=x.dtype)*regularization, rhs).T
        weights = prior+delta
        validation = float((x[held_out] @ weights.T-y[held_out]).square().mean())
        trials.append(dict(multiplier=multiplier, regularization=float(regularization),
                           validation_mse=validation, delta_l2=float(delta.norm())))
        if best is None or validation < best[0]:
            best = (validation, weights, multiplier)
    return best[1], dict(trials=trials, selected_multiplier=best[2],
        prior_validation_mse=float((x[held_out] @ prior.T-y[held_out]).square().mean()),
        validation_mse=best[0], training_mse=float((train_x @ best[1].T-train_y).square().mean()))


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if sha256(Path(saved['contract']['source'])) != saved['contract']['source_bundle_sha256']:
        raise ValueError('Source bundle identity mismatch')
    model, bundle = load_model(saved)
    cases = prepare_cases(args.cases, Path(saved['contract']['dataset']))
    head = model.codec.decoder.model[-1]
    if not isinstance(head, torch.nn.Conv1d) or head.out_channels != len(model.inference.local_motion_rep.indices['all']):
        raise ValueError('Expected the existing complete local-motion output projection')
    original_weight, original_bias = head.weight.clone(), head.bias.clone()
    head_addresses = {head.weight.data_ptr(), head.bias.data_ptr()}
    immutable_keys = [key for key,value in model.state_dict().items() if value.data_ptr() not in head_addresses]
    immutable_digest = state_digest(model, immutable_keys)
    prior = torch.cat((original_weight.flatten(1), original_bias[:, None]), -1)
    records, inputs, targets, held = [], [], [], []
    calls = []
    hook = head.register_forward_hook(lambda module, values, output: calls.append((values[0].detach().clone(), output.detach().clone())))
    try:
        for case in cases:
            source = case['world'][None].cuda()
            for block in range(case['training_segments']):
                window = source[:, block*8:block*8+33]
                history = window[:, :4]
                intent = SoftCommandState().from_controls(history, control(case, block), 8)
                calls.clear()
                result = model(history, intent, [case['scene']])
                if len(calls) != 2:
                    raise ValueError('Expected one proposal and one final causal decode')
                x, actual = calls[-1]
                design = convolution_design(x, head)[0, :32]
                reference = training_targets(model.inference, window, history, result['observation'])['local_motions'][0]
                predicted = design.double() @ prior.double().T
                error = float((predicted-actual[0, :, :32].T.double()).abs().max())
                if error > 1e-4:
                    raise ValueError('Linear design does not reproduce the actual decoder output')
                inputs.append(design.cpu())
                targets.append(reference.cpu())
                held.append(torch.full((32,), block%3 == 1, dtype=torch.bool))
                records.append(dict(label=case['label'], source_window_start=block*8,
                                    held_out=block%3 == 1, design_replay_error=error))
    finally:
        hook.remove()
    x, y, validation = torch.cat(inputs).cuda(), torch.cat(targets).cuda(), torch.cat(held).cuda()
    indices = model.inference.local_motion_rep.indices
    root_rows = [int(indices['global_root_y'][0]), *map(int, indices['local_root_vel'])]
    rotation_rows = list(map(int, indices['global_rot_data']))
    fitted = {}
    for name, rows in [('root', root_rows), ('rotation', rotation_rows)]:
        weights, report = ridge_delta(x, y[:, rows], prior[rows], validation)
        fitted[name] = (rows, weights.float(), report)
    immutable = {name:state_digest(model.get_submodule(name)) for name in
        ('root','pose','codec.encoder','codec.quantizer','control','sdf_encoder','sdf_root','sdf_pose','sdf_decoder')}
    contract = dict(source_sha256=sha256(args.checkpoint), source_step=saved['step'],
        operation='closed-form ridge delta on existing decoder output rows, no optimizer updates',
        features='actual predicted Root/codes from factual source4 history, command and observed terrain; no oracle features',
        targets='unchanged real source local Root/rotations, labels only',
        validation='source-prefix windows with block%3==1 held out of the linear fit; overlapping motions and prior training mean this is NOT unseen-data validation',
        source_windows=records, frames=len(x), training_frames=int((~validation).sum()), validation_frames=int(validation.sum()),
        root_fit=fitted['root'][2], rotation_fit=fitted['rotation'][2], frozen_module_hashes=immutable,
        all_non_head_state_sha256=immutable_digest,
        caveat='Updated head also changes the model proposal used for terrain queries; only actual closed-loop reviews can establish improvement.',
        postprocessing=None, future_truth_in_actor=False, quality_approved=False)
    write_json(args.output/'contract.json', contract)
    geometry = JointGeometryObjective(model.inference.global_motion_rep, 'cuda:0')
    for arm in ('root', 'rendered'):
        output = args.output/arm
        output.mkdir()
        head.weight.copy_(original_weight)
        head.bias.copy_(original_bias)
        changed = []
        for group in ('root',) if arm=='root' else ('root','rotation'):
            rows, weights, _ = fitted[group]
            head.weight[rows] = weights[:, :-1].reshape(len(rows), *head.weight.shape[1:])
            head.bias[rows] = weights[:, -1]
            changed += rows
        frozen_rows = sorted(set(range(head.out_channels))-set(changed))
        if not torch.equal(head.weight[frozen_rows], original_weight[frozen_rows]) or not torch.equal(head.bias[frozen_rows], original_bias[frozen_rows]):
            raise ValueError('Unexpected output-row mutation')
        if {name:state_digest(model.get_submodule(name)) for name in immutable} != immutable:
            raise ValueError('Frozen model components changed')
        if state_digest(model, immutable_keys) != immutable_digest:
            raise ValueError('Decoder trunk or other non-head tensors changed')
        arm_contract = dict(contract, arm=arm, changed_output_rows=changed,
            updated_parameters=len(changed)*(head.weight[0].numel()+1))
        write_json(output/'contract.json', arm_contract)
        torch.save(dict(schema=SDF_VERSION, model_state=model.state_dict(), step=0,
            inherited_optimizer_updates=saved['step'], fit_operations=1,
            contract=dict(saved['contract'], render_head_fit=arm_contract), coefficients=saved['coefficients']), output/'weights.pt')
        final = exact_reviews(model, cases, geometry, output/'final')
        varied = evaluate(model, bundle, cases, geometry, output/'varied')
        write_json(output/'complete.json', dict(arm=arm, optimizer_updates=0, fit_operations=1,
            final=final, varied=varied, quality_approved=False))
    write_json(args.output/'complete.json', dict(status='complete', arms=['root','rendered'],
                                              optimizer_updates=0, quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','cases','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
