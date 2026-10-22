"""Native Root then source-Root Pose capacity gates on four fixed real windows."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_root_supervision import factual_history_root_targets
from motionbricks.data.unified_pose_training import pose_example, pose_memory, pose_teacher
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.played_motion_review import played_window, PUBLICATION_GEOMETRY
from scripts.audit_native_root_validation import source_root_intervention
from scripts.audit_unified_stair_gait import difference, generate, summarize
from scripts.check_unified_pose_preconditions import scores
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


SCHEMA = 'motionbricks.unified.native_stage_capacity.v1'


def fresh_kwargs(kwargs):
    result = dict(kwargs)
    if 'terrain_observation' in result:
        result['terrain_observation'] = {k: v for k, v in result['terrain_observation'].items()
                                         if k != '_inference_token_cache'}
    if 'config' in result:
        result['config'] = dict(result['config'], cache_terrain_encoder=False)
    return result


@torch.no_grad()
def prepare(controller, cases, windows, device):
    by_label = {c['label']: c for c in cases}
    prepared = []
    for window in windows:
        case = by_label[window['label']]
        source = case['world'][None, window['start']:window['start'] + 33].to(device)
        calls = []
        handle = controller.root.register_forward_pre_hook(lambda _m, a, k: calls.append((a, fresh_kwargs(k))), with_kwargs=True)
        try:
            example = pose_example(controller, source, case['scene'])
        finally:
            handle.remove()
        if len(calls) != 1:
            raise ValueError('Expected one causal Root request')
        root_args, root_kwargs = calls[0]
        if (root_kwargs['global_root_component_mask'][:, 4:, 1].any()
                or root_kwargs['local_root_component_mask'][:, 3:, 3].any()
                or root_args[5][:, 4:].any()):
            raise ValueError('Future height/pose entered causal Root inputs')
        target = factual_history_root_targets(controller.inference, example['target'], source[:, :4], example['result']['observation'])
        with source_root_intervention(controller, target['global_motions']):
            oracle = generate(controller, source[:, :4], example['command'], case['scene'])
            args, kwargs = controller.last_pose_request
            full_example = dict(example, args=args, kwargs=fresh_kwargs(kwargs))
        # Native training uses the actual token length; validate against padded runtime.
        cropped_args = (args[0][:, :8], args[1][:, :32], args[2][:, :32], args[3][:, :32], *args[4:])
        cropped_kwargs = fresh_kwargs(kwargs)
        if 'root_height_trajectory' in cropped_kwargs:
            cropped_kwargs['root_height_trajectory'] = cropped_kwargs['root_height_trajectory'][:, :32]
        source_teacher = dict(example, args=cropped_args, kwargs=cropped_kwargs)
        _, full, _, _ = pose_teacher(controller, full_example, 17117, all_mask=True)
        _, short, _, _ = pose_teacher(controller, source_teacher, 17117, all_mask=True)
        torch.testing.assert_close(short, full, atol=5e-5, rtol=1e-5)
        prepared.append(dict(case=case, source=source, example=example, root_args=root_args,
            root_kwargs=root_kwargs, root_target=target, source_teacher=source_teacher,
            full_source_teacher=full_example, teacher_padding_logit_max=float((short - full).abs().max()),
            window=window))
    return prepared


@torch.no_grad()
def root_review(controller, prepared):
    rows = []
    rep = controller.inference.global_motion_rep
    for item in prepared:
        with pose_memory(controller, item['example']):
            output = controller.root(*item['root_args'], **fresh_kwargs(item['root_kwargs']))
        predicted = rep.unnormalize(output['pred_global_root_values'][:, :32])
        truth = rep.unnormalize(item['root_target']['global_motions'])
        error = predicted[:, 4:28, :3] - truth[:, 4:28, :3]
        rms = float(error.square().mean().sqrt())
        max_y = float(error[..., 1].abs().max())
        rows.append(dict(**item['window'], root_xyz_rms_m=rms, max_root_y_error_m=max_y,
            gate_pass=rms <= .01 and max_y <= .02))
    return rows


@torch.no_grad()
def pose_review(controller, prepared, geometry, folder):
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    rep = controller.inference.global_motion_rep
    for item in prepared:
        case, source, example = item['case'], item['source'], item['example']
        actor = generate(controller, source[:, :4], example['command'], case['scene'])
        with source_root_intervention(controller, item['root_target']['global_motions']):
            oracle = generate(controller, source[:, :4], example['command'], case['scene'])
        teacher = dict(item['source_teacher'], kwargs=fresh_kwargs(item['source_teacher']['kwargs']))
        _, logits, _, _ = pose_teacher(controller, teacher, 17117, all_mask=True)
        values = {}
        for label, result in (('actor', actor), ('offline_source_root', oracle)):
            future = result['world'][:, 4:28]
            error = difference(rep, future, source[:, 4:28])
            raw_skin = exact_skin_metrics(geometry, result['world'][:, :28], case['scene'])
            skin = exact_skin_metrics(geometry, played_window(source[:, :4], result['world']), case['scene'])
            rms = float((future[..., :3] - source[:, 4:28, :3]).square().mean().sqrt())
            values[label] = dict(errors=error, root_xyz_rms_m=rms, skin=skin,
                raw_reconstruction_skin=raw_skin, geometry_contract=PUBLICATION_GEOMETRY,
                gait=summarize(rep, future, case['scene']), tokens=scores(result['pose_logits'][:, :8], example['tokens']),
                gate_pass=error['leg_relative_mpjpe_m'] <= .015 and rms <= .02
                    and skin['body_prism_max_m'] <= .02 and skin['unknown_body_fraction'] == 0
                    and skin['uncertain_inside_samples'] == 0)
        for label, world in (('source', source[:, :28]), ('actor', played_window(source[:, :4], actor['world'])),
                             ('offline_source_root', played_window(source[:, :4], oracle['world']))):
            out = folder / label; out.mkdir(exist_ok=True)
            pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
            import json
            np.savez_compressed(out / (case['label'] + '.npz'), local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(),
                roots=world[0, :, :3].cpu().numpy(), heightfield_y_m=np.asarray(case['scene']['heightfield_y_m']),
                min_point_world_xyz_m=np.asarray(case['scene']['min_point_world_xyz_m']), dx_m=np.float32(case['scene']['dx_m']),
                metadata_json=np.array(json.dumps(dict(role='unchanged_reference' if label == 'source' else 'generated',
                    actual_frames=28, quality_approved=False, offline_source_root=label == 'offline_source_root',
                    raw_reconstructed_history=False, postprocessing=None))))
        rows.append(dict(**item['window'], values=values, source_gait=summarize(rep, source[:, 4:28], case['scene']),
            source_root_teacher_all_mask=scores(logits, example['tokens'])))
    return rows


def main(a):
    if not 1 <= a.steps <= 300:
        raise ValueError('Native stage capacity tests are bounded to300updates each')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(17117)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    codec = torch.load(a.codec, map_location='cpu', weights_only=False)
    import json
    codec_report = json.loads(a.codec.with_name('complete.json').read_text())
    if not codec_report['capacity_gate_pass'] or codec['schema'] != SCHEMA:
        raise ValueError('Source codec capacity must pass before native Root/Pose fitting')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    initial = torch.load(a.initial, map_location='cpu', weights_only=False)
    if codec['contract']['initial_sha256'] != sha256(a.initial) or codec['contract']['bundle_sha256'] != sha256(a.bundle):
        raise ValueError('Stage checkpoint baseline changed')
    controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
    controller.codec.decoder.load_state_dict(codec['decoder'], strict=True)
    if a.stage == 'pose':
        if a.root is None:
            raise ValueError('Pose capacity requires a passed native Root stage')
        root_report = json.loads(a.root.with_name('complete.json').read_text())
        root_saved = torch.load(a.root, map_location='cpu', weights_only=False)
        if not root_report['capacity_gate_pass'] or root_saved['contract']['codec_sha256'] != sha256(a.codec):
            raise ValueError('Root stage did not pass or uses a different codec')
        controller.root.load_state_dict(root_saved['root'], strict=True)
        controller.adaptation.load_state_dict(root_saved['adaptation'], strict=True)
    controller.eval().requires_grad_(False)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    if sha256(a.cases) != codec['contract']['cases_sha256']:
        raise ValueError('Capacity source set changed')
    prepared = prepare(controller, cases, codec['contract']['source_windows'], a.device)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    baseline = root_review(controller, prepared) if a.stage == 'root' else pose_review(controller, prepared, geometry, a.output / 'baseline')
    module = controller.root if a.stage == 'root' else controller.pose
    module.requires_grad_(True)
    for key in (('root_stage1', 'root_stage2') if a.stage == 'root' else ('pose',)):
        controller.adaptation[key].requires_grad_(True)
    names = [n for n, p in controller.named_parameters() if p.requires_grad]
    protected_keys = [n for n in controller.state_dict() if n not in names]
    protected = state_digest(controller, protected_keys)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    contract = dict(schema=SCHEMA, stage=a.stage, initial_sha256=sha256(a.initial), bundle_sha256=sha256(a.bundle),
        codec_sha256=sha256(a.codec), root_sha256=sha256(a.root) if a.root else None,
        source_windows=codec['contract']['source_windows'], steps=a.steps, batch=4, exposures=a.steps * 4,
        trainable=names, trainable_count=sum(p.numel() for p in parameters),
        objective='native Root loss with causal inputs' if a.stage == 'root' else 'native Pose CE with source Root teacher as MB paper',
        actor_future_truth=False, pose_teacher_future_root=a.stage == 'pose',
        dropout='eval for explicit memorization test', padded_teacher_parity=[p['teacher_padding_logit_max'] for p in prepared],
        geometry_loss_added=False, fixed_case_capacity_only=True, quality_approved=False)
    write_json(a.output / 'contract.json', contract)
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    losses = []
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        values = []
        for index, item in enumerate(prepared):
            if a.stage == 'root':
                with pose_memory(controller, item['example']):
                    prediction = controller.root(*item['root_args'], **fresh_kwargs(item['root_kwargs']))
                loss = inference._root_model.loss(item['root_target'], prediction)['loss']
            else:
                teacher = dict(item['source_teacher'], kwargs=fresh_kwargs(item['source_teacher']['kwargs']))
                loss, _, _, _ = pose_teacher(controller, teacher, 17117 + step * 4 + index)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite native stage loss')
            (loss / 4).backward()
            values.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid native stage gradient')
        if any(p.grad is not None for p in controller.parameters() if not p.requires_grad):
            raise ValueError('Protected module acquired gradient')
        optimizer.step(); losses.append(float(np.mean(values)))
        if (step + 1) % 20 == 0:
            write_json(a.output / 'progress.json', dict(stage=a.stage, step=step + 1, exposures=(step + 1) * 4,
                mean_last20_loss=float(np.mean(losses[-20:]))))
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected modules changed during native stage')
    final = root_review(controller, prepared) if a.stage == 'root' else pose_review(controller, prepared, geometry, a.output / 'final')
    passed = all(r['gate_pass'] if a.stage == 'root' else r['values']['actor']['gate_pass'] for r in final)
    checkpoint = a.output / 'diagnostic_weights.pt'
    payload = dict(schema=SCHEMA, contract=contract, adaptation=controller.adaptation.state_dict(),
        root=controller.root.state_dict())
    if a.stage == 'pose':
        payload['pose'] = controller.pose.state_dict()
    torch.save(payload, checkpoint)
    write_json(a.output / 'complete.json', dict(status='complete', contract=contract, baseline=baseline, final=final,
        optimizer_updates=a.steps, losses=losses, protected_hash_unchanged=True, candidate_sha256=sha256(checkpoint),
        capacity_gate_pass=passed, quality_approved=False,
        limitation='Four-window memorization only. Passing does not certify closed-loop travel or novel commands/terrain.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'codec', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--root', type=Path)
    p.add_argument('--stage', choices=('root', 'pose'), required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
