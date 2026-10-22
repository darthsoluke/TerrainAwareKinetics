"""Convergence-gated native curriculum over complete real up/down and BONES clips."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_pose_training import pose_memory, pose_teacher
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.played_motion_review import played_window, PUBLICATION_GEOMETRY
from motionbricks.data.native_codec_fidelity import source_fidelity_losses, VERSION as FIDELITY_VERSION
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.audit_unified_stair_gait import difference, generate, source_token_intervention
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.probe_unified_native_codec import source_decode, evaluate as codec_review
from scripts.probe_unified_native_stages import prepare, fresh_kwargs, root_review, pose_review
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


SCHEMA = 'motionbricks.unified.native_route_curriculum.v1'


def balanced_schedule(prepared, steps):
    groups, entries = {}, {}
    for index, item in enumerate(prepared):
        label = item['window']['label']
        groups.setdefault(label, []).append(index)
        entries[label] = item['case'].get('entry', 'flat')
    up = [k for k in groups if entries[k] == 'up']
    down = [k for k in groups if entries[k] == 'down']
    flat = [k for k in groups if entries[k] == 'flat']
    if len(up) != 1 or len(down) != 1 or len(flat) != 6:
        raise ValueError('Route curriculum requires one up, one down and six real BONES clips')
    visits = Counter(); result = []
    for step in range(steps):
        labels = [up[0], down[0], flat[(2 * step) % len(flat)], flat[(2 * step + 1) % len(flat)]]
        batch = []
        for label in labels:
            # Advance each source independently; global-step modulo aliases with source selection.
            batch.append(groups[label][visits[label] % len(groups[label])])
            visits[label] += 1
        result.append(batch)
    return result


def main(a):
    if not 100 <= a.max_steps <= 1000 or a.max_steps % 100:
        raise ValueError('Use a bounded100..1000updates per native stage')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'checks').mkdir()
    torch.set_num_threads(4); torch.manual_seed(17118)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    if a.fidelity and (a.stage != 'codec' or a.fidelity_preflight is None):
        raise ValueError('Source fidelity is an explicit gated codec-only objective')
    if a.stage == 'codec' and a.continue_codec:
        previous = torch.load(a.continue_codec, map_location='cpu', weights_only=False)
        if (previous['schema'] != SCHEMA or previous['contract']['stage'] != 'codec'
                or previous['contract']['cases_sha256'] != sha256(a.cases)
                or previous['contract']['bundle_sha256'] != sha256(a.bundle)):
            raise ValueError('Codec continuation identity differs')
        controller.load_state_dict(previous['model_state'], strict=True)
        windows, anchors = previous['contract']['source_windows'], previous['contract']['capacity_anchors']
        parent_hash = sha256(a.continue_codec)
    elif a.stage == 'codec':
        if a.parent_pose is None or a.parent_codec is None:
            raise ValueError('First route stage must continue the passed native capacity model')
        parent = torch.load(a.parent_pose, map_location='cpu', weights_only=False)
        codec = torch.load(a.parent_codec, map_location='cpu', weights_only=False)
        if (parent['contract']['codec_sha256'] != sha256(a.parent_codec)
                or codec['contract']['cases_sha256'] != sha256(a.cases)
                or parent['contract']['bundle_sha256'] != sha256(a.bundle)):
            raise ValueError('Parent chain identity differs')
        if not json.loads(a.parent_pose.with_name('complete.json').read_text())['capacity_gate_pass']:
            raise ValueError('Four-window native capacity must pass before expansion')
        controller.codec.decoder.load_state_dict(codec['decoder'], strict=True)
        controller.root.load_state_dict(parent['root'], strict=True)
        controller.pose.load_state_dict(parent['pose'], strict=True)
        controller.adaptation.load_state_dict(parent['adaptation'], strict=True)
        anchors = parent['contract']['source_windows']
        selected_stairs = {r['label'] for r in anchors if 'bones:' not in r['label']}
        selected = [c for c in cases if c['label'] in selected_stairs or c['row']['origin'] == 'bones_seed']
        anchor_map = {r['label']: r['start'] for r in anchors}
        windows = [dict(label=c['label'], start=start) for c in selected for start in
            sorted(set(range(0, len(c['world']) - 32, 8)) | {len(c['world']) - 33, anchor_map.get(c['label'], 0)})]
        parent_hash = sha256(a.parent_pose)
    else:
        if a.previous is None:
            raise ValueError('The preceding native route stage is required')
        previous = torch.load(a.previous, map_location='cpu', weights_only=False)
        prior_report = json.loads(a.previous.with_name('complete.json').read_text())
        upstream_pass = prior_report['training_bank_gate_pass']
        if a.previous_gate:
            gate = json.loads(a.previous_gate.read_text())
            if (a.stage != 'root' or gate['schema'] != 'motionbricks.codec_publication_gate.v1'
                    or gate['geometry_contract'] != PUBLICATION_GEOMETRY
                    or gate['source_checkpoint_sha256'] != sha256(a.previous)
                    or gate['cases_sha256'] != sha256(a.cases)
                    or gate['source_windows'] != previous['contract']['source_windows']
                    or len(gate['rows']) != len(gate['source_windows'])):
                raise ValueError('Publication gate does not bind the complete preceding codec bank')
            upstream_pass = gate['training_bank_gate_pass'] and all(
                row['errors']['leg_relative_mpjpe_m'] <= .015 and row['root_xyz_rms_m'] <= .02
                and row['skin']['body_prism_max_m'] <= .02 and row['skin']['unknown_body_fraction'] == 0
                and row['skin']['uncertain_inside_samples'] == 0 for row in gate['rows'])
        expected = 'codec' if a.stage == 'root' else 'root'
        if (previous['schema'] != SCHEMA or previous['contract']['stage'] != expected
                or not upstream_pass
                or previous['contract']['cases_sha256'] != sha256(a.cases)
                or previous['contract']['bundle_sha256'] != sha256(a.bundle)):
            raise ValueError('Preceding stage did not pass or has incompatible inputs')
        controller.load_state_dict(previous['model_state'], strict=True)
        windows, anchors = previous['contract']['source_windows'], previous['contract']['capacity_anchors']
        parent_hash = sha256(a.previous)
    controller.eval().requires_grad_(False)
    if a.fidelity:
        gate = json.loads(a.fidelity_preflight.read_text())
        if (not gate['all_checks_passed'] or gate['optimizer_updates']
                or gate['parent_sha256'] != parent_hash or gate['cases_sha256'] != sha256(a.cases)):
            raise ValueError('Fidelity gradient gate does not match this source/model')
    prepared = prepare(controller, cases, windows, a.device)
    for item in prepared:
        item['case'] = dict(item['case'], label=item['window']['label'] + f'-f{item["window"]["start"]}')
        item['example']['target'] = {k: v for k, v in item['example']['target'].items()
                                    if k not in ('motion_frame_mask', 'motion_interval_mask')}
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)

    def review(folder):
        if a.stage == 'codec':
            rows = codec_review(controller, [(dict(i['case'], probe_start=i['window']['start']),
                i['source'], i['example']) for i in prepared], geometry, folder)
            passed = all(r['capacity_gate_pass'] for r in rows)
        elif a.stage == 'root':
            rows = root_review(controller, prepared); passed = all(r['gate_pass'] for r in rows)
            if passed:
                # A numerically accurate Root must also support the codec's actual source tokens.
                with torch.no_grad():
                    for row, item in zip(rows, prepared):
                        with source_token_intervention(inference, item['example']['tokens']):
                            result = generate(controller, item['source'][:, :4], item['example']['command'],
                                item['case']['scene'], item['example']['tokens'])
                        skin = exact_skin_metrics(geometry, played_window(item['source'][:, :4], result['world']), item['case']['scene'])
                        errors = difference(inference.global_motion_rep, result['world'][:, 4:28], item['source'][:, 4:28])
                        chain_pass = errors['leg_relative_mpjpe_m'] <= .015 and skin['body_prism_max_m'] <= .02 \
                            and skin['unknown_body_fraction'] == 0 and skin['uncertain_inside_samples'] == 0
                        row['offline_source_token_chain'] = dict(skin=skin, errors=errors, gate_pass=chain_pass)
                        row['gate_pass'] = row['gate_pass'] and chain_pass
                passed = all(r['gate_pass'] for r in rows)
        else:
            rows = pose_review(controller, prepared, geometry, folder)
            passed = all(r['values']['actor']['gate_pass'] for r in rows)
        return rows, passed

    baseline, _ = review(a.output / 'baseline')
    if a.stage == 'codec':
        controller.codec.decoder.requires_grad_(True)
    elif a.stage == 'root':
        controller.root.requires_grad_(True)
        for key in ('root_stage1', 'root_stage2'):
            controller.adaptation[key].requires_grad_(True)
    else:
        controller.pose.requires_grad_(True); controller.adaptation['pose'].requires_grad_(True)
    names = [n for n, p in controller.named_parameters() if p.requires_grad]
    protected_keys = [n for n in controller.state_dict() if n not in names]
    protected = state_digest(controller, protected_keys)
    parameters = [p for p in controller.parameters() if p.requires_grad]
    schedule = balanced_schedule(prepared, a.max_steps)
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    contract = dict(schema=SCHEMA, stage=a.stage, bundle_sha256=sha256(a.bundle), cases_sha256=sha256(a.cases),
        parent_sha256=parent_hash, max_updates=a.max_steps, batch=4, max_exposures=4 * a.max_steps,
        upstream_publication_receipt_sha256=sha256(a.previous_gate) if a.previous_gate else None,
        source_windows=windows, capacity_anchors=anchors, unique_clips=len({w['label'] for w in windows}),
        overlapping_windows=len(windows), source_frames_modified=False, schedule_sha256=schedule_hash,
        trainable=names, trainable_count=sum(p.numel() for p in parameters),
        geometry_objective_added=False, native_coefficients_unchanged=True, source_root_pose_teacher=a.stage == 'pose',
        root_gate_includes_source_token_codec_chain=a.stage == 'root',
        source_fidelity_objective=FIDELITY_VERSION if a.fidelity else None,
        source_fidelity_coefficients=dict(global_root=float(inference._root_model.args['global_root_loss_coeff']),
            local_root=float(inference._root_model.args['local_root_loss_coeff']), rendered_fk=1.) if a.fidelity else None,
        actual_actor_future_truth=False, full_campaign=False, quality_approved=False,
        stop_rule='Check the complete training bank every100updates; advance stages only when all training gates pass',
        limits='Training bank convergence is not closed-loop or held-out quality approval')
    write_json(a.output / 'contract.json', contract)
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    losses, curve = [], []
    final, passed = [], False
    for step, batch in enumerate(schedule):
        optimizer.zero_grad(set_to_none=True); values = []
        for slot, index in enumerate(batch):
            item = prepared[index]
            if a.stage == 'codec':
                result = source_decode(controller, item['example'])
                loss = controller.loss_model.loss(item['example']['target'], result, None)['loss']
                if a.fidelity:
                    fidelity = source_fidelity_losses(inference, result['recon_state'], item['example']['target'],
                        item['source'][:, :4], item['example']['result']['observation'])
                    loss = loss + fidelity['total']
            elif a.stage == 'root':
                with pose_memory(controller, item['example']):
                    result = controller.root(*item['root_args'], **fresh_kwargs(item['root_kwargs']))
                loss = inference._root_model.loss(item['root_target'], result)['loss']
            else:
                example = dict(item['source_teacher'], kwargs=fresh_kwargs(item['source_teacher']['kwargs']))
                loss, _, _, _ = pose_teacher(controller, example, 17118 + step * 4 + slot)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite native route objective')
            (loss / 4).backward(); values.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid native route gradients')
        if any(p.grad is not None for p in controller.parameters() if not p.requires_grad):
            raise ValueError('Frozen module acquired gradient')
        optimizer.step(); losses.append(float(np.mean(values)))
        if (step + 1) % 20 == 0:
            write_json(a.output / 'progress.json', dict(stage=a.stage, step=step + 1, max_steps=a.max_steps,
                primitive_exposures=(step + 1) * 4, mean_last20_loss=float(np.mean(losses[-20:])), gates=curve))
        if (step + 1) % 100 == 0:
            final, passed = review(a.output / 'checks' / f'step{step + 1:04d}')
            curve.append(dict(step=step + 1, all_training_windows_pass=passed))
            write_json(a.output / 'checks' / f'step{step + 1:04d}-report.json', dict(rows=final, passed=passed))
            if passed:
                break
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected weights changed')
    checkpoint = a.output / 'weights.pt'
    torch.save(dict(schema=SCHEMA, contract=contract, model_state=controller.state_dict(), actual_updates=step + 1), checkpoint)
    write_json(a.output / 'complete.json', dict(contract=contract, baseline=baseline, final=final, losses=losses, gates=curve,
        actual_updates=step + 1, primitive_exposures=(step + 1) * 4, training_bank_gate_pass=passed,
        protected_hash_unchanged=True, checkpoint_sha256=sha256(checkpoint), quality_approved=False,
        final_recordings=str(a.output / 'checks' / f'step{step + 1:04d}')))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    for key in ('parent-pose', 'parent-codec', 'previous', 'continue-codec', 'fidelity-preflight', 'previous-gate'):
        p.add_argument('--' + key, type=Path)
    p.add_argument('--stage', choices=('codec', 'root', 'pose'), required=True)
    p.add_argument('--fidelity', action='store_true')
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
