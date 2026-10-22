"""First native-stage gate: four real windows, source codec reconstruction only."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_pose_training import pose_example
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.played_motion_review import played_window, PUBLICATION_GEOMETRY
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.audit_native_root_validation import source_root_intervention
from scripts.audit_unified_stair_gait import difference, generate, source_token_intervention, summarize
from scripts.check_unified_pose_preconditions import restore
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@contextmanager
def source_codec_context(controller, example):
    previous = (controller.memory, controller.valid, controller.last_root, controller.frames, controller._adapting)
    controller.memory, controller.valid = example['memory'], example['valid']
    controller.last_root = dict(pred_global_root_values=example['target']['global_motions'][..., :5])
    controller.frames, controller._adapting = 32, True
    try:
        yield
    finally:
        controller.memory, controller.valid, controller.last_root, controller.frames, controller._adapting = previous


def source_decode(controller, example, *, full_encoder=False):
    local = example['target']['local_motions']
    root = extract_feature_from_motion_rep(local, controller.codec.motion_rep, controller.codec.decoder_external_cond_feature_mode)
    mask = torch.zeros(local.shape[:2], dtype=torch.bool, device=local.device)
    mask[:, :4] = True
    with source_codec_context(controller, example):
        if full_encoder:
            return controller.codec(local, target_cond=local, has_target_cond=mask, external_cond=root)
        result = controller.codec.forward_decoder(example['tokens'], target_cond=local,
            has_target_cond=mask, external_cond=root, token_mask=torch.ones(example['tokens'].shape[:2],
                dtype=torch.bool, device=local.device), use_overall_indices=False)
    return dict(result, l_commit=result['recon_state'].sum() * 0, perplexity=0.)


@torch.no_grad()
def evaluate(controller, prepared, geometry, folder):
    folder.mkdir(parents=True, exist_ok=False)
    rep = controller.inference.global_motion_rep
    rows = []
    for case, source, example in prepared:
        local = source_decode(controller, example)['recon_state']
        world = restore(controller.inference, local, source, example['result']['observation'])
        error = difference(rep, world[:, 4:28], source[:, 4:28])
        rms = float((world[:, 4:28, :3] - source[:, 4:28, :3]).square().mean().sqrt())
        raw_skin = exact_skin_metrics(geometry, world[:, :28], case['scene'])
        played = played_window(source[:, :4], world)
        skin = exact_skin_metrics(geometry, played, case['scene'])
        source_skin = exact_skin_metrics(geometry, source[:, :28], case['scene'])
        passed = (error['leg_relative_mpjpe_m'] <= .015 and rms <= .02
            and skin['body_prism_max_m'] <= .02 and skin['unknown_body_fraction'] == 0
            and skin['uncertain_inside_samples'] == 0)
        row = dict(label=case['label'], start=case['probe_start'], errors=error, root_xyz_rms_m=rms,
            skin=skin, raw_reconstruction_skin=raw_skin, geometry_contract=PUBLICATION_GEOMETRY,
            source_skin=source_skin, gait=summarize(rep, world[:, 4:28], case['scene']),
            source_gait=summarize(rep, source[:, 4:28], case['scene']), capacity_gate_pass=passed)
        rows.append(row)
        for arm, value in (('source', source[:, :28]), ('reconstruction', played), ('raw_reconstruction', world[:, :28])):
            pose = rep.inverse(value, False, return_all=True, joint_positions_from='global_rot_data')
            out = folder / arm
            out.mkdir(exist_ok=True)
            np.savez_compressed(out / (case['label'] + '.npz'),
                local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(), roots=value[0, :, :3].cpu().numpy(),
                heightfield_y_m=np.asarray(case['scene']['heightfield_y_m']),
                min_point_world_xyz_m=np.asarray(case['scene']['min_point_world_xyz_m']), dx_m=np.float32(case['scene']['dx_m']),
                metadata_json=np.array(json.dumps(dict(row, role='unchanged_reference' if arm == 'source' else 'generated',
                    offline_source_root_tokens=True, actual_frames=28, raw_reconstructed_history=arm == 'raw_reconstruction',
                    quality_approved=False, postprocessing=None))))
    return rows


def main(a):
    if not 1 <= a.steps <= 300:
        raise ValueError('This capability test is bounded to300updates')
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17116)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    initial = torch.load(a.initial, map_location='cpu', weights_only=False)
    if initial['contract']['bundle_sha256'] != sha256(a.bundle):
        raise ValueError('Initial bundle mismatch')
    controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
    controller.eval().requires_grad_(False)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    selected = [next(c for c in cases if c.get('entry') == entry) for entry in ('up', 'down')]
    selected += [next(c for c in cases if c.get('name') == name) for name in ('idle', 'walk')]
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    prepared, alignment = [], []
    with torch.no_grad():
        for case in selected:
            start = case['interval']['first_riser'] - case['interval']['start'] - 4 if 'interval' in case else 0
            source = case['world'][None, start:start + 33].to(a.device)
            example = pose_example(controller, source, case['scene'])
            # Native autoencoder loss supervises every real frame, including the played prefix.
            example['target'] = {k: v for k, v in example['target'].items()
                                 if k not in ('motion_frame_mask', 'motion_interval_mask')}
            encoded = source_decode(controller, example, full_encoder=True)['recon_state']
            indexed = source_decode(controller, example)['recon_state']
            with source_root_intervention(controller, example['target']['global_motions'][..., :5]), \
                    source_token_intervention(inference, example['tokens']):
                oracle = generate(controller, source[:, :4], example['command'], case['scene'], example['tokens'])
            ae_world = restore(inference, indexed, source, example['result']['observation'])
            row = dict(label=case['label'], start=start,
                encode_vs_index_local_max=float((encoded - indexed).abs().max()),
                source_decode_vs_runtime_local_max=float((indexed[:, :28] - oracle['local_prediction'][:, :28]).abs().max()),
                source_decode_vs_runtime_world_max=float((ae_world[:, :28] - oracle['world'][:, :28]).abs().max()))
            alignment.append(row)
            prepared.append((dict(case, probe_start=start), source, example))
    write_json(a.output / 'alignment.json', dict(rows=alignment))
    if any(max(r[k] for k in ('encode_vs_index_local_max', 'source_decode_vs_runtime_local_max',
                             'source_decode_vs_runtime_world_max')) > 5e-5 for r in alignment):
        write_json(a.output / 'complete.json', dict(status='alignment_failed_before_optimizer',
            optimizer_updates=0, rows=alignment, quality_approved=False))
        return
    baseline = evaluate(controller, prepared, geometry, a.output / 'baseline')
    controller.codec.decoder.requires_grad_(True)
    protected_keys = [k for k in controller.state_dict() if not k.startswith('codec.decoder.')]
    protected = state_digest(controller, protected_keys)
    contract = dict(schema='motionbricks.unified.native_stage_capacity.v1', stage='codec',
        initial_sha256=sha256(a.initial), bundle_sha256=sha256(a.bundle), cases_sha256=sha256(a.cases),
        steps=a.steps, batch=4, exposures=a.steps * 4, teacher_only=True, full_campaign=False,
        trainable='native codec.decoder only', source_windows=[dict(label=c['label'], start=c['probe_start']) for c, _, _ in prepared],
        objective='unchanged native MotionVQVAE.loss, all32real frames and31intervals, source Root + first4poses',
        frozen='native Root/Pose, encoder, codebook, every unified adapter',
        source_root='training autoencoder only, never actual actor', runtime_changed=False,
        gate=dict(leg_relative_mpjpe_m=.015, root_xyz_rms_m=.02, skin_peak_m=.02), quality_approved=False)
    write_json(a.output / 'contract.json', contract)
    parameters = list(controller.codec.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-5)
    losses = []
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        values = []
        for _, _, example in prepared:
            output = source_decode(controller, example)
            loss = controller.loss_model.loss(example['target'], output, None)['loss']
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite source reconstruction')
            (loss / 4).backward()
            values.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid decoder gradient')
        if any(p.grad is not None for p in controller.parameters() if not p.requires_grad):
            raise ValueError('Frozen parameter acquired gradient')
        optimizer.step()
        losses.append(float(np.mean(values)))
        if (step + 1) % 20 == 0:
            write_json(a.output / 'progress.json', dict(stage='native_source_codec_fit', step=step + 1,
                exposures=(step + 1) * 4, mean_last20_loss=float(np.mean(losses[-20:]))))
    if state_digest(controller, protected_keys) != protected:
        raise ValueError('Protected modules changed')
    checkpoint = a.output / 'diagnostic_weights.pt'
    torch.save(dict(schema=contract['schema'], contract=contract, decoder=controller.codec.decoder.state_dict()), checkpoint)
    final = evaluate(controller, prepared, geometry, a.output / 'final')
    write_json(a.output / 'complete.json', dict(status='complete', contract=contract, baseline=baseline, final=final,
        alignment=alignment, optimizer_updates=a.steps, primitive_exposures=a.steps * 4, losses=losses,
        frozen_hash_unchanged=True, candidate_sha256=sha256(checkpoint),
        capacity_gate_pass=all(r['capacity_gate_pass'] for r in final), quality_approved=False,
        limitation='Deliberate four-window memorization diagnostic. No generalization, full-route or WASD claim.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
