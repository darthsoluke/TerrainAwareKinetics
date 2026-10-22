"""Frozen final-model actor windows versus source autoencoder reconstruction."""

import argparse
import os
from pathlib import Path

from scripts.run_pfnn_parallel_workflow import gpu_snapshot
from scripts.run_pfnn_bias_workflow import save, read, digest


def main(a):
    snapshot, available = gpu_snapshot()
    if a.gpu not in available:
        raise ValueError('Selected diagnostic GPU is no longer idle')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    import torch
    from motionbricks.data.full_corpus_inference import inference_receipt, load_candidate, tensor_digest
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    from motionbricks.data.unified_pose_training import pose_example
    from motionbricks.data.terrain_joint_objective import JointGeometryObjective
    from motionbricks.data.terrain_decoder_pilot import vertical_origin
    from scripts.portable_parc_online_soma import load_terrain_bundle
    from scripts.probe_unified_native_codec import source_decode
    from scripts.check_unified_pose_preconditions import restore
    from scripts.audit_unified_stair_gait import difference, capture_tokens
    from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
    from scripts.audit_native_terrain_gaps import save_world

    torch.set_num_threads(4); torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    inference, bundle = load_terrain_bundle(a.bundle, 'cuda:0')
    controller = UnifiedTerrainController(inference, bundle)
    checkpoint_hash, bundle_hash = digest(a.checkpoint), digest(a.bundle)
    load_candidate(controller, saved, inference_receipt(saved, checkpoint_hash, checkpoint_hash, bundle_hash), bundle_hash)
    before = tensor_digest(controller.state_dict())
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    heldout = torch.load(a.evaluation/'heldout-cases.pt', map_location='cpu', weights_only=False)
    report = read(a.route_report)
    lookup = {case['label']: case for case in cases+heldout}
    selected = [next(case for case in cases if case.get('entry') == entry) for entry in ('up', 'down')]
    selected += [case for name in ('idle', 'walk') for case in cases if case.get('name') == name][:2]
    worst = sorted(report['routes']['rows'], key=lambda row: max(x['skin']['body_prism_max_m'] for x in row['rollouts']), reverse=True)
    selected += [lookup[row['label']] for row in worst[:2]]
    unique = {case['label']: case for case in selected}
    geometry = JointGeometryObjective(inference.global_motion_rep, 'cuda:0')
    rows = []
    try:
        with torch.no_grad():
            for case in unique.values():
                start = max(0, min(case['interval']['first_riser']-case['interval']['start']-4, len(case['world'])-33)) if 'interval' in case else 0
                source = case['world'][None, start:start+33].to('cuda:0')
                with capture_tokens(controller.codec) as actor_tokens:
                    example = pose_example(controller, source, case['scene'])
                actor = example['result']['world']
                encoded = source_decode(controller, example, full_encoder=True)['recon_state']
                indexed = source_decode(controller, example)['recon_state']
                reconstruction = restore(inference, indexed, source, example['result']['observation'])
                planner_y = inference.global_motion_rep.unnormalize(
                    example['result']['root_output']['pred_global_root_values'][:, :32])[..., 1]
                planner_y += vertical_origin(inference, source[:, :4], example['result']['observation'])[:, None]
                row = dict(label=case['label'], start=start,
                    selection='fixed legacy examples plus the two largest observed route penetrations',
                    encoder_vs_index_decoder_max_abs=float((encoded-indexed).abs().max()),
                    actor_source_token_agreement=float((actor_tokens[0][:, :8] == example['tokens']).float().mean()),
                    planner_root_y_mae_m=float((planner_y[:, 4:28]-source[:, 4:28, 1]).abs().mean()),
                    planner_root_y_max_error_m=float((planner_y[:, 4:28]-source[:, 4:28, 1]).abs().max()))
                for name, world in [('actor', actor), ('source_autoencoder', reconstruction)]:
                    played = torch.cat((source[:, :4], world[:, 4:28]), 1)
                    row[name] = dict(errors=difference(inference.global_motion_rep, world[:, 4:28], source[:, 4:28]),
                        root_xyz_rms_m=float((world[:, 4:28, :3]-source[:, 4:28, :3]).square().mean().sqrt()),
                        skin=exact_skin_metrics(geometry, played, case['scene']))
                    save_world(a.output/'recordings', case['label']+'-'+name, played, case['scene'],
                        'actual_actor' if name == 'actor' else 'source_root_and_source_tokens_autoencoder_only_not_actor')
                rows.append(row);save(a.output/'progress.json',dict(rows=rows,actual_optimizer_updates=0))
        if tensor_digest(controller.state_dict()) != before:
            raise ValueError('Diagnostic changed model tensors')
        save(a.output/'complete.json',dict(completed=True,rows=rows,weights_unchanged=True,
            actual_optimizer_updates=0,checkpoint_sha256=checkpoint_hash,gpu=a.gpu,gpu_snapshot_before=snapshot,
            source_autoencoder_is_not_actor=True,actor_root_and_pose_inputs_unmodified=True,
            quality_approved=False,delivery_approved=False))
    finally:
        controller.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','checkpoint','cases','evaluation','route-report','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    try:main(args)
    except Exception as error:
        save(args.output/'failure.json',dict(error=repr(error),actual_optimizer_updates=0));raise
