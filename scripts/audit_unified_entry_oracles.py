"""Separate source-entry token selection and codec ability with OFFLINE oracles."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_decoder_pilot import training_targets
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_pose_training import load_pose_checkpoint, pose_teacher
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.audit_native_root_validation import source_root_intervention
from scripts.audit_unified_stair_gait import difference, generate, source_token_intervention, summarize
from scripts.check_unified_pose_preconditions import scores
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    saved = torch.load(a.candidate, map_location='cpu', weights_only=False)
    controller = UnifiedTerrainController(inference, bundle,
        pose_condition_stage=saved['contract'].get('pose_condition_stage', 'input'))
    load_pose_checkpoint(controller, saved, sha256(a.bundle))
    rep = inference.global_motion_rep
    frozen = state_digest(controller, list(controller.state_dict()))
    geometry = JointGeometryObjective(rep, a.device)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    archived = json.loads(a.reference_report.read_text())
    reference = {r['label']: r for r in archived['rows'] if r['arm'] == 'pose' and r['source_history']}
    rows = []
    for case in cases:
        if case.get('entry') not in ('up', 'down'):
            continue
        first = case['interval']['first_riser'] - case['interval']['start']
        for start in sorted({max(0, first - 16), max(0, first - 4)}):
            source = case['world'][None, start:start + 33].to(a.device)
            if source.shape[1] != 33:
                raise ValueError('Only complete real source windows')
            history, scene = source[:, :4], case['scene']
            command = source_terminal_goal(source[..., :5])
            actual = generate(controller, history, command, scene)
            target = training_targets(inference, source, history, actual['observation'])
            tokens = controller.codec.encode_into_idx(target['local_motions'], fetch_overall_indices=False)
            example = dict(result=actual, target=target, tokens=tokens, args=controller.last_pose_request[0],
                kwargs=controller.last_pose_request[1], memory=controller.memory, valid=controller.valid)
            _, teacher_logits, _, _ = pose_teacher(controller, example, 16115)
            _, mask_logits, _, _ = pose_teacher(controller, example, 16115, all_mask=True)
            with source_root_intervention(controller, target['global_motions'][..., :5]):
                correct_root = generate(controller, history, command, scene)
                root_example = dict(example, args=controller.last_pose_request[0], kwargs=controller.last_pose_request[1],
                    memory=controller.memory, valid=controller.valid)
                _, root_teacher, _, _ = pose_teacher(controller, root_example, 16115)
                _, root_mask, _, _ = pose_teacher(controller, root_example, 16115, all_mask=True)
            with source_token_intervention(inference, tokens):
                correct_tokens = generate(controller, history, command, scene, tokens)
                with source_root_intervention(controller, target['global_motions'][..., :5]):
                    correct_both = generate(controller, history, command, scene, tokens)
            replay = generate(controller, history, command, scene)
            torch.testing.assert_close(replay['world'], actual['world'], atol=0, rtol=0)
            label = f'{case["label"]}-start{start}'
            values = {}
            modes = dict(actor=actual, offline_source_root=correct_root,
                offline_source_tokens=correct_tokens, offline_source_root_tokens=correct_both)
            for arm, result in modes.items():
                future = result['world'][:, 4:28]
                skin = exact_skin_metrics(geometry, future, scene)
                values[arm] = dict(errors=difference(rep, future, source[:, 4:28]), skin=skin,
                    gait=summarize(rep, future, scene), token_scores=scores(result['pose_logits'][:, :8], tokens),
                    tokens_overridden=arm in ('offline_source_tokens', 'offline_source_root_tokens'))
                if arm == 'actor' and skin != reference[label]['future_skin']:
                    raise ValueError('Normal actor differs from archived factual entry review')
                pose = rep.inverse(result['world'][:, :28], False, return_all=True, joint_positions_from='global_rot_data')
                folder = a.output / 'recordings' / arm
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / (label + '.npz'),
                    local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(),
                    roots=result['world'][0, :28, :3].cpu().numpy(),
                    heightfield_y_m=np.asarray(scene['heightfield_y_m']),
                    min_point_world_xyz_m=np.asarray(scene['min_point_world_xyz_m']), dx_m=np.float32(scene['dx_m']),
                    metadata_json=np.array(json.dumps(dict(role='generated', offline_oracle=arm != 'actor',
                        actual_frames=28, quality_approved=False, postprocessing=None))))
            source_folder = a.output / 'recordings/source'
            source_folder.mkdir(parents=True, exist_ok=True)
            pose = rep.inverse(source[:, :28], False, return_all=True, joint_positions_from='global_rot_data')
            np.savez_compressed(source_folder / (label + '.npz'),
                local_rot_mats=pose['local_joint_rots'][0].cpu().numpy(), roots=source[0, :28, :3].cpu().numpy(),
                heightfield_y_m=np.asarray(scene['heightfield_y_m']),
                min_point_world_xyz_m=np.asarray(scene['min_point_world_xyz_m']), dx_m=np.float32(scene['dx_m']),
                metadata_json=np.array(json.dumps(dict(role='unchanged_reference', actual_frames=28, postprocessing=None))))
            rows.append(dict(label=label, entry=case['entry'], start=start, values=values,
                source_gait=summarize(rep, source[:, 4:28], scene),
                source_skin=exact_skin_metrics(geometry, source[:, 4:28], scene),
                teacher_predicted_root=scores(teacher_logits, tokens), teacher_source_root=scores(root_teacher, tokens),
                all_mask_predicted_root=scores(mask_logits, tokens), all_mask_source_root=scores(root_mask, tokens),
                actor_replay_difference=0., archived_actor_metrics_equal=True))
            write_json(a.output / 'progress.json', dict(rows=rows))
    if state_digest(controller, list(controller.state_dict())) != frozen:
        raise ValueError('Read-only oracle diagnosis changed weights')
    write_json(a.output / 'complete.json', dict(rows=rows, optimizer_updates=0, weights_unchanged=True,
        runtime_changed=False, candidate_sha256=sha256(a.candidate), cases_sha256=sha256(a.cases),
        scope='Source Root/tokens are OFFLINE oracle interventions. Actual actor stays causal. '
            'Token logits in token-oracle arms are diagnostic predictions before overriding IDs. '
            'No full-route or online improvement is inferred from any oracle result.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'candidate', 'cases', 'reference-report', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
