"""Audit causal Root input/target alignment and actual codec composition, without updates."""
import argparse
from pathlib import Path

import torch

from motionbricks.data.played_motion_review import played_window
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.audit_unified_stair_gait import difference, generate, source_token_intervention
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_stages import prepare, root_review
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.parent, map_location='cpu', weights_only=False)
    if saved['contract']['bundle_sha256'] != sha256(a.bundle) or saved['contract']['cases_sha256'] != sha256(a.cases):
        raise ValueError('Source and bundle provenance changed')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True); controller.eval().requires_grad_(False)
    prepared = prepare(controller, torch.load(a.cases, map_location='cpu', weights_only=False),
                       saved['contract']['source_windows'], a.device)
    rows = root_review(controller, prepared)
    features = torch.cat([torch.cat([item['root_args'][j].flatten(1) for j in (0, 2, 4)]
                                  + [item['example']['memory'].flatten(1)], 1) for item in prepared])
    distances = torch.cdist(features.double(), features.double()) / features.shape[1] ** .5
    distances.fill_diagonal_(torch.inf)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    for index, (row, item) in enumerate(zip(rows, prepared)):
        args, target = item['root_args'], item['example']['target']
        expected_pose = extract_feature_from_motion_rep(target['local_motions'], inference.local_motion_rep,
                                                       inference.INTERNAL_POSE_FEATURE_MODE)
        checks = dict(global_root_history_max=float((args[0][:, :4] - item['root_target']['global_motions'][:, :4]).abs().max()),
            local_root_history_max=float((args[2][:, :3] - item['root_target']['local_motions'][:, :3]).abs().max()),
            local_pose_history_max=float((args[4][:, :4] - expected_pose[:, :4]).abs().max()),
            known_pose_frames=args[5][0].nonzero().flatten().tolist())
        neighbor = int(distances[index].argmin())
        row['input_alignment'] = checks
        row['nearest_condition'] = dict(window=prepared[neighbor]['window'],
            normalized_input_rms=float(distances[index, neighbor]),
            normalized_input_max=float((features[index] - features[neighbor]).abs().max()))
        with source_token_intervention(inference, item['example']['tokens']):
            output = generate(controller, item['source'][:, :4], item['example']['command'], item['case']['scene'],
                              item['example']['tokens'])
        future = output['world'][:, 4:28]
        errors = difference(inference.global_motion_rep, future, item['source'][:, 4:28])
        rms = float((future[..., :3] - item['source'][:, 4:28, :3]).square().mean().sqrt())
        skin = exact_skin_metrics(geometry, played_window(item['source'][:, :4], output['world']), item['case']['scene'])
        passed = errors['leg_relative_mpjpe_m'] <= .015 and rms <= .02 and skin['body_prism_max_m'] <= .02 \
            and skin['unknown_body_fraction'] == 0 and skin['uncertain_inside_samples'] == 0
        row['offline_source_token_chain'] = dict(errors=errors, root_xyz_rms_m=rms, skin=skin, gate_pass=passed)
        write_json(a.output / 'progress.json', dict(completed=index + 1, total=len(rows), last=row))
    write_json(a.output / 'complete.json', dict(schema='motionbricks.native_root_bank_chain_audit.v1',
        parent_sha256=sha256(a.parent), cases_sha256=sha256(a.cases), rows=rows,
        root_trajectory_pass=sum(row['gate_pass'] for row in rows),
        offline_source_token_chain_pass=sum(row['offline_source_token_chain']['gate_pass'] for row in rows),
        optimizer_updates=0, quality_approved=False, actual_pose_prediction_tested=False,
        note='Offline source-token causal isolation only; no changes to prior gates, actor, source or targets'))
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
