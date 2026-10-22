"""Preserved28941 versus full native-bank actor on unchanged flat WASD schedules."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_unified_flat import runs
from scripts.review_unified_native_feedback import support_diagnostics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.candidate, map_location='cpu', weights_only=False)
    initial = torch.load(a.initial, map_location='cpu', weights_only=False)
    if (saved['schema'] != 'motionbricks.unified.native_route_curriculum.v1'
            or saved['contract']['stage'] != 'pose' or saved['contract']['bundle_sha256'] != sha256(a.bundle)
            or saved['contract']['cases_sha256'] != sha256(a.cases)
            or not json.loads(a.candidate.with_name('complete.json').read_text())['training_bank_gate_pass']
            or sha256(a.initial) != 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979'):
        raise ValueError('Native bank/baseline provenance differs')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.adaptation.load_state_dict(initial['adaptation'], strict=True)
    controller.eval().requires_grad_(False)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    idle = next(case for case in cases if case['name'] == 'idle')
    height = idle['scene']['heightfield_y_m']
    if not torch.isfinite(height).all() or not height.eq(height[0, 0]).all():
        raise ValueError('Real idle seed must have its known flat plane')
    floor = float(height[0, 0])
    scene = dict(heightfield_y_m=torch.full((201, 201), floor), min_point_world_xyz_m=torch.tensor([-50., floor, -50.]), dx_m=torch.tensor(.5))
    seed = inference.global_motion_rep.change_first_heading(idle['world'][None, :4].to(a.device), 0., False, False)
    seed[..., [0, 2]] -= seed[:, -1:, [0, 2]].clone()
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    reports = {}
    for arm in ('before', 'native'):
        if arm == 'native':
            controller.load_state_dict(saved['model_state'], strict=True)
        write_json(a.output / 'state.json', dict(arm=arm, stage='rollouts'))
        rows = runs(inference, controller, seed, scene, a.output / arm)
        for index, row in enumerate(rows):
            with np.load(a.output / arm / (row['case'] + '.npz'), allow_pickle=False) as data:
                world = torch.from_numpy(data['global_features'].copy())[None].to(a.device)
            row['skin'] = exact_skin_metrics(geometry, world, scene)
            row['support'] = support_diagnostics(geometry, world, scene)
            write_json(a.output / 'state.json', dict(arm=arm, stage='geometry', completed=index + 1, total=len(rows)))
        reports[arm] = rows
        write_json(a.output / (arm + '-report.json'), dict(rows=rows))
    write_json(a.output / 'complete.json', dict(**reports, initial_sha256=sha256(a.initial),
        candidate_sha256=sha256(a.candidate), seed_source=idle['label'], source_floor_m=floor,
        optimizer_updates=0, no_padding=True, quality_approved=False,
        note='All17four-yaw/start-after-idle/run/turn/release schedules and full sampled-skin geometry retained. '
             'Foot range/support diagnostics alone do not certify natural gait or contact forces.'))
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'initial', 'candidate', 'cases', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
