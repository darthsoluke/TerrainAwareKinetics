"""Separate reconstructed past from actual published-prefix/future geometry."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.terrain_distance_volume import swept_surface_points
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_codec import source_decode
from scripts.probe_unified_native_stages import prepare
from scripts.check_unified_pose_preconditions import restore
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.weights, map_location='cpu', weights_only=False)
    report = json.loads(a.weights.with_name('complete.json').read_text())
    by_key = {f'{w["label"]}-f{w["start"]}': w for w in saved['contract']['source_windows']}
    failures = [row for row in report['final'] if not row['capacity_gate_pass']]
    windows = [by_key[row['label']] for row in failures]
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True); controller.eval().requires_grad_(False)
    prepared = prepare(controller, torch.load(a.cases, map_location='cpu', weights_only=False), windows, a.device)
    geometry = JointGeometryObjective(inference.global_motion_rep, a.device)
    rows = []
    for item, old in zip(prepared, failures):
        local = source_decode(controller, item['example'])['recon_state']
        world = restore(inference, local, item['source'], item['example']['result']['observation'])[:, :28]
        raw = exact_skin_metrics(geometry, world, item['case']['scene'])
        if abs(raw['body_prism_max_m'] - old['skin']['body_prism_max_m']) > 1e-6:
            raise ValueError('Archived reconstruction mismatch')
        played = torch.cat((item['source'][:, :4], world[:, 4:]), 1)
        actual = exact_skin_metrics(geometry, played, item['case']['scene'])
        pose = inference.global_motion_rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
        points = geometry.skin.skin(pose['local_joint_rots'][0], world[0, :, :3, None].transpose(-1, -2).expand(-1, 30, -1), rot_is_global=False)
        swept = swept_surface_points(points, 8)
        gap, known = heightfield_gap(swept, item['case']['scene'])
        inside = known & (gap < 0); depth = torch.zeros_like(gap)
        distance, _, certified = HeightfieldPrismSDF(item['case']['scene']).query(swept[inside])
        depth[inside] = torch.where(certified, (-distance).clamp_min(0), -gap[inside])
        peaks = depth.amax(-1)
        worst = int(peaks.argmax())
        indices = torch.argsort(peaks, descending=True)[:8]
        rows.append(dict(**item['window'], raw_skin=raw, played_skin=actual,
            worst_fractional_frame=worst / 8, worst_vertex=int(depth[worst].argmax()),
            top_frames=[dict(frame=float(i) / 8, depth_m=float(peaks[i])) for i in indices],
            future_skin=exact_skin_metrics(geometry, world[:, 4:], item['case']['scene']),
            raw_prefix_skin=exact_skin_metrics(geometry, world[:, :4], item['case']['scene'])))
    write_json(a.output / 'complete.json', dict(rows=rows, optimizer_updates=0, weights_sha256=sha256(a.weights),
        runtime_changed=False, scope='Actual playback uses observed history4then generated future. '
        'This audit distinguishes past reconstruction from published geometry without modifying any future frame.'))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'weights', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.exists():
            write_json(args.output / 'failure.json', dict(error=repr(error)))
        raise
