"""Verify full real source route and exact skin before extending collection."""
import argparse
import json
from pathlib import Path
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.pilot_joint_sdf_full_context import exact_skin_metrics
from scripts.prepare_unified_pose_cases import stair_intervals
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    rep = build_native_representation(a.dataset / 'stats')
    data = TerrainLearningDataset(a.dataset, rep, 'train')
    geometry = JointGeometryObjective(rep, a.device)
    selected = {r['source_id'] for r in json.loads(a.selection_catalog.read_text())['rows'] if r['disposition'] == 'admitted'}
    rows = []
    for index, record in enumerate(data.records):
        if record['source_id'] not in selected:
            continue
        item = data.full_reference_item(index)
        world, scene = item['world'].to(a.device), item['scene']
        metrics = exact_skin_metrics(geometry, world[None], scene)
        gap, known = heightfield_gap(world[:, :3], scene)
        ground = world[:, 1] - gap
        changes = torch.diff(ground)
        up = (changes > .05).nonzero().flatten() + 1
        down = (changes < -.05).nonzero().flatten() + 1
        intervals = stair_intervals(world.cpu(), scene)
        velocity = torch.diff(world[:, [0, 2]], dim=0).norm(dim=-1) * 30
        displacement = world[-1, [0, 2]] - world[0, [0, 2]]
        direction = displacement / displacement.norm()
        side = torch.stack((direction[1], -direction[0]))
        lateral = float(((world[:, [0, 2]] - world[0, [0, 2]]) @ side).abs().max())
        complete = bool(len(up) >= 3 and len(down) >= 3 and up[0] < down[0]
                        and (ground[-24:] - ground[0]).abs().max() <= .03 and known.all())
        clean = metrics['body_prism_max_m'] <= .02 and metrics['unknown_body_fraction'] == 0 and metrics['uncertain_inside_samples'] == 0
        row = dict(source_id=record['source_id'], frames=len(world), metrics=metrics, root_route_complete=complete,
            rising_edges=len(up), descending_edges=len(down), initial_flat_frames=int(up[0]) if len(up) else 0,
            flat_exit_frames=len(world) - int(down[-1]) if len(down) else 0, lateral_max_m=lateral,
            terminal_speed_quantiles_mps=torch.quantile(velocity, velocity.new_tensor([.1, .5, .9])).tolist(),
            eligible_real_entry_intervals=intervals, full_route_exact_skin_pass=clean,
            ready_for_source_window_selection=bool(complete and clean and lateral <= .35 and intervals),
            training_started=False)
        rows.append(row)
        write_json(a.output / 'progress.json', dict(rows=rows))
    write_json(a.output / 'complete.json', dict(rows=rows, source_manifest_sha256=sha256(a.dataset / 'native_manifest.json'),
        ready_records=sum(r['ready_for_source_window_selection'] for r in rows),
        independent_terrain_groups=len({r['source_group'] for r in data.records if r['source_id'] in selected}), optimizer_updates=0,
        scope='Real physically tracked/retargeted sources. Exact prism/sampled skin is not proof of continuous triangle-mesh contact forces.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'output', 'selection-catalog'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
