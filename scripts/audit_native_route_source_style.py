"""Describe real source stair foot alternation; no data/threshold modification."""
import argparse
from pathlib import Path
import torch

from motionbricks.data.native_mb_data import build_native_representation
from scripts.audit_unified_stair_gait import gait_features
from scripts.train_joint_terrain_v3 import write_json


def main(a):
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    rep = build_native_representation(a.stats)
    cases = torch.load(a.cases, map_location='cpu', weights_only=False)
    selected = [next(c for c in cases if c.get('entry') == kind) for kind in ('up', 'down')]
    rows = []
    with torch.no_grad():
        for case in selected:
            world = case['world'][None]
            values = gait_features(rep, world)
            axis = world[0, -6, [0, 2]] - world[0, 3, [0, 2]]; axis /= axis.norm()
            delta = values['world_feet'][0, :, 0, [0, 2]] - values['world_feet'][0, :, 1, [0, 2]]
            separation = delta @ axis
            first = case['interval']['first_riser'] - case['interval']['start']
            last = case['interval']['plateau'] - case['interval']['start']
            segment = separation[first:last + 1]
            signs = torch.where(segment > .1, 1, torch.where(segment < -.1, -1, 0))
            nonzero = signs[signs != 0]
            switches = int((nonzero[1:] != nonzero[:-1]).sum())
            rows.append(dict(label=case['label'], frames=world.shape[1], stair_root_frames=[first, last],
                recorded_riser_crossings=case['interval']['steps'],
                foot_foreaft_separation_abs_median_m=float(segment.abs().median()),
                close_feet_10cm_fraction=float((segment.abs() < .1).float().mean()),
                alternating_sign_changes=switches, signed_separation_m=segment.tolist(),
                source_group=case['row']['source_group'], terrain_family=case['row'].get('terrain_family'),
                limitation='Foot-spacing/sign diagnostics do not certify contact sequencing or aesthetic gait.'))
    write_json(a.output / 'complete.json', dict(rows=rows, source_modified=False, optimizer_updates=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('cases', 'stats', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    main(p.parse_args())
