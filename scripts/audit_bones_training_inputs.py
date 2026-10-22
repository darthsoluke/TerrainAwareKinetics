"""Audit actual sampled controls and played-history feature alignment on CPU."""
import argparse
from collections import defaultdict
from pathlib import Path

import torch

from motionbricks.data.bones_seed_rehearsal import BonesSeedPool
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_observed_history import reencode_played_history
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(args):
    torch.set_num_threads(4)
    rep = build_native_representation(args.stats)
    pool = BonesSeedPool(args.dataset, rep, args.stats)
    records = []
    feature_errors = defaultdict(list)
    for step in range(200):
        for window, control, scene, record in pool.sample(step):
            history = window[None, :4]
            actual = (history[:, 3, [0, 2]] - history[:, 0, [0, 2]]) * 10
            requested = control[:, :2]
            observed = reencode_played_history(history, rep)
            for name in ('global_root_pos', 'global_root_pos_2d', 'ric_data',
                         'global_rot_data', 'local_vel', 'foot_contacts'):
                index = rep.indices[name]
                feature_errors[name].append(float((observed[..., index] - history[..., index]).abs().max()))
            records.append(dict(record, update=step+1, speed=float(requested.norm()),
                initial_speed=float(actual.norm()),
                initial_heading=float(torch.atan2(history[0, 3, 4], history[0, 3, 3])),
                requested_heading=float(torch.atan2(control[0, 3], control[0, 2])),
                request_velocity=requested[0].tolist(),
                source_end_velocity=((window[31, [0, 2]]-window[30, [0, 2]])*30).tolist(),
                history_feature_error=float((observed-history).abs().max())))
    groups = {}
    for category in pool.groups:
        rows = [r for r in records if r['category'] == category]
        speed = torch.tensor([r['speed'] for r in rows])
        groups[category] = dict(count=len(rows), speed_quantiles=torch.quantile(
            speed, torch.tensor([0., .25, .5, .75, 1.])).tolist(),
            stationary_history=sum(r['initial_speed'] < .12 for r in rows),
            stationary_to_move=sum(r['initial_speed'] < .12 and r['speed'] > .3 for r in rows),
            zero_requests=sum(r['speed'] < 1e-5 for r in rows),
            requests_at_least5mps=int((speed >= 5).sum()))
    write_json(args.output, dict(groups=groups, records=records,
        feature_max={k:max(v) for k,v in feature_errors.items()},
        generated_history_training=False, persistent_stop_goal_training=False,
        optimizer_updates=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'stats', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
