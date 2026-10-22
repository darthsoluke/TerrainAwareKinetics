"""Measure feet and source-certified stance on saved P1 trajectories; no inference."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset, sha256
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap, source_support_masks, SOURCE_SUPPORT_MASK_VERSION


def stats(values):
    if not values.numel():
        return dict(count=0, mean=None, p95=None)
    return dict(count=values.numel(), mean=float(values.mean()), p95=float(torch.quantile(values.flatten(), .95)))


@torch.no_grad()
def main(args):
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    rep = build_native_representation(args.stats)
    geometry = JointGeometryObjective(rep, 'cpu')
    report = dict(optimizer_updates=0, model_inference_calls=0, test_windows=0,
                  stance_mask_version=SOURCE_SUPPORT_MASK_VERSION, sets={})
    for split, directory in zip(('validation', 'train'), (args.validation, args.train)):
        dataset = PFNNFrameDataset(args.cache, args.pfnn, rep, split)
        by_id = {row['cache_row_index']: i for i, row in enumerate(dataset.records)}
        review = json.loads((directory / 'windows.json').read_text())
        groups = {}
        for mode in review['modes']:
            archive = directory / (mode + '.pt')
            raw = torch.load(archive, map_location='cpu', weights_only=False)
            rows = []
            for begin in range(0, len(raw['selection']), 8):
                selected = raw['selection'][begin:begin + 8]
                items = [dataset[by_id[row['cache_row_index']]] for row in selected]
                if any(item['record'] != row for item, row in zip(items, selected)):
                    raise ValueError('Saved trajectory does not match the fixed source identity')
                pred = raw['world'][begin:begin + 8]
                source = torch.stack([item['world'][:pred.shape[1]] for item in items])
                pj, pp = geometry.geometry(pred)
                sj, sp = geometry.geometry(source)
                feet = rep.skeleton.foot_joint_idx
                pr = pj - pred[..., :3].unsqueeze(-2)
                sr = sj - source[..., :3].unsqueeze(-2)
                pr, sr = pr[:, 3:, feet], sr[:, 3:, feet]
                pa = (pr.amax(1) - pr.amin(1)).norm(dim=-1).mean(-1)
                sa = (sr.amax(1) - sr.amin(1)).norm(dim=-1).mean(-1)
                pv = (torch.diff(pr, dim=1) * 30).square().mean((1, 2, 3)).sqrt()
                sv = (torch.diff(sr, dim=1) * 30).square().mean((1, 2, 3)).sqrt()
                for i, item in enumerate(items):
                    target = sp[i, 3:, :geometry.sole_count]
                    predicted = pp[i, 3:, :geometry.sole_count]
                    gap, valid = heightfield_gap(target, item['scene'])
                    active, intervals = source_support_masks(target, gap, valid)
                    pred_gap, pred_valid = heightfield_gap(predicted, item['scene'])
                    velocities = torch.diff(predicted, dim=0) * 30
                    source_velocities = torch.diff(target, dim=0) * 30
                    row = item['record']
                    rows.append(dict(cache_row_index=row['cache_row_index'], category=row['category'],
                        terrain=dataset.index.sources[row['source_id']]['terrain_construction']['kind'],
                        foot_span_m=float(pa[i]), source_foot_span_m=float(sa[i]),
                        foot_span_ratio=float(pa[i] / sa[i]) if sa[i] > .005 else None,
                        foot_velocity_rms_mps=float(pv[i]), source_foot_velocity_rms_mps=float(sv[i]),
                        foot_velocity_ratio=float(pv[i] / sv[i]) if sv[i] > .01 else None,
                        source_stance_velocity=stats(source_velocities.norm(dim=-1)[intervals]),
                        predicted_stance_velocity=stats(velocities.norm(dim=-1)[intervals]),
                        stance_velocity_error=stats((velocities - source_velocities).norm(dim=-1)[intervals]),
                        stance_gap_error=stats((pred_gap - gap).abs()[active & pred_valid]),
                        unknown_predicted_stance_points=int((active & ~pred_valid).sum())))
            grouped = defaultdict(list)
            for row in rows:
                for name in ('all', 'terrain:' + row['terrain'], 'category:' + row['category']):
                    grouped[name].append(row)
            summary = {}
            for name, samples in grouped.items():
                entry = dict(windows=len(samples))
                for key in ('foot_span_ratio', 'foot_velocity_ratio'):
                    values = [s[key] for s in samples if s[key] is not None]
                    entry[key] = dict(windows=len(values), median=float(np.median(values)) if values else None,
                                      p10=float(np.quantile(values, .1)) if values else None,
                                      p90=float(np.quantile(values, .9)) if values else None)
                for key in ('source_stance_velocity', 'predicted_stance_velocity', 'stance_velocity_error', 'stance_gap_error'):
                    count = sum(s[key]['count'] for s in samples)
                    entry[key] = dict(count=count, mean=sum(s[key]['mean'] * s[key]['count'] for s in samples
                                                            if s[key]['count']) / count if count else None)
                entry['unknown_predicted_stance_points'] = sum(s['unknown_predicted_stance_points'] for s in samples)
                summary[name] = entry
            groups[mode] = dict(trajectory_sha256=sha256(archive), rows=rows, groups=summary)
            print(json.dumps(dict(split=split, mode=mode, summary=summary['all'])), flush=True)
        report['sets'][split] = groups
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output / 'complete.json').write_text(json.dumps(dict(completed=True, optimizer_updates=0,
        model_inference_calls=0, script_sha256=sha256(__file__)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stats', 'cache', 'pfnn', 'validation', 'train', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
