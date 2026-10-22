"""Audit actual sole surfaces and reciprocal stair progression without editing motion."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def true_runs(flags, minimum=3):
    result, start = [], None
    for index, active in enumerate(list(flags) + [False]):
        if active and start is None:
            start = index
        if not active and start is not None:
            if index - start >= minimum:
                result.append((start, index))
            start = None
    return result


def reciprocal_failures(events, flight_levels, tolerance=.01):
    """Screen measured same-tread catch-ups only inside a known ascending flight."""
    rows, prior = [], None
    for event in sorted(events, key=lambda row: row['start_frame']):
        distances = np.abs(np.asarray(flight_levels) - event['surface_height_m'])
        level = int(distances.argmin())
        if distances[level] > tolerance:
            prior = None
            continue
        if prior is not None and event['side'] != prior['side']:
            previous_level = prior['flight_level']
            if 0 < previous_level < len(flight_levels)-1 and level <= previous_level:
                rows.append(dict(previous=prior, current=dict(event, flight_level=level),
                    reason='opposite_foot_did_not_reach_a_higher_tread_inside_flight'))
        prior = dict(event, flight_level=level)
    return rows


def longest_regular_ascent(heights, tolerance=.005):
    segments = []
    for value in heights:
        if not np.isfinite(value):
            if not segments or segments[-1] is not None:
                segments.append(None)
        elif not segments or segments[-1] is None or abs(value-segments[-1]) > tolerance:
            segments.append(float(value))
    best = []
    for start in range(len(segments)-1):
        if segments[start] is None or segments[start+1] is None:
            continue
        rise = segments[start+1]-segments[start]
        if rise <= .05:
            continue
        end = start+1
        while end+1 < len(segments) and segments[end+1] is not None and abs(segments[end+1]-segments[end]-rise) <= tolerance:
            end += 1
        if end-start >= 3 and end-start+1 > len(best):
            best = segments[start:end+1]
    if not best:
        raise ValueError('No regular ascending flight found; never infer stair labels on a flat/unknown scene')
    return best


def main(args):
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from motionbricks.data.native_mb_data import build_native_representation
    from motionbricks.data.terrain_joint_objective import JointGeometryObjective
    from motionbricks.data.terrain_response_loss import heightfield_gap
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.sole_region_contact import sole_region_indices

    torch.set_num_threads(4)
    rep = build_native_representation()
    geometry = JointGeometryObjective(rep, 'cpu')
    surfaces, *_ = official_sole_surfaces(rep.skeleton)
    left_count = len(sole_region_indices(surfaces, 'Left'))
    axis = torch.tensor(args.direction, dtype=torch.float32)
    axis /= axis.norm()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = {}
    fig, axes = plt.subplots(len(args.recordings), 1, figsize=(11, 3.7*len(args.recordings)), constrained_layout=True, squeeze=False)
    for plot, item in zip(axes[:, 0], args.recordings):
        label, filename = item.split('=', 1)
        path = Path(filename)
        with np.load(path, allow_pickle=False) as archive:
            world = torch.from_numpy(archive['world'].copy())[None]
            scene = {key: archive[key].copy() for key in ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
        with torch.no_grad():
            _, points = geometry.geometry(world)
        soles = points[0, :, :geometry.sole_count]
        gaps, known = heightfield_gap(soles, scene)
        surface = soles[..., 1]-gaps
        origin = world[0, 0, [0, 2]]
        distance = torch.arange(-.5, float((world[0, -1, [0, 2]]-origin) @ axis)+1., float(scene['dx_m'])/4)
        line = torch.zeros(len(distance), 3)
        line[:, [0, 2]] = origin+distance[:, None]*axis
        line_gap, line_known = heightfield_gap(line, scene)
        line_height = -line_gap.numpy()
        line_height[~line_known.numpy()] = np.nan
        levels = longest_regular_ascent(line_height)
        events, traces = [], {}
        for side, indices, color in [('L', slice(0, left_count), '#287cab'), ('R', slice(left_count, geometry.sole_count), '#c26524')]:
            foot = soles[:, indices]
            near = known[:, indices] & (gaps[:, indices].abs() <= .02)
            speed = torch.diff(foot, dim=0).norm(dim=-1)*30
            stable = near & torch.cat((torch.zeros_like(near[:1]), speed <= .1)) & torch.cat((speed <= .1, torch.zeros_like(near[:1])))
            for kind, mask in [('near_surface', near), ('strict_stationary', stable)]:
                for start, end in true_runs((mask.sum(-1) >= 3).tolist()):
                    selected = mask[start:end]
                    selected_points = foot[start:end][selected]
                    events.append(dict(side=side, kind=kind, start_frame=start, end_frame=end-1,
                        surface_height_m=float(surface[start:end, indices][selected].median()),
                        forward_m=float(((selected_points[:, [0, 2]]-origin) @ axis).mean()),
                        material_samples=int(selected.sum())))
            bottom = foot[:, :, 1].amin(-1).numpy()
            traces[side] = dict(bottom_y_m=bottom.tolist(), nearby_vertices=near.sum(-1).tolist(),
                                stationary_vertices=stable.sum(-1).tolist())
            plot.plot(np.arange(len(foot))/30, bottom, color=color, alpha=.7, label=side+' lowest sole')
            for event in [e for e in events if e['side'] == side and e['kind'] == 'near_surface']:
                x = (event['start_frame']+event['end_frame'])/60
                y = event['surface_height_m']
                plot.scatter([x], [y], color=color, zorder=4)
                plot.annotate(f'{side}:{y:.3f}', (x, y), xytext=(0, 10 if side == 'L' else -17),
                              textcoords='offset points', ha='center', color=color, fontsize=8)
        near_events = sorted([e for e in events if e['kind'] == 'near_surface'], key=lambda e: e['start_frame'])
        violations = reciprocal_failures(near_events, levels)
        rows[label] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frames=world.shape[1], flight_levels_m=levels, events=sorted(events, key=lambda e: e['start_frame']),
            interior_same_tread_events=violations, strict_stationary_runs=sum(e['kind']=='strict_stationary' for e in events),
            traces=traces, quality_approved=False)
        for level in levels:
            plot.axhline(level, color='#888888', linewidth=.6, alpha=.35)
        plot.set_title(label+' | actual sole geometry; marked dots are near-surface episodes, not measured force contacts')
        plot.set_xlabel('Original time (s), no retiming')
        plot.set_ylabel('World height (m)')
        low, high = plot.get_ylim()
        plot.set_ylim(low-.08, high+.03)
        plot.legend(loc='upper left')
    report = dict(rows=rows, optimizer_updates=0, motion_modified=False,
        criteria='At least3 sole material vertices within2cm for at least3 consecutive frames. '
                 'Strict stationary additionally requires the same vertices moving<=0.1m/s in adjacent intervals. '
                 'These are kinematic evidence, not force contacts. Interior equal/lower opposite-foot tread is rejected; '
                 'same-level catch-up on a top landing is permitted. Missing contacts never imply success.')
    (args.output/'complete.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    fig.savefig(args.output/'footfalls.png', dpi=160)
    fig.savefig(args.output/'footfalls.pdf')
    print(json.dumps({label: {key:value for key,value in row.items() if key not in ('traces','events')}
                      for label,row in rows.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', nargs='+', required=True)
    parser.add_argument('--direction', nargs=2, type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
