"""Select complete, unchanged regular stair passages from paired PARC motion."""

import math

import torch

from motionbricks.data.terrain_response_loss import heightfield_gap


def regular_passages(world, scene, entry, *, maximum_frames=161):
    if entry not in ('up', 'down'):
        raise ValueError('Expected a physical ascent or descent, never reversed motion')
    sign = 1 if entry == 'up' else -1
    gap, known = heightfield_gap(world[:, :3], scene)
    level = world[:, 1] - gap
    delta = torch.diff(level) * sign
    edges = (torch.nonzero((delta > .05) & (delta < .34)).flatten() + 1).tolist()
    groups = []
    for edge in edges:
        if not groups or edge - groups[-1][-1] > 18 or (delta[groups[-1][-1]:edge] < -.04).any():
            groups.append([])
        groups[-1].append(edge)
    passages = []
    for group in groups:
        if len(group) < 3:
            continue
        first, last = group[0], group[-1]
        begin, goal = first - 20, last + 15
        last_offset = 8 * math.ceil(max(0, goal - begin - 11) / 8)
        end = begin + last_offset + 33
        if begin < 0 or end > len(world) or end - begin > maximum_frames:
            continue
        if not known[begin:end].all():
            continue
        if level[begin:first].max() - level[begin:first].min() > .03:
            continue
        if level[last:goal + 1].max() - level[last:goal + 1].min() > .03:
            continue
        risers = delta[torch.tensor(group) - 1]
        if risers.max() - risers.min() > .025:
            continue
        travel = world[goal, [0, 2]] - world[begin + 3, [0, 2]]
        if travel.norm() < 1. or sign * (level[goal] - level[begin + 3]) < .4:
            continue
        axis = travel / travel.norm()
        side = torch.stack((-axis[1], axis[0]))
        lateral = ((world[begin:goal + 1, [0, 2]] - world[begin + 3, [0, 2]]) @ side).abs().max()
        if lateral > .25:
            continue
        passages.append(dict(start=begin, end=end, goal=goal, edge=first,
                             risers=len(group), entry=entry, lateral_m=float(lateral)))
    return passages


def validate_pure_cases(cases):
    sources = {'train': set(), 'heldout': set()}
    counts = {(split, entry): 0 for split in sources for entry in ('up', 'down')}
    for case in cases:
        row = case['row']
        if not row.get('origin', '').startswith('parc'):
            raise ValueError('PARC-only training excludes BONES and synthetic labels')
        split, entry = case['pilot_split'], case['entry']
        counts[split, entry] += 1
        sources[split].add(row['source_id'])
        frames = len(case['world'])
        if frames != 33 + (case['training_segments'] - 1) * 8:
            raise ValueError('Incomplete real source window; padding is forbidden')
        if case['evaluation_goal_source_frame'] > case['start_frame'] + case['training_segments'] * 8 + 3:
            raise ValueError('Published passage does not cover its source goal')
    if sources['train'] & sources['heldout']:
        raise ValueError('Source identity leaks across the pilot holdout')
    if counts != {('train', 'up'): 2, ('train', 'down'): 2,
                  ('heldout', 'up'): 1, ('heldout', 'down'): 1}:
        raise ValueError('Expected four training passages and two independent holdouts')
    return counts
