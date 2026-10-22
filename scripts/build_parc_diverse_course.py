"""Versioned terrain distributions using PARC's own terrain builders and A*."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import yaml

from build_parc_stair_course import _load_official_terrain_util, PARC_REPO


FAMILIES = ('stairs', 'platforms', 'blocks', 'gaps', 'rough', 'mixed')


def course_spec(family, index):
    if family not in FAMILIES or not 0 <= index < 16:
        raise ValueError('Expected a declared family and a layout index in 0..15')
    seed = 20261100 + 100 * FAMILIES.index(family) + index
    return {'version': 'parc-diverse-scenes-v1', 'family': family, 'layout_seed': seed,
            'layout_index': index, 'scene_id': f'{family}_{seed}',
            'split': 'validation' if index < 2 else 'test' if index < 4 else 'train',
            'cells': [160, 100] if family == 'mixed' else 32 + 8 * (index % 2),
            'spacing_m': .2 if family == 'mixed' else .4,
            'difficulty': index % 3, 'motion_seed': seed + 100000}


def expanded_course_spec(family, index):
    if family not in FAMILIES or not 0 <= index < 100000:
        raise ValueError('Invalid expanded layout identity')
    seed = 40000000 + FAMILIES.index(family) * 100000 + index
    rng = random.Random(seed)
    return {'version': 'parc-diverse-scenes-v2', 'family': family, 'layout_seed': seed,
            'layout_index': index, 'scene_id': f'{family}_{seed}',
            'split': 'validation' if index % 10 == 0 else 'test' if index % 10 == 1 else 'train',
            'cells': [160, 100] if family == 'mixed' else rng.choice([24, 32, 40, 48]),
            'spacing_m': .2 if family == 'mixed' else .4,
            'difficulty': index % 3, 'motion_seed': seed + 1000000,
            'reverse_route': bool(index % 2), 'parameter_seed': seed + 2000000}


def build_diverse_course(spec, device='cpu'):
    family, seed = spec['family'], spec['layout_seed']
    if family not in FAMILIES or spec['version'] not in ('parc-diverse-scenes-v1', 'parc-diverse-scenes-v2'):
        raise ValueError('Unknown terrain distribution')
    expanded = spec['version'] == 'parc-diverse-scenes-v2'
    parameters_rng = random.Random(spec.get('parameter_seed', seed))
    terrain_util = _load_official_terrain_util()
    from parc.motion_synthesis.procgen import astar
    cfg = yaml.safe_load((PARC_REPO / 'data/configs/parc_2_kin_gen_default.yaml').read_text())
    rng_state = np.random.get_state()
    python_rng_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if family == 'mixed':
                from build_parc_mixed_course import build_mixed_course
                rng = random.Random(seed)
                overrides = {'stair_riser_m': rng.uniform(.12, .20),
                    'obstacle_height_m': rng.uniform(.55, 1.1),
                    'rock_step_m': rng.uniform(.04, .12), 'trench_depth_m': rng.uniform(.35, .65)}
                if expanded:
                    overrides.update(ledge_riser_m=parameters_rng.uniform(.12, .22) if spec['difficulty'] < 2
                                     else parameters_rng.uniform(.5, .75),
                                     obstacle_height_m=parameters_rng.uniform(.45, 1.15))
                terrain, nodes, course = build_mixed_course('cpu', spec['difficulty'], seed,
                                                           profile_overrides=overrides)
                parameters = overrides
            else:
                terrain = terrain_util.SubTerrain(spec['scene_id'], x_dim=spec['cells'],
                    y_dim=spec['cells'], dx=spec['spacing_m'], dy=spec['spacing_m'],
                    min_x=0., min_y=0., device='cpu')
                difficulty = spec['difficulty']
                if family == 'stairs':
                    parameters = dict(min_stair_start_height=-.4, max_stair_start_height=.2,
                        min_step_height=.12, max_step_height=.16+.04*difficulty,
                        num_stairs=2+difficulty, min_stair_thickness=1.6, max_stair_thickness=3.2)
                    if expanded:
                        parameters.update(min_step_height=parameters_rng.uniform(.08, .14),
                            max_step_height=parameters_rng.uniform(.15, .25),
                            num_stairs=parameters_rng.randint(2, 5),
                            min_stair_thickness=parameters_rng.uniform(1.2, 1.8),
                            max_stair_thickness=parameters_rng.uniform(2.4, 4.0))
                    terrain_util.add_stairs_to_hf(terrain, **parameters)
                elif family == 'gaps':
                    parameters = dict(num_paths=4+difficulty, maxpool_size=1, floor_height=-.7,
                                      path_min_height=-.05, path_max_height=.35+.15*difficulty)
                    if expanded:
                        parameters.update(num_paths=parameters_rng.randint(3, 8),
                            floor_height=-parameters_rng.uniform(.35, 1.0),
                            path_max_height=parameters_rng.uniform(.2, .65))
                    terrain_util.gen_paths_hf(terrain, **parameters)
                else:
                    limits = {'platforms': (8, 14, 10, 0., .5+.35*difficulty),
                              'blocks': (3, 7, 22, 0., .5+.35*difficulty),
                              'rough': (2, 5, 80, -.12, .08+.06*difficulty)}
                    minimum, maximum, count, low, high = limits[family]
                    parameters = dict(box_min_len=minimum, box_max_len=maximum, num_boxes=count,
                        box_min_height=low, box_max_height=high, min_angle=0., max_angle=0.)
                    if expanded:
                        parameters.update(num_boxes=parameters_rng.randint(max(3, count//2), count*2),
                            box_max_height=parameters_rng.uniform(.06, .23) if family == 'rough'
                            else parameters_rng.uniform(.25, 1.2),
                            min_angle=-np.pi/4, max_angle=np.pi/4)
                    terrain_util.add_boxes_to_hf2(terrain.hf, **parameters)
                settings = astar.AStarSettings()
                for name, value in cfg['astar'].items():
                    setattr(settings, name, value)
                original = terrain.hf.clone()
                nodes = None
                for attempt in range(10):
                    terrain.hf = original.clone()
                    start, end = astar.pick_random_start_end_nodes_on_edges(
                        terrain, min_dist=settings.min_start_end_xy_dist)
                    # Same simplification and spawn/exit preparation as parc_2_kin_gen.py.
                    terrain_util.flat_maxpool_2x2(terrain)
                    for point in (start, end):
                        terrain_util.flatten_4x4_near_edge(terrain, point, float(terrain.hf[point[0], point[1]]))
                    nodes = astar.run_a_star_on_start_end_nodes(terrain, start, end, settings)
                    if nodes is not False and nodes is not None:
                        break
                if nodes is False or nodes is None:
                    raise ValueError('No path found under official PARC A* constraints')
                length = float(torch.diff(nodes[:, :2], dim=0).norm(dim=-1).sum())
                course = {'schema_version': 1, 'course_kind': 'diverse', 'scenario_id': spec['scene_id'],
                    'stair_flights': [], 'course': {'planar_length_m': length, 'surface_length_m':
                        float(torch.diff(nodes, dim=0).norm(dim=-1).sum())},
                    'route_segments': [{'name': family, 'stage': family, 'path_node_start': 0,
                                        'path_node_end': len(nodes)-1}],
                    'astar': cfg['astar'], 'route_attempt': attempt}
            if spec.get('reverse_route'):
                nodes = nodes.flip(0)
                last = len(nodes) - 1
                course['route_segments'] = [dict(segment,
                    name='reverse_' + segment['name'],
                    path_node_start=last-segment['path_node_end'],
                    path_node_end=last-segment['path_node_start'])
                    for segment in reversed(course['route_segments'])]
            if not torch.isfinite(terrain.hf).all() or not torch.isfinite(nodes).all():
                raise ValueError('Nonfinite generated terrain/path')
            gap = (terrain.get_hf_val_from_points(nodes[:, :2])-nodes[:, 2]).abs().max()
            endpoint_gap = (terrain.get_hf_val_from_points(nodes[[0, -1], :2])-nodes[[0, -1], 2]).abs().max()
            if endpoint_gap > 1e-5 or (family == 'mixed' and gap > 1e-5):
                raise ValueError('Route support endpoints do not match the scene surface')
            terrain.hf_maxmin[..., 0] = terrain.hf
            terrain.hf_maxmin[..., 1] = terrain.hf
            course.update(distribution=deepcopy(spec), terrain_parameters=parameters,
                terrain_sha256=hashlib.sha256(terrain.hf.numpy().tobytes()).hexdigest(),
                requested_family=family, generated_behavior_verified=False,
                path_surface_max_error_m=float(endpoint_gap),
                path_guide_surface_delta_max_m=float(gap),
                path_height_semantics=('terrain surface endpoints' if family == 'mixed' else
                    'official A* includes interpolated guidance across jump edges; not all nodes are supports'))
            terrain.set_device(device)
            return terrain, nodes.to(device), course
    finally:
        np.random.set_state(rng_state)
        random.setstate(python_rng_state)


def write_plan(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    # Interleave families, so partial collections cannot silently become one terrain type.
    for index in range(16):
        for family in FAMILIES:
            spec = course_spec(family, index)
            path = output / (spec['scene_id']+'.json')
            path.write_text(json.dumps(spec, indent=2))
            jobs.append({'id': spec['scene_id'], 'spec': path.name, 'family': family,
                         'split': spec['split'], 'motion_seed': spec['motion_seed']})
    manifest = {'version': 'parc-diverse-scenes-v1', 'jobs': jobs, 'layouts': len(jobs),
                'candidate_batch': 128, 'retained_per_layout': 2, 'max_attempts_per_layout': 3,
                'split_rule': 'whole new layout groups, assigned before generation or quality scores',
                'physics_required': True, 'training_approved': False}
    (output/'plan.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight-plan', type=Path)
    args = parser.parse_args()
    if args.preflight_plan:
        plan = json.loads(args.preflight_plan.read_text())
        hashes = set()
        for job in plan['jobs']:
            _, _, course = build_diverse_course(json.loads((args.preflight_plan.parent/job['spec']).read_text()))
            if course['terrain_sha256'] != job['terrain_sha256']:
                raise ValueError('Generation runtime preflight differs: '+job['id'])
            hashes.add(course['terrain_sha256'])
        print(json.dumps({'checked':len(plan['jobs']),'unique_heightfields':len(hashes),
                          'generation_runtime_matches_plan':True}))
    elif args.output:
        print(json.dumps({'layouts': write_plan(args.output)['layouts']}))
    else:
        parser.error('--output or --preflight-plan required')
