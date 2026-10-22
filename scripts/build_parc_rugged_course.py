"""Small versioned rough-terrain routes with genuine flat-platform turns."""

import hashlib
import torch

FAMILIES = ('stairs', 'rough', 'ramp', 'mixed')


def terrain_arrays(index):
    if isinstance(index, bool) or index not in range(8):
        raise ValueError('Expected one of eight disjoint layouts')
    family, variant = FAMILIES[index % 4], index // 4
    profile = torch.zeros(161)
    x = torch.arange(161)
    riser, tread = (.12 + .025 * variant), (4 + variant)
    if family in ('stairs', 'mixed'):
        count = 4 if family == 'stairs' else 3
        up = ((x - 30) // tread + 1).clamp(0, count)
        down = ((x - 80) // tread + 1).clamp(0, count)
        profile = (up - down).float() * riser
        if family == 'mixed':
            height = count * riser
            profile[x >= 80] = (height * (110 - x[x >= 80]) / 30).clamp_min(0)
            profile[113:117] = .05 + .01 * variant
            profile[117:121] = .10 + .02 * variant
            profile[121:125] = .05 + .01 * variant
    elif family == 'ramp':
        height = .48 + .12 * variant
        profile = height * torch.minimum(((x-30)/30).clamp(0, 1), ((110-x)/30).clamp(0, 1))
    else:
        levels = (.06, .14, .03, .18, .08)
        for start in (30, 85):
            values = levels if start == 30 else tuple(reversed(levels))
            for block, value in enumerate(values):
                profile[start+block*6:start+(block+1)*6] = value + .015 * variant
    hf = profile[:, None].repeat(1, 81)
    route = [(i, 20) for i in range(10, 66)]
    route += [(65, j) for j in range(21, 51)]
    route += [(i, 50) for i in range(66, 141)]
    grid = torch.tensor(route, dtype=torch.long)
    if index % 4 >= 2:
        hf = hf.flip(0).contiguous()
        grid[:, 0] = hf.shape[0] - 1 - grid[:, 0]
    if index % 2:
        hf = hf.T.contiguous()
        grid = grid[:, [1, 0]]
    return hf, grid, dict(family=family, variant=variant, spacing_m=.1,
        split='train' if index < 4 else ('validation' if index < 6 else 'test'),
        reverse_axis=index % 4 >= 2, swap_xy=bool(index % 2))


def build_rugged_course(index, device='cpu'):
    from build_parc_stair_course import _load_official_terrain_util
    hf, grid, parameters = terrain_arrays(index)
    terrain = _load_official_terrain_util().SubTerrain('rugged32_'+str(index),
        x_dim=hf.shape[0], y_dim=hf.shape[1], dx=.1, dy=.1, min_x=0., min_y=0., device='cpu')
    terrain.hf.copy_(hf)
    terrain.hf_maxmin[..., 0] = hf
    terrain.hf_maxmin[..., 1] = hf
    nodes = terrain.get_xyz_point(grid)
    surface = terrain.get_hf_val_from_points(nodes[:, :2])
    if not torch.allclose(surface, nodes[:, 2], atol=1e-6, rtol=0):
        raise ValueError('Route nodes differ from actual terrain samples')
    if nodes[[0, -1], 2].abs().max() > 1e-6:
        raise ValueError('Real flat start and exit required')
    segments = [(0, 20, 'approach'), (20, 50, parameters['family']+'_entry'),
        (50, 55, 'platform_before_turn'), (55, 85, 'platform_turn'),
        (85, 99, 'platform_after_turn'), (99, 145, parameters['family']+'_exit'),
        (145, len(nodes)-1, 'flat_exit')]
    report = dict(schema_version=1, course_kind='rugged32_v1', scenario_id='rugged32_'+str(index),
        layout_index=index, split=parameters['split'], distribution=parameters,
        requested_family=parameters['family'], stair_flights=[],
        route_segments=[dict(name=stage, stage=stage, path_node_start=a, path_node_end=b,
            start_xyz_m=nodes[a].tolist(), end_xyz_m=nodes[b].tolist()) for a,b,stage in segments],
        course=dict(planar_length_m=float(torch.diff(nodes[:, :2], dim=0).norm(dim=-1).sum()),
                    surface_length_m=float(torch.diff(nodes, dim=0).norm(dim=-1).sum())),
        terrain_sha256=hashlib.sha256(hf.numpy().tobytes()).hexdigest(),
        path_surface_max_error_m=float((surface-nodes[:, 2]).abs().max()),
        terrain_span_m=[float(hf.min()), float(hf.max())],
        history_reset_between_sections=False, requested_stops=[],
        initialization='native PARC flat start; no mid-route pose placement',
        generated_behavior_verified=False)
    terrain.set_device(device)
    return terrain, nodes.to(device), report
