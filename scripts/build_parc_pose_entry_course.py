"""New source-only stairs with long real approaches; no character pose edits."""
import hashlib
import torch


def terrain_arrays(index):
    if isinstance(index, bool) or index not in range(8):
        raise ValueError('Eight declared source geometries only')
    riser = (.12, .14, .16, .18)[index % 4]
    tread_cells = 3 + index // 4
    steps = 5 + index % 2
    first = 70
    top = first + steps * tread_cells
    down = top + 50
    bottom = down + steps * tread_cells
    end = bottom + 60
    axis = torch.arange(end + 11)
    ascent = ((axis - first) // tread_cells + 1).clamp(0, steps)
    descent = ((axis - down) // tread_cells + 1).clamp(0, steps)
    hf = (ascent - descent).float()[:, None].repeat(1, 81) * riser
    route = torch.stack((torch.arange(10, end + 1), torch.full((end - 9,), 40)), -1)
    if index % 2:
        hf = hf.T.contiguous(); route = route[:, [1, 0]]
    return hf, route, dict(riser_m=riser, tread_m=tread_cells * .1, steps=steps,
        approach_m=6., top_flat_m=5., exit_m=6., split='train', axis='y' if index % 2 else 'x',
        stages=[(0, first - 10, 'flat_approach'), (first - 10, top - 10, 'ascent'),
                (top - 10, down - 10, 'top_flat'), (down - 10, bottom - 10, 'descent'),
                (bottom - 10, end - 10, 'flat_exit')])


def build_pose_entry_course(index, device='cpu'):
    from build_parc_stair_course import _load_official_terrain_util
    hf, grid, parameters = terrain_arrays(index)
    terrain = _load_official_terrain_util().SubTerrain('pose_entry_' + str(index),
        x_dim=hf.shape[0], y_dim=hf.shape[1], dx=.1, dy=.1, min_x=0., min_y=0., device='cpu')
    terrain.hf.copy_(hf); terrain.hf_maxmin[..., 0] = hf; terrain.hf_maxmin[..., 1] = hf
    nodes = terrain.get_xyz_point(grid)
    torch.testing.assert_close(terrain.get_hf_val_from_points(nodes[:, :2]), nodes[:, 2], atol=1e-6, rtol=0)
    report = dict(schema_version=1, course_kind='pose_entry_20261014_v1', scenario_id='pose_entry_' + str(index),
        layout_index=index, split='train', distribution=parameters, requested_family='stairs', stair_flights=[],
        route_segments=[dict(name=name, stage=name, path_node_start=a, path_node_end=b,
            start_xyz_m=nodes[a].tolist(), end_xyz_m=nodes[b].tolist()) for a, b, name in parameters['stages']],
        course=dict(planar_length_m=float(torch.diff(nodes[:, :2], dim=0).norm(dim=-1).sum()),
                    surface_length_m=float(torch.diff(nodes, dim=0).norm(dim=-1).sum())),
        terrain_sha256=hashlib.sha256(hf.numpy().tobytes()).hexdigest(), path_surface_max_error_m=0.,
        terrain_span_m=[float(hf.min()), float(hf.max())], history_reset_between_sections=False,
        requested_stops=[], initialization='native PARC flat start; unchanged generated motion', generated_behavior_verified=False)
    terrain.set_device(device)
    return terrain, nodes.to(device), report
