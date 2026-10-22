import hashlib
import torch
from scripts.build_parc_rugged_course import terrain_arrays


def test_rugged_routes_are_disjoint_and_have_real_flat_turns():
    hashes = set()
    for index in range(8):
        hf, grid, spec = terrain_arrays(index)
        heights = hf[grid[:, 0], grid[:, 1]]
        assert heights[[0, -1]].abs().max() == 0
        assert (grid[1:]-grid[:-1]).abs().sum(-1).eq(1).all()
        assert torch.diff(heights[50:100]).abs().max() == 0
        direction = grid[1:]-grid[:-1]
        assert ((direction[1:]*direction[:-1]).sum(-1) == 0).sum() == 2
        assert torch.diff(heights).abs().max() < .22
        assert heights.max()-heights.min() >= .18
        assert spec['split'] == ('train' if index < 4 else ('validation' if index < 6 else 'test'))
        hashes.add(hashlib.sha256(hf.numpy().tobytes()).hexdigest())
    assert len(hashes) == 8


def test_rugged_geometry_is_deterministic_and_spans_the_route_width():
    for index in range(8):
        a, grid, spec = terrain_arrays(index)
        b, repeat, _ = terrain_arrays(index)
        assert torch.equal(a, b) and torch.equal(grid, repeat)
        canonical = a.T if spec['swap_xy'] else a
        assert torch.equal(canonical[:, 0], canonical[:, -1])
