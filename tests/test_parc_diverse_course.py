"""Geometry randomness cannot inherit motion-generation seeds or leak layouts."""

import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_parc_diverse_course import FAMILIES, course_spec, build_diverse_course, write_plan
from build_parc_diverse_course import expanded_course_spec


def test_plan_assigns_whole_layouts_and_covers_all_families(tmp_path):
    plan = write_plan(tmp_path/'plan')
    assert len(plan['jobs']) == len({j['id'] for j in plan['jobs']}) == 96
    for family in FAMILIES:
        rows = [j for j in plan['jobs'] if j['family'] == family]
        assert [j['split'] for j in rows].count('train') == 12
        assert [j['split'] for j in rows].count('validation') == 2
        assert [j['split'] for j in rows].count('test') == 2


@pytest.mark.parametrize('family', ['gaps', 'stairs', 'mixed'])
def test_scene_and_route_are_independent_of_motion_rng(family):
    states, results = [], []
    for seed in (91, 671):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        terrain, nodes, manifest = build_diverse_course(course_spec(family, 4))
        assert torch.equal(torch.get_rng_state(), before[0])
        assert np.array_equal(np.random.get_state()[1], before[1][1])
        assert random.getstate() == before[2]
        assert manifest['path_surface_max_error_m'] < 1e-5
        results.append((terrain.hf, nodes, manifest['terrain_sha256']))
    assert results[0][2] == results[1][2]
    torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
    torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)


def test_official_astar_jump_guides_are_not_mistaken_for_support_surfaces():
    terrain, nodes, manifest = build_diverse_course(course_spec('stairs', 0))
    assert manifest['path_surface_max_error_m'] < 1e-5
    assert 'interpolated' in manifest['path_height_semantics']
    torch.testing.assert_close(terrain.get_hf_val_from_points(nodes[[0, -1], :2]), nodes[[0, -1], 2])


def test_expanded_ids_do_not_collide_and_splits_precede_quality():
    specs=[expanded_course_spec(family,index) for family in FAMILIES for index in range(64)]
    assert len({spec['scene_id'] for spec in specs})==384
    assert not ({spec['scene_id'] for spec in specs} & {course_spec(f,i)['scene_id'] for f in FAMILIES for i in range(16)})
    for spec in specs:
        assert spec['split']==('validation' if spec['layout_index']%10==0 else 'test' if spec['layout_index']%10==1 else 'train')


@pytest.mark.parametrize('family', FAMILIES)
def test_expanded_geometry_and_reverse_route(family):
    spec=expanded_course_spec(family,3)
    terrain,backwards,report=build_diverse_course(spec)
    forward_spec=dict(spec,reverse_route=False)
    other,forwards,_=build_diverse_course(forward_spec)
    torch.testing.assert_close(terrain.hf,other.hf,atol=0,rtol=0)
    torch.testing.assert_close(backwards,forwards.flip(0),atol=0,rtol=0)
    assert report['path_surface_max_error_m']<1e-5
    assert report['route_segments'][0]['path_node_start']==0
    assert report['route_segments'][-1]['path_node_end']==len(backwards)-1
