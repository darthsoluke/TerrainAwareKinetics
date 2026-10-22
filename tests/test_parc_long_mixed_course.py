"""Long-route geometry is continuous; motion is never looped or spliced."""

from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_parc_long_mixed_course import build_long_mixed_course


def test_independent_sections_form_one_long_route():
    terrain,nodes,report=build_long_mixed_course(20261010,3)
    assert report['course']['planar_length_m']>100
    assert report['section_count']==3
    assert len({row['terrain_sha256'] for row in report['sections']})==3
    assert not report['motion_concatenation']
    assert not report['history_reset_between_sections']
    assert report['route_segments'][0]['path_node_start']==0
    assert report['route_segments'][-1]['path_node_end']==len(nodes)-1
    for a,b in zip(report['route_segments'][:-1],report['route_segments'][1:]):
        assert a['path_node_end']==b['path_node_start']
    torch.testing.assert_close(terrain.get_hf_val_from_points(nodes[:,:2]),nodes[:,2],atol=1e-5,rtol=0)
    assert all(stage in {r['stage'] for r in report['route_segments']}
               for stage in ['obstacle_detour','jump_candidate','stairs_up','platform_walk','rough_rock','stairs_down'])


def test_long_geometry_is_repeatable_and_does_not_depend_on_motion_rng():
    torch.manual_seed(17)
    a,pa,ra=build_long_mixed_course(20261011,2)
    torch.manual_seed(971)
    b,pb,rb=build_long_mixed_course(20261011,2)
    assert ra['terrain_sha256']==rb['terrain_sha256']
    torch.testing.assert_close(a.hf,b.hf,atol=0,rtol=0)
    torch.testing.assert_close(pa,pb,atol=0,rtol=0)
