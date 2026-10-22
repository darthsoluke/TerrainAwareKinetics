import numpy as np
import pytest
import torch

from motionbricks.data.bvh_heightfield import BVHHeightfieldPrismSDF
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from scripts.package_pfnn_demo_map import convert_heightmap, display_mesh


def test_pfnn_axes_units_and_fixed_vertical_translation():
    raw = np.arange(35, dtype=np.float32).reshape(5,7)
    height, origin, dx, zero = convert_heightmap(raw, 10., 3., [0.,0.])
    assert height.shape == (7,5)
    np.testing.assert_allclose(origin, [-.2,0.,-.3])
    np.testing.assert_allclose(height, (raw-raw[2,3]).T*.03, atol=1e-7)
    mesh = display_mesh(height, origin, float(dx), np.ones_like(raw))
    assert len(mesh['vertices']) == 12
    assert mesh['faces'].max() < len(mesh['vertices'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA conformance test')
def test_bvh_matches_native_prisms_and_captured_queries():
    torch.manual_seed(173)
    for heights in (np.zeros((7,9),np.float32),
                    np.round(np.random.default_rng(3).normal(size=(15,17)),1).astype(np.float32),
                    np.random.default_rng(5).normal(size=(24,29)).astype(np.float32)):
        scene = dict(heightfield_y_m=heights, min_point_world_xyz_m=np.array([-1.,0.,-1.],np.float32), dx_m=.13)
        native, fast = HeightfieldPrismSDF(scene), BVHHeightfieldPrismSDF(scene)
        points = torch.randn(1300,3,device='cuda')*2
        points = torch.cat((points, torch.tensor([[-1.,0.,-1.],[-1.065,0.,-1.],
                                                 [-.935,0.,-1.],[0.,0.,0.]],device='cuda')))
        expected = native.query(points)
        actual = fast.query(points)
        for left,right in zip(actual,expected):
            torch.testing.assert_close(left,right,atol=0,rtol=0)
        if heights.shape == (24,29):
            from motionbricks.data.soma_static_execution import trace_fixed_call
            from motionbricks.data.soma_cuda_graph import StaticCudaCall
            def query(value):
                distance, normal, valid = fast.query(value)
                return dict(distance=distance, normal=normal, valid=valid)
            with torch.inference_mode():
                traced = trace_fixed_call(query, (points,), cuda_constants=True)
                runner = StaticCudaCall(traced)
                captured = runner(points)
                for left,right in zip(captured.values(),expected):
                    torch.testing.assert_close(left,right,atol=0,rtol=0)
                moved = points+.031
                for left,right in zip(runner(moved).values(),native.query(moved)):
                    torch.testing.assert_close(left,right,atol=0,rtol=0)
