import torch

from motionbricks.data.terrain_exact_interior import ExactTerrainInterior,PackedExactInteriors
from motionbricks.data.terrain_response_loss import terrain_clearance


def test_analytic_step_distances_and_gradients():
    scene={'heightfield_y_m':torch.tensor([[0.,1.,1.]]),
           'min_point_world_xyz_m':[0.,0.,0.],'dx_m':1.}
    points=torch.tensor([[.6,.5,0.],[.4,.5,0.],[1.,1.1,0.],[1.,.9,0.]],requires_grad=True)
    distance,known=ExactTerrainInterior(scene).query(points)
    assert known.all()
    torch.testing.assert_close(distance,torch.tensor([-.1,.5,.1,-.1]),atol=1e-6,rtol=0)
    (-distance).clamp_min(0).square().sum().backward()
    assert points.grad[0,0]>.19
    assert points.grad[3,1]<-.19
    torch.testing.assert_close(points.grad[1:3],torch.zeros(2,3))


def test_exact_negative_part_matches_parc_reference_and_keeps_unknown_mask():
    torch.manual_seed(4)
    scene={'heightfield_y_m':torch.tensor([[0.,.3,.8],[0.,.3,.8],[0.,.3,.8]]),
           'min_point_world_xyz_m':[-.2,0.,-.2],'dx_m':.2}
    points=torch.rand(64,3)*torch.tensor([.6,1.,.6])+torch.tensor([-.3,-.1,-.3])
    expected,known=terrain_clearance(points,scene)
    actual,mask=ExactTerrainInterior(scene).query(points)
    assert torch.equal(mask,known)
    torch.testing.assert_close(actual[known].clamp_max(0),expected[known].clamp_max(0),atol=1e-6,rtol=0)
    outside=torch.tensor([[-10.,0.,-10.]])
    _,mask=ExactTerrainInterior(scene).query(outside)
    assert not mask.any()
    batched,batch_mask=PackedExactInteriors([scene,scene]).query(points[None].expand(2,-1,-1))
    torch.testing.assert_close(batched,actual[None].expand(2,-1))
    assert torch.equal(batch_mask,known[None].expand(2,-1))


def test_active_prism_gradient_matches_full_reference():
    torch.manual_seed(17)
    scene={'heightfield_y_m':torch.tensor([[0.,.3,.8],[0.,.3,.8],[0.,.3,.8]]),
           'min_point_world_xyz_m':[-.2,0.,-.2],'dx_m':.2}
    points=(torch.rand(80,3)*torch.tensor([.55,.9,.55])+torch.tensor([-.27,-.1,-.27])).requires_grad_()
    reference,known=terrain_clearance(points,scene)
    actual,mask=ExactTerrainInterior(scene).query(points)
    expected_grad=torch.autograd.grad(reference[known].clamp_max(0).square().sum(),points)[0]
    actual_grad=torch.autograd.grad(actual[mask].clamp_max(0).square().sum(),points)[0]
    torch.testing.assert_close(actual_grad,expected_grad,atol=1e-6,rtol=1e-5)
