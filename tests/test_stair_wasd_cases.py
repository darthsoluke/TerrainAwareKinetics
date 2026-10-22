import torch

from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.prepare_stair_wasd_trial import descending_passages


def example():
    heights = torch.ones(3,30)
    for x in range(6,30):
        heights[:,x] -= .15*min(4,(x-6)//2+1)
    scene = dict(heightfield_y_m=heights,min_point_world_xyz_m=torch.tensor([0.,0.,-1.]),dx_m=torch.tensor(1.))
    world = torch.zeros(120,366)
    world[:,0] = torch.arange(120)*.2
    gap, known = heightfield_gap(world[:,:3],scene)
    assert known.all()
    world[:,1] = .9-gap
    return world,scene


def test_real_order_descent_with_complete_context():
    world,scene = example()
    rows = descending_passages(world,scene)
    assert len(rows)==1 and rows[0]['risers']==4
    assert rows[0]['end']<=len(world)
    assert (rows[0]['end']-rows[0]['start']-33)%8==0


def test_ascent_or_missing_continuation_not_admitted():
    world,scene = example()
    assert descending_passages(world.flip(0),scene)==[]
    assert descending_passages(world[:62],scene)==[]
