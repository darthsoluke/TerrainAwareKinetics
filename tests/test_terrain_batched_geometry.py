from collections import OrderedDict
from types import SimpleNamespace

import torch

from motionbricks.data.terrain_batched_geometry import PackedDistanceVolumes,PackedHeightfields
from motionbricks.data.terrain_distance_volume import TerrainDistanceVolume
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_model_v3 import pack_scene_chunks
from motionbricks.data.terrain_response_loss import heightfield_gap


def scenes():
    flat=torch.zeros(21,31);stairs=torch.zeros(31,41);stairs[:,20:]=.8
    hole=torch.zeros(17,23);hole[3:5,7:9]=torch.nan;hole[7:]=-.3
    return [{'heightfield_y_m':h,'min_point_world_xyz_m':[-1.,0.,-1.],'dx_m':.1} for h in (flat,stairs,hole)]


def test_batched_queries_preserve_values_validity_and_point_gradients():
    torch.set_num_threads(2);torch.manual_seed(32)
    data=scenes();volumes=[TerrainDistanceVolume(s) for s in data]
    points=torch.randn(3,9,7,3)*1.7;points[:,0,:,1]=-2.;points[:,1,:,1]=6.
    points.requires_grad_();packed=PackedDistanceVolumes(volumes,'cpu',cell_budget=10000)
    distance,known=packed.query(points)
    refs=[v.query(p) for v,p in zip(volumes,points)]
    torch.testing.assert_close(distance,torch.stack([r[0] for r in refs]),atol=2e-6,rtol=2e-6)
    assert torch.equal(known,torch.stack([r[1] for r in refs]))
    weight=torch.randn_like(distance)
    actual=torch.autograd.grad((distance*weight).sum(),points,retain_graph=True)[0]
    expected=torch.autograd.grad((torch.stack([r[0] for r in refs])*weight).sum(),points)[0]
    torch.testing.assert_close(actual,expected,atol=5e-6,rtol=5e-6)
    gaps,masks=PackedHeightfields(data,'cpu').query(points)
    for i,s in enumerate(data):
        gap,mask=heightfield_gap(points[i],s)
        torch.testing.assert_close(gaps[i],gap,atol=0,rtol=0);assert torch.equal(masks[i],mask)


def test_cached_geometry_keeps_all_loss_values_and_gradients_on_same_window():
    torch.set_num_threads(2);torch.manual_seed(41)
    class Rep:
        def inverse(self,world,*args,**kwargs):return {'posed_joints':world.reshape(*world.shape[:2],6,3)}
    objective=JointGeometryObjective.__new__(JointGeometryObjective)
    objective.rep=Rep();objective.sole_count=2;objective.volumes=OrderedDict();objective.max_volumes=96
    objective.geometry=lambda x:(objective.rep.inverse(x)['posed_joints'],objective.rep.inverse(x)['posed_joints'])
    source=torch.randn(3,1,6,3).expand(-1,65,-1,-1).clone()*.15
    source[:,:,:,1]=.1;source[:,:,0:2,1]=.01
    source=source.flatten(-2);data=scenes()
    episode=SimpleNamespace(physical_source=source,scenes=data,identities=['a','b','c'],packed_scene=pack_scene_chunks(data,'cpu'))
    prepared=objective.prepare_episode(episode)
    offset=8;truth=source[:,offset:offset+33];history=truth[:,:4].clone();history[:,:,0]+=.03
    predicted=(truth[:,:32]+torch.randn_like(truth[:,:32])*.035).requires_grad_()
    old=objective.losses(predicted,truth,history,data,episode.identities,volume_source=source)
    new=objective.prepared_losses(predicted,history,prepared,offset=offset)
    names=['fk_m','boundary_mps','collision_m2','support_gap_m','source_support_velocity_mps']
    for name in names:torch.testing.assert_close(new[name],old[name],atol=2e-6,rtol=2e-6)
    a=torch.autograd.grad(sum(new[k] for k in names),predicted,retain_graph=True)[0]
    b=torch.autograd.grad(sum(old[k] for k in names),predicted)[0]
    torch.testing.assert_close(a,b,atol=5e-6,rtol=5e-6)
    for key in ('known_surface_points','unknown_surface_points','support_vertices'):assert new[key]==old[key]
    assert abs(new['source_collision_p99_m']-old['source_collision_p99_m'])<2e-6
