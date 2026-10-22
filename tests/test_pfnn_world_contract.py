import torch
import json
from types import SimpleNamespace
import pytest

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_world_contract import encode_pfnn_world,scene_contact_targets,apply_scene_contacts


def fixture():
    rep=build_native_representation()
    local=torch.eye(3).repeat(40,30,1,1)
    angle=torch.tensor(.8)
    local[:,0]=torch.tensor([[angle.cos(),0,angle.sin()],[0,1,0],[-angle.sin(),0,angle.cos()]])
    roots=torch.tensor([8.,1.,-3.]).repeat(40,1)
    global_rot,positions,_=rep.skeleton.fk(local,roots)
    return rep,dict(posed_joints=positions,global_rot_mats=global_rot,local_rot_mats=local,root_positions=roots)


def test_encoding_keeps_original_scene_position_heading_and_body():
    rep,data=fixture()
    world,proof=encode_pfnn_world(rep,data)
    assert proof['canonical_root_max_error_m']>5
    assert proof['fk_joint_max_m']<5e-5
    torch.testing.assert_close(world[:,:3],data['root_positions'],atol=5e-5,rtol=0)
    joints=rep.inverse(world,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints']
    torch.testing.assert_close(joints,data['posed_joints'],atol=5e-5,rtol=0)


def test_scene_contacts_are_invariant_to_joint_motion_and_floor_height_shift():
    rep,data=fixture()
    world,_=encode_pfnn_world(rep,data)
    feet=list(rep.skeleton.left_foot_joint_idx)+list(rep.skeleton.right_foot_joint_idx)
    floor=float(data['posed_joints'][:,feet,1].min())
    scene=dict(heightfield_y_m=torch.full((2,2),floor),min_point_world_xyz_m=torch.tensor([-10.,0.,-10.]),dx_m=20.)
    first=scene_contact_targets(rep,world,scene)
    assert first.any()
    shifted=world.clone()
    shifted[:,1]+=.7
    second=scene_contact_targets(rep,shifted,dict(scene,heightfield_y_m=scene['heightfield_y_m']+.7))
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    attached=apply_scene_contacts(world,first,0,rep.indices['foot_contacts'])
    rest=[i for i in range(world.shape[-1]) if i not in rep.indices['foot_contacts']]
    torch.testing.assert_close(attached[:,rest],world[:,rest],atol=0,rtol=0)


def test_old_canonicalized_pfnn_manifest_cannot_enter_training_cache(tmp_path):
    from scripts.cache_full_terrain_primitives import sources
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema='motionbricks.pfnn_static_pairs.v1',pilot=False,records=[])))
    with pytest.raises(ValueError,match='factual world'):
        next(sources(SimpleNamespace(mixed=None,pfnn=tmp_path,pilot=False),None))
