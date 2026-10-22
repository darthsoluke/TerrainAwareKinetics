"""Restore native PFNN encoding to its fitted scene and bind scene contact labels."""

import torch

from .native_mb_data import _native_root_world


VERSION = 'pfnn-world-and-scene-contacts-v2'


@torch.no_grad()
def encode_pfnn_world(rep,data):
    positions=torch.as_tensor(data['posed_joints'])
    value,frame=rep.dual_rep(dict(posed_joints=positions,
        global_joint_rots=torch.as_tensor(data['global_rot_mats']),
        local_joint_rots=torch.as_tensor(data['local_rot_mats']),
        translation=torch.as_tensor(data['root_positions']),
        # Contact labels depend on each fitted scene and are supplied separately.
        foot_contacts=positions.new_zeros(len(positions),4)),
        to_normalize=False,return_init_heading_info=True)
    canonical=rep.dual_rep.get_feature_subset(value,mode='global')
    roots=_native_root_world(canonical,frame['init_heading_quat'],frame['root_pos_init_xz'],
        rep.indices['global_root_pos'],rep.indices['global_root_heading'])
    world=rep.change_first_heading(canonical[None],torch.atan2(roots[:1,4],roots[:1,3]),False,False)
    world[...,[0,2]]+=roots[:1,[0,2]][None]
    source_roots=torch.as_tensor(data['root_positions']).to(world)
    torch.testing.assert_close(world[0,:,:3],source_roots,atol=5e-5,rtol=1e-5)
    restored=rep.inverse(world,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints'][0]
    fk_error=float((restored-positions.to(restored)).norm(dim=-1).max())
    if fk_error>5e-5:
        raise ValueError('PFNN world encoding differs from factual retarget FK: '+str(fk_error))
    proof=dict(version=VERSION,root_max_abs_m=float((world[0,:,:3]-source_roots).abs().max()),
        fk_joint_max_m=fk_error,canonical_root_max_error_m=float((canonical[:,:3]-source_roots).norm(dim=-1).max()),
        actual_frames=len(positions),contact_channels_require_scene_sidecar=True)
    return world[0].cpu(),proof


@torch.no_grad()
def scene_contact_targets(rep,world,scene,*,posed_joints=None):
    from scripts.build_parc_native_dataset import _terrain_relative_foot_contacts
    if world.ndim!=2:
        raise ValueError('A real single source episode is required')
    joints=rep.inverse(world,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints'] \
        if posed_joints is None else torch.as_tensor(posed_joints)
    if joints.shape!=(len(world),rep.skeleton.nbjoints,3) or not torch.isfinite(joints).all():
        raise ValueError('Contact labels require aligned factual FK positions')
    return _terrain_relative_foot_contacts(joints,rep.skeleton,
        clip_heightfield_y_m=scene['heightfield_y_m'],
        clip_origin_xyz_m=scene['min_point_world_xyz_m'],dx_m=scene['dx_m']).cpu()


def apply_scene_contacts(world,contacts,offset,indices):
    if (contacts.ndim!=2 or contacts.shape[1]!=4 or offset<0 or offset+len(contacts)>len(world)
            or not torch.isfinite(contacts).all() or not ((contacts==0)|(contacts==1)).all()):
        raise ValueError('Scene contacts must cover declared real source frames')
    result=world.clone()
    result[offset:offset+len(contacts),indices]=contacts.to(result)
    return result
