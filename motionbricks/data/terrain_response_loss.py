"""Training-only geometry objectives, adapted from PARC's inverted-air-box SDF.

PARC util/terrain_util.py:points_hf_sdf(inverted=True) measures distance to the
union of air boxes above terrain. Inverting that distance avoids treating cell
boundaries inside a solid wall as free surfaces. Actor observations stay local;
static scene geometry here is target-side supervision only.
"""

import torch

SOURCE_SUPPORT_MASK_VERSION='certified-source-velocity-intervals-v2'


@torch.no_grad()
def source_support_masks(source_sole,gap,valid,fps=30):
    """Separate source support instants from certified velocity intervals."""
    if (source_sole.ndim<3 or source_sole.shape[-1]!=3 or source_sole.shape[-3]<1
            or gap.shape!=source_sole.shape[:-1] or valid.shape!=gap.shape
            or valid.dtype!=torch.bool or fps<=0):
        raise ValueError('Expected matching [...,T,sole,XYZ] source and surface queries')
    speed=torch.diff(source_sole.detach(),dim=-3).norm(dim=-1)*fps
    intervals=valid[...,1:,:] & valid[...,:-1,:] & (gap[...,1:,:].abs()<=.02) & (gap[...,:-1,:].abs()<=.02) & (speed<=.1)
    frame=torch.zeros_like(valid)
    frame[...,:-1,:]|=intervals;frame[...,1:,:]|=intervals
    return frame,intervals


def terrain_clearance(points, scene, *, point_chunk=256, cell_chunk=2048):
    if points.shape[-1]!=3 or not torch.isfinite(points).all():
        raise ValueError('Body surface points must be finite XYZ')
    heights=torch.as_tensor(scene['heightfield_y_m'],device=points.device,dtype=points.dtype).detach()
    origin=torch.as_tensor(scene['min_point_world_xyz_m'],device=points.device,dtype=points.dtype).detach()
    dx=float(scene['dx_m'])
    if heights.ndim!=2 or dx<=0:
        raise ValueError('Expected a square-cell heightfield')
    known=torch.isfinite(heights)
    if 'heightfield_valid' in scene:known=known & torch.as_tensor(scene['heightfield_valid'],device=points.device).bool()
    z,x=torch.where(known)
    if x.numel()==0:
        return points[...,0]*0.,torch.zeros_like(points[...,0],dtype=torch.bool)
    ceiling=torch.maximum(heights[known].max(),points[...,1].detach().max())+2.
    h=heights[z,x]
    centers=torch.stack((origin[0]+x*dx,(h+ceiling)/2,origin[2]+z*dx),-1)
    halves=torch.stack((torch.full_like(h,dx/2),(ceiling-h)/2,torch.full_like(h,dx/2)),-1)
    flat=points.reshape(-1,3)
    outputs=[]
    for chunk in flat.split(point_chunk):
        closest=None
        for c,b in zip(centers.split(cell_chunk),halves.split(cell_chunk)):
            q=(chunk[:,None]-c[None]).abs()-b[None]
            distances=q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
            value=distances.amin(-1)
            closest=value if closest is None else torch.minimum(closest,value)
        outputs.append(-closest)
    cell=torch.floor((flat[:,[0,2]]-origin[[0,2]])/dx+.5).long()
    ix,iz=cell.unbind(-1)
    valid=(ix>=0)&(ix<heights.shape[1])&(iz>=0)&(iz<heights.shape[0])
    valid=valid & known[iz.clamp(0,heights.shape[0]-1),ix.clamp(0,heights.shape[1]-1)]
    return torch.cat(outputs).reshape(points.shape[:-1]),valid.reshape(points.shape[:-1])


def heightfield_gap(points, scene):
    hf=torch.as_tensor(scene['heightfield_y_m'],device=points.device,dtype=points.dtype).detach()
    origin=torch.as_tensor(scene['min_point_world_xyz_m'],device=points.device,dtype=points.dtype).detach()
    cell=torch.floor((points[..., [0,2]]-origin[[0,2]])/float(scene['dx_m'])+.5).long()
    x,z=cell.unbind(-1)
    valid=(x>=0)&(x<hf.shape[1])&(z>=0)&(z<hf.shape[0])
    safe_z,safe_x=z.clamp(0,hf.shape[0]-1),x.clamp(0,hf.shape[1]-1)
    ground=hf[safe_z,safe_x]
    valid=valid & torch.isfinite(ground)
    if 'heightfield_valid' in scene:
        valid=valid & torch.as_tensor(scene['heightfield_valid'],device=points.device).bool()[safe_z,safe_x]
    return points[...,1]-torch.nan_to_num(ground),valid


def source_support_evidence(source_sole, scene, fps=30):
    gap,valid=heightfield_gap(source_sole.detach(),scene)
    frame,_=source_support_masks(source_sole,gap,valid,fps)
    return frame.detach(),gap.detach()


def surface_training_losses(predicted, source, scene, sole_count, *, collision_stride=1):
    clearance,known=terrain_clearance(predicted[:,::collision_stride],scene)
    penetration=(-clearance-.005).clamp_min(0)
    collision=(penetration.square()*known).sum()/known.sum().clamp_min(1)
    source_mask,source_gap=source_support_evidence(source[:,:sole_count],scene)
    pred_gap,pred_known=heightfield_gap(predicted[:,:sole_count],scene)
    eligible=source_mask & pred_known
    contact=((pred_gap-source_gap).abs()*eligible).sum()/eligible.sum().clamp_min(1)
    return {'collision_m2':collision,'source_support_gap_m':contact,
            'known_points':int(known.sum()),'unknown_points':int((~known).sum()),
            'support_points':int(eligible.sum())}
