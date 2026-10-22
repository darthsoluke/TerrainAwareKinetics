"""Batched versions of the existing nearest-height and trilinear SDF queries."""

import math

import torch
from torch.nn import functional as F

from motionbricks.data.terrain_model_v3 import pack_scene_chunks


class PackedHeightfields:
    def __init__(self,scenes,device,packed=None):
        self.chunks=pack_scene_chunks(scenes,device) if packed is None else packed
        self.sizes=torch.tensor([s['heightfield_y_m'].shape for s in scenes],device=device)

    def query(self,points):
        gaps=[];masks=[]
        for start,stop,hf,origin,dx in self.chunks:
            part=points[start:stop]
            shape=(len(part),)+(1,)*(part.ndim-2)
            cell=torch.floor((part[...,[0,2]]-origin[:,[0,2]].reshape(*shape,2))/dx.reshape(*shape,1)+.5).long()
            x,z=cell.unbind(-1)
            height=self.sizes[start:stop,0].reshape(shape);width=self.sizes[start:stop,1].reshape(shape)
            known=(x>=0)&(x<width)&(z>=0)&(z<height)
            row=torch.arange(len(part),device=points.device).reshape(shape)
            ground=hf[row,torch.minimum(z.clamp_min(0),height-1),torch.minimum(x.clamp_min(0),width-1)]
            masks.append(known & torch.isfinite(ground));gaps.append(part[...,1]-torch.nan_to_num(ground))
        return torch.cat(gaps),torch.cat(masks)


class PackedDistanceVolumes:
    def __init__(self,volumes,device,cell_budget=64_000_000):
        scenes=[{'heightfield_y_m':torch.where(v.known,v.heights,torch.nan),
            'min_point_world_xyz_m':v.origin,'dx_m':v.dx} for v in volumes]
        self.heights=PackedHeightfields(scenes,device);self.groups=[]
        buckets={}
        for i,v in enumerate(volumes):
            if v.plane is None:
                key=tuple(2**math.ceil(math.log2(max(2,n))) for n in v.volume.shape[2:])
                buckets.setdefault(key,[]).append(i)
        self.planes=torch.tensor([v.plane if v.plane is not None else 0. for v in volumes],device=device)
        for dimensions,indices in buckets.items():
            count=max(1,cell_budget//math.prod(dimensions))
            for start in range(0,len(indices),count):
                ids=indices[start:start+count];d,h,w=dimensions
                packed=torch.zeros(len(ids),1,d,h,w)
                for row,index in enumerate(ids):
                    value=volumes[index].volume[0,0];nd,nh,nw=value.shape
                    packed[row,0,:nd,:nh,:nw]=value
                origin=torch.tensor([volumes[i].origin.tolist() for i in ids],device=device)
                sizes=torch.tensor([volumes[i].volume.shape[2:] for i in ids],device=device)
                dx=torch.tensor([volumes[i].dx for i in ids],device=device)
                dy=torch.tensor([volumes[i].dy for i in ids],device=device)
                y0=torch.tensor([volumes[i].y0 for i in ids],device=device)
                self.groups.append((torch.tensor(ids,device=device),packed.to(device),origin,sizes,dx,dy,y0))

    def query(self,points):
        if points.shape[-1]!=3 or not torch.isfinite(points).all():raise ValueError('Finite body XYZ required')
        fallback,known=self.heights.query(points)
        result=points[...,1]-self.planes.reshape((len(points),)+(1,)*(points.ndim-2))
        for ids,volume,origin,sizes,dx,dy,y0 in self.groups:
            p=points[ids];shape=(len(ids),)+(1,)*(p.ndim-2)
            x=(p[...,0]-origin[:,0].reshape(shape))/dx.reshape(shape)
            z=(p[...,2]-origin[:,2].reshape(shape))/dx.reshape(shape)
            y=(p[...,1]-y0.reshape(shape))/dy.reshape(shape)
            # Clamp at each original volume's border, before padding normalization.
            # Padding must never change the original border interpolation.
            xyz=[]
            for a,axis in ((x,2),(z,1),(y,0)):
                upper=(sizes[:,axis]-1).reshape(shape)
                xyz.append(torch.where(a<=0,torch.zeros_like(a),torch.where(a>=upper,upper,a)))
            grid=torch.stack([2*a/(volume.shape[axis+2]-1)-1 for a,axis in zip(xyz,(2,1,0))],-1)
            distance=F.grid_sample(volume,grid.reshape(len(ids),-1,1,1,3).float(),
                mode='bilinear',padding_mode='border',align_corners=True).reshape(p.shape[:-1])
            distance=torch.where(y<0,fallback[ids],distance)
            distance=torch.where(y>(sizes[:,0]-1).reshape(shape),fallback[ids].clamp_min(0),distance)
            result=result.index_copy(0,ids,distance)
        return result,known


def batch_swept_surface_points(points,substeps=4):
    alpha=torch.arange(substeps,device=points.device,dtype=points.dtype)/substeps
    swept=points[:,:-1,None]+alpha[None,None,:,None,None]*(points[:,1:]-points[:,:-1])[:,:,None]
    return torch.cat((swept.flatten(1,2),points[:,-1:]),1)
