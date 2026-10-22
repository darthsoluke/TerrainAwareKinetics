"""Target-side distance geometry using SciPy EDT and PyTorch trilinear queries.

The field approximates the union of terrain solids, including vertical sides.
It is never supplied as full-scene actor input. Flat planes use their exact
analytic distance. Masked scene exterior is reported, not treated as free ground.
"""

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt


class TerrainDistanceVolume:
    def __init__(self,scene,vertical_spacing=.1):
        heights=torch.as_tensor(scene['heightfield_y_m']).detach().cpu().numpy()
        self.origin=np.asarray(torch.as_tensor(scene['min_point_world_xyz_m']).cpu(),dtype=np.float32)
        self.dx=float(scene['dx_m']);self.dy=float(vertical_spacing)
        self.heights=torch.from_numpy(np.nan_to_num(heights).astype(np.float32))
        self.known=torch.from_numpy(np.isfinite(heights))
        if 'heightfield_valid' in scene:self.known&=torch.as_tensor(scene['heightfield_valid']).cpu().bool()
        if not self.known.any() or min(self.dx,self.dy)<=0:raise ValueError('Known positive-spacing scene required')
        known=heights[self.known.numpy()]
        self.plane=float(known[0]) if self.known.all() and np.ptp(known)==0 else None
        self.device_cache={}
        if self.plane is not None:self.volume=None;return
        self.y0=np.floor((float(known.min())-1)/self.dy)*self.dy+self.dy/2
        count=int(np.ceil((float(known.max())+3-self.y0)/self.dy))+1
        y=self.y0+np.arange(count)*self.dy
        solid=(y[:,None,None]<np.nan_to_num(heights)[None]) & self.known.numpy()[None]
        inside=distance_transform_edt(solid,sampling=(self.dy,self.dx,self.dx))
        outside=distance_transform_edt(~solid,sampling=(self.dy,self.dx,self.dx))
        # Voxel centers border a surface midway between inside/outside samples.
        correction=min(self.dx,self.dy)/2
        sdf=np.where(solid,-np.maximum(inside-correction,0),np.maximum(outside-correction,0))
        self.volume=torch.from_numpy(sdf.astype(np.float32))[None,None]

    def _tensors(self,device):
        key=str(device)
        if key not in self.device_cache:
            self.device_cache[key]=(self.heights.to(device),self.known.to(device),
                torch.as_tensor(self.origin,device=device),self.volume.to(device) if self.volume is not None else None)
        return self.device_cache[key]

    def query(self,points):
        if points.shape[-1]!=3 or not torch.isfinite(points).all():raise ValueError('Finite body XYZ required')
        hf,known,origin,volume=self._tensors(points.device)
        cell=torch.floor((points.detach()[...,[0,2]]-origin[[0,2]])/self.dx+.5).long()
        x,z=cell.unbind(-1)
        valid=(x>=0)&(x<hf.shape[1])&(z>=0)&(z<hf.shape[0])
        sx,sz=x.clamp(0,hf.shape[1]-1),z.clamp(0,hf.shape[0]-1)
        valid&=known[sz,sx]
        ground=hf[sz,sx]
        if self.plane is not None:return points[...,1]-self.plane,valid
        px=2*(points[...,0]-origin[0])/(self.dx*(hf.shape[1]-1))-1
        pz=2*(points[...,2]-origin[2])/(self.dx*(hf.shape[0]-1))-1
        py=2*(points[...,1]-self.y0)/(self.dy*(volume.shape[2]-1))-1
        grid=torch.stack((px,pz,py),-1).reshape(1,-1,1,1,3)
        distances=F.grid_sample(volume,grid.float(),mode='bilinear',padding_mode='border',align_corners=True).reshape(points.shape[:-1])
        # Deeply fallen points cannot evade collision by leaving the Y volume.
        distances=torch.where(points[...,1]<self.y0,points[...,1]-ground,distances)
        distances=torch.where(py>1,(points[...,1]-ground).clamp_min(0),distances)
        return distances,valid


def swept_surface_points(points,substeps=4):
    alpha=torch.arange(substeps,device=points.device,dtype=points.dtype)/substeps
    swept=points[:-1,None]+alpha[:,None,None]*(points[1:]-points[:-1])[:,None]
    return torch.cat((swept.flatten(0,1),points[-1:]),0)
