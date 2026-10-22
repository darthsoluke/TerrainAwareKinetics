"""Rigid coordinate augmentation; the measured scene and motion stay paired."""

from dataclasses import dataclass

import torch


@dataclass
class HorizontalSceneFrame:
    yaw: torch.Tensor
    pivot_xz: torch.Tensor

    def _rotate(self,value,angle):
        c,s=angle.cos(),angle.sin()
        shape=(len(angle),)+(1,)*(value.ndim-2)
        c,s=c.reshape(shape),s.reshape(shape)
        return torch.stack((c*value[...,0]+s*value[...,1],-s*value[...,0]+c*value[...,1]),-1)

    def vectors(self,value,*,inverse=False):
        return self._rotate(value,(-self.yaw if inverse else self.yaw).to(value))

    def points(self,value,*,inverse=False):
        pivot=self.pivot_xz.to(value).reshape((len(value),)+(1,)*(value.ndim-2)+(2,))
        return self.vectors(value-pivot,inverse=inverse)+pivot

    def task(self,value,*,inverse=False):
        result=value.clone()
        result[:,:2]=self.points(value[:,:2],inverse=inverse)
        result[:,2:4]=self.vectors(value[:,2:4],inverse=inverse)
        heading=torch.atan2(value[:,5],value[:,4])+(-self.yaw if inverse else self.yaw).to(value)
        result[:,4]=heading.cos();result[:,5]=heading.sin()
        return result

    def motion(self,representation,value,*,inverse=False):
        offset=(-self.yaw if inverse else self.yaw).to(value)
        heading=torch.atan2(value[:,0,4],value[:,0,3])+offset
        transformed=representation.change_first_heading(value,heading,False,False)
        # The native helper recenters XZ. Restore the same rigid world transform
        # for history, references and predictions, including drifting outputs.
        first=self.points(value[:,0,[0,2]],inverse=inverse)
        transformed[...,[0,2]]+=first[:,None]
        return transformed

    def row(self,index):
        return HorizontalSceneFrame(self.yaw[index:index+1],self.pivot_xz[index:index+1])


def known_history_centers(history,packed_scene):
    """Same nearest-cell validity as the near observation center; no height fill."""
    rows=[]
    for start,stop,heights,origins,spacing in packed_scene:
        cell=torch.floor((history[start:stop,3,[0,2]]-origins[:,[0,2]])/spacing[:,None]+.5).long()
        x,z=cell.unbind(-1)
        known=(x>=0)&(x<heights.shape[2])&(z>=0)&(z<heights.shape[1])
        samples=heights[torch.arange(stop-start,device=history.device),z.clamp(0,heights.shape[1]-1),x.clamp(0,heights.shape[2]-1)]
        rows.append(known & torch.isfinite(samples))
    return torch.cat(rows)
