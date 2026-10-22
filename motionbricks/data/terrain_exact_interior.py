"""Target-side penetration into heightfield solids using PARC air-box geometry."""

import torch

from .terrain_response_loss import heightfield_gap,terrain_clearance


class ExactTerrainInterior:
    """Exact interior distance; exterior values are vertical gap, not a full SDF.

    Penetration losses only use the negative part. Restricting the expensive
    union-of-air-box query to true interior points avoids voxel interpolation
    treating free space near a step as solid. No actor output is modified.
    """
    def __init__(self,scene):
        self.scene=scene
        self.cache={}

    def interior(self,points):
        key=(points.device,points.dtype)
        if key not in self.cache:
            hf=torch.as_tensor(self.scene['heightfield_y_m'],device=points.device,dtype=points.dtype).detach()
            known=torch.isfinite(hf)
            if 'heightfield_valid' in self.scene:
                known&=torch.as_tensor(self.scene['heightfield_valid'],device=points.device).bool()
            origin=torch.as_tensor(self.scene['min_point_world_xyz_m'],device=points.device,dtype=points.dtype)
            dx=float(self.scene['dx_m']);z,x=torch.where(known)
            centers=torch.stack((origin[0]+x*dx,origin[2]+z*dx),-1)
            self.cache[key]=(centers,hf[z,x],dx)
        centers,heights,dx=self.cache[key]
        pieces=[]
        for part in points.split(256):
            # Argmin chooses the active air prism. Autograd only needs that
            # prism's distance, the exact minimum's usual subgradient.
            with torch.no_grad():
                best=torch.full((len(part),),torch.inf,device=part.device)
                index=torch.zeros(len(part),device=part.device,dtype=torch.long)
                for start in range(0,len(centers),2048):
                    horizontal=(part[:,None,[0,2]]-centers[None,start:start+2048]).abs().sub(dx/2).clamp_min(0)
                    vertical=(heights[None,start:start+2048]-part[:,None,1]).clamp_min(0)
                    squared=horizontal.square().sum(-1)+vertical.square()
                    distance,choice=squared.min(-1)
                    better=distance<best
                    index=torch.where(better,choice+start,index);best=torch.minimum(best,distance)
            horizontal=(part[:,[0,2]]-centers[index]).abs().sub(dx/2).clamp_min(0)
            vertical=(heights[index]-part[:,1]).clamp_min(0)
            pieces.append(-torch.cat((horizontal,vertical[:,None]),-1).norm(dim=-1))
        return torch.cat(pieces)

    def query(self,points):
        gap,known=heightfield_gap(points,self.scene)
        inside=known & (gap<0)
        result=gap.clamp_min(0)
        if inside.any():
            exact=self.interior(points[inside])
            result=result.masked_scatter(inside,exact.clamp_max(0))
        return result,known


class PackedExactInteriors:
    def __init__(self,scenes):
        self.queries=[ExactTerrainInterior(scene) for scene in scenes]

    def query(self,points):
        values=[query.query(row) for query,row in zip(self.queries,points)]
        return torch.stack([value[0] for value in values]),torch.stack([value[1] for value in values])
