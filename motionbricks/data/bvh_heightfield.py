"""Exact native heightfield-prism selection accelerated with a spatial BVH."""

import itertools
import weakref

import numpy as np
import torch
import warp as wp

from .joint_sdf_observation import HeightfieldPrismSDF


wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})
_fields = weakref.WeakValueDictionary()
_ids = itertools.count(1)


@wp.func
def prism_squared(p: wp.vec3, center: wp.vec2, half: wp.vec2, height: float, interior: int):
    x = wp.max(wp.abs(p[0]-center[0])-half[0], 0.0)
    z = wp.max(wp.abs(p[2]-center[1])-half[1], 0.0)
    vertical = p[1]-height
    if interior != 0:
        vertical = -vertical
    vertical = wp.max(vertical, 0.0)
    return (x*x+z*z)+vertical*vertical


@wp.kernel
def select_prism(points: wp.array(dtype=wp.vec3), owners: wp.array(dtype=wp.int32),
                 inside: wp.array(dtype=wp.int32), valid: wp.array(dtype=wp.int32),
                 certainty_radius: wp.array(dtype=wp.float32),
                 centers: wp.array(dtype=wp.vec2), halves: wp.array(dtype=wp.vec2),
                 levels: wp.array(dtype=wp.float32), solid_tree: wp.uint64, air_tree: wp.uint64,
                 indices: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = points[i]
    best_id = owners[i]
    best = prism_squared(p, centers[best_id], halves[best_id], levels[best_id], inside[i])
    if valid[i] != 0:
        tree=solid_tree
        if inside[i] != 0:
            tree=air_tree
        # Search near the query first. A distance within the visited radius is
        # globally minimal; beyond the native certainty radius output is unknown.
        limit = wp.max(certainty_radius[i],0.0)
        limit = limit+1.e-4*wp.max(1.0,limit)
        radius = wp.min(wp.min(wp.sqrt(best),0.5),limit)+1.e-4
        searching = int(1)
        while searching != 0:
            query_radius = radius+1.e-4*wp.max(1.0,radius)
            query = wp.bvh_query_aabb(tree, wp.vec3(p[0]-query_radius, p[1]-query_radius, p[2]-query_radius),
                                     wp.vec3(p[0]+query_radius, p[1]+query_radius, p[2]+query_radius))
            candidate = int(0)
            while wp.bvh_query_next(query, candidate):
                value = prism_squared(p, centers[candidate], halves[candidate], levels[candidate], inside[i])
                if value < best or (value == best and candidate < best_id):
                    best, best_id = value, candidate
            if radius >= wp.sqrt(best) or radius >= limit:
                searching = 0
            else:
                radius = wp.min(wp.min(radius*2.0,wp.sqrt(best)+1.e-4),limit)
    indices[i] = best_id


@torch.library.custom_op('motionbricks::prism_bvh_indices', mutates_args=())
def prism_bvh_indices(points: torch.Tensor, owners: torch.Tensor, inside: torch.Tensor,
                      valid: torch.Tensor, certainty_radius: torch.Tensor, field_id: int) -> torch.Tensor:
    field = _fields[field_id]
    output = torch.empty_like(owners)
    wp.launch(select_prism, dim=points.shape[0], inputs=[
        wp.from_torch(points, dtype=wp.vec3), wp.from_torch(owners),
        wp.from_torch(inside), wp.from_torch(valid), wp.from_torch(certainty_radius),field.w_centers, field.w_halves,
        field.w_levels, field.solid_tree.id, field.air_tree.id], outputs=[wp.from_torch(output)],
        stream=wp.stream_from_torch(torch.cuda.current_stream(points.device)), record_tape=False)
    return output


@prism_bvh_indices.register_fake
def _fake_indices(points, owners, inside, valid, certainty_radius, field_id):
    return torch.empty_like(owners)


class BVHHeightfieldPrismSDF(HeightfieldPrismSDF):
    """All-known CUDA float32 scenes; same prisms, ties, normals and boundary gate."""

    def __init__(self, scene, device='cuda:0'):
        wp.init()
        super().__init__(scene)
        if not self.known.all() or torch.device(device).type != 'cuda':
            raise ValueError('BVH heightfields require fully known terrain and a CUDA device')
        sample = torch.zeros(1, 3, device=device, dtype=torch.float32)
        boxes, _, _, _, _ = self._tensors(sample)
        self.centers, self.halves, self.levels = boxes
        owners = np.empty(self.heights.shape, dtype=np.int32)
        for index, (x0, x1, z0, z1, _) in enumerate(self.rectangles):
            owners[z0:z1, x0:x1] = index
        self.owners = torch.from_numpy(owners).to(device)
        # Outside queries cannot be below the minimum terrain height; inside
        # queries cannot be above its maximum. These finite BVH caps are exact.
        minimum = torch.full_like(self.levels,float(self.levels.min())-1.)
        maximum = torch.full_like(self.levels,float(self.levels.max())+1.)
        self.lowers = torch.stack((self.centers[:,0]-self.halves[:,0], minimum,
                                   self.centers[:,1]-self.halves[:,1]), -1).contiguous()
        self.uppers = torch.stack((self.centers[:,0]+self.halves[:,0], self.levels,
                                   self.centers[:,1]+self.halves[:,1]), -1).contiguous()
        self.air_lowers=self.lowers.clone();self.air_lowers[:,1]=self.levels
        self.air_uppers=self.uppers.clone();self.air_uppers[:,1]=maximum
        self.w_centers = wp.from_torch(self.centers, dtype=wp.vec2)
        self.w_halves = wp.from_torch(self.halves, dtype=wp.vec2)
        self.w_levels = wp.from_torch(self.levels)
        self.solid_tree = wp.Bvh(wp.from_torch(self.lowers, dtype=wp.vec3),
                          wp.from_torch(self.uppers, dtype=wp.vec3), constructor='sah')
        self.air_tree = wp.Bvh(wp.from_torch(self.air_lowers,dtype=wp.vec3),
                              wp.from_torch(self.air_uppers,dtype=wp.vec3),constructor='sah')
        self.field_id = next(_ids)
        _fields[self.field_id] = self

    def query(self, points):
        if not points.is_cuda or points.dtype != torch.float32 or points.shape[-1] != 3:
            raise ValueError('BVH scene queries require CUDA float32 XYZ points')
        if not torch.isfinite(points).all():
            raise ValueError('Finite XYZ queries required')
        boxes, _, hf, known, origin = self._tensors(points)
        centers, halves, levels = boxes
        p = points.reshape(-1,3).contiguous()
        cell = torch.floor((p[:,[0,2]].detach()-origin[[0,2]])/self.dx+.5).long()
        x, z = cell.unbind(-1)
        valid = (x>=0)&(z>=0)&(x<hf.shape[1])&(z<hf.shape[0])
        cx, cz = x.clamp(0,hf.shape[1]-1), z.clamp(0,hf.shape[0]-1)
        inside = p[:,1].detach()<hf[cz,cx]
        edge = torch.stack((p[:,0]-(origin[0]-self.dx/2),origin[0]+(hf.shape[1]-.5)*self.dx-p[:,0],
                            p[:,2]-(origin[2]-self.dx/2),origin[2]+(hf.shape[0]-.5)*self.dx-p[:,2]),-1).amin(-1)
        indices = prism_bvh_indices(p, self.owners[cz,cx], inside.to(torch.int32),
                                    valid.to(torch.int32),edge+1e-6,self.field_id).long()
        # Retain the native Torch calculation after exact active-prism selection.
        difference = p[:,[0,2]]-centers[indices]
        horizontal = difference.abs().sub(halves[indices]).clamp_min(0)*difference.sign()
        sign = torch.where(inside,-torch.ones_like(p[:,1]),torch.ones_like(p[:,1]))
        vertical = ((p[:,1]-levels[indices])*sign).clamp_min(0)
        vector = torch.stack((horizontal[:,0]*sign,vertical,horizontal[:,1]*sign),-1)
        magnitude = vector.norm(dim=-1)
        valid &= magnitude.detach() <= edge.detach()+1e-6
        distance = (sign*magnitude).reshape(points.shape[:-1])
        normal = (vector/magnitude.clamp_min(1e-8)[:,None]).reshape(points.shape)
        valid = valid.reshape(points.shape[:-1])
        return torch.where(valid,distance,0),torch.where(valid[...,None],normal,0),valid
