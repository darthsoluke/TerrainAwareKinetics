"""Explicit sphere-broadphase candidate; original prism geometry stays unchanged."""

import torch
import warp as wp

from motionbricks.data.bvh_heightfield import (
    BVHHeightfieldPrismSDF, _fields, prism_bvh_indices, prism_squared,
)

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})
SPHERE_BACKEND = 'warp_bvh_sphere_exact'
SPHERE_PROOF_SCHEMA = 'motionbricks.mb_sonic.sphere_bvh_admission.v1'


@wp.kernel
def select_prism_sphere(points: wp.array(dtype=wp.vec3), owners: wp.array(dtype=wp.int32),
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
        limit = wp.max(certainty_radius[i],0.0)
        limit = limit+1.e-4*wp.max(1.0,limit)
        radius = wp.min(wp.min(wp.sqrt(best),0.5),limit)+1.e-4
        searching = int(1)
        while searching != 0:
            query_radius = radius+1.e-4*wp.max(1.0,radius)
            query = wp.bvh_query_sphere(tree, p, query_radius)
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


@torch.library.custom_op('motionbricks::prism_bvh_sphere_indices', mutates_args=())
def prism_bvh_sphere_indices(points: torch.Tensor, owners: torch.Tensor, inside: torch.Tensor,
                             valid: torch.Tensor, certainty_radius: torch.Tensor, field_id: int) -> torch.Tensor:
    field = _fields[field_id]
    output = torch.empty_like(owners)
    wp.launch(select_prism_sphere, dim=points.shape[0], inputs=[
        wp.from_torch(points, dtype=wp.vec3), wp.from_torch(owners),
        wp.from_torch(inside), wp.from_torch(valid), wp.from_torch(certainty_radius),field.w_centers, field.w_halves,
        field.w_levels, field.solid_tree.id, field.air_tree.id], outputs=[wp.from_torch(output)],
        stream=wp.stream_from_torch(torch.cuda.current_stream(points.device)), record_tape=False)
    return output


@prism_bvh_sphere_indices.register_fake
def _fake_indices(points, owners, inside, valid, certainty_radius, field_id):
    return torch.empty_like(owners)


class SphereBVHHeightfieldPrismSDF(BVHHeightfieldPrismSDF):
    """Same field/trees and Torch post-selection math; sphere selector only."""

    def query(self, points):
        return self._query(points, prism_bvh_sphere_indices)

    def reference_query(self, points):
        """Proof-only old selector on this same owner; never an automatic fallback."""
        return self._query(points, prism_bvh_indices)

    def _query(self, points, selector):
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
        indices = selector(p, self.owners[cz,cx], inside.to(torch.int32),
                           valid.to(torch.int32),edge+1e-6,self.field_id).long()
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
