"""CPU triangle/volume queries in explicit object frames; no motion correction."""

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import trimesh
import warp as wp


QUERY_SCHEMA = "motionbricks.scene_control_v2.query.v1"


def _empty(n: int) -> dict:
    return {
        "distance_m": np.zeros(n, np.float32),
        "normal_world": np.zeros((n, 3), np.float32),
        "closest_point_world_m": np.zeros((n, 3), np.float32),
        "closest_point_valid": np.zeros(n, bool),
        "distance_gradient_world": np.zeros((n, 3), np.float32),
        "distance_gradient_valid": np.zeros(n, bool),
        "valid": np.zeros(n, bool),
        "object_id": np.full(n, -1, np.int32),
    }


def rigid_transform(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Object transforms must be finite rigid SE(3), without scale")
    return matrix


@wp.kernel
def _query_mesh(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    use_signed: int,
    max_distance: float,
    distances: wp.array(dtype=float),
    normals: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    gradients: wp.array(dtype=wp.vec3),
    hits: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh_id, points[i], max_distance)
    if use_signed != 0:
        q = wp.mesh_query_point_sign_winding_number(mesh_id, points[i], max_distance, 2.0, 0.5)
    if q.result:
        closest = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        distance = wp.length(points[i] - closest)
        if use_signed != 0:
            distance = distance * q.sign
        distances[i] = distance
        normals[i] = wp.mesh_eval_face_normal(mesh_id, q.face)
        closest_points[i] = closest
        delta = points[i] - closest
        magnitude = wp.length(delta)
        if magnitude > 1e-7:
            gradient = delta / magnitude
            if use_signed != 0:
                gradient = gradient * q.sign
            gradients[i] = gradient
        hits[i] = 1


class LocalField(Protocol):
    distance_semantics: str

    def query_local(self, points: np.ndarray) -> dict: ...


class MeshField:
    """BVH closest triangles with Warp winding-number sign, float32 on CPU.

    Magnitudes are triangle distances, not vertex/height approximations. Sign is
    Warp's fast winding-number classification, not a certified exact solid test.
    Signed inputs must be watertight, consistently wound, outward volumes.
    """

    def __init__(self, mesh: trimesh.Trimesh, *, signed: bool = True,
                 max_distance_m: float = 1e6):
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            raise ValueError("MeshField requires a nonempty triangle mesh")
        if (not np.isfinite(mesh.vertices).all() or np.any(mesh.area_faces <= 1e-12)
                or (signed and not mesh.is_volume)):
            raise ValueError("Signed meshes require outward watertight volumes and nondegenerate faces")
        if not np.isfinite(max_distance_m) or max_distance_m <= 0:
            raise ValueError("max_distance_m must be positive and finite")
        self.signed = signed
        self.max_distance_m = float(max_distance_m)
        self.distance_semantics = ("mesh_triangle_distance_warp_winding_sign" if signed
                                   else "unsigned_mesh_triangle_distance")
        wp.init()
        self._mesh = wp.Mesh(
            wp.array(np.asarray(mesh.vertices, np.float32), dtype=wp.vec3, device="cpu"),
            wp.array(np.asarray(mesh.faces, np.int32).ravel(), dtype=wp.int32, device="cpu"),
            support_winding_number=signed,
            bvh_constructor="sah",
        )

    def query_local(self, points: np.ndarray) -> dict:
        points = np.asarray(points, np.float32)
        out = _empty(len(points))
        finite = np.isfinite(points).all(axis=1)
        n = int(finite.sum())
        if n:
            distances = wp.zeros(n, dtype=float, device="cpu")
            normals = wp.zeros(n, dtype=wp.vec3, device="cpu")
            closest = wp.zeros(n, dtype=wp.vec3, device="cpu")
            gradients = wp.zeros(n, dtype=wp.vec3, device="cpu")
            hits = wp.zeros(n, dtype=wp.int32, device="cpu")
            wp.launch(_query_mesh, dim=n,
                      inputs=[self._mesh.id, wp.array(points[finite], dtype=wp.vec3, device="cpu"),
                              int(self.signed), self.max_distance_m],
                      outputs=[distances, normals, closest, gradients, hits], device="cpu")
            out["distance_m"][finite] = distances.numpy()
            out["normal_world"][finite] = normals.numpy()
            out["closest_point_world_m"][finite] = closest.numpy()
            out["closest_point_valid"][finite] = hits.numpy().astype(bool)
            out["distance_gradient_world"][finite] = gradients.numpy()
            out["valid"][finite] = hits.numpy().astype(bool)
            out["distance_gradient_valid"][finite] = (hits.numpy() != 0) & (np.abs(distances.numpy()) > 1e-7)
        return out


class VolumeField:
    """Trilinear interpolation of supplied XYZ SDF/normal/valid grids.

    This queries a sampled volume, not an exact mesh distance. No occupancy to
    distance conversion or fabricated signs are performed.
    """

    distance_semantics = "trilinear_supplied_sdf_m"

    def __init__(self, axes_m, distance_m, normal_local, valid):
        axes = tuple(np.asarray(axis, np.float64) for axis in axes_m)
        shape = tuple(len(axis) for axis in axes)
        if len(axes) != 3 or any(len(a) < 2 or np.any(np.diff(a) <= 0) for a in axes):
            raise ValueError("Volume axes must be three strictly increasing metric axes")
        distance = np.asarray(distance_m, np.float32)
        normals = np.asarray(normal_local, np.float32)
        valid = np.asarray(valid, bool)
        if distance.shape != shape or normals.shape != (*shape, 3) or valid.shape != shape:
            raise ValueError("Distance, normal and validity grids must share XYZ axes")
        if not np.isfinite(distance[valid]).all() or not np.isfinite(normals[valid]).all():
            raise ValueError("Valid volume nodes must be finite")
        values = np.concatenate((np.where(valid, distance, 0)[..., None],
                                 np.where(valid[..., None], normals, 0), valid[..., None]), axis=-1)
        self._interp = RegularGridInterpolator(axes, values, bounds_error=False, fill_value=0)
        self._axes = axes
        self._distance = np.where(valid, distance, 0)

    def query_local(self, points: np.ndarray) -> dict:
        values = self._interp(points)
        norm = np.linalg.norm(values[:, 1:4], axis=-1)
        valid = (values[:, 4] >= 1 - 1e-7) & (norm > 1e-8) & np.isfinite(values).all(axis=1)
        out = _empty(len(values))
        out["distance_m"][valid] = values[valid, 0]
        out["normal_world"][valid] = values[valid, 1:4] / norm[valid, None]
        out["valid"] = valid
        # Analytic derivative of the same trilinear interpolant, not its normal grid.
        p = np.asarray(points)[valid]
        indices = [np.clip(np.searchsorted(axis, p[:, d], side="right") - 1, 0, len(axis) - 2)
                   for d, axis in enumerate(self._axes)]
        widths = [axis[index + 1] - axis[index] for axis, index in zip(self._axes, indices)]
        fraction = [(p[:, d] - axis[indices[d]]) / widths[d] for d, axis in enumerate(self._axes)]
        gradient = np.zeros((len(p), 3))
        for corner in np.ndindex(2, 2, 2):
            node = self._distance[tuple(indices[d] + corner[d] for d in range(3))]
            for d in range(3):
                weight = np.full(len(p), 1 if corner[d] else -1, dtype=float) / widths[d]
                for other in range(3):
                    if other != d:
                        weight *= fraction[other] if corner[other] else 1 - fraction[other]
                gradient[:, d] += node * weight
        out["distance_gradient_world"][valid] = gradient
        out["distance_gradient_valid"] = valid.copy()
        return out


class LegacyHeightfieldField:
    """Explicit legacy selection: vertical clearance, never a 3-D SDF."""

    distance_semantics = "legacy_heightfield_vertical_clearance_m"

    def __init__(self, x_m, z_m, height_y_m, valid):
        heights = np.asarray(height_y_m, np.float32)
        valid = np.asarray(valid, bool)
        dx, dz = np.gradient(heights, x_m, z_m)
        normals = np.stack((-dx, np.ones_like(heights), -dz), axis=-1)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
        values = np.concatenate((heights[..., None], normals, valid[..., None]), axis=-1)
        self._interp = RegularGridInterpolator((x_m, z_m), values, bounds_error=False, fill_value=0)

    def query_local(self, points: np.ndarray) -> dict:
        values = self._interp(np.asarray(points)[:, [0, 2]])
        out = _empty(len(points))
        valid = (values[:, 4] >= 1 - 1e-7) & np.isfinite(points).all(axis=1)
        out["valid"] = valid
        out["distance_m"][valid] = points[valid, 1] - values[valid, 0]
        out["normal_world"][valid] = values[valid, 1:4]
        return out


@dataclass(frozen=True)
class SceneObject:
    object_id: int
    name: str
    field: LocalField
    object_to_world: np.ndarray
    valid: np.ndarray  # Known transform/geometry track; False means unknown, never absent.

    def __post_init__(self):
        if (self.object_id < 0 or len(self.object_to_world) != len(self.valid)
                or np.asarray(self.valid).dtype != np.bool_):
            raise ValueError("Objects require nonnegative ID and aligned transform validity")
        for matrix in np.asarray(self.object_to_world)[np.asarray(self.valid, bool)]:
            rigid_transform(matrix)


class SceneQuery:
    """Minimum per-object distances, with selected object's outward face normal.

    Overlap interiors use minimum component SDF, not exact union distance.
    `valid` requires a known world region and every declared component's track
    and local distance to be known. An unknown component invalidates the whole
    scene distance, even if a floor or another component can be queried.
    Presence/absence is not encoded by SceneObject.valid in this schema.
    Actor callers use sample_scene_queries for causal track checks; these
    low-level queries also serve explicitly offline training objectives.
    """

    def __init__(self, objects: tuple[SceneObject, ...], valid_bounds_world_m,
                 *, validity_fn: Callable | None = None):
        ids = [obj.object_id for obj in objects]
        bounds = np.asarray(valid_bounds_world_m, np.float32)
        if (len(ids) != len(set(ids)) or not ids or bounds.shape != (2, 3)
                or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0])):
            raise ValueError("Scene requires unique objects and finite nonempty known bounds")
        semantics = {obj.field.distance_semantics for obj in objects}
        if len(semantics) != 1:
            raise ValueError("Do not mix mesh, unsigned, volume or legacy distance semantics")
        self.objects = objects
        self.valid_bounds_world_m = bounds
        self.validity_fn = validity_fn
        self.distance_semantics = "minimum_component:" + next(iter(semantics))

    def query_world(self, world_points_m, time_index) -> dict:
        points = np.asarray(world_points_m, np.float32)
        times = np.asarray(time_index)
        if points.ndim != 2 or points.shape[1] != 3 or times.shape != (len(points),):
            raise ValueError("Queries require points[Q,3] and time_index[Q]")
        if not np.issubdtype(times.dtype, np.integer) or np.any(times < 0):
            raise ValueError("time_index must be nonnegative integer indices")
        out = _empty(len(points))
        known = (np.isfinite(points).all(axis=1)
                 & (points >= self.valid_bounds_world_m[0]).all(axis=1)
                 & (points <= self.valid_bounds_world_m[1]).all(axis=1))
        if self.validity_fn is not None:
            known &= np.asarray(self.validity_fn(points, times), bool)
        best = np.full(len(points), np.inf)
        for obj in self.objects:
            if len(times) and times.max() >= len(obj.valid):
                raise ValueError("Requested time is outside object transform track")
            known &= np.asarray(obj.valid)[times]
        for obj in self.objects:
            for t in np.unique(times[known]):
                indices = np.flatnonzero(known & (times == t))
                matrix = obj.object_to_world[t]
                local = (points[indices] - matrix[:3, 3]) @ matrix[:3, :3]
                result = obj.field.query_local(local)
                known[indices] &= result["valid"]
                use = result["valid"] & (result["distance_m"] < best[indices])
                chosen = indices[use]
                best[chosen] = result["distance_m"][use]
                out["distance_m"][chosen] = result["distance_m"][use]
                out["normal_world"][chosen] = result["normal_world"][use] @ matrix[:3, :3].T
                out["closest_point_world_m"][chosen] = (
                    result["closest_point_world_m"][use] @ matrix[:3, :3].T + matrix[:3, 3])
                out["closest_point_valid"][chosen] = result["closest_point_valid"][use]
                out["distance_gradient_world"][chosen] = result["distance_gradient_world"][use] @ matrix[:3, :3].T
                out["distance_gradient_valid"][chosen] = result["distance_gradient_valid"][use]
                out["object_id"][chosen] = obj.object_id
                out["valid"][chosen] = True
        out["valid"] &= known
        out["distance_gradient_valid"] &= out["valid"]
        out["closest_point_valid"] &= out["valid"]
        for key in ("distance_m", "normal_world", "distance_gradient_world"):
            out[key][~out["valid"]] = 0
        out["object_id"][~out["valid"]] = -1
        out["closest_point_world_m"][~out["closest_point_valid"]] = 0
        out.update(schema=QUERY_SCHEMA, world_points_m=points.copy(),
                   time_index=times.astype(np.int32), distance_semantics=self.distance_semantics,
                   valid_semantics="all_declared_components_known")
        return out

    def query_world_torch(self, world_points_m, time_index) -> dict:
        """First-order distance VJP for predicted points, on their Torch device.

        CPU BVH lookup synchronizes/copies inputs. The returned SDF derivative is
        geometric `(p-closest)/norm * sign`, never the nearest triangle normal.
        Mesh/transform derivatives and second derivatives are not implemented.
        """
        import torch

        if any(isinstance(obj.field, LegacyHeightfieldField) for obj in self.objects):
            raise ValueError("Torch physical objectives require mesh or supplied volume, not legacy heightfield")
        if world_points_m.ndim != 2 or world_points_m.shape[1] != 3 or not world_points_m.is_floating_point():
            raise ValueError("Torch queries require floating points[Q,3]")
        result = self.query_world(world_points_m.detach().cpu().numpy(), time_index.detach().cpu().numpy())
        out = {key: torch.as_tensor(value, device=world_points_m.device)
               if isinstance(value, np.ndarray) else value for key, value in result.items()}
        gradient = out["distance_gradient_world"].to(world_points_m)
        gradient = gradient * out["distance_gradient_valid"].unsqueeze(-1)
        # Geometry supplies the local VJP; invalid/nondifferentiable slots have zero VJP.
        delta = torch.where(out["valid"].unsqueeze(-1), world_points_m - world_points_m.detach(),
                            torch.zeros_like(world_points_m))
        out["distance_m"] = out["distance_m"].to(world_points_m) + (delta * gradient).sum(dim=-1)
        out["world_points_m"] = world_points_m
        return out

    def anchors_world(self, object_ids, time_index, anchor_local_m, normal_local) -> dict:
        ids, times = np.asarray(object_ids), np.asarray(time_index)
        local, normals = np.asarray(anchor_local_m), np.asarray(normal_local)
        points_out = np.zeros_like(local, dtype=np.float32)
        normals_out = np.zeros_like(normals, dtype=np.float32)
        valid = np.zeros(len(ids), bool)
        for obj in self.objects:
            for t in np.unique(times[ids == obj.object_id]):
                if t < 0 or t >= len(obj.valid) or not obj.valid[t]:
                    continue
                use = (ids == obj.object_id) & (times == t)
                matrix = obj.object_to_world[t]
                points_out[use] = local[use] @ matrix[:3, :3].T + matrix[:3, 3]
                normals_out[use] = normals[use] @ matrix[:3, :3].T
                valid[use] = np.isfinite(local[use]).all(axis=1) & np.isfinite(normals[use]).all(axis=1)
        return {"world_points_m": points_out, "normal_world": normals_out, "valid": valid}
