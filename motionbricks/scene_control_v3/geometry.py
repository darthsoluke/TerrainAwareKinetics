"""Fixed-budget mesh queries for the untrained v3 performance probe."""

import hashlib
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp


wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


MAX_OBJECTS = 8
MAX_QUERIES = 512
FEATURE_LAYOUT = (
    "ego_xyz_m:3", "mesh_distance_m:1", "ego_face_normal:3", "body_part_id:1",
    "query_time_offset_s:1", "ego_closest_xyz_m:3", "ego_surface_velocity_m_s:3",
    "distance_signed_flag:1",
)


@wp.kernel(enable_backward=False)
def _query_scene(
    points: wp.array2d(dtype=wp.vec3),
    poses: wp.array2d(dtype=wp.transform),
    known: wp.array2d(dtype=wp.bool),
    enabled: wp.array2d(dtype=wp.bool),
    query_valid: wp.array2d(dtype=wp.bool),
    mesh_ids: wp.array(dtype=wp.uint64),
    use_signed: wp.array(dtype=wp.bool),
    search_radii: wp.array(dtype=float),
    bounds: wp.array(dtype=wp.vec3),
    ground_cells: wp.array2d(dtype=wp.bool),
    ground_origin: wp.vec3,
    ground_dx: float,
    has_ground: int,
    object_count: int,
    max_distance: float,
    distance: wp.array2d(dtype=float),
    normal: wp.array2d(dtype=wp.vec3),
    closest: wp.array2d(dtype=wp.vec3),
    gradient: wp.array2d(dtype=wp.vec3),
    gradient_valid: wp.array2d(dtype=wp.bool),
    valid: wp.array2d(dtype=wp.bool),
    object_id: wp.array2d(dtype=wp.int32),
    distance_signed: wp.array2d(dtype=wp.bool),
    triangle_id: wp.array2d(dtype=wp.int32),
    barycentric_uv: wp.array2d(dtype=wp.vec2),
):
    b, i = wp.tid()
    distance[b, i] = 0.0
    normal[b, i] = wp.vec3(0.0)
    closest[b, i] = wp.vec3(0.0)
    gradient[b, i] = wp.vec3(0.0)
    gradient_valid[b, i] = False
    valid[b, i] = False
    object_id[b, i] = -1
    distance_signed[b, i] = False
    triangle_id[b, i] = -1
    barycentric_uv[b, i] = wp.vec2(0.0)
    p = points[b, i]
    usable = query_valid[b, i]
    for d in range(3):
        usable = usable and wp.isfinite(p[d]) and p[d] >= bounds[0][d] and p[d] <= bounds[1][d]
    for j in range(object_count):
        usable = usable and known[b, j]
        if enabled[b, j]:
            pose = poses[b, j]
            for d in range(7):
                usable = usable and wp.isfinite(pose[d])
            rotation = wp.transform_get_rotation(pose)
            usable = usable and wp.abs(wp.dot(rotation, rotation) - 1.0) < 1.0e-4
    if usable and has_ground != 0 and enabled[b, 0]:
        local = wp.transform_point(wp.transform_inverse(poses[b, 0]), p)
        x = (local[0] - ground_origin[0]) / ground_dx
        z = (local[2] - ground_origin[2]) / ground_dx
        nx = ground_cells.shape[1]
        nz = ground_cells.shape[0]
        if x < 0.0 or z < 0.0 or x > float(nx) or z > float(nz):
            usable = False
        else:
            ix = wp.min(int(wp.floor(x)), nx - 1)
            iz = wp.min(int(wp.floor(z)), nz - 1)
            usable = ground_cells[iz, ix]
    if usable:
        best = max_distance
        for j in range(object_count):
            if enabled[b, j]:
                pose = poses[b, j]
                local = wp.transform_point(wp.transform_inverse(pose), p)
                if use_signed[j]:
                    q = wp.mesh_query_point_sign_winding_number(mesh_ids[j], local, search_radii[j], 2.0, 0.5)
                else:
                    q = wp.mesh_query_point_no_sign(mesh_ids[j], local, max_distance)
                if q.result:
                    local_closest = wp.mesh_eval_position(mesh_ids[j], q.face, q.u, q.v)
                    delta = local - local_closest
                    magnitude = wp.length(delta)
                    sign = float(1.0)
                    if use_signed[j]:
                        sign = q.sign
                    candidate = magnitude * sign
                    if candidate < best:
                        best = candidate
                        distance[b, i] = candidate
                        closest[b, i] = wp.transform_point(pose, local_closest)
                        normal[b, i] = wp.transform_vector(pose, wp.mesh_eval_face_normal(mesh_ids[j], q.face))
                        gradient[b, i] = wp.vec3(0.0)
                        gradient_valid[b, i] = magnitude > 1.0e-7
                        if magnitude > 1.0e-7:
                            gradient[b, i] = wp.transform_vector(pose, sign * delta / magnitude)
                        valid[b, i] = True
                        object_id[b, i] = j
                        distance_signed[b, i] = use_signed[j]
                        triangle_id[b, i] = q.face
                        barycentric_uv[b, i] = wp.vec2(q.u, q.v)


class WarpScene:
    """Own immutable local meshes/BVHs; change only tensor rigid-body state.

    Object IDs are fixed slots 0..O-1. Query takes minimum component distance,
    so solid penetration overrides any nearer positive open surface. Overlapping
    solids are not an exact CSG union distance. Open meshes have unsigned
    distance; only outward watertight volumes use winding sign.
    A missing active transform invalidates the scene, never removes an object.
    """

    def __init__(self, meshes, *, signed, known_bounds_m, batch_size=1,
                 query_count=512, device="cpu", max_distance_m=10.0,
                 object_pose=None, ground_domain=None):
        meshes, signed = tuple(meshes), tuple(signed)
        bounds = np.asarray(known_bounds_m, np.float32)
        if not 1 <= len(meshes) <= MAX_OBJECTS or len(signed) != len(meshes):
            raise ValueError("Scene requires 1..8 meshes and an explicit signed flag per mesh")
        if batch_size < 1 or not 1 <= query_count <= MAX_QUERIES:
            raise ValueError("Scene requires positive batch_size and query_count in 1..512")
        if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
                or np.any(bounds[1] <= bounds[0]) or not 0 < max_distance_m < np.inf):
            raise ValueError("Scene requires finite nonempty known bounds and positive search radius")
        wp.init()
        owned = []
        counts = []
        radii = []
        for mesh, use_sign in zip(meshes, signed):
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                raise ValueError("Scene requires actual nonempty triangle geometry")
            if not np.isfinite(mesh.vertices).all() or np.any(mesh.area_faces <= 1e-12):
                raise ValueError("Mesh vertices and triangles must be finite and nondegenerate")
            if use_sign and not mesh.is_volume:
                raise ValueError("Signed queries require an outward watertight volume")
            vertices = np.asarray(mesh.vertices, np.float32)
            indices = np.asarray(mesh.faces, np.int32).ravel()
            radii.append(max(float(max_distance_m), float(np.linalg.norm(mesh.extents)) + 1e-4))
            owned.append(wp.Mesh(
                wp.array(vertices, dtype=wp.vec3, device=device),
                wp.array(indices, dtype=wp.int32, device=device),
                support_winding_number=bool(use_sign), bvh_constructor="sah",
            ))
            counts.append({"vertices": len(vertices), "triangles": len(indices) // 3,
                           "vertex_bytes": vertices.nbytes, "index_bytes": indices.nbytes,
                           "triangle_bounds_bytes": (len(indices) // 3) * 24,
                           "signed_search_radius_m": radii[-1] if use_sign else None,
                           "signed": bool(use_sign)})
        self._meshes = tuple(owned)
        self.batch_size, self.query_count = batch_size, query_count
        self.object_count = len(owned)
        self.device = torch.device(str(self._meshes[0].device))
        self.max_distance_m = float(max_distance_m)
        self._ids = wp.array([mesh.id for mesh in owned], dtype=wp.uint64, device=device)
        self._signed = wp.array(signed, dtype=wp.bool, device=device)
        self._search_radii = wp.array(radii, dtype=float, device=device)
        self._bounds = wp.array(bounds, dtype=wp.vec3, device=device)
        self._has_ground = int(ground_domain is not None)
        origin, dx, cells = ((np.zeros(3), 1.0, np.ones((1, 1), bool))
                             if ground_domain is None else ground_domain)
        self._ground_cells = wp.array(np.asarray(cells, bool), dtype=wp.bool, device=device)
        self._ground_origin = wp.vec3(*np.asarray(origin, np.float32))
        self._ground_dx = float(dx)
        pose = np.zeros((self.object_count, 7), np.float32)
        pose[:, 6] = 1
        if object_pose is not None:
            pose[:] = np.asarray(object_pose, np.float32)
        if (not np.isfinite(pose).all()
                or not np.allclose(np.linalg.norm(pose[:, 3:], axis=-1), 1.0, atol=1e-5)):
            raise ValueError("Object poses require finite XYZ and unit XYZW quaternions")
        self.initial_object_pose = torch.tensor(pose, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        self.initial_object_known = torch.ones((batch_size, self.object_count), dtype=torch.bool, device=self.device)
        self.initial_object_enabled = self.initial_object_known.clone()
        self.storage_summary = {
            "objects": tuple(counts),
            "explicit_geometry_bytes": sum(c["vertex_bytes"] + c["index_bytes"] + c["triangle_bounds_bytes"]
                                           for c in counts),
            "domain_mask_bytes": np.asarray(cells, bool).nbytes,
            "bvh_and_winding_bytes": None,
            "total_device_bytes_measured": None,
            "bvh_constructor": "cpu_sah_cold_path",
        }

    def query(self, points_world_m, object_pose, object_known, object_enabled, query_valid):
        """Query fixed float32[B,Q,3] points using tensor-only inputs/outputs.

        Poses are float32[B,O,7] XYZ+XYZW; masks are bool[B,O]/bool[B,Q].
        Inputs reside on this scene's device. No CPU reads, refits or transfers.
        A known disabled slot means absent. Any unknown declared slot invalidates
        every point. Positive hits use the strict search radius; signed interiors
        also search the object diagonal so penetration cannot hide beyond it.
        No hit yields zero+invalid, not flat.
        Only distance has a first-order point VJP; pose/normal/closest are detached.
        """
        shape = (self.batch_size, self.query_count)
        specs = {
            "distance_m": (shape, torch.float32),
            "normal_world": ((*shape, 3), torch.float32),
            "closest_point_world_m": ((*shape, 3), torch.float32),
            "distance_gradient_world": ((*shape, 3), torch.float32),
            "distance_gradient_valid": (shape, torch.bool),
            "valid": (shape, torch.bool),
            "object_id": (shape, torch.int32),
            "distance_signed": (shape, torch.bool),
            "triangle_id": (shape, torch.int32),
            "barycentric_uv": ((*shape, 2), torch.float32),
        }
        out = {name: torch.empty(size, dtype=dtype, device=self.device)
               for name, (size, dtype) in specs.items()}
        vec3_keys = ("normal_world", "closest_point_world_m", "distance_gradient_world")
        outputs = [wp.from_torch(out[key], dtype=wp.vec3 if key in vec3_keys else
                                 (wp.vec2 if key == "barycentric_uv" else None),
                                 requires_grad=False) for key in specs]
        stream = wp.stream_from_torch(points_world_m.device) if self.device.type == "cuda" else None
        wp.launch(_query_scene, dim=shape, inputs=[
            wp.from_torch(points_world_m, dtype=wp.vec3, requires_grad=False),
            wp.from_torch(object_pose, dtype=wp.transform, requires_grad=False),
            wp.from_torch(object_known, requires_grad=False),
            wp.from_torch(object_enabled, requires_grad=False),
            wp.from_torch(query_valid, requires_grad=False), self._ids, self._signed, self._search_radii, self._bounds,
            self._ground_cells, self._ground_origin, self._ground_dx, self._has_ground,
            self.object_count, self.max_distance_m,
        ], outputs=outputs, device=self._meshes[0].device, stream=stream)
        out["scene_known"] = object_known.all(dim=-1) & object_enabled.any(dim=-1)
        if points_world_m.requires_grad:
            delta = torch.where(out["valid"].unsqueeze(-1), points_world_m - points_world_m.detach(),
                                torch.zeros_like(points_world_m))
            out["distance_m"] = out["distance_m"] + (delta * out["distance_gradient_world"]).sum(-1)
        return out


def encode_scene_features(points_world_m, query_result, root_world_m, heading,
                          body_part_id, query_time_offset_s, object_pose, object_velocity):
    """16D features from current/causally predicted body points, never future GT.

    Heading[B] follows native Y rotation: ego x=c*dx-s*dz, z=s*dx+c*dz.
    Velocity[B,O,6] is world linear XYZ followed by world angular XYZ (rad/s).
    Rigid surface velocity is v + omega cross (closest - object origin).
    Time offsets describe body-point horizons against the supplied object snapshot;
    they do not advance objects. Callers supply their causal poses explicitly.
    Slot 3 is signed only for closed solids, unsigned for open PFNN surfaces;
    feature 15 distance_signed_flag carries that distinction into the actor.
    """
    ids = query_result["object_id"].clamp_min(0).long()
    origins = object_pose[..., :3].gather(1, ids.unsqueeze(-1).expand(-1, -1, 3))
    velocities = object_velocity.gather(1, ids.unsqueeze(-1).expand(-1, -1, 6))
    closest = query_result["closest_point_world_m"]
    surface_velocity = velocities[..., :3] + torch.cross(velocities[..., 3:], closest - origins, dim=-1)
    c, s = heading.cos().unsqueeze(-1), heading.sin().unsqueeze(-1)

    def ego(v):
        return torch.stack((c * v[..., 0] - s * v[..., 2], v[..., 1],
                            s * v[..., 0] + c * v[..., 2]), dim=-1)

    features = torch.cat((
        ego(points_world_m - root_world_m.unsqueeze(1)), query_result["distance_m"].unsqueeze(-1),
        ego(query_result["normal_world"]), body_part_id.to(points_world_m).unsqueeze(-1),
        query_time_offset_s.unsqueeze(-1), ego(closest - root_world_m.unsqueeze(1)), ego(surface_velocity),
        query_result["distance_signed"].to(points_world_m).unsqueeze(-1),
    ), dim=-1)
    valid = (query_result["valid"] & query_result["scene_known"].unsqueeze(-1)
             & torch.isfinite(features).all(-1) & (body_part_id >= 0) & (body_part_id < 15))
    return torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features)), valid


def build_pfnn_scene(scene_npz_path, *, batch_size=1, query_count=512, device="cpu", max_distance_m=10.0):
    """Full-resolution triangle ground, not old piecewise-constant prism parity.

    Missing nodes produce unknown cells, not a flat fill. Ground remains open
    and unsigned; no side walls, bottom cap, face-sign inside or decimation.
    """
    path = Path(scene_npz_path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as data:
        height = np.asarray(data["heightfield_y_m"], np.float32)
        origin = np.asarray(data["min_point_world_xyz_m"], np.float32)
        dx = float(data["dx_m"])
    if (height.ndim != 2 or min(height.shape) < 2 or origin.shape != (3,)
            or not np.isfinite(origin).all() or not 0 < dx < np.inf):
        raise ValueError("PFNN requires metric height[z,x], origin XYZ and positive dx")
    finite = np.isfinite(height)
    cells = finite[:-1, :-1] & finite[1:, :-1] & finite[:-1, 1:] & finite[1:, 1:]
    if not cells.any():
        raise ValueError("PFNN geometry is entirely unknown; a flat replacement is forbidden")
    nz, nx = height.shape
    z, x = np.meshgrid(np.arange(nz, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing="ij")
    vertices = np.stack((origin[0] + x * dx, np.where(finite, height, 0), origin[2] + z * dx), axis=-1)
    ids = np.arange(nz * nx, dtype=np.int32).reshape(nz, nx)
    a, b, c, d = ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:], ids[1:, 1:]
    faces = np.stack((np.stack((a, b, c), -1), np.stack((c, b, d), -1)), -2)[cells].reshape(-1, 3)
    bounds = [[vertices[0, 0, 0], float(height[finite].min()) - 10, vertices[0, 0, 2]],
              [vertices[-1, -1, 0], float(height[finite].max()) + 10, vertices[-1, -1, 2]]]
    scene = WarpScene((trimesh.Trimesh(vertices.reshape(-1, 3), faces, process=False),),
                      signed=(False,), known_bounds_m=bounds, batch_size=batch_size,
                      query_count=query_count, device=device, max_distance_m=max_distance_m,
                      ground_domain=(origin, dx, cells))
    scene.storage_summary.update(source_path=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 representation="full_resolution_open_triangle_ground_unsigned",
                                 old_prism_exact_parity=False, source_shape_zx=height.shape)
    return scene


def build_true3d_scene(*, batch_size=1, query_count=512, device="cpu", max_distance_m=10.0):
    """Independent floor, overhead slab and side wall, each a closed mesh."""
    extents = ((8, 0.2, 8), (2.4, 0.2, 2.0), (0.2, 2.4, 3.0))
    poses = np.zeros((3, 7), np.float32)
    poses[:, 6] = 1
    poses[:, :3] = ((0, -0.1, 0), (0, 1.8, 0), (1.5, 1.2, 0))
    scene = WarpScene(tuple(trimesh.creation.box(extents=e) for e in extents),
                      signed=(True, True, True), known_bounds_m=[[-4, -1, -4], [4, 4, 4]],
                      batch_size=batch_size, query_count=query_count, device=device,
                      max_distance_m=max_distance_m, object_pose=poses)
    scene.storage_summary.update(representation="independent_closed_floor_overhead_side_meshes",
                                 old_prism_exact_parity=False)
    return scene
