"""Export exact E2 scenes and run bounded, headless CPU MuJoCo checks."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from motionbricks.mb_sonic.scene import export_catalog, load_scene


def _model_xml(scene, probe=None):
    root = ET.Element("mujoco", model="mb_sonic_scene_check")
    option = ET.SubElement(root, "option", timestep="0.005")
    ET.SubElement(option, "flag", autoreset="disable")
    ET.SubElement(root, "include", file=scene.scene_include_path.as_posix())
    if probe is not None:
        world = ET.SubElement(root, "worldbody")
        body = ET.SubElement(world, "body", name="probe", pos=" ".join(map(str, probe["position"])),
                             quat=" ".join(map(str, probe["quat"])))
        ET.SubElement(body, "freejoint", name="probe_free")
        shapes = {"sphere": ("sphere", ".04"), "foot": ("box", ".11 .05 .025"),
                  "palm": ("ellipsoid", ".045 .025 .065")}
        kind, size = shapes[probe["kind"]]
        ET.SubElement(body, "geom", name="probe_geom", type=kind, size=size, mass="1", group="1",
                      contype="1", conaffinity="1", condim="3", friction=" ".join(map(str, scene.manifest.friction["coefficients"])))
    return ET.tostring(root, encoding="unicode")


def _sample_points(scene, count):
    rng = np.random.default_rng(20261018)
    nr, nc = scene.height.shape
    cells = rng.integers([0, 0], [nc - 1, nr - 1], size=(count, 2))
    nodes = rng.integers([0, 0], [nc, nr], size=(count, 2))
    lower, upper = scene.bounds_xy_m
    interiors = np.concatenate((cells + [.73, .19], cells + [.17, .79]))
    t = np.linspace(0, 1, count)
    boundary = np.concatenate((np.column_stack((t * (nc - 1), np.zeros(count))),
                               np.column_stack((t * (nc - 1), np.full(count, nr - 1))),
                               np.column_stack((np.zeros(count), t * (nr - 1))),
                               np.column_stack((np.full(count, nc - 1), t * (nr - 1)))))
    midpoint = (lower + upper) / 2
    outside = np.array([[lower[0] - scene.dx, midpoint[1]], [upper[0] + scene.dx, midpoint[1]],
                        [midpoint[0], lower[1] - scene.dx], [midpoint[0], upper[1] + scene.dx]])
    return {"nodes": lower + nodes * scene.dx, "interiors": lower + interiors * scene.dx,
            "boundary": lower + boundary * scene.dx, "unknown": outside}


def verify_scene(manifest_path, samples_per_kind=32):
    """Compare actual native hfield rays/normals with the canonical triangles."""
    import mujoco

    if not 1 <= samples_per_kind <= 1024:
        raise ValueError("samples_per_kind must be between 1 and 1024")
    scene = load_scene(manifest_path)
    model = mujoco.MjModel.from_xml_string(_model_xml(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mb_ground")
    hid = int(model.geom_dataid[gid])
    lo, hi = scene.manifest.heightfield["normalization"]["min_m"], scene.manifest.heightfield["normalization"]["max_m"]
    tolerance_m = float(max(2e-6, (hi - lo) * float(np.finfo(np.float32).eps) * 3))
    max_height_error, max_normal_error = 0., 0.
    results = {}
    direction = np.array([0., 0., -1.])
    normalized = model.hfield_data[model.hfield_adr[hid]:model.hfield_adr[hid] + scene.height.size].reshape(scene.height.shape)
    native_nodes = normalized.astype(float) * model.hfield_size[hid, 2] + data.geom_xpos[gid, 2]
    node_error = float(np.abs(native_nodes - scene.height).max())
    if node_error > tolerance_m:
        raise AssertionError(f"Native full-resolution node reconstruction error: {node_error}")
    boundary_grazing_errors = []
    for label, points in _sample_points(scene, samples_per_kind).items():
        query = scene.query_height(points)
        errors, normals, hits = [], [], []
        for i, point in enumerate(points):
            normal = np.zeros(3)
            ray_start = np.array([point[0], point[1], hi + 1.])
            ray_direction = direction.copy()
            if label == "boundary":
                grazing = mujoco.mj_rayHfield(model, data, gid, ray_start, direction)
                if grazing < 0 or abs(ray_start[2] - grazing - query.height_m[i]) > tolerance_m:
                    boundary_grazing_errors.append(dict(xy_m=point.tolist(), distance_m=float(grazing),
                                                        expected_height_m=float(query.height_m[i])))
                # A vertical ray exactly on a finite wall is tangential. Use an
                # explicit inward incidence, and query at the actual hit XY.
                center = scene.bounds_xy_m.mean(axis=0)
                inward = center - point
                ray_direction[:2] = inward / np.linalg.norm(inward) * (scene.dx * .01 / (hi - lo + 1.))
            distance = mujoco.mj_rayHfield(model, data, gid, ray_start, ray_direction, normal)
            hit = distance >= 0
            hits.append(hit)
            if hit != query.known[i]:
                raise AssertionError(f"{label}: native hit disagrees with known domain at {point}")
            if hit:
                at_hit = scene.query_height((ray_start + distance * ray_direction)[:2])
                if not at_hit.known:
                    raise AssertionError("Native ray hit outside the canonical known domain")
                errors.append(float(abs(ray_start[2] - distance - at_hit.height_m)))
                if at_hit.normal_unique:
                    normals.append(float(np.linalg.norm(normal - at_hit.normal_xyz)))
        error = max(errors, default=0.)
        normal_error = max(normals, default=0.)
        max_height_error = max(max_height_error, error)
        max_normal_error = max(max_normal_error, normal_error)
        if error > tolerance_m or normal_error > .0005:
            raise AssertionError(f"{label}: height error {error}, normal error {normal_error}")
        results[label] = dict(points=len(points), native_hits=sum(hits), max_height_error_m=error,
                              unique_normal_comparisons=len(normals), max_normal_vector_error=normal_error)
    asset_shape = [int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])]
    if asset_shape != list(scene.height.shape):
        raise AssertionError("Native hfield resolution differs from the original")
    invalid = scene.query_height([[np.nan, 0.]])
    if invalid.known.any() or not np.isnan(invalid.height_m).all():
        raise AssertionError("Nonfinite query must remain unknown")
    return dict(scene_id=scene.manifest.scene_id, passed=True, mujoco_version=mujoco.__version__,
                shape_yx=asset_shape, groups=results, max_height_error_m=max_height_error,
                height_tolerance_m=tolerance_m, max_normal_vector_error=max_normal_error,
                all_node_reconstruction_max_error_m=node_error,
                boundary_ray_contract="explicit_inward_incidence; compare_at_actual_hit_XY; exact_node_data_checked_separately",
                exact_boundary_vertical_ray_mismatches=boundary_grazing_errors,
                actual_optimizer_updates=0)


def contact_probe(manifest_path, *, kind, position, quat=(1., 0., 0., 0.), steps=0):
    """Genuine free-body contacts from one XML initial pose, with no state overwrites."""
    import mujoco

    if not 0 <= steps <= 2000:
        raise ValueError("Probe steps must be between 0 and 2000 (at most 10 simulated seconds)")
    if np.asarray(position).shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Probe position must be finite XYZ metres")
    if np.asarray(quat).shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1):
        raise ValueError("Probe orientation must be a unit WXYZ quaternion")
    scene = load_scene(manifest_path)
    model = mujoco.MjModel.from_xml_string(_model_xml(scene, dict(kind=kind, position=position, quat=quat)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    probe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
    object_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, obj["geom_name"]): obj["object_id"]
                  for obj in scene.manifest.objects}
    contacts = []
    max_contacts = 0
    min_distance = None
    for step in range(steps + 1):
        if step:
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
            raise RuntimeError("MuJoCo probe became unstable; no automatic reset is allowed")
        current = []
        for index, contact in enumerate(data.contact):
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if probe_id not in (geom1, geom2):
                continue
            other = geom2 if geom1 == probe_id else geom1
            if other not in object_ids:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            current.append(dict(object_id=object_ids[other], geom_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other),
                                position_m=contact.pos.tolist(), normal_geom1_to_geom2=contact.frame[:3].tolist(),
                                distance_m=float(contact.dist), wrench_contact_frame=force.tolist(),
                                time_s=float(data.time)))
        max_contacts = max(max_contacts, len(current))
        if current:
            contacts = current
            minimum = min(contact["distance_m"] for contact in current)
            min_distance = minimum if min_distance is None else min(min_distance, minimum)
    return dict(scene_id=scene.manifest.scene_id, kind=kind, initial_position_xyz_m=list(position),
                initial_quat_wxyz=list(quat), steps=steps, timestep_s=float(model.opt.timestep),
                simulation_time_s=float(data.time), max_simultaneous_contacts=max_contacts,
                minimum_contact_distance_m=min_distance, last_observed_contacts=contacts,
                final_position_xyz_m=data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")].tolist(),
                passed=max_contacts > 0, actual_optimizer_updates=0, external_forces=False,
                qpos_qvel_overwrites=False, automatic_reset=False, mujoco_version=mujoco.__version__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Export all 15 source maps and the independent rigid-box scene")
    export.add_argument("--catalog", type=Path, default=Path("out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json"))
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--friction", type=float, nargs=3, required=True, metavar=("SLIDING", "TORSIONAL", "ROLLING"))
    export.add_argument("--friction-source", required=True)
    verify = commands.add_parser("verify", help="Native MuJoCo ray height and normal checks; CPU only")
    verify.add_argument("manifests", type=Path, nargs="+")
    verify.add_argument("--samples-per-kind", type=int, default=32)
    probe = commands.add_parser("probe", help="Bounded physical sphere, foot or palm contact check; no viewer")
    probe.add_argument("manifest", type=Path)
    probe.add_argument("--kind", choices=("sphere", "foot", "palm"), required=True)
    probe.add_argument("--position", type=float, nargs=3, required=True)
    probe.add_argument("--quat", type=float, nargs=4, default=(1., 0., 0., 0.))
    probe.add_argument("--steps", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "export":
        paths = export_catalog(args.catalog, args.output, friction=args.friction, friction_source=args.friction_source)
        report = dict(manifests=[str(path) for path in paths], actual_optimizer_updates=0)
    elif args.command == "verify":
        report = dict(scenes=[verify_scene(path, args.samples_per_kind) for path in args.manifests], actual_optimizer_updates=0)
    else:
        report = contact_probe(args.manifest, kind=args.kind, position=args.position, quat=args.quat, steps=args.steps)
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(args.command == "probe" and not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
