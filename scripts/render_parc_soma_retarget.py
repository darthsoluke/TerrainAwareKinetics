"""Render an unmodified PARC-to-SOMA retarget clip beside its PARC source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import GifImagePlugin, Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CHARACTER_XML = ROOT / "out/vendor/PARC/data/assets/humanoid.xml"
Y_TO_Z = np.asarray(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)), dtype=np.float64)
HEADER_HEIGHT = 58
FOOTER_HEIGHT = 46
PANEL_AZIMUTH = 60
PANEL_ELEVATION = -19
MIN_CAMERA_DISTANCE = 4.6

NPZ_FIELDS = (
    "source_positions_zup", "source_rotations_xyzw", "source_names", "source_contacts",
    "target_positions_yup", "target_names", "target_parents",
    "target_sole_vertices_yup", "hf", "min_point", "dx", "uniform_scale", "source_fps",
)


def _scalar(value, field):
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must be a numeric scalar")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _string_names(value, field, count):
    names = np.asarray(value)
    if names.shape != (count,) or names.dtype.kind not in "US":
        raise ValueError(f"{field} must contain {count} strings")
    decoded = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names.tolist()]
    if any(not name for name in decoded) or len(set(decoded)) != count:
        raise ValueError(f"{field} must contain {count} nonempty unique strings")
    return decoded


def _validate_tree(parents, count):
    parents = np.asarray(parents)
    if parents.shape != (count,) or parents.dtype.kind not in "iu":
        raise ValueError(f"target_parents must contain {count} integer parent indices")
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1 or np.any((parents < -1) | (parents >= count)):
        raise ValueError("target_parents must describe one acyclic tree")
    root = int(roots[0])
    for joint in range(count):
        seen = set()
        current = joint
        while current != -1:
            if current in seen:
                raise ValueError("target_parents contains a cycle")
            seen.add(current)
            current = int(parents[current])
        if root not in seen:
            raise ValueError("target_parents contains a joint disconnected from its root")
    return parents.astype(np.int64, copy=True)


def load_clip(path):
    """Load and validate the renderer-only NPZ contract without pickle support."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [field for field in NPZ_FIELDS if field not in archive.files]
        if missing:
            raise ValueError(f"Retarget NPZ is missing fields: {', '.join(missing)}")
        data = {field: archive[field].copy() for field in NPZ_FIELDS}

    source = data["source_positions_zup"]
    source_rot = data["source_rotations_xyzw"]
    source_contacts = data["source_contacts"]
    target = data["target_positions_yup"]
    soles = data["target_sole_vertices_yup"]
    if source.ndim != 3 or source.shape[1:] != (15, 3) or len(source) == 0:
        raise ValueError("source_positions_zup must have shape [T,15,3] with T > 0")
    frames = len(source)
    if source_rot.shape != (frames, 15, 4):
        raise ValueError("source_rotations_xyzw must have shape [T,15,4]")
    if source_contacts.shape != (frames, 15):
        raise ValueError("source_contacts must have shape [T,15]")
    data["source_names"] = _string_names(data["source_names"], "source_names", 15)
    if target.shape != (frames, 30, 3):
        raise ValueError("target_positions_yup must have shape [T,30,3]")
    data["target_names"] = _string_names(data["target_names"], "target_names", 30)
    data["target_parents"] = _validate_tree(data["target_parents"], 30)
    if soles.ndim != 3 or soles.shape[0] != frames or soles.shape[2] != 3 or soles.shape[1] == 0:
        raise ValueError("target_sole_vertices_yup must have shape [T,N,3] with N > 0")
    if data["hf"].ndim != 2 or min(data["hf"].shape) < 1:
        raise ValueError("hf must be a nonempty two-dimensional terrain height array")
    if data["min_point"].shape not in ((2,), (3,)):
        raise ValueError("min_point must contain the terrain X/Y origin")

    scale = _scalar(data["uniform_scale"], "uniform_scale")
    fps = _scalar(data["source_fps"], "source_fps")
    dx = _scalar(data["dx"], "dx")
    if scale <= 0:
        raise ValueError("uniform_scale must be positive")
    if dx <= 0:
        raise ValueError("dx must be positive")
    if fps <= 0 or fps > 100:
        raise ValueError("source_fps must be in (0,100] for frame-accurate GIF timing")

    numeric_fields = ("source_positions_zup", "source_rotations_xyzw", "source_contacts",
                      "target_positions_yup", "target_sole_vertices_yup", "hf", "min_point")
    for field in numeric_fields:
        values = data[field]
        is_boolean_contact = field == "source_contacts" and values.dtype.kind == "b"
        if (not np.issubdtype(values.dtype, np.number) and not is_boolean_contact) or not np.isfinite(values).all():
            raise ValueError(f"{field} must contain only finite numeric values")
    quaternion_norms = np.linalg.norm(source_rot, axis=-1)
    if np.max(np.abs(quaternion_norms - 1.0)) > 1e-4:
        raise ValueError("source_rotations_xyzw must contain unit xyzw quaternions")
    return data


def target_yup_to_zup(points_yup):
    """Express Y-up SOMA positions in the source/terrain Z-up display frame."""
    points = np.asarray(points_yup, dtype=np.float64)
    if points.shape[-1:] != (3,):
        raise ValueError("positions must end in a three-coordinate axis")
    return points @ Y_TO_Z.T


def scale_primitive_xml(root, scale):
    """Return a scaled in-memory XML copy; body poses are intentionally untouched."""
    from copy import deepcopy

    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    scaled = deepcopy(root)
    geom_count = 0
    for geom in scaled.iter("geom"):
        geom_count += 1
        for attribute in ("pos", "fromto", "size"):
            raw = geom.get(attribute)
            if raw is None:
                continue
            try:
                values = np.asarray([float(value) for value in raw.replace(",", " ").split()])
            except ValueError as exc:
                raise ValueError(f"Invalid numeric geom {attribute}: {raw!r}") from exc
            if not values.size or not np.isfinite(values).all():
                raise ValueError(f"Invalid numeric geom {attribute}: {raw!r}")
            geom.set(attribute, " ".join(f"{value * scale:.9g}" for value in values))
    if geom_count == 0:
        raise ValueError("PARC character XML contains no primitive geoms")
    return scaled


def shared_camera_lookat(source_root_zup, target_root_zup, vertical_offset=0.18):
    """Follow the midpoint of both roots with a camera-only offset."""
    source_root = np.asarray(source_root_zup, dtype=np.float64)
    target_root = np.asarray(target_root_zup, dtype=np.float64)
    if source_root.shape != (3,) or target_root.shape != (3,):
        raise ValueError("camera roots must each have shape [3]")
    if not np.isfinite(source_root).all() or not np.isfinite(target_root).all():
        raise ValueError("camera roots must be finite")
    focus = (source_root + target_root) / 2.0
    focus[2] += float(vertical_offset)
    return focus


def shared_camera_distance(source_positions, target_positions_zup, target_soles_zup,
                           source_root_indices, target_root_index):
    """Choose one fixed scale that contains both poses while the camera follows roots."""
    source = np.asarray(source_positions, dtype=np.float64)
    target = np.asarray(target_positions_zup, dtype=np.float64)
    soles = np.asarray(target_soles_zup, dtype=np.float64)
    source_roots = source[np.arange(len(source)), source_root_indices]
    target_roots = target[:, target_root_index]
    midpoint = (source_roots + target_roots) / 2.0
    extents = [source - midpoint[:, None, :], target - midpoint[:, None, :],
               soles - midpoint[:, None, :]]
    max_radius = max(float(np.linalg.norm(points, axis=-1).max()) for points in extents)
    # MuJoCo's default vertical field of view is 45 degrees; retain a modest margin.
    return max(MIN_CAMERA_DISTANCE, max_radius / math.tan(math.radians(45.0) / 2.0) * 1.15)


def _numbers(values):
    return " ".join(f"{float(value):.9g}" for value in np.ravel(values))


def _font(size, bold=False):
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _new_scene(model_name, width, height):
    scene = ET.Element("mujoco", model=model_name)
    ET.SubElement(scene, "compiler", angle="radian")
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "headlight", diffuse=".35 .35 .35", ambient=".25 .25 .25",
                  specular=".1 .1 .1")
    assets = ET.SubElement(scene, "asset")
    ET.SubElement(assets, "texture", type="skybox", builtin="gradient", rgb1=".86 .9 .93",
                  rgb2=".96 .97 .98", width="128", height="128")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(world, "light", pos="0 -6 12", dir="0 .4 -1", directional="true",
                  diffuse=".5 .5 .5", castshadow="true")
    return scene, world


def _add_terrain(world, hf, origin, dx):
    for x, y, center, halfsize in terrain_cells(hf, origin, dx):
        color = ".71 .75 .77 1" if (x + y) % 2 else ".78 .81 .82 1"
        ET.SubElement(world, "geom", name=f"terrain_{x}_{y}", type="box",
                      pos=_numbers(center),
                      size=_numbers(halfsize), rgba=color,
                      contype="0", conaffinity="0")


def terrain_cells(hf, origin, dx):
    """Yield exact PARC cell-center coordinates and box half-sizes in Z-up metres."""
    heights = np.asarray(hf, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    dx = float(dx)
    if (heights.ndim != 2 or min(heights.shape) < 1 or origin.shape not in ((2,), (3,)) or
            not np.isfinite(heights).all() or not np.isfinite(origin).all() or
            not math.isfinite(dx) or dx <= 0):
        raise ValueError("terrain requires a 2-D height array and a two- or three-coordinate min_point")
    bottom = float(np.min(heights)) - 0.6
    for x, y in np.ndindex(heights.shape):
        top = float(heights[x, y])
        halfheight = (top - bottom) / 2.0
        center = (origin[0] + x * dx, origin[1] + y * dx, bottom + halfheight)
        yield x, y, center, (dx / 2.0, dx / 2.0, halfheight)


def make_source_scene(character_xml, body_names, hf, origin, dx, scale, width, height):
    """Build the PARC primitive replay with its local geom dimensions scaled once."""
    from io import StringIO
    from scripts.render_parc_pilot import make_scene

    character_root = scale_primitive_xml(ET.parse(character_xml).getroot(), scale)
    source_bodies = {body.attrib["name"]: body for body in character_root.findall(".//body")}
    missing = [name for name in body_names if name not in source_bodies]
    if missing:
        raise ValueError(f"PARC character XML is missing source bodies: {missing}")
    xml_stream = StringIO(ET.tostring(character_root, encoding="unicode"))
    return make_scene(xml_stream, body_names, hf, origin, dx, width, height)


def _free_body(world, body_name, joint_name, geom_attributes):
    body = ET.SubElement(world, "body", name=body_name)
    ET.SubElement(body, "freejoint", name=joint_name)
    ET.SubElement(body, "geom", **geom_attributes)


def _joint_color(name):
    lowered = name.lower()
    if "left" in lowered:
        return ".08 .55 .64 1"
    if "right" in lowered:
        return ".8 .25 .2 1"
    return ".82 .86 .84 1"


def make_target_scene(target_names, target_parents, sole_count, hf, origin, dx, width, height):
    """Build a schematic capsule skeleton and one marker for every saved sole vertex."""
    import mujoco

    scene, world = _new_scene("SOMA_skeleton_and_exact_sole_points", width, height)
    _add_terrain(world, hf, origin, dx)
    edges = [(int(parent), child) for child, parent in enumerate(target_parents) if parent >= 0]
    for index, name in enumerate(target_names):
        _free_body(world, f"soma_joint_{index:02d}", f"soma_joint_free_{index:02d}", {
            "name": f"soma_joint_geom_{index:02d}", "type": "sphere", "size": ".024",
            "rgba": _joint_color(name), "contype": "0", "conaffinity": "0",
        })
    for edge_index, (_, child) in enumerate(edges):
        color = _joint_color(target_names[child])
        _free_body(world, f"soma_link_{edge_index:02d}", f"soma_link_free_{edge_index:02d}", {
            "name": f"soma_link_geom_{edge_index:02d}", "type": "capsule", "size": ".014 .001",
            "rgba": color, "contype": "0", "conaffinity": "0",
        })
    for index in range(sole_count):
        _free_body(world, f"soma_sole_{index:04d}", f"soma_sole_free_{index:04d}", {
            "name": f"soma_sole_geom_{index:04d}", "type": "sphere", "size": ".009",
            "rgba": ".98 .73 .12 1", "contype": "0", "conaffinity": "0",
        })
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    state = mujoco.MjData(model)

    def addresses_for(prefix, count):
        body_ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                                f"{prefix}_{index:02d}" if prefix != "soma_sole"
                                                else f"{prefix}_{index:04d}")
                               for index in range(count)], dtype=np.int32)
        return body_ids, model.jnt_qposadr[model.body_jntadr[body_ids]]

    joint_ids, joint_addresses = addresses_for("soma_joint", len(target_names))
    link_ids, link_addresses = addresses_for("soma_link", len(edges))
    sole_ids, sole_addresses = addresses_for("soma_sole", sole_count)
    link_geom_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"soma_link_geom_{index:02d}")
        for index in range(len(edges))], dtype=np.int32)
    return (model, state, edges, joint_ids, joint_addresses, link_ids, link_addresses,
            link_geom_ids, sole_ids, sole_addresses)


def _quat_z_to_direction(direction):
    direction = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return np.asarray((1., 0., 0., 0.))
    unit = direction / length
    dot = float(unit[2])
    if dot < -0.999999:
        return np.asarray((0., 1., 0., 0.))
    quat = np.asarray((1.0 + dot, -unit[1], unit[0], 0.0), dtype=np.float64)
    return quat / np.linalg.norm(quat)


def set_target_frame(mujoco, model, state, edges, joint_ids, joint_addresses,
                     link_ids, link_addresses, link_geom_ids, sole_ids, sole_addresses,
                     positions_zup, sole_vertices_zup):
    expected = np.empty((len(edges), 3), dtype=np.float64)
    for index, address in enumerate(joint_addresses):
        state.qpos[address:address + 3] = positions_zup[index]
        state.qpos[address + 3:address + 7] = (1., 0., 0., 0.)
    for edge_index, (parent, child) in enumerate(edges):
        start, end = positions_zup[parent], positions_zup[child]
        expected[edge_index] = (start + end) / 2.0
        address = link_addresses[edge_index]
        state.qpos[address:address + 3] = expected[edge_index]
        state.qpos[address + 3:address + 7] = _quat_z_to_direction(end - start)
        model.geom_size[link_geom_ids[edge_index], 1] = max(
            float(np.linalg.norm(end - start)) / 2.0, 1e-4)
    for index, address in enumerate(sole_addresses):
        state.qpos[address:address + 3] = sole_vertices_zup[index]
        state.qpos[address + 3:address + 7] = (1., 0., 0., 0.)
    mujoco.mj_forward(model, state)
    joint_error = float(np.max(np.abs(state.xpos[joint_ids] - positions_zup)))
    sole_error = float(np.max(np.abs(state.xpos[sole_ids] - sole_vertices_zup)))
    link_error = float(np.max(np.abs(state.xpos[link_ids] - expected))) if len(edges) else 0.0
    if max(joint_error, sole_error, link_error) > 1e-6:
        raise AssertionError("SOMA display geoms do not match saved joint/sole coordinates")
    return joint_error, sole_error, link_error


def _compose_frame(source_rgb, target_rgb, source_frame, total_frames, fps):
    source = Image.fromarray(source_rgb)
    target = Image.fromarray(target_rgb)
    panel_width, viewport_height = source.size
    if target.size != source.size:
        raise ValueError("Source and target panels must share dimensions")
    canvas = Image.new("RGB", (panel_width * 2, viewport_height + HEADER_HEIGHT + FOOTER_HEIGHT),
                       "#20282d")
    canvas.paste(source, (0, HEADER_HEIGHT))
    canvas.paste(target, (panel_width, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(19, True)
    panel_font = _font(14, True)
    footer_font = _font(12)
    info_font = _font(14)
    draw.rectangle((0, 0, canvas.width, HEADER_HEIGHT), fill="#20282d")
    draw.text((16, 5), "PARC SOURCE / SOMA RETARGET REPLAY", font=title_font, fill="white")
    clock = f"Frame {source_frame + 1}/{total_frames} | {source_frame / fps:.2f} s | 1.00x"
    bbox = draw.textbbox((0, 0), clock, font=info_font)
    draw.text((canvas.width - (bbox[2] - bbox[0]) - 16, 9), clock, font=info_font, fill="white")
    draw.text((16, 34), "OFFICIAL PARC PRIMITIVES", font=panel_font, fill="#d4dce1")
    draw.text((panel_width + 16, 34), "SOMA SKELETON + ACTUAL SOLE SKIN POINTS",
              font=panel_font, fill="#d4dce1")
    draw.rectangle((0, HEADER_HEIGHT + viewport_height, canvas.width, canvas.height), fill="#20282d")
    draw.text((16, canvas.height - 32), "Yellow: PARC contact labels; not contact validation",
              font=footer_font, fill="#f5cf53")
    draw.text((panel_width + 16, canvas.height - 32),
              "Gold: exact selected SOMA sole vertices; skeleton is schematic",
              font=footer_font, fill="#f4d485")
    draw.line((panel_width, HEADER_HEIGHT, panel_width, canvas.height), fill="#d7dde0", width=2)
    return canvas


def _gif_durations(frame_count, fps):
    boundaries = np.rint(np.arange(frame_count + 1, dtype=np.float64) * 100.0 / fps).astype(np.int64)
    durations = (np.diff(boundaries) * 10).tolist()
    if (len(durations) != frame_count or
            any(duration <= 0 or duration > 655350 for duration in durations)):
        raise ValueError("GIF timing cannot represent every source frame at the requested FPS")
    return durations


def _frame_tags(frame_count):
    middle = frame_count // 2
    tags = {}
    for tag, index in (("first", 0), ("mid", middle), ("last", frame_count - 1)):
        tags.setdefault(index, []).append(tag)
    return tags


def _write_gif_frame(stream, palette_frame, image, duration):
    if palette_frame is None:
        quantized = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        header, _ = GifImagePlugin.getheader(quantized, info={"loop": 0})
        for block in header:
            stream.write(block)
        palette_frame = quantized
    else:
        quantized = image.quantize(palette=palette_frame, dither=Image.Dither.NONE)
    for block in GifImagePlugin.getdata(quantized, duration=duration, disposal=2):
        stream.write(block)
    return palette_frame


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_clip(input_path, output_dir, *, character_xml=DEFAULT_CHARACTER_XML,
                start_frame=0, max_frames=None, width=1600, height=640):
    """Replay every frame in the selected contiguous source range at input FPS."""
    import mujoco
    from scripts.render_parc_pilot import set_frame as set_source_frame

    input_path = Path(input_path).resolve()
    character_xml = Path(character_xml).resolve()
    output_dir = Path(output_dir).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not character_xml.is_file():
        raise FileNotFoundError(character_xml)
    if output_dir.exists():
        raise FileExistsError(f"Output must be a new directory: {output_dir}")
    if width < 960 or width % 2 or height < 360:
        raise ValueError("width must be even and at least 960; height must be at least 360")
    if start_frame < 0:
        raise ValueError("start_frame must be nonnegative")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    source_hash_before = _sha256_file(input_path)
    data = load_clip(input_path)
    source_pose_baseline = (data["source_positions_zup"].copy(),
                            data["source_rotations_xyzw"].copy())
    total_frames = len(data["source_positions_zup"])
    if start_frame >= total_frames:
        raise ValueError(f"start_frame {start_frame} is outside the {total_frames}-frame clip")
    stop_frame = total_frames if max_frames is None else min(total_frames, start_frame + max_frames)
    source_fps = _scalar(data["source_fps"], "source_fps")
    uniform_scale = _scalar(data["uniform_scale"], "uniform_scale")
    dx = _scalar(data["dx"], "dx")
    frame_count = stop_frame - start_frame
    durations = _gif_durations(frame_count, source_fps)

    source_positions = data["source_positions_zup"][start_frame:stop_frame].astype(np.float64) * uniform_scale
    source_rotations = data["source_rotations_xyzw"][start_frame:stop_frame].astype(np.float64, copy=True)
    source_contacts = data["source_contacts"][start_frame:stop_frame].copy()
    target_positions = target_yup_to_zup(data["target_positions_yup"][start_frame:stop_frame])
    target_soles = target_yup_to_zup(data["target_sole_vertices_yup"][start_frame:stop_frame])
    source_names = data["source_names"]
    target_names = data["target_names"]
    target_parents = data["target_parents"]
    source_root_index = next((i for i, name in enumerate(source_names)
                              if name.lower().replace("_", "") in {"pelvis", "hips", "root"}), 0)
    target_root_index = int(np.flatnonzero(target_parents == -1)[0])
    camera_distance = shared_camera_distance(
        source_positions, target_positions, target_soles,
        np.full(frame_count, source_root_index, dtype=np.int64), target_root_index)

    output_dir.mkdir(parents=True, exist_ok=False)
    panel_width = width // 2
    view_height = height - HEADER_HEIGHT - FOOTER_HEIGHT
    terrain = (data["hf"], data["min_point"], dx)
    source_scene = make_source_scene(character_xml, source_names, *terrain, uniform_scale,
                                     panel_width, view_height)
    target_scene = make_target_scene(target_names, target_parents,
                                     target_soles.shape[1], *terrain, panel_width, view_height)
    source_model, source_state, source_body_ids, source_addresses = source_scene
    (target_model, target_state, edges, joint_ids, joint_addresses, link_ids, link_addresses,
     link_geom_ids, sole_ids, sole_addresses) = target_scene
    source_base_rgba = source_model.geom_rgba.copy()

    camera = mujoco.MjvCamera()
    camera.distance = camera_distance
    camera.elevation = PANEL_ELEVATION
    camera.azimuth = PANEL_AZIMUTH
    min_panel_std = float("inf")
    max_source_fk_error = 0.0
    max_target_joint_error = 0.0
    max_target_sole_error = 0.0
    max_target_link_center_error = 0.0
    saved_tags = _frame_tags(frame_count)

    gif_path = output_dir / "parc_source_vs_soma.gif"
    with gif_path.open("wb") as gif_stream:
        palette_frame = None
        with mujoco.Renderer(source_model, height=view_height, width=panel_width) as source_renderer, \
             mujoco.Renderer(target_model, height=view_height, width=panel_width) as target_renderer:
            for local_index in range(frame_count):
                source_frame = start_frame + local_index
                fk_error = set_source_frame(
                    source_model, source_state, source_body_ids, source_addresses,
                    source_positions[local_index], source_rotations[local_index],
                    source_contacts[local_index], source_base_rgba)
                target_errors = set_target_frame(
                    mujoco, target_model, target_state, edges, joint_ids, joint_addresses,
                    link_ids, link_addresses, link_geom_ids, sole_ids, sole_addresses,
                    target_positions[local_index], target_soles[local_index])
                max_source_fk_error = max(max_source_fk_error, fk_error)
                max_target_joint_error = max(max_target_joint_error, target_errors[0])
                max_target_sole_error = max(max_target_sole_error, target_errors[1])
                max_target_link_center_error = max(max_target_link_center_error, target_errors[2])

                camera.lookat[:] = shared_camera_lookat(
                    source_positions[local_index, source_root_index],
                    target_positions[local_index, target_root_index])
                # The same camera instance and parameters are passed to both panels.
                source_renderer.update_scene(source_state, camera=camera)
                target_renderer.update_scene(target_state, camera=camera)
                source_rgb = source_renderer.render()
                target_rgb = target_renderer.render()
                deviations = (float(source_rgb.std()), float(target_rgb.std()))
                if min(deviations) < 5.0:
                    raise AssertionError(f"Blank render panel at source frame {source_frame}")
                min_panel_std = min(min_panel_std, *deviations)
                image = _compose_frame(source_rgb, target_rgb, source_frame, total_frames, source_fps)
                for tag in saved_tags.get(local_index, ()):
                    image.save(output_dir / f"{tag}.png")
                # Stream frames so long excerpts do not retain every rendered image in memory.
                palette_frame = _write_gif_frame(gif_stream, palette_frame, image, durations[local_index])
        gif_stream.write(b";")
    with Image.open(gif_path) as saved:
        if saved.n_frames != frame_count or saved.size != (width, height):
            raise AssertionError("GIF frame count or dimensions differ from the rendered excerpt")
        actual_durations = []
        for index in range(saved.n_frames):
            saved.seek(index)
            actual_durations.append(saved.info.get("duration", 0))
    if actual_durations != durations:
        raise AssertionError("GIF playback timing differs from the source FPS frame schedule")

    source_hash_after = _sha256_file(input_path)
    if source_hash_after != source_hash_before:
        raise AssertionError("Input NPZ changed during the render")
    if not (np.array_equal(data["source_positions_zup"], source_pose_baseline[0]) and
            np.array_equal(data["source_rotations_xyzw"], source_pose_baseline[1])):
        raise AssertionError("Loaded source poses changed during the render")

    manifest = {
        "status": "renderer_only_replay",
        "input_npz": str(input_path),
        "input_sha256_before": source_hash_before,
        "input_sha256_after": source_hash_after,
        "input_unchanged": source_hash_before == source_hash_after,
        "source_pose_arrays_modified": False,
        "excerpt": {
            "is_excerpt": start_frame != 0 or stop_frame != total_frames,
            "start_frame_inclusive": start_frame,
            "stop_frame_exclusive": stop_frame,
            "first_and_last_source_frames_inclusive": [start_frame, stop_frame - 1],
            "source_total_frames": total_frames,
            "rendered_frame_count": frame_count,
            "requested_max_frames": max_frames,
            "contiguous_source_frame_indices": True,
        },
        "source_fps": source_fps,
        "playback_speed": 1.0,
        "gif_duration_ms": int(sum(actual_durations)),
        "uniform_scale": uniform_scale,
        "coordinate_conventions": {
            "source": "Z-up positions in raw source metres, uniformly scaled once for display",
            "target_archive": "Y-up positions; rotated to Z-up for display only",
            "terrain": "already uniformly scaled Z-up terrain, not scaled again",
            "camera_follow": "camera only; midpoint of source and converted target root positions",
            "pose_or_ground_translation": "none",
        },
        "terrain": {
            "same_array_values_in_both_panels": True,
            "hf_shape": list(data["hf"].shape),
            "min_point": data["min_point"].tolist(),
            "dx_m": dx,
            "cell_centers": "x/y = min_point[:2] + [x,y]*dx; z centered between decorative bottom and hf[x,y]",
            "cell_tops": "exact hf values",
        },
        "source_geometry": {
            "character_xml": str(character_xml),
            "character_xml_sha256": _sha256_file(character_xml),
            "scaling": "in-memory geom pos/fromto/size; saved world positions scaled once",
            "xml_body_poses_scaled": False,
            "source_bodies": source_names,
        },
        "target_geometry": {
            "target_joint_count": len(target_names),
            "target_names": target_names,
            "target_parents": target_parents.tolist(),
            "sole_vertex_count": int(data["target_sole_vertices_yup"].shape[1]),
            "sole_vertices": "all exact target_sole_vertices_yup points from the NPZ",
            "skeleton_representation": "schematic joint spheres and parent-child capsules; not a full mesh",
        },
        "motion_postprocessing": "none",
        "ik": False,
        "smoothing": False,
        "physics_simulation": False,
        "camera": {
            "shared_instance_per_frame": True,
            "azimuth_deg": PANEL_AZIMUTH,
            "elevation_deg": PANEL_ELEVATION,
            "distance_m": camera_distance,
            "root_follow_vertical_offset_m": 0.18,
        },
        "checks": {
            "max_source_fk_position_error_m": max_source_fk_error,
            "max_target_joint_position_error_m": max_target_joint_error,
            "max_target_sole_vertex_position_error_m": max_target_sole_error,
            "max_target_link_center_position_error_m": max_target_link_center_error,
            "pixel": {
                "panels_checked": frame_count * 2,
                "nonblank_threshold_rgb_std": 5.0,
                "minimum_panel_rgb_std": min_panel_std,
                "passed": min_panel_std >= 5.0,
            },
            "gif_frame_count_matches_excerpt": True,
            "gif_durations_match_source_fps_schedule": True,
        },
        "outputs": {
            "gif": str(gif_path),
            "first_png": str((output_dir / "first.png")),
            "mid_png": str((output_dir / "mid.png")),
            "last_png": str((output_dir / "last.png")),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Renderer-contract retarget NPZ")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--character-xml", type=Path, default=DEFAULT_CHARACTER_XML)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional contiguous excerpt length; no frames are skipped")
    parser.add_argument("--width", type=int, default=1600, help="Combined side-by-side width")
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args(argv)
    result = render_clip(args.input, args.output, character_xml=args.character_xml,
                         start_frame=args.start_frame, max_frames=args.max_frames,
                         width=args.width, height=args.height)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
