"""Render an unmodified PARC-generated course motion as a kinematic candidate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import NamedTuple

import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = ROOT / "out" / "vendor" / "PARC"
MAX_GIF_PREVIEW_SECONDS = 10.0
MAX_GIF_PREVIEW_FRAMES = 300
MAX_MOTION_BYTES = 128 * 1024**2
HEADER_HEIGHT = 82
PANEL_HEADER_HEIGHT = 30
HEADER_TITLE = "PARC GENERATED KINEMATIC RAW CANDIDATE"
STATUS_LINE_1 = "PHYSICS NOT YET VERIFIED | SOURCE CONTACT LABELS ONLY, NOT VERIFIED"
STATUS_LINE_2 = "{fps:g} FPS, 1x | NO POSE MODIFICATION"
OVERVIEW_PANEL_TITLE = "FULL COURSE OVERVIEW"
FOLLOW_PANEL_TITLE = "FOLLOW | OFFICIAL PARC HUMANOID"

for _path in (str(ROOT), str(SCRIPT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from motionbricks.data.parc_paired_source import _decode_section, _restricted_load
from render_parc_pilot import (
    ROTATION_PARITY_TOLERANCE,
    font,
    normalized_mujoco_quaternion_wxyz,
    numbers,
    set_frame,
)


class TerrainRectangle(NamedTuple):
    """Half-open heightfield cell bounds for one constant-height rectangle."""

    i0: int
    i1: int
    j0: int
    j1: int
    height: float


def merge_heightfield_rectangles(heightfield: np.ndarray) -> list[TerrainRectangle]:
    """Merge equal-height horizontal runs across adjacent rows without changing cells."""
    hf = np.asarray(heightfield)
    if (hf.ndim != 2 or min(hf.shape) < 1 or not np.issubdtype(hf.dtype, np.number)
            or not np.isrealobj(hf)):
        raise ValueError("heightfield must be a nonempty numeric two-dimensional array")
    if not np.isfinite(hf).all():
        raise ValueError("heightfield must contain only finite heights")

    finished = []
    active = {}
    for i, row in enumerate(hf):
        cuts = np.flatnonzero(np.r_[True, row[1:] != row[:-1], True])
        current = {}
        for j0, j1 in zip(cuts[:-1], cuts[1:]):
            key = (int(j0), int(j1), float(row[j0]))
            rectangle = active.pop(key, None)
            if rectangle is None:
                rectangle = [i, i + 1, int(j0), int(j1), float(row[j0])]
            else:
                rectangle[1] = i + 1
            current[key] = rectangle
        finished.extend(active.values())
        active = current
    finished.extend(active.values())
    return [TerrainRectangle(*values) for values in sorted(
        finished, key=lambda rect: (rect[0], rect[2], rect[1], rect[3], rect[4])
    )]


def terrain_rectangle_xy_geometry(rectangle: TerrainRectangle, min_point, dx: float):
    """Return (center_xy, half_size_xy) preserving the source cell-center bounds."""
    origin = np.asarray(min_point, dtype=np.float64)
    spacing = float(dx)
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("terrain min_point must be a finite [x, y] origin")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("terrain spacing must be finite and positive")
    center = origin + np.asarray((
        (rectangle.i0 + rectangle.i1 - 1) / 2,
        (rectangle.j0 + rectangle.j1 - 1) / 2,
    )) * spacing
    half_size = np.asarray((rectangle.i1 - rectangle.i0,
                            rectangle.j1 - rectangle.j0), dtype=np.float64) * spacing / 2
    return center, half_size


def video_duration_seconds(frame_count: int, fps: float) -> float:
    """Duration of a constant-rate stream containing every source frame."""
    if isinstance(frame_count, bool) or int(frame_count) != frame_count or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    return float(frame_count) / float(fps)


def gif_preview_frame_count(frame_count: int, fps: float) -> int:
    if frame_count <= 0 or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("frame_count and fps must be positive")
    time_limit = max(1, int(math.floor(MAX_GIF_PREVIEW_SECONDS * float(fps) + 1e-9)))
    return min(int(frame_count), time_limit, MAX_GIF_PREVIEW_FRAMES)


def gif_durations_ms(frame_count: int, fps: float) -> list[int]:
    """Return cumulative-rounded GIF durations at the source playback rate."""
    if frame_count <= 0 or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("frame_count and fps must be positive")
    boundaries = np.rint(np.arange(frame_count + 1) * 100.0 / float(fps)).astype(np.int64)
    durations = (np.diff(boundaries) * 10).tolist()
    if any(duration <= 0 for duration in durations):
        raise ValueError("GIF timing cannot represent the source frame rate")
    return durations


def _read_motion_file(path: Path):
    path = Path(path).resolve()
    if path.stat().st_size > MAX_MOTION_BYTES:
        raise ValueError(f"{path} exceeds the {MAX_MOTION_BYTES}-byte MS file limit")
    payload = path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    container = _restricted_load(payload)
    if not isinstance(container, dict):
        raise ValueError("PARC MS file root must be a dictionary")
    motion = _decode_section(container, "motion_data")
    terrain = _decode_section(container, "terrain_data")
    misc = container.get("misc_data", {})
    if isinstance(misc, bytes):
        misc = _restricted_load(misc)
    if not isinstance(misc, dict):
        raise ValueError("PARC misc_data must be a dictionary")

    required = ("root_pos", "root_rot", "joint_rot", "body_contacts", "fps")
    missing = [name for name in required if name not in motion or motion[name] is None]
    if missing:
        raise ValueError(f"PARC motion_data is missing fields: {', '.join(missing)}")
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float32)
    joint_rot = np.asarray(motion["joint_rot"], dtype=np.float32)
    contacts = np.asarray(motion["body_contacts"], dtype=np.float32)
    if root_pos.ndim != 2 or root_pos.shape[1:] != (3,) or len(root_pos) == 0:
        raise ValueError("root_pos must have shape [frames, 3] with at least one frame")
    count = len(root_pos)
    if root_rot.shape != (count, 4) or joint_rot.shape != (count, 14, 4):
        raise ValueError("root_rot/joint_rot must have shapes [frames,4] and [frames,14,4]")
    if contacts.ndim != 2 or contacts.shape[0] != count:
        raise ValueError("body_contacts must have shape [frames, bodies]")
    if not all(np.isfinite(values).all() for values in
               (root_pos, root_rot, joint_rot, contacts)):
        raise ValueError("motion arrays must contain only finite values")
    if np.max(np.abs(np.linalg.norm(root_rot, axis=-1) - 1.0)) > 1e-4:
        raise ValueError("root_rot must contain unit xyzw quaternions")
    if np.max(np.abs(np.linalg.norm(joint_rot, axis=-1) - 1.0)) > 1e-4:
        raise ValueError("joint_rot must contain unit xyzw quaternions")
    fps = float(motion["fps"])
    if not math.isfinite(fps) or not 0 < fps <= 100:
        raise ValueError("source FPS must be finite and in (0, 100]")

    hf = np.asarray(terrain.get("hf"), dtype=np.float32)
    origin = np.asarray(terrain.get("min_point"), dtype=np.float64)
    dx = float(terrain.get("dx", 0))
    if hf.ndim != 2 or min(hf.shape) < 1 or not np.isfinite(hf).all():
        raise ValueError("terrain hf must be a finite, nonempty 2D array")
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("terrain min_point must be a finite [x, y] cell-center origin")
    if not math.isfinite(dx) or dx <= 0:
        raise ValueError("terrain dx must be finite and positive")
    if "path_nodes" not in misc:
        raise ValueError("PARC misc_data is missing path_nodes required for the course overview")
    route = np.asarray(misc["path_nodes"], dtype=np.float64)
    if route.ndim != 2 or route.shape[1:] != (3,) or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("path_nodes must be a finite [nodes, 3] array with at least two nodes")
    grid_float = (route[:, :2] - origin) / dx
    grid = np.rint(grid_float).astype(np.int64)
    if np.max(np.abs(grid_float - grid)) > 1e-3:
        raise ValueError("path_nodes do not align with the terrain cell-center grid")
    if np.any(grid[:, 0] < 0) or np.any(grid[:, 0] >= hf.shape[0]) or \
            np.any(grid[:, 1] < 0) or np.any(grid[:, 1] >= hf.shape[1]):
        raise ValueError("path_nodes leave the terrain bounds")
    route_heights = hf[grid[:, 0], grid[:, 1]]
    if not np.allclose(route[:, 2], route_heights, rtol=0, atol=1e-4):
        raise ValueError("path_nodes elevations disagree with the embedded terrain")
    source_metadata = {
        str(key): value.item() if isinstance(value, np.generic) else value
        for key, value in misc.items()
        if key != "path_nodes" and isinstance(value, (str, bool, int, float, np.generic))
    }

    return {
        "path": path,
        "source_sha256": source_hash,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "joint_rot": joint_rot,
        "contacts": contacts,
        "fps": fps,
        "hf": hf,
        "origin": origin,
        "dx": dx,
        "route": route.astype(np.float32),
        "source_metadata": source_metadata,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_fk(motion, repo: Path, character_xml: Path):
    repo = Path(repo).resolve()
    character_xml = Path(character_xml).resolve()
    if not (repo / "PARC").is_dir():
        raise FileNotFoundError(f"Official PARC checkout not found: {repo}")
    if not character_xml.is_file():
        raise FileNotFoundError(f"Official PARC humanoid XML not found: {character_xml}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    sys.modules.setdefault("parc", importlib.import_module("PARC"))
    import torch
    from parc.anim.kin_char_model import KinCharModel

    character = KinCharModel(torch.device("cpu"))
    character.load_char_file(str(character_xml))
    names = [str(name) for name in character.get_body_names()]
    if len(names) != len(set(names)) or "pelvis" not in names:
        raise ValueError("Official PARC humanoid body names must be unique and include pelvis")
    tensors = [torch.as_tensor(motion[key], dtype=torch.float32, device="cpu")
               for key in ("root_pos", "root_rot", "joint_rot")]
    with torch.no_grad():
        positions, rotations = [np.asarray(value) for value in
                                character.forward_kinematics(*tensors)]
    if positions.shape != (len(motion["root_pos"]), len(names), 3) or \
            rotations.shape != (len(motion["root_pos"]), len(names), 4):
        raise ValueError("Official PARC FK returned unexpected body/frame dimensions")
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError("Official PARC FK returned nonfinite body poses")
    if np.max(np.abs(np.linalg.norm(rotations, axis=-1) - 1.0)) > 1e-4:
        raise ValueError("Official PARC FK rotations are not unit xyzw quaternions")
    if motion["contacts"].shape != positions.shape[:2]:
        raise ValueError("body_contacts do not match official PARC FK body dimensions")
    return character, names, positions, rotations, character_xml


def _add_terrain_geometry(world, hf, origin, dx):
    rectangles = merge_heightfield_rectangles(hf)
    bottom = float(np.min(hf)) - 0.6
    for index, rectangle in enumerate(rectangles):
        center_xy, half_xy = terrain_rectangle_xy_geometry(rectangle, origin, dx)
        half_height = (rectangle.height - bottom) / 2
        center = (center_xy[0], center_xy[1], bottom + half_height)
        shade = 0.76 - min(max(rectangle.height, 0.0), 2.5) * 0.025
        ET.SubElement(
            world,
            "geom",
            name=f"terrain_rect_{index}",
            type="box",
            pos=numbers(center),
            size=numbers((half_xy[0], half_xy[1], half_height)),
            rgba=numbers((shade, shade + 0.025, shade + 0.035, 1)),
            contype="0",
            conaffinity="0",
        )
    return rectangles


def make_scene(character_xml, body_names, hf, origin, dx, route, width, height):
    """Build render-only terrain and official source body geoms in a static world."""
    source = ET.parse(character_xml).getroot()
    source_bodies = {body.attrib["name"]: body for body in source.findall(".//body")
                     if "name" in body.attrib}
    missing = [name for name in body_names if name not in source_bodies]
    if missing:
        raise ValueError(f"Official humanoid XML is missing FK bodies: {missing}")

    scene = ET.Element("mujoco", model="PARC_long_course_kinematic_render_only")
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
    ET.SubElement(world, "light", pos="-6 -10 18", dir=".25 .35 -1", directional="true",
                  diffuse=".7 .7 .7", castshadow="true")
    rectangles = _add_terrain_geometry(world, hf, origin, dx)

    route_geom_names = []
    for index, point in enumerate(route):
        name = f"route_node_{index}"
        route_geom_names.append(name)
        radius = 0.07 if index in (0, len(route) - 1) else 0.045
        color = ".12 .48 .28 1" if index == 0 else (
            ".67 .20 .18 1" if index == len(route) - 1 else ".16 .22 .25 1"
        )
        ET.SubElement(world, "geom", name=name, type="sphere",
                      pos=numbers((point[0], point[1], point[2] + 0.035)),
                      size=numbers((radius,)), rgba=color, contype="0", conaffinity="0")

    for name in body_names:
        body = ET.SubElement(world, "body", name=name)
        ET.SubElement(body, "freejoint", name=name + "_render_pose")
        for geom in source_bodies[name].findall("geom"):
            copied = deepcopy(geom)
            copied.set("type", copied.get("type", "capsule"))
            copied.set("contype", "0")
            copied.set("conaffinity", "0")
            color = ".08 .55 .64 1" if name.startswith("left") else (
                ".8 .25 .2 1" if name.startswith("right") else ".88 .88 .87 1"
            )
            copied.set("rgba", color)
            body.append(copied)

    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding="unicode"))
    data = mujoco.MjData(model)
    body_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names
    ], dtype=np.int32)
    if np.any(body_ids <= 0):
        raise ValueError("MuJoCo could not resolve every official PARC body")
    addresses = model.jnt_qposadr[model.body_jntadr[body_ids]]
    route_geom_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in route_geom_names
    ], dtype=np.int32)
    if np.any(route_geom_ids < 0):
        raise ValueError("MuJoCo could not resolve the course route markers")
    return model, data, body_ids, addresses, route_geom_ids, rectangles


def _rotation_parity_error(data, body_ids, rotations):
    maximum = 0.0
    for index, body_id in enumerate(body_ids):
        expected = np.empty(9, dtype=np.float64)
        quaternion_wxyz = normalized_mujoco_quaternion_wxyz(rotations[index])
        mujoco.mju_quat2Mat(expected, quaternion_wxyz)
        maximum = max(maximum, float(np.max(np.abs(data.xmat[body_id] - expected))))
    if maximum > ROTATION_PARITY_TOLERANCE:
        raise AssertionError(f"Rendered body orientation disagrees with saved FK: {maximum}")
    return maximum


def _nearest_route_node(root, route):
    delta = route[:, :2].astype(np.float64) - np.asarray(root[:2], dtype=np.float64)
    return int(np.argmin(np.einsum("ij,ij->i", delta, delta)))


def compose_frame(overview_rgb, follow_rgb, frame_index, frame_count, fps):
    overview = Image.fromarray(overview_rgb)
    follow = Image.fromarray(follow_rgb)
    width, height = overview.size
    if follow.size != overview.size:
        raise ValueError("overview and follow views must have identical dimensions")
    canvas = Image.new("RGB", (width * 2, HEADER_HEIGHT + PANEL_HEADER_HEIGHT + height), "#20282d")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 8), HEADER_TITLE,
              font=font(22, True), fill="white")
    status_font = font(12, True)
    draw.text((19, 34), STATUS_LINE_1,
              font=status_font, fill="#f1cf61")
    draw.text((19, 52), STATUS_LINE_2.format(fps=fps),
              font=status_font, fill="#f1cf61")
    panel_y = HEADER_HEIGHT
    draw.rectangle((0, panel_y, width, panel_y + PANEL_HEADER_HEIGHT), fill="#344148")
    draw.rectangle((width, panel_y, width * 2, panel_y + PANEL_HEADER_HEIGHT), fill="#344148")
    draw.text((12, panel_y + 7), OVERVIEW_PANEL_TITLE, font=font(15, True), fill="white")
    draw.text((width + 12, panel_y + 8), FOLLOW_PANEL_TITLE,
              font=font(13, True), fill="white")
    view_y = panel_y + PANEL_HEADER_HEIGHT
    canvas.paste(overview, (0, view_y))
    canvas.paste(follow, (width, view_y))
    clock = f"{frame_index / fps:.2f} / {video_duration_seconds(frame_count, fps):.2f} s"
    clock_font = font(13, True)
    bounds = draw.textbbox((0, 0), clock, font=clock_font)
    draw.text((width * 2 - (bounds[2] - bounds[0]) - 18, 53), clock,
              font=clock_font, fill="white")
    return canvas


def _gif_preview_frame(frame):
    scale = min(1.0, 720 / frame.width)
    size = (max(2, int(round(frame.width * scale))), max(2, int(round(frame.height * scale))))
    image = frame.resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 26), fill="#20282d")
    draw.text((9, 5), "GIF PREVIEW EXCERPT | INITIAL COURSE SEGMENT, MAX 10 SEC",
              font=font(13, True), fill="white")
    palette = getattr(Image, "Palette", Image)
    adaptive = getattr(palette, "ADAPTIVE", getattr(Image, "ADAPTIVE", 1))
    return image.convert("P", palette=adaptive, colors=128)


class _FFmpegPipeWriter:
    def __init__(self, process):
        self.process = process

    def append_data(self, frame):
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("video frames must be uint8 RGB arrays")
        self.process.stdin.write(np.ascontiguousarray(array).tobytes())

    def close(self):
        self.process.stdin.close()
        error = self.process.stderr.read().decode("utf-8", errors="replace")
        code = self.process.wait()
        if code:
            raise RuntimeError(f"FFmpeg failed with exit code {code}: {error.strip()}")


class _ImageIOFFmpegWriter:
    def __init__(self, stream):
        self.stream = stream
        self.stream.send(None)

    def append_data(self, frame):
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("video frames must be uint8 RGB arrays")
        self.stream.send(np.ascontiguousarray(array).tobytes())

    def close(self):
        self.stream.close()


def _ffmpeg_backend():
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
        return "imageio_ffmpeg", executable
    except (ImportError, RuntimeError, OSError):
        executable = shutil.which("ffmpeg")
        if executable:
            return "system_ffmpeg", executable
    raise RuntimeError(
        "MP4 output requires bundled imageio_ffmpeg or an ffmpeg executable on PATH; "
        "neither is available in the selected Python environment"
    )


@contextmanager
def open_mp4_writer(path: Path, width: int, height: int, fps: float):
    backend, executable = _ffmpeg_backend()
    path = Path(path)
    if backend == "imageio_ffmpeg":
        import imageio_ffmpeg

        stream = imageio_ffmpeg.write_frames(
            str(path),
            (width, height),
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=float(fps),
            quality=8,
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )
        writer = _ImageIOFFmpegWriter(stream)
        try:
            yield writer
        except BaseException:
            stream.close()
            raise
        else:
            writer.close()
        return

    command = [
        executable, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}", "-r", f"{fps:.12g}",
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    writer = _FFmpegPipeWriter(process)
    try:
        yield writer
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    else:
        writer.close()


def render_long_course(motion_path: Path, output: Path, repo: Path = DEFAULT_REPO,
                       panel_width: int = 720, panel_height: int = 540,
                       gif_preview: bool = False, azimuth: float = 60.0):
    motion_path = Path(motion_path).resolve()
    output = Path(output).resolve()
    if panel_width < 320 or panel_height < 240 or panel_height % 2:
        raise ValueError("panel dimensions must be at least 320 x 240 with an even height")
    if not math.isfinite(float(azimuth)):
        raise ValueError("camera azimuth must be finite")
    _ffmpeg_backend()
    motion = _read_motion_file(motion_path)
    character, names, positions, rotations, character_xml = _official_fk(
        motion, repo, Path(repo) / "data" / "assets" / "humanoid.xml"
    )
    output.mkdir(parents=True, exist_ok=False)
    model, state, body_ids, addresses, route_geom_ids, rectangles = make_scene(
        character_xml, names, motion["hf"], motion["origin"], motion["dx"],
        motion["route"], panel_width, panel_height,
    )
    base_rgba = model.geom_rgba.copy()
    base_geom_size = model.geom_size.copy()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = float(azimuth)
    camera.elevation = -19
    camera.distance = 5.5
    overview_camera = mujoco.MjvCamera()
    overview_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    overview_camera.azimuth = 135
    overview_camera.elevation = -55
    overview_camera.distance = max(motion["hf"].shape) * motion["dx"] * 1.9
    overview_camera.lookat[:] = (
        motion["origin"][0] + (motion["hf"].shape[0] - 1) * motion["dx"] / 2,
        motion["origin"][1] + (motion["hf"].shape[1] - 1) * motion["dx"] / 2,
        (float(np.min(motion["hf"])) + float(np.max(motion["hf"]))) / 2,
    )

    frame_count = len(positions)
    selected_pngs = {0, frame_count // 2, frame_count - 1}
    preview_count = gif_preview_frame_count(frame_count, motion["fps"]) if gif_preview else 0
    preview_frames = []
    overview_stds = []
    follow_stds = []
    translation_errors = []
    rotation_errors = []
    pelvis_index = names.index("pelvis")
    route_geom_index = _nearest_route_node(positions[0, pelvis_index], motion["route"])
    source_pose_baseline = tuple(motion[key].copy() for key in ("root_pos", "root_rot", "joint_rot"))
    video_path = output / "generated_kinematic_candidate_1x.mp4"
    with mujoco.Renderer(model, height=panel_height, width=panel_width) as renderer:
        with open_mp4_writer(video_path, panel_width * 2,
                             HEADER_HEIGHT + PANEL_HEADER_HEIGHT + panel_height,
                             motion["fps"]) as writer:
            for index in range(frame_count):
                translation_errors.append(set_frame(
                    model, state, body_ids, addresses, positions[index], rotations[index],
                    motion["contacts"][index], base_rgba,
                ))
                rotation_errors.append(_rotation_parity_error(state, body_ids, rotations[index]))
                model.geom_size[route_geom_ids, 0] = base_geom_size[route_geom_ids, 0]
                pelvis = positions[index, pelvis_index]
                camera.lookat[:] = pelvis + np.asarray((0.0, 0.0, 0.22))
                nearest = _nearest_route_node(pelvis, motion["route"])
                route_geom_index = nearest
                model.geom_rgba[route_geom_ids[route_geom_index]] = (1.0, 0.62, 0.08, 1.0)
                model.geom_size[route_geom_ids[route_geom_index], 0] = max(
                    0.09, float(base_geom_size[route_geom_ids[route_geom_index], 0])
                )
                renderer.update_scene(state, camera=overview_camera)
                overview_rgb = renderer.render()
                renderer.update_scene(state, camera=camera)
                follow_rgb = renderer.render()
                overview_std = float(overview_rgb.std())
                follow_std = float(follow_rgb.std())
                if overview_std < 5 or follow_std < 5:
                    raise AssertionError("A course view rendered as a blank image")
                overview_stds.append(overview_std)
                follow_stds.append(follow_std)
                frame = compose_frame(overview_rgb, follow_rgb, index, frame_count,
                                      motion["fps"])
                if index in selected_pngs:
                    frame.save(output / f"frame_{index:04d}.png")
                if index < preview_count:
                    preview_frames.append(_gif_preview_frame(frame))
                writer.append_data(np.asarray(frame, dtype=np.uint8))
                if (index + 1) % 300 == 0 or index + 1 == frame_count:
                    print(json.dumps({"rendered_frames": index + 1,
                                      "total_frames": frame_count}), flush=True)

    if not all(np.array_equal(motion[key], baseline) for key, baseline in
               zip(("root_pos", "root_rot", "joint_rot"), source_pose_baseline)):
        raise AssertionError("Renderer modified source motion pose arrays")
    if _sha256_file(motion_path) != motion["source_sha256"]:
        raise AssertionError("Source MS file changed during rendering")
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise AssertionError("FFmpeg did not produce a nonempty MP4")

    gif_path = None
    gif_duration_ms = None
    if gif_preview:
        gif_path = output / "gif_preview_first_10s.gif"
        gif_durations = gif_durations_ms(preview_count, motion["fps"])
        preview_frames[0].save(
            gif_path, save_all=True, append_images=preview_frames[1:],
            duration=gif_durations, loop=0, optimize=False, disposal=2,
        )
        gif_duration_ms = sum(gif_durations)

    manifest = {
        "status": "raw_generated_kinematic_candidate_physics_not_yet_verified",
        "motion_kind": "raw_generated_kinematic_candidate",
        "source_metadata": motion["source_metadata"],
        "source_motion": str(motion_path),
        "source_motion_sha256": motion["source_sha256"],
        "source_sha256_unchanged_after_render": True,
        "frames": frame_count,
        "source_fps": motion["fps"],
        "playback_speed": 1.0,
        "video_stream_duration_seconds": video_duration_seconds(frame_count, motion["fps"]),
        "motion_time_span_seconds": (frame_count - 1) / motion["fps"],
        "pose_modification": "none",
        "ik": "not run",
        "smoothing_or_interpolation": "none",
        "mj_step_called": False,
        "physics_verified": False,
        "classification": "generated kinematic candidate; physics not yet verified",
        "kinematics": "official PARC KinCharModel CPU forward_kinematics",
        "body_geometry": "official PARC humanoid.xml primitive body geoms",
        "terrain_geometry": "exact cell-center bounds; equal-height cells merged into boxes",
        "terrain_shape_ij": list(motion["hf"].shape),
        "terrain_dx_m": motion["dx"],
        "terrain_rectangle_count": len(rectangles),
        "terrain_cell_count": int(motion["hf"].size),
        "terrain_rectangle_cell_coverage_exact": True,
        "route_node_count": int(len(motion["route"])),
        "rendered_body_origin_parity_max_abs_m": max(translation_errors),
        "rendered_body_rotation_parity_max_abs_matrix_error": max(rotation_errors),
        "rendered_body_rotation_parity_checked_each_frame": True,
        "nonblank_pixel_stats": {
            "overview_rgb_std_min": min(overview_stds),
            "overview_rgb_std_mean": float(np.mean(overview_stds)),
            "follow_rgb_std_min": min(follow_stds),
            "follow_rgb_std_mean": float(np.mean(follow_stds)),
        },
        "video": str(video_path),
        "gif_preview": None if gif_path is None else str(gif_path),
        "gif_preview_frames": preview_count,
        "gif_preview_duration_ms": gif_duration_ms,
        "gif_preview_note": "initial excerpt, capped at 10 seconds and 300 frames",
        "character_xml_sha256": _sha256_file(character_xml),
        "renderer_sha256": _sha256_file(Path(__file__)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True,
                        help="current PARC MSfile.pkl generated for the long course")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--panel-width", type=int, default=720)
    parser.add_argument("--panel-height", type=int, default=540)
    parser.add_argument("--azimuth", type=float, default=60)
    parser.add_argument("--gif-preview", action="store_true",
                        help="also save a bounded GIF excerpt from the first 10 seconds")
    args = parser.parse_args(argv)
    manifest = render_long_course(args.motion, args.output, args.repo, args.panel_width,
                                  args.panel_height, args.gif_preview, args.azimuth)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
