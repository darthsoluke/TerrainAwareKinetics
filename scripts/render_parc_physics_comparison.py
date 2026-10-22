"""Compare immutable raw PARC motion with actual official physical recordings."""

import argparse
from contextlib import ExitStack
import html
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from render_parc_long_course import (
    DEFAULT_REPO, _official_fk, _rotation_parity_error, _sha256_file,
    make_scene, open_mp4_writer,
)
from render_parc_pilot import font, set_frame
from motionbricks.data.parc_paired_source import _decode_section, _restricted_load


def load_motion(path):
    container = _restricted_load(path.read_bytes())
    motion = _decode_section(container, "motion_data")
    terrain = _decode_section(container, "terrain_data")
    result = {key: np.asarray(motion[key], dtype=np.float32)
              for key in ("root_pos", "root_rot", "joint_rot")}
    result.update(contacts=np.asarray(motion["body_contacts"]), fps=float(motion["fps"]),
                  hf=np.asarray(terrain["hf"]), origin=np.asarray(terrain["min_point"]),
                  dx=float(terrain["dx"]))
    if not all(np.isfinite(result[key]).all() for key in
               ("root_pos", "root_rot", "joint_rot", "contacts", "hf", "origin")):
        raise ValueError("Non-finite paired motion or terrain")
    return result


def verify_pair(recording, source_path, physical_path, source, physical):
    if recording.get("recorded_terrain_error_m", float("inf")) > 1e-5:
        raise ValueError("Physical recording lacks validated terrain pairing")
    if not recording["whole_motion"] or recording["start_fraction"] != 0.0:
        raise ValueError("Partial recording requires a separate explicit time alignment")
    if _sha256_file(source_path) != recording["source_sha256"]:
        raise ValueError("Raw source hash mismatch")
    if _sha256_file(physical_path) != recording["converted_sha256"]:
        raise ValueError("Physical export hash mismatch")
    if (len(source["root_pos"]) != recording["frames"] or
            len(physical["root_pos"]) != recording["frames"] or
            source["fps"] != physical["fps"] or physical["fps"] != recording["fps"]):
        raise ValueError("Source and simulation time axes differ")


def render_pair(recording, source_path, physical_path, output, repo=DEFAULT_REPO,
                raw_route_covered=None, source_stage="RAW CANDIDATE"):
    source, physical = load_motion(source_path), load_motion(physical_path)
    verify_pair(recording, source_path, physical_path, source, physical)
    output.mkdir(parents=True, exist_ok=False)
    width, height, header = 640, 480, 108
    fps, count = physical["fps"], recording["frames"]
    video = output / "raw_vs_physics_1x.mp4"
    parity, rotation_parity, pixel_std = [], [], []
    with ExitStack() as stack:
        views = []
        for motion in (source, physical):
            _, names, positions, rotations, xml = _official_fk(
                motion, repo, repo / "data/assets/humanoid.xml")
            model, state, ids, addresses, _, _ = make_scene(
                xml, names, motion["hf"], motion["origin"], motion["dx"],
                np.empty((0, 3)), width, height)
            renderer = stack.enter_context(mujoco.Renderer(model, height=height, width=width))
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.azimuth, camera.elevation, camera.distance = 60, -19, 5.0
            views.append((motion, names, positions, rotations, model, state, ids,
                          addresses, renderer, camera, model.geom_rgba.copy()))
        writer = stack.enter_context(open_mp4_writer(video, width * 2, height + header, fps))
        for frame_id in range(count):
            canvas = Image.new("RGB", (width * 2, height + header), "#20282d")
            draw = ImageDraw.Draw(canvas)
            draw.text((14, 7), source_stage + "  |  OFFICIAL PHYSICS RECORDING", font=font(22, True), fill="white")
            draw.text((14, 37), "30 FPS / 1x | NO IK, FOOT LOCK OR SMOOTHING | NOT TRAINING-APPROVED",
                      font=font(13, True), fill="#f1cf61")
            route_status = "NOT ASSESSED" if raw_route_covered is None else ("COVERED" if raw_route_covered else "NOT COVERED")
            draw.text((14, 58), f"REFERENCE ROUTE: {route_status} | SIMULATED ROUTE COMPLETION NOT YET SCORED",
                      font=font(13, True), fill="#f1cf61")
            labels = (source_stage + " / SOURCE TERRAIN", "PHYSICS: SIMULATED STATES / RECORDED TERRAIN")
            for side, view in enumerate(views):
                motion, names, pos, rot, model, state, ids, addresses, renderer, camera, rgba = view
                parity.append(set_frame(model, state, ids, addresses, pos[frame_id], rot[frame_id],
                                        motion["contacts"][frame_id], rgba))
                rotation_parity.append(_rotation_parity_error(state, ids, rot[frame_id]))
                camera.lookat[:] = pos[frame_id, names.index("pelvis")] + [0, 0, .15]
                renderer.update_scene(state, camera=camera)
                rgb = renderer.render()
                pixel_std.append(float(rgb.std()))
                if pixel_std[-1] < 5:
                    raise AssertionError("Blank comparison view")
                canvas.paste(Image.fromarray(rgb), (side * width, header))
                draw.text((side * width + 14, 86), labels[side], font=font(13, True), fill="white")
            draw.text((1090, 38), f"{frame_id / fps:.2f}/{count / fps:.2f} s",
                      font=font(13, True), fill="white")
            if frame_id in {0, count // 2, count - 1}:
                canvas.save(output / f"frame_{frame_id:04d}.png")
            writer.append_data(np.asarray(canvas))
    verify_pair(recording, source_path, physical_path, source, physical)
    decoded, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    if decoded != count or abs(seconds - count / fps) > 1 / fps:
        raise AssertionError("Encoded playback frame count/rate mismatch")
    report = dict(recording, video=video.name, video_sha256=_sha256_file(video),
                  decoded_frames=decoded, duration_s=count / fps,
                  render_fk_error_m=max(parity), render_rotation_error=max(rotation_parity),
                  min_pixel_std=min(pixel_std), pose_modification="none",
                  renderer_simulates_physics=False, motion_origin="official Isaac Gym recording",
                  camera_alignment="same follow distance/orientation, independently centered",
                  terrain_localization="original paired terrain for each panel, no guessed alignment",
                  terrain_rendering="heightfield cell columns, not engine triangle-mesh visualization",
                  training_approved=False)
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def render_showcase(export, collection, output, repo):
    manifest = json.loads((export / "manifest.json").read_text())
    catalog = json.loads((collection / "gallery/catalog.json").read_text())
    source_rows = {row["candidate_key"]: row for row in catalog["candidates"]}
    original = json.loads((collection / "summary.json").read_text())
    preferred = {row["candidate_key"] for row in original["showcase"]}
    groups = {}
    for row in manifest["recordings"]:
        if row["whole_motion"]:
            source = source_rows[row["candidate_key"]]
            groups.setdefault((source["course_kind"], source["course_variant"]), []).append((row, source))
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for (kind, variant), rows in sorted(groups.items()):
        record, source = min(rows, key=lambda pair: (
            pair[0]["candidate_key"] not in preferred,
            not pair[1]["geometric_route_covered"], pair[0]["candidate_key"]))
        name = f"{kind}_{variant}"
        result = render_pair(record, collection / "downloads" / source["source_path"],
                             export / record["converted_path"], output / name, repo,
                             source["geometric_route_covered"])
        result.update(name=name, raw_geometric_route_covered=source["geometric_route_covered"])
        results.append(result)
        print(json.dumps({"rendered": name, "frames": result["frames"]}), flush=True)
    summary = {"recordings_in_export": len(manifest["recordings"]), "showcase": results,
               "selection": "one full-start recording per available terrain family; prefer original showcase, then raw route coverage, then candidate key",
               "training_approved": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sections = []
    for row in results:
        sections.append(f'<h2>{html.escape(row["name"])}</h2><video controls preload="metadata" '
                        f'poster="{row["name"]}/frame_{row["frames"] // 2:04d}.png" '
                        f'src="{row["name"]}/raw_vs_physics_1x.mp4"></video>')
    (output / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>PARC Raw / Physics</title><style>body{font:16px system-ui;background:#f4f6f6;'
        'color:#20282d;max-width:1280px;margin:24px auto;padding:0 16px}h1{font-size:26px}'
        'h2{font-size:18px;margin-top:30px}video{display:block;width:100%;height:auto}</style>'
        '<h1>PARC Raw / Official Physics Recording</h1>'
        '<p>Full-start recordings only. No IK or smoothing. Training approval pending.</p>' +
        ''.join(sections) + '</html>', encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    args = parser.parse_args()
    render_showcase(args.export.resolve(), args.collection.resolve(), args.output.resolve(), args.repo.resolve())
