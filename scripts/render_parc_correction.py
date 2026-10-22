"""Render bounded, side-by-side PARC raw and optimized saved-FK candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CASE_SETTINGS = {
    "stairs_up": ("STAIRS / ASCENT", 60),
    "stairs_down": ("STAIRS / DESCENT", 60),
    "hand_contact_continuation": ("HAND-CONTACT / CONTINUATION", 15),
}
STAGE_PAIR = ("raw", "optimized")
MAX_CASES = len(CASE_SETTINGS)
MAX_REPORT_CASES = 16
MAX_FRAMES = 300
MAX_MOTION_BYTES = 128 * 1024 * 1024
HEIGHT = 600
FPS = 30
PLAYBACK_SPEED = 0.5


def selected_frame_indices(frame_count):
    if frame_count <= 0:
        raise ValueError("saved FK contains no frames")
    return sorted({0, frame_count // 2, frame_count - 1})


def gif_durations(frame_count, fps=FPS, speed=PLAYBACK_SPEED):
    """Return centisecond-aligned durations at the requested playback speed."""
    if frame_count <= 0 or fps <= 0 or speed <= 0:
        raise ValueError("frame count, fps and speed must be positive")
    boundaries = np.round(np.arange(frame_count + 1) * 100 / (fps * speed)).astype(int)
    durations = (np.diff(boundaries) * 10).tolist()
    if len(durations) != frame_count or any(value <= 0 for value in durations):
        raise ValueError("GIF timing could not represent the requested playback speed")
    return durations


def read_stage(case, stage):
    spec = case.get("stages", {}).get(stage)
    if not isinstance(spec, dict):
        return None, [f"stages.{stage}"]
    missing = [key for key in ("motion_file", "diagnostics_file")
               if not spec.get(key) or not Path(spec[key]).is_file()]
    if missing:
        return None, [f"stages.{stage}.{key}" for key in missing]
    return {
        "motion_file": Path(spec["motion_file"]).resolve(),
        "diagnostics_file": Path(spec["diagnostics_file"]).resolve(),
    }, []


def motion_fps(motion_file):
    from motionbricks.data.parc_paired_source import _restricted_load

    motion_file = Path(motion_file)
    if motion_file.stat().st_size > MAX_MOTION_BYTES:
        raise ValueError(f"{motion_file} exceeds the bounded motion-file read limit")
    outer = _restricted_load(motion_file.read_bytes())
    if not isinstance(outer, dict) or "motion_data" not in outer:
        raise ValueError(f"{motion_file} is missing motion_data")
    motion = outer["motion_data"]
    if isinstance(motion, bytes):
        motion = _restricted_load(motion)
    if not isinstance(motion, dict) or "fps" not in motion:
        raise ValueError(f"{motion_file} motion_data is missing fps")
    fps = float(motion["fps"])
    if not math.isfinite(fps) or not np.isclose(fps, FPS, rtol=0, atol=1e-6):
        raise ValueError(f"expected authoritative motion fps {FPS}, got {fps}")
    return fps


def load_saved_fk(stage, expected_fps=FPS):
    source_fps = motion_fps(stage["motion_file"])
    with np.load(stage["diagnostics_file"], allow_pickle=False) as saved:
        required = ("body_names", "body_world_pos", "body_world_rot", "body_contacts")
        missing = [key for key in required if key not in saved]
        if missing:
            raise ValueError(f"{stage['diagnostics_file']} is missing keys: {missing}")
        names = [str(name) for name in saved["body_names"].tolist()]
        positions = saved["body_world_pos"].copy()
        rotations = saved["body_world_rot"].copy()
        contacts = saved["body_contacts"].copy()
        diagnostic_fps = float(saved["fps"].item()) if "fps" in saved else None
    if not np.isclose(source_fps, expected_fps, rtol=0, atol=1e-6):
        raise ValueError(f"expected saved motion at {expected_fps} fps, got {source_fps}")
    if diagnostic_fps is not None and not np.isclose(
            diagnostic_fps, source_fps, rtol=0, atol=1e-6):
        raise ValueError(
            f"diagnostic fps {diagnostic_fps} disagrees with motion fps {source_fps}")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("body_world_pos must have shape [frames, bodies, 3]")
    if rotations.shape != (*positions.shape[:2], 4):
        raise ValueError("body_world_rot must have shape [frames, bodies, 4] in xyzw order")
    if contacts.shape != positions.shape[:2] or len(names) != positions.shape[1]:
        raise ValueError("saved body names/contact labels do not match body FK dimensions")
    count = len(positions)
    if count == 0:
        raise ValueError("saved FK contains no frames")
    if not all(np.isfinite(array).all() for array in (positions, rotations, contacts)):
        raise ValueError("saved FK or contact labels contain nonfinite values")
    if np.max(np.abs(np.linalg.norm(rotations, axis=-1) - 1.0)) > 1e-4:
        raise ValueError("saved FK quaternions are not unit xyzw")
    if count > MAX_FRAMES:
        raise ValueError(f"frame count {count} exceeds worker limit {MAX_FRAMES}")
    # Independent read-only baselines make accidental in-place pose edits detectable.
    poses = (positions.copy(), rotations.copy())
    return {"names": names, "positions": positions, "rotations": rotations,
            "contacts": contacts, "fps": source_fps, "pose_baseline": poses}


def verify_pose_unchanged(saved):
    if not (np.array_equal(saved["positions"], saved["pose_baseline"][0]) and
            np.array_equal(saved["rotations"], saved["pose_baseline"][1])):
        raise AssertionError("renderer changed saved source poses")


def terrain_for(stage):
    from scripts.render_parc_pilot import load_terrain

    return load_terrain(stage["motion_file"])


def assert_same_terrain(a, b):
    if not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]):
        raise ValueError("raw and optimized motion files do not contain identical terrain")


def panel_image(rgb, banner, title, frame_index, frame_count, contacts, names):
    from PIL import Image, ImageDraw
    from scripts.render_parc_pilot import font

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle((0, 0, width, 80), fill="#20282d")
    draw.text((18, 10), banner, font=font(21, True), fill="white")
    draw.text((18, 48), title, font=font(14), fill="#d4dce1")
    draw.rectangle((0, height - 57, width, height), fill="#20282d")
    active = [name.replace("_", " ") for name, value in zip(names, contacts) if value > .5]
    draw.text((18, height - 47), "Predicted contact: " + (", ".join(active) or "none"),
              font=font(14), fill="#f5cf53")
    clock = f"{frame_index / FPS:.2f} / {(frame_count - 1) / FPS:.2f} s / {PLAYBACK_SPEED:g}x"
    bounds = draw.textbbox((0, 0), clock, font=font(16, True))
    draw.text((width - (bounds[2] - bounds[0]) - 16, 48), clock,
              font=font(16, True), fill="white")
    return image


def compose_frame(raw_rgb, optimized_rgb, title, frame_index, frame_count,
                  raw_contacts, optimized_contacts, names):
    from PIL import Image, ImageDraw
    from scripts.render_parc_pilot import font

    raw_panel = panel_image(raw_rgb, "RAW CANDIDATE", title, frame_index, frame_count,
                            raw_contacts, names)
    optimized_panel = panel_image(optimized_rgb, "OFFICIAL KINEMATIC OPTIMIZATION",
                                  title, frame_index, frame_count,
                                  optimized_contacts, names)
    canvas = Image.new("RGB", (raw_panel.width + optimized_panel.width, HEIGHT), "#20282d")
    canvas.paste(raw_panel, (0, 0))
    canvas.paste(optimized_panel, (raw_panel.width, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, HEIGHT - 24, canvas.width, HEIGHT), fill="#151b1f")
    disclaimer = "NO PHYSICS TRACKING / NOT MB OR SOMA"
    bounds = draw.textbbox((0, 0), disclaimer, font=font(13, True))
    draw.text(((canvas.width - bounds[2] + bounds[0]) // 2, HEIGHT - 22),
              disclaimer, font=font(13, True), fill="#f2f3f4")
    draw.rectangle((0, 80, canvas.width, 82), fill="#d7dde0")
    return canvas


def render_pair(case, raw_stage, optimized_stage, output, character_xml, width):
    import mujoco
    from PIL import Image
    from scripts.render_parc_pilot import load_terrain, make_scene, set_frame

    title, azimuth = CASE_SETTINGS[case["case"]]
    raw_fk = load_saved_fk(raw_stage)
    optimized_fk = load_saved_fk(optimized_stage)
    count = len(raw_fk["positions"])
    if len(optimized_fk["positions"]) != count:
        raise ValueError("raw and optimized saved FK frame counts differ")
    if raw_fk["names"] != optimized_fk["names"]:
        raise ValueError("raw and optimized saved FK body names/order differ")
    raw_terrain, optimized_terrain = terrain_for(raw_stage), terrain_for(optimized_stage)
    assert_same_terrain(raw_terrain, optimized_terrain)
    if raw_fk["fps"] != optimized_fk["fps"]:
        raise ValueError("raw and optimized saved FK frame rates differ")

    panel_width = width // 2
    models = [make_scene(character_xml, raw_fk["names"], raw_terrain[0], raw_terrain[1],
                         raw_terrain[2], panel_width, HEIGHT),
              make_scene(character_xml, optimized_fk["names"], optimized_terrain[0],
                         optimized_terrain[1], optimized_terrain[2], panel_width, HEIGHT)]
    rgba = [model[0].geom_rgba.copy() for model in models]
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 4.6, -19, azimuth
    output.mkdir(parents=True, exist_ok=True)
    stem = case["case"]
    chosen = set(selected_frame_indices(count))
    gif_frames = []
    gif_palette = None
    max_fk_error = 0.0
    min_image_std = float("inf")
    with mujoco.Renderer(models[0][0], height=HEIGHT, width=panel_width) as raw_renderer, \
         mujoco.Renderer(models[1][0], height=HEIGHT, width=panel_width) as optimized_renderer:
        for frame_index in range(count):
            for model, fk, color in zip(models, (raw_fk, optimized_fk), rgba):
                error = set_frame(model[0], model[1], model[2], model[3],
                                  fk["positions"][frame_index], fk["rotations"][frame_index],
                                  fk["contacts"][frame_index], color)
                max_fk_error = max(max_fk_error, error)
            pelvis = raw_fk["names"].index("pelvis") if "pelvis" in raw_fk["names"] else 0
            camera.lookat[:] = (raw_fk["positions"][frame_index, pelvis] +
                                optimized_fk["positions"][frame_index, pelvis]) / 2.0
            raw_renderer.update_scene(models[0][1], camera=camera)
            optimized_renderer.update_scene(models[1][1], camera=camera)
            raw_rgb, optimized_rgb = raw_renderer.render(), optimized_renderer.render()
            deviation = min(float(raw_rgb[82:HEIGHT - 57].std()),
                            float(optimized_rgb[82:HEIGHT - 57].std()))
            if deviation < 5:
                raise AssertionError("blank rendered panel")
            min_image_std = min(min_image_std, deviation)
            image = compose_frame(raw_rgb, optimized_rgb, title, frame_index, count,
                                  raw_fk["contacts"][frame_index],
                                  optimized_fk["contacts"][frame_index], raw_fk["names"])
            if frame_index in chosen:
                for tag, selected in (("first", 0), ("mid", count // 2),
                                      ("last", count - 1)):
                    if frame_index == selected:
                        image.save(output / f"{stem}_{tag}.png")
            if gif_palette is None:
                gif_frame = image.quantize(colors=256)
                gif_palette = gif_frame
            else:
                gif_frame = image.quantize(palette=gif_palette)
            gif_frames.append(gif_frame)
    verify_pose_unchanged(raw_fk)
    verify_pose_unchanged(optimized_fk)
    gif_path = output / f"{stem}_raw_vs_optimized.gif"
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=gif_durations(count), loop=0, optimize=False, disposal=2)
    with Image.open(gif_path) as animation:
        if animation.n_frames != count:
            raise AssertionError("saved GIF frame count differs from saved FK")
    return {
        "case": case["case"], "status": "rendered", "input_status": case.get("status"),
        "frame_count": count, "fps": FPS, "gif_playback_speed": "0.5x",
        "gif": str(gif_path.resolve()),
        "pngs": [str((output / f"{stem}_{tag}.png").resolve())
                 for tag in ("first", "mid", "last")],
        "raw_motion_file": str(raw_stage["motion_file"]),
        "raw_diagnostics_file": str(raw_stage["diagnostics_file"]),
        "optimized_motion_file": str(optimized_stage["motion_file"]),
        "optimized_diagnostics_file": str(optimized_stage["diagnostics_file"]),
        "max_saved_fk_origin_error_m": max_fk_error,
        "min_panel_pixel_std": min_image_std,
        "pose_arrays_unchanged": True,
        "physics_corrected": False,
        "terrain": "identical terrain serialized in each stage motion file",
        "camera": {"azimuth_degrees": azimuth, "shared_per_frame_root_midpoint": True},
    }


def run(report_path, output, character_xml, width):
    if width not in (1440, 1600):
        raise ValueError("width must be 1440 or 1600; height is fixed at 600")
    report_path = report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("physics_corrected") is not False:
        raise ValueError("input report must explicitly declare physics_corrected: false")
    cases = report.get("cases")
    if not isinstance(cases, list) or len(cases) > MAX_REPORT_CASES:
        raise ValueError(f"report cases must be a list of at most {MAX_REPORT_CASES} entries")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_report": str(report_path), "physics_corrected": False,
                "render_limits": {"primary_cases": MAX_CASES,
                                  "report_cases": MAX_REPORT_CASES,
                                  "frames_per_case": MAX_FRAMES,
                                  "width": width, "height": HEIGHT},
                "results": []}
    seen = set()
    for case in cases:
        name = case.get("case") if isinstance(case, dict) else None
        if name not in CASE_SETTINGS:
            is_variant = (isinstance(name, str) and any(
                name.startswith(primary + "_variant_") and
                name[len(primary + "_variant_"):].isdigit()
                for primary in CASE_SETTINGS))
            status = "skipped_unrequested_variant" if is_variant else "skipped_unknown_case"
            manifest["results"].append({"case": name, "status": status})
            continue
        if name in seen:
            manifest["results"].append({"case": name, "status": "skipped_duplicate_case"})
            continue
        seen.add(name)
        available, missing = {}, []
        for stage in ("raw", "optimized", "final"):
            available[stage], stage_missing = read_stage(case, stage)
            missing.extend(stage_missing)
        if missing:
            print(json.dumps({"case": name, "status": "missing_stage_inputs",
                              "missing": missing}), flush=True)
        if any(available[key] is None for key in STAGE_PAIR):
            manifest["results"].append({"case": name, "status": "skipped_missing_required_stages",
                                        "missing": missing})
            continue
        try:
            row = render_pair(case, available["raw"], available["optimized"], output,
                              character_xml, width)
            if missing:
                row["status"] = "rendered_with_missing_stages"
            row["missing_stages"] = missing
        except Exception as error:
            row = {"case": name, "status": "render_failed",
                   "error": f"{type(error).__name__}: {error}", "missing_stages": missing}
        manifest["results"].append(row)
        print(json.dumps(row), flush=True)
    manifest_path = output / "render_manifest.json"
    manifest["manifest"] = str(manifest_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--character-xml", type=Path,
                        default=ROOT / "out/vendor/PARC/data/assets/humanoid.xml")
    parser.add_argument("--width", type=int, choices=(1440, 1600), default=1440)
    args = parser.parse_args()
    manifest = run(args.report, args.output, args.character_xml, args.width)
    if any(row['status'] in ('render_failed', 'skipped_missing_required_stages',
                             'rendered_with_missing_stages') for row in manifest['results']):
        raise SystemExit('Incomplete rendering; see render_manifest.json for failed or missing stages')
    if not any(row["status"].startswith("rendered") for row in manifest["results"]):
        raise SystemExit("No case was rendered; see render_manifest.json for explicit skip/failure reasons")


if __name__ == "__main__":
    main()
