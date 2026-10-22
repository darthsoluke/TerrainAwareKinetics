"""Render saved PARC FK and exact terrain cells without changing the motion."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motionbricks.data.parc_paired_source import _restricted_load


ROTATION_PARITY_TOLERANCE = 1e-6
QUATERNION_NORM_TOLERANCE = 1e-4


def numbers(values):
    return ' '.join(f'{float(value):.9g}' for value in np.ravel(values))


def normalized_mujoco_quaternion_wxyz(rotation_xyzw):
    quaternion = np.asarray(rotation_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError('Saved FK quaternion must contain four finite xyzw values')
    norm = math.hypot(*(float(value) for value in quaternion))
    if not math.isfinite(norm) or norm == 0.0 or abs(norm - 1.0) > QUATERNION_NORM_TOLERANCE:
        raise ValueError(
            f'Saved FK quaternion norm must be within {QUATERNION_NORM_TOLERANCE:g} of 1'
        )
    return quaternion[[3, 0, 1, 2]] / norm


def load_terrain(path):
    outer = _restricted_load(path.read_bytes())
    terrain = outer['terrain_data']
    if isinstance(terrain, bytes):
        terrain = _restricted_load(terrain)
    return np.asarray(terrain['hf']), np.asarray(terrain['min_point']), float(terrain['dx'])


def make_scene(character_xml, body_names, hf, origin, dx, width, height):
    source = ET.parse(character_xml).getroot()
    source_bodies = {body.attrib['name']: body for body in source.findall('.//body')}
    scene = ET.Element('mujoco', model='PARC_saved_FK_render_only')
    ET.SubElement(scene, 'compiler', angle='radian')
    visual = ET.SubElement(scene, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, 'quality', shadowsize='2048', offsamples='4')
    ET.SubElement(visual, 'headlight', diffuse='.35 .35 .35', ambient='.25 .25 .25', specular='.1 .1 .1')
    assets = ET.SubElement(scene, 'asset')
    ET.SubElement(assets, 'texture', type='skybox', builtin='gradient', rgb1='.86 .9 .93',
                  rgb2='.96 .97 .98', width='128', height='128')
    world = ET.SubElement(scene, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -6 12', dir='0 .4 -1', directional='true',
                  diffuse='.5 .5 .5', castshadow='true')
    bottom = float(np.min(hf)) - .6
    for x, y in np.ndindex(hf.shape):
        halfheight = (float(hf[x, y]) - bottom) / 2
        # PARC uses nearest-cell lookup: min_point is a cell center, not a corner.
        color = '.71 .75 .77 1' if (x + y) % 2 else '.78 .81 .82 1'
        ET.SubElement(world, 'geom', name=f'terrain_{x}_{y}', type='box',
            pos=numbers([origin[0] + x * dx, origin[1] + y * dx, bottom + halfheight]),
            size=numbers([dx / 2, dx / 2, halfheight]), rgba=color,
            contype='0', conaffinity='0')
    for name in body_names:
        body = ET.SubElement(world, 'body', name=str(name))
        ET.SubElement(body, 'freejoint', name=str(name) + '_render_pose')
        for geom in source_bodies[name].findall('geom'):
            copied = deepcopy(geom)
            copied.set('type', copied.get('type', 'capsule'))
            copied.set('contype', '0')
            copied.set('conaffinity', '0')
            color = '.08 .55 .64 1' if name.startswith('left') else (
                '.8 .25 .2 1' if name.startswith('right') else '.88 .88 .87 1')
            copied.set('rgba', color)
            body.append(copied)
    model = mujoco.MjModel.from_xml_string(ET.tostring(scene, encoding='unicode'))
    data = mujoco.MjData(model)
    body_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names])
    addresses = model.jnt_qposadr[model.body_jntadr[body_ids]]
    return model, data, body_ids, addresses


def set_frame(model, data, body_ids, addresses, positions, rotations, contacts, base_rgba):
    expected_quaternions = np.asarray([
        normalized_mujoco_quaternion_wxyz(rotation) for rotation in rotations
    ])
    for index, address in enumerate(addresses):
        data.qpos[address:address + 3] = positions[index]
        # Official PARC FK stores xyzw; MuJoCo freejoint stores wxyz.
        data.qpos[address + 3:address + 7] = rotations[index, [3, 0, 1, 2]]
    model.geom_rgba[:] = base_rgba
    for index, body_id in enumerate(body_ids):
        if contacts[index] > .5:
            begin, count = model.body_geomadr[body_id], model.body_geomnum[body_id]
            model.geom_rgba[begin:begin + count] = [.96, .71, .08, 1.]
    mujoco.mj_forward(model, data)  # FK/render update only; never mj_step.
    error = float(np.abs(data.xpos[body_ids] - positions).max())
    if error > 1e-6:
        raise AssertionError(f'Rendered body origins disagree with saved FK: {error}')
    for index, body_id in enumerate(body_ids):
        expected = np.empty(9)
        mujoco.mju_quat2Mat(expected, expected_quaternions[index])
        if not np.allclose(data.xmat[body_id], expected, atol=ROTATION_PARITY_TOLERANCE, rtol=0):
            raise AssertionError('Rendered body orientation disagrees with saved FK')
    return error


def font(size, bold=False):
    path = Path('C:/Windows/Fonts') / ('arialbd.ttf' if bold else 'arial.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def caption(rgb, title, frame_index, frame_count, speed, contacts, body_names, label, fps=30):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    draw.rectangle((0, 0, w, 79), fill='#20282d')
    draw.text((22, 12), title, font=font(25, True), fill='white')
    draw.text((22, 47), f'PARC source humanoid | {label} | no IK / no smoothing',
              font=font(16), fill='#d4dce1')
    draw.rectangle((0, h - 68, w, h), fill='#20282d')
    active = [str(name).replace('_', ' ') for name, value in zip(body_names, contacts)
              if value > .5]
    draw.text((22, h - 58), 'Yellow: contact label, not verified contact', font=font(16), fill='#f5cf53')
    draw.text((22, h - 32), 'Active: ' + (', '.join(active) or 'none'), font=font(15), fill='white')
    clock = f'{frame_index / fps:.2f} / {(frame_count - 1) / fps:.2f} s | {speed:g}x'
    bbox = draw.textbbox((0, 0), clock, font=font(17))
    draw.text((w - (bbox[2] - bbox[0]) - 20, 17), clock, font=font(17), fill='white')
    return image


def render_case(case_dir, output, character_xml, title, speed, width, height,
                preview_only=False, source_reference=False):
    hand = case_dir.name == 'hand_contact_continuation'
    diag_name = ('source_kinematic_world_terrain_diagnostics.npz' if source_reference else
                 'generated_kinematic_world_terrain_diagnostics.npz') if hand else 'kinematic_world_terrain_diagnostics.npz'
    diag_path = case_dir / diag_name
    data_file = 'source_climb_motion_and_terrain.pkl' if source_reference else 'generated_raw_kinematic.pkl'
    hf, origin, dx = load_terrain(case_dir / data_file)
    with np.load(diag_path, allow_pickle=False) as saved:
        names = saved['body_names'].tolist()
        positions, rotations, contacts = (saved[key].copy() for key in
            ('body_world_pos', 'body_world_rot', 'body_contacts'))
    if source_reference:
        condition_frame = json.loads((case_dir / 'result.json').read_text())['source_frame_index']
        start, end = max(0, condition_frame - 30), min(len(positions), condition_frame + 45)
        positions, rotations, contacts = positions[start:end], rotations[start:end], contacts[start:end]
    else:
        start, end = 0, len(positions)
    if not all(np.isfinite(array).all() for array in (positions, rotations, contacts, hf)):
        raise ValueError('Nonfinite source arrays cannot be rendered')
    if np.max(np.abs(np.linalg.norm(rotations, axis=-1) - 1)) > 1e-4:
        raise ValueError('Saved FK quaternions are not unit xyzw')
    model, state, body_ids, addresses = make_scene(character_xml, names, hf, origin, dx, width, height)
    base_rgba = model.geom_rgba.copy()
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation = (4.6, -22) if hand else (4.6, -19)
    # Keep camera orientation constant within a clip, so changes in heading remain visible.
    camera.azimuth = 15 if hand else 60
    frames, frame_indices, pixel_stds, errors = [], [], [], []
    selected = sorted({0, len(positions) // 2, len(positions) - 1}) if preview_only else list(range(len(positions)))
    stem = 'hand_source_reference' if source_reference else case_dir.name
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index in selected:
            errors.append(set_frame(model, state, body_ids, addresses, positions[index],
                                    rotations[index], contacts[index], base_rgba))
            root = positions[index, names.index('pelvis')]
            camera.lookat[:] = root + np.array([0., 0., .02])
            renderer.update_scene(state, camera=camera)
            rgb = renderer.render()
            deviation = float(rgb[80:height - 68].std())
            if deviation < 5:
                raise AssertionError('Blank rendered frame')
            pixel_stds.append(deviation)
            label = 'SOURCE reference' if source_reference else 'RAW generated candidate'
            image = caption(rgb, title, index, len(positions), speed, contacts[index], names, label)
            if index in {0, len(positions) // 2, len(positions) - 1}:
                image.save(output / f'{stem}_{index:04d}.png')
            frames.append(image)
            frame_indices.append(index)
    if not preview_only:
        # GIF durations are centiseconds; cumulative rounding avoids playback drift.
        boundaries = np.round(np.arange(len(frames) + 1) * 100 / (30 * speed)).astype(int) * 10
        durations = np.diff(boundaries).tolist()
        frames[0].save(output / f'{stem}.gif', save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=False, disposal=2)
        with Image.open(output / f'{stem}.gif') as animation:
            assert animation.n_frames == len(frames)
    return {
        'case': stem, 'source_diagnostics': str(diag_path.resolve()),
        'source_sha256': hashlib.sha256(diag_path.read_bytes()).hexdigest(),
        'source_frame_range': [start, end], 'rendered_frames': len(frames), 'motion_fps': 30,
        'playback_speed': speed, 'max_render_fk_error_m': max(errors),
        'min_image_std': min(pixel_stds), 'motion_postprocessing': 'none',
        'body_geometry': 'official PARC humanoid.xml sphere/capsule/box geoms',
        'terrain_geometry': 'nearest-cell columns, exact dx and cell-center origin',
        'gif': None if preview_only else str((output / f'{stem}.gif').resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--character-xml', type=Path, default=ROOT / 'out/vendor/PARC/data/assets/humanoid.xml')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, title, speed in [('stairs_up', 'STAIRS / ASCENT', .5),
                               ('stairs_down', 'STAIRS / DESCENT', .5),
                               ('hand_contact_continuation', 'HAND-CONTACT / CONTINUATION', .25)]:
        row = render_case(args.run / case, args.output, args.character_xml,
                          title, speed, args.width, args.height, args.preview_only)
        rows.append(row)
        print(json.dumps(row), flush=True)
    rows.append(render_case(args.run / 'hand_contact_continuation', args.output, args.character_xml,
        'HAND-CONTACT / SOURCE CONTEXT', .5, args.width, args.height, args.preview_only, source_reference=True))
    (args.output / 'render_manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
