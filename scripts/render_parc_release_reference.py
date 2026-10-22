"""Replay a released PARC motion/terrain pair, without generation or correction."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image
import torch

from render_parc_pilot import ROOT, caption, make_scene, set_frame
from motionbricks.data.parc_paired_source import _decode_section, _restricted_load


def render_reference(source, output, repo, title, release_member, azimuth=60):
    sys.path.insert(0, str(repo.resolve()))
    sys.modules.setdefault('parc', importlib.import_module('PARC'))
    from parc.anim.kin_char_model import KinCharModel

    payload = source.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    container = _restricted_load(payload)
    motion = _decode_section(container, 'motion_data')
    terrain = _decode_section(container, 'terrain_data')
    fps = float(motion['fps'])
    if not np.isfinite(fps) or not 0 < fps <= 100:
        raise ValueError('Replay GIF requires a finite source FPS in (0, 100]')
    tensors = [torch.as_tensor(motion[key], dtype=torch.float32) for key in
               ('root_pos', 'root_rot', 'joint_rot')]
    character_xml = repo / 'data/assets/humanoid.xml'
    character = KinCharModel(torch.device('cpu'))
    character.load_char_file(str(character_xml))
    names = character.get_body_names()
    with torch.no_grad():
        positions, rotations = [value.numpy() for value in character.forward_kinematics(*tensors)]
    contacts = np.asarray(motion['body_contacts'])
    hf, origin, dx = np.asarray(terrain['hf']), np.asarray(terrain['min_point']), float(terrain['dx'])
    count = len(positions)
    if not count or contacts.shape != (count, len(names)):
        raise ValueError('Contact and FK frame/body axes disagree')
    if not all(np.isfinite(value).all() for value in (positions, rotations, contacts, hf, origin)):
        raise ValueError('Nonfinite source data')
    if np.max(np.abs(np.linalg.norm(rotations, axis=-1) - 1)) > 1e-4:
        raise ValueError('Source FK rotations are not unit quaternions')
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError('Invalid terrain spacing')

    output.mkdir(parents=True, exist_ok=False)
    width, height = 960, 600
    model, state, body_ids, addresses = make_scene(
        character_xml, names, hf, origin, dx, width, height)
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 4.6, -19, azimuth
    base_rgba = model.geom_rgba.copy()
    images, errors, deviations = [], [], []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for index in range(count):
            errors.append(set_frame(model, state, body_ids, addresses, positions[index],
                                    rotations[index], contacts[index], base_rgba))
            camera.lookat[:] = positions[index, names.index('pelvis')] + [0, 0, .02]
            renderer.update_scene(state, camera=camera)
            rgb = renderer.render()
            deviations.append(float(rgb[80:height - 68].std()))
            if deviations[-1] < 5:
                raise AssertionError('Blank rendered frame')
            frame = caption(rgb, title, index, count, 1, contacts[index], names,
                            'RELEASE replay, not new generation', fps=fps)
            images.append(frame)
            if index in {0, count // 2, count - 1}:
                frame.save(output / f'frame_{index:04d}.png')

    # Use actual source FPS; no interpolation, frame dropping, or motion changes.
    boundaries = np.round(np.arange(count + 1) * 100 / fps).astype(int) * 10
    gif = output / 'release_reference_1x.gif'
    images[0].save(gif, save_all=True, append_images=images[1:],
                   duration=np.diff(boundaries).tolist(), loop=0, optimize=False, disposal=2)
    with Image.open(gif) as saved:
        if saved.n_frames != count:
            raise AssertionError('GIF frame count mismatch')
        duration_ms = 0
        for index in range(saved.n_frames):
            saved.seek(index)
            duration_ms += saved.info['duration']
    if duration_ms != boundaries[-1]:
        raise AssertionError('GIF playback duration mismatch')
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise AssertionError('Source changed during replay')
    manifest = {
        'status': 'official_release_replayed_not_new_generation',
        'source': str(source.resolve()), 'source_sha256': source_hash,
        'release_member': release_member, 'frames': count, 'source_fps': fps,
        'playback_speed': 1.0, 'gif_duration_ms': duration_ms,
        'motion_postprocessing': 'none', 'retargeting': 'none',
        'physics_correction_provenance': 'not_established_by_replay_or_filename',
        'training_approved': False, 'max_render_fk_error_m': max(errors),
        'min_image_std': min(deviations), 'gif': str(gif.resolve()),
        'character_geometry': 'official humanoid.xml body geoms',
        'terrain_geometry': 'source cell tops and footprints; decorative column bottoms',
        'character_xml_sha256': hashlib.sha256(character_xml.read_bytes()).hexdigest(),
        'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=ROOT / 'out/vendor/PARC')
    parser.add_argument('--title', default='OFFICIAL RELEASE / REFERENCE')
    parser.add_argument('--release-member', required=True)
    parser.add_argument('--azimuth', type=float, default=60)
    args = parser.parse_args()
    print(json.dumps(render_reference(args.source, args.output, args.repo, args.title,
                                     args.release_member, args.azimuth)), flush=True)


if __name__ == '__main__':
    main()
