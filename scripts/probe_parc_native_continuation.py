"""One native PARC generation call from real source history; no SOMA admission."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['WANDB_MODE'] = 'disabled'
    torch.set_num_threads(4)
    torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(args.parc.resolve()))
    sys.path.insert(0, str(args.helpers.resolve()))
    sys.modules.setdefault('parc', importlib.import_module('PARC'))
    from parc.motion_generator import gen_util
    from parc.motion_generator.diffusion_util import MDMKeyType
    from parc.util import geom_util, terrain_util, torch_util
    from parc.util.motion_util import MotionFrames
    from run_parc_local_terrain_pilot import load_official_model
    specification = importlib.util.spec_from_file_location('diagnostic_exact_prism', args.sdf_path)
    sdf_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(sdf_module)
    receipt = json.loads((args.input.parent / 'complete.json').read_text())
    if digest(args.input) != receipt['input_sha256']:
        raise ValueError('Native prefix input changed')
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != 'a03a9d4cc7b3a43b217d67116429d241c92e31e7de93e02849832b0371860e2e':
        raise ValueError('Released PARC checkpoint identity changed')
    model = load_official_model(args.parc.resolve(), args.checkpoint.resolve(), 'cuda:0', weight_key='ema_model')
    if model._num_prev_states != 2 or model._sequence_fps != 30 or model._seq_len != 15:
        raise ValueError('Native checkpoint history/time contract differs')
    with np.load(args.input, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    terrain = terrain_util.SubTerrain(x_dim=arrays['hf'].shape[0], y_dim=arrays['hf'].shape[1],
                                     dx=float(arrays['dx']), dy=float(arrays['dx']),
                                     min_x=float(arrays['min_point'][0]), min_y=float(arrays['min_point'][1]), device='cuda:0')
    terrain.hf = torch.from_numpy(arrays['hf']).float().cuda()
    terrain.hf_maxmin = torch.stack((terrain.hf, terrain.hf), -1)
    batch = 4
    def tensor(name):
        return torch.from_numpy(arrays[name][:2]).float().cuda()[None].expand(batch, *arrays[name][:2].shape).clone()
    previous = MotionFrames(root_pos=tensor('root_pos'), root_rot=tensor('root_rot'),
                            joint_rot=tensor('joint_rot'), contacts=tensor('body_contacts'))
    world_direction = torch.from_numpy(arrays['command_world_xy']).float().cuda()
    world_direction /= world_direction.norm()
    destination = previous.root_pos[:, -1].clone()
    destination[:, :2] += world_direction * 5.
    settings = gen_util.MDMGenSettings()
    settings.ddim_stride = 10
    settings.cfg_scale = .65
    settings.use_prev_state = True
    flag_keys = (MDMKeyType.PREV_STATE_NOISE_IND_KEY, MDMKeyType.PREV_STATE_FLAG_KEY)
    captured_flags = []
    native_sequence = model.gen_sequence
    def record_sequence(conditions, *positional, **keywords):
        before = {str(key): conditions[key].clone().cpu().tolist() for key in flag_keys}
        result = native_sequence(conditions, *positional, **keywords)
        after = {str(key): conditions[key].clone().cpu().tolist() for key in flag_keys}
        captured_flags.append({'before': before, 'after': after})
        return result
    model.gen_sequence = record_sequence
    native_predict = model.predict_x0
    def restore_condition_flags(x_t, conditions, *positional, **keywords):
        original_flags = {key: conditions[key].clone() for key in flag_keys}
        try:
            return native_predict(x_t, conditions, *positional, **keywords)
        finally:
            for key, flag in original_flags.items():
                conditions[key].copy_(flag)
    generated_runs = []
    differences = {}
    timings = {}
    for label in ('released', 'released_repeat', 'preserved_condition_flags'):
        model.predict_x0 = restore_condition_flags if label == 'preserved_condition_flags' else native_predict
        torch.manual_seed(2333)
        started = time.perf_counter()
        generated = gen_util.gen_mdm_motion(target_world_pos=destination, prev_frames=previous, terrain=terrain,
                                           mdm_model=model, char_model=model._kin_char_model, mdm_settings=settings)
        torch.cuda.synchronize()
        timings[label] = time.perf_counter() - started
        differences[label] = {}
        for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts'):
            a, b = getattr(generated, key)[:, :2], getattr(previous, key)
            if key in ('root_rot', 'joint_rot'):
                difference = torch.minimum((a - b).abs().amax(-1), (a + b).abs().amax(-1)).max()
            else:
                difference = (a - b).abs().max()
            differences[label][key] = float(difference)
        # Save before any comparison assertion. Failed prefix contracts remain
        # inspectable and no generated candidate is dropped.
        np.savez_compressed(args.output / (label + '.npz'),
                            **{key: getattr(generated, key).cpu().numpy() for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts')},
                            hf=arrays['hf'], min_point=arrays['min_point'], dx=arrays['dx'])
        generated_runs.append((label, generated))
    future_differences = {key: float((getattr(generated_runs[0][1], key)[:, 2:]
                                    - getattr(generated_runs[2][1], key)[:, 2:]).abs().max())
                          for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts')}
    repeated_differences = {key: float((getattr(generated_runs[0][1], key)[:, 2:]
                                      - getattr(generated_runs[1][1], key)[:, 2:]).abs().max())
                            for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts')}
    character = model._kin_char_model
    local_points = geom_util.get_char_point_samples(character)
    # Swap height into Y solely for the scalar distance query. This is an
    # isometry of the same raster; it is not a SOMA skeleton conversion.
    field = sdf_module.HeightfieldPrismSDF({'heightfield_y_m': terrain.hf.T,
                                           'min_point_world_xyz_m': [float(terrain.min_point[0]), 0., float(terrain.min_point[1])],
                                           'dx_m': float(terrain.dxdy[0])})
    source = MotionFrames(root_pos=torch.from_numpy(arrays['root_pos'][:model._seq_len])[None].float().cuda(),
                          root_rot=torch.from_numpy(arrays['root_rot'][:model._seq_len])[None].float().cuda(),
                          joint_rot=torch.from_numpy(arrays['joint_rot'][:model._seq_len])[None].float().cuda(),
                          contacts=torch.from_numpy(arrays['body_contacts'][:model._seq_len])[None].float().cuda())
    rows = []
    for label, frames in [('source', source)] + generated_runs:
        body_pos, body_rot = character.forward_kinematics(frames.root_pos, frames.root_rot, frames.joint_rot)
        points = []
        for body, samples in enumerate(local_points):
            points.append(torch_util.quat_rotate(body_rot[..., body, None, :], samples[None, None]) + body_pos[..., body, None, :])
        points = torch.cat(points, -2)
        for index in range(len(frames.root_pos)):
            evaluated = points[index, 2:]
            distance, _, known = field.query(evaluated[..., [0, 2, 1]])
            depths = (-distance[known]).clamp_min(0)
            delta = frames.root_pos[index, -1] - previous.root_pos[0, 1]
            first_delta = frames.root_pos[index, 2] - previous.root_pos[0, 1]
            source_first_delta = source.root_pos[0, 2] - source.root_pos[0, 1]
            row = {'label': label, 'candidate': index, 'frames_including_history': model._seq_len,
                   'new_frames': model._seq_len - model._num_prev_states,
                   'native_sampled_surface_peak_m': float(depths.max()),
                   'native_sampled_surface_p99_m': float(torch.quantile(depths, .99)),
                   'known_query_fraction': float(known.float().mean()), 'body_points_per_frame': points.shape[-2],
                   'progress_m': float(delta[:2].dot(world_direction)), 'root_rise_m': float(delta[2]),
                   'first_step_norm_m': float(first_delta.norm()),
                   'first_step_difference_from_source_m': float((first_delta - source_first_delta).norm()),
                   'quality_approved': False, 'physics_corrected': False}
            rows.append(row)
        np.savez_compressed(args.output / (label + '.npz'),
                            **{key: getattr(frames, key).cpu().numpy() for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts')},
                            hf=arrays['hf'], min_point=arrays['min_point'], dx=arrays['dx'])
    report = {'rows': rows, 'generation_calls': 3, 'generated_candidates_per_call': 4, 'seed': 2333,
              'generation_seconds': timings, 'prefix_max_errors': differences,
              'condition_flags': captured_flags, 'future_max_differences': future_differences,
              'released_repeat_future_max_differences': repeated_differences,
              'prefix_preserved_after_flag_repair': all(value <= 1e-5 for value in differences['preserved_condition_flags'].values()),
              'checkpoint_sha256': checkpoint_hash, 'input_sha256': digest(args.input),
              'ddim_stride': settings.ddim_stride, 'cfg_scale': settings.cfg_scale,
              'quality_approved': False, 'soma_recovery_labels_created': False,
              'scope': 'Matched original-PARC-prefix continuation calls, no future source conditions, no candidate filtering '
                       'or physics correction. Native sparse character surface samples only, not SOMA skin '
                       'or complete-route validation. Does not repair the failed SOMA-to-PARC bridge.'}
    (args.output / 'complete.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--parc', type=Path, required=True)
    parser.add_argument('--helpers', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sdf-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
