"""Read-only causal stop/resume comparison of PARC's native history guidance."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def prefix_comparison(actual, factual, rotation=False):
    if rotation:
        # Quaternions q and -q denote the same rotation. Comparison only.
        sign = torch.where((actual*factual).sum(-1, keepdim=True) < 0, -1., 1.)
        return actual*sign, factual
    return actual, factual


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.helpers))
    sys.path.insert(0, str(args.repo))
    sys.modules.setdefault('parc', importlib.import_module('PARC'))
    import run_parc_local_terrain_pilot as pilot
    from generate_parc_stair64 import preserve_flags
    from parc.motion_generator import gen_util
    from parc.motion_generator.diffusion_util import MDMKeyType
    from parc.util import file_io_helper, terrain_util
    from parc.util.motion_util import MotionFrames

    torch.set_num_threads(4)
    digest = pilot.sha256(args.checkpoint)
    if digest != 'a03a9d4cc7b3a43b217d67116429d241c92e31e7de93e02849832b0371860e2e':
        raise ValueError('Unexpected expert checkpoint')
    model = pilot.load_official_model(args.repo, args.checkpoint, 'cuda:0', 'ema_model')
    model.predict_x0 = preserve_flags(model.predict_x0,
        (MDMKeyType.PREV_STATE_NOISE_IND_KEY, MDMKeyType.PREV_STATE_FLAG_KEY))
    rows = []
    for layout in args.layouts:
        folder = args.batch / f'layout_{layout:02d}' / 'generation'
        report = json.loads((folder / 'generation.json').read_text())
        row = next(r for r in report['candidates'] if r['candidate_id'] == 0)
        source = file_io_helper.load_ms_file(row['motion'], 'cuda:0')
        td = source.terrain_data
        terrain = terrain_util.SubTerrain('history-guidance', x_dim=td.hf.shape[0],
            y_dim=td.hf.shape[1], dx=float(td.dx), dy=float(td.dx),
            min_x=float(td.min_point[0]), min_y=float(td.min_point[1]), device='cuda:0')
        terrain.hf.copy_(td.hf)
        terrain.hf_maxmin.copy_(td.hf_maxmin)
        course = json.loads((folder / 'course/course_manifest.json').read_text())
        for event in row['stop_response']['stops']:
            frame = event['start_frame']
            motion = source.motion_data
            prefix = MotionFrames(root_pos=motion.root_pos[None, frame-2:frame],
                root_rot=motion.root_rot[None, frame-2:frame],
                joint_rot=motion.joint_rot[None, frame-2:frame],
                contacts=motion.body_contacts[None, frame-2:frame])
            with np.load(row['intent'], allow_pickle=False) as data:
                direction = torch.from_numpy(data['command_world_xy'][frame-1].copy()).float().cuda()
            if not torch.isclose(direction.norm(), direction.new_tensor(1.), atol=1e-4):
                raise ValueError('Pre-stop command is not a recorded move direction')
            segment = next(p for p in course['route_segments'] if p['stage'] == event['phase'])
            a = torch.tensor(segment['start_xyz_m'][:2], device='cuda:0')
            b = torch.tensor(segment['end_xyz_m'][:2], device='cuda:0')
            axis = (b-a) / (b-a).norm()
            for scale in (.65, 1.):
                torch.manual_seed(20261013 + layout)
                history = prefix
                parts = []
                prefix_error = 0.
                boundary_angles = []
                settled = False
                settings = gen_util.MDMGenSettings()
                settings.ddim_stride = 5
                settings.cfg_scale = scale
                settings.use_cfg = True

                def generate(stopping):
                    nonlocal history, prefix_error
                    target = history.root_pos[:, -1].clone()
                    if not stopping:
                        target[:, :2] += direction * 3.
                    with torch.no_grad():
                        generated = gen_util.gen_mdm_motion(target_world_pos=target,
                            prev_frames=history, terrain=terrain, mdm_model=model,
                            char_model=model._kin_char_model, mdm_settings=settings)
                    for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts'):
                        actual = getattr(generated, key)[:, :2]
                        factual = getattr(history, key)
                        actual, factual = prefix_comparison(actual, factual,
                            rotation=key in ('root_rot', 'joint_rot'))
                        error = float((actual-factual).abs().max())
                        prefix_error = max(prefix_error, error)
                        torch.testing.assert_close(actual, factual, atol=3e-5, rtol=0)
                    q0 = history.joint_rot[:, -1]
                    q1 = generated.joint_rot[:, 2]
                    dot = (q0*q1).sum(-1).abs().clamp(0., 1.)
                    boundary_angles.append(float((2*dot.acos()*180/torch.pi).max()))
                    parts.append(generated.get_slice(slice(2, 9)))
                    history = generated.get_slice(slice(7, 9))

                for block in range(26):
                    generate(True)
                    roots = torch.cat([p.root_pos for p in parts], 1)
                    if roots.shape[1] >= 30:
                        speed = torch.diff(roots[0, -15:, :2], dim=0).norm(dim=-1)*30
                        drift = (roots[0, -1, :2]-roots[0, -15, :2]).norm()
                        settled = bool(torch.quantile(speed, .9) <= .15 and drift <= .08)
                        if settled:
                            break
                stop_blocks = block+1
                stopped = history.root_pos[0, -1].clone()
                progress = float((stopped[:2]-a) @ axis)
                inside = .15 < progress < float((b-a).norm())-.15
                stop_angles = boundary_angles.copy()
                for _ in range(4):
                    generate(False)
                resume = float((history.root_pos[0, -1, :2]-stopped[:2]) @ axis)
                value = dict(layout=layout, phase=event['phase'], cfg_scale=scale,
                    stop_seconds=stop_blocks*7/30, settled=settled, stopped_inside_flight=inside,
                    stop_displacement_m=float((stopped[:2]-prefix.root_pos[0, -1, :2]).norm()),
                    tail_speed_p90_mps=float(torch.quantile(speed, .9)), tail_drift_m=float(drift),
                    resume_progress_m=resume, stop_resume_screen=settled and inside and resume >= .25,
                    max_stop_boundary_joint_angle_deg=max(stop_angles),
                    max_all_boundary_joint_angle_deg=max(boundary_angles), prefix_error=prefix_error,
                    input_speed_mps=float(torch.diff(prefix.root_pos[0, :, :2], dim=0).norm()*30))
                rows.append(value)
                name = f"layout{layout}-{event['phase']}-cfg{scale:g}"
                np.savez_compressed(args.output/(name+'.npz'),
                    **{key:torch.cat([getattr(prefix, key)]+[getattr(p, key) for p in parts], 1).cpu().numpy()
                       for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts')})
                (args.output/'progress.json').write_text(json.dumps(dict(rows=rows), indent=2))
                print(json.dumps(value), flush=True)
    result = dict(rows=rows, checkpoint_sha256=digest, updates=0, postprocessing=None,
        quality_approved=False, note='Native CFG .65 mixes history-conditioned and no-history predictions; 1 is the no-history prediction endpoint. Both receive actual terrain and zero stop target. No limb/root output was overwritten. This screen is not physics or SOMA certification.')
    (args.output/'complete.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('helpers', 'repo', 'checkpoint', 'batch', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--layouts', nargs='+', type=int, default=[3, 6])
    main(parser.parse_args())
