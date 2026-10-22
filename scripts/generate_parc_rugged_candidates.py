"""Continuous PARC candidates; preserved factual prefix, no STOP or pose edits."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import torch

import generate_parc_long_stairs as generation
import build_parc_stair_course as course_base
from build_parc_rugged_course import build_rugged_course


def preserve_flags(predict, keys):
    def wrapped(x, conditions, *args, **kwargs):
        flags={key:conditions[key].clone() for key in keys}
        try:
            return predict(x, conditions, *args, **kwargs)
        finally:
            for key,value in flags.items():conditions[key].copy_(value)
    return wrapped


def main(args):
    if generation.pilot.sha256(args.checkpoint) != 'a03a9d4cc7b3a43b217d67116429d241c92e31e7de93e02849832b0371860e2e':
        raise ValueError('Unexpected expert checkpoint')
    course_base.PARC_REPO = args.repo
    sys.path.insert(0, str(args.repo))
    sys.modules.setdefault('parc', importlib.import_module('PARC'))
    from parc.motion_generator import gen_util
    from parc.motion_generator.diffusion_util import MDMKeyType
    audit = dict(calls=0, checked_history_calls=0, max_prefix_error=0., stop_task=False)
    native_load = generation.pilot.load_official_model
    def load(*a, **kw):
        model = native_load(*a, **kw)
        model.predict_x0 = preserve_flags(model.predict_x0,
            (MDMKeyType.PREV_STATE_NOISE_IND_KEY, MDMKeyType.PREV_STATE_FLAG_KEY))
        return model
    generation.pilot.load_official_model = load
    generation.build_course = lambda device: build_rugged_course(args.layout, device)
    native_generate = gen_util.gen_mdm_motion
    def checked_generate(*a, **kw):
        if a:
            raise ValueError('Explicit arguments required')
        result = native_generate(**kw)
        audit['calls'] += 1
        settings = kw['mdm_settings']
        if bool(settings.use_prev_state) and bool(settings.prev_state_ind_key):
            history = kw['mdm_model']._num_prev_states
            for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts'):
                x, y = getattr(result, key)[:, :history], getattr(kw['prev_frames'], key)[:, :history]
                error = torch.minimum((x-y).abs().amax(-1), (x+y).abs().amax(-1)).max() if key.endswith('rot') else (x-y).abs().max()
                audit['max_prefix_error'] = max(audit['max_prefix_error'], float(error))
                if error > 3e-5:
                    raise ValueError('Factual history changed: '+key)
            audit['checked_history_calls'] += 1
        return result
    gen_util.gen_mdm_motion = checked_generate
    args.kind='stairs'; args.device='cuda:0'; args.batch_size=4; args.max_seconds=30.
    args.threads=4; args.score_frame_chunk=4; args.course_spec=None; args.course_sections=1
    args.variant=0; args.sampling_profile='official_data'; args.expected_terrain_sha256=args.terrain_sha256
    try:
        generation.run(args)
    finally:
        if args.output.exists():
            audit.update(wrapper_sha256=generation.pilot.sha256(Path(__file__)),
                course_sha256=generation.pilot.sha256(Path(__file__).with_name('build_parc_rugged_course.py')))
            (args.output/'prefix-contract.json').write_text(json.dumps(audit, indent=2))
    report = json.loads((args.output/'generation.json').read_text())
    if len(report['candidates']) != 4 or audit['checked_history_calls'] == 0:
        raise ValueError('Incomplete candidate/prefix inventory')
    report.update(prefix_contract=audit, course_wrapper='rugged32_v1', layout_index=args.layout,
        command_semantics='unit direction issued before diffusion; NOT metric velocity',
        source_future_pose_condition=False, training_approved=False)
    (args.output/'generation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','checkpoint','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--layout',type=int,choices=range(8),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--terrain-sha256',required=True)
    main(parser.parse_args())
