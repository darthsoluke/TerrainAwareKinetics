"""Versioned regular-stair generation with preserved native history flags."""

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch

import generate_parc_long_stairs as generation
import build_parc_stair_course as course_base
from build_parc_stair64_course import build_stair64_course
from parc_stair_command_schedule import StairCommandSchedule,stop_response_report


def preserve_flags(predict, keys):
    def wrapped(x, conditions, *args, **kwargs):
        flags={key:conditions[key].clone() for key in keys}
        try:
            return predict(x,conditions,*args,**kwargs)
        finally:
            for key,value in flags.items():
                conditions[key].copy_(value)
    return wrapped


def main(args):
    if generation.pilot.sha256(args.checkpoint)!='a03a9d4cc7b3a43b217d67116429d241c92e31e7de93e02849832b0371860e2e':
        raise ValueError('Unexpected PARC expert checkpoint')
    course_base.PARC_REPO=args.repo
    sys.path.insert(0,str(args.repo))
    sys.modules.setdefault('parc',importlib.import_module('PARC'))
    from parc.motion_generator import gen_util
    from parc.motion_generator.diffusion_util import MDMKeyType
    audit=dict(calls=0,checked_history_calls=0,max_prefix_error=0.,flag_policy='restore after predict_x0; vendor unchanged')
    native_load=generation.pilot.load_official_model
    def load(*a,**kw):
        model=native_load(*a,**kw)
        keys=(MDMKeyType.PREV_STATE_NOISE_IND_KEY,MDMKeyType.PREV_STATE_FLAG_KEY)
        model.predict_x0=preserve_flags(model.predict_x0,keys)
        return model
    generation.pilot.load_official_model=load
    generation.build_course=lambda device:build_stair64_course(args.layout,device)
    native_generate=gen_util.gen_mdm_motion
    schedule=None
    def command_transform(kw):
        nonlocal schedule
        if schedule is None:
            _,nodes,course=build_stair64_course(args.layout,kw['prev_frames'].root_pos.device)
            model=kw['mdm_model']
            stride=model._seq_len-1-5-model._num_prev_states
            schedule=StairCommandSchedule(nodes,course,fps=model._sequence_fps,stride=stride)
        return schedule.transform(kw)
    args.command_transform=command_transform
    def checked_generate(*a,**kw):
        if a:raise ValueError('Explicit arguments required for provenance')
        result=native_generate(**kw)
        audit['calls']+=1
        settings=kw['mdm_settings']
        if bool(settings.use_prev_state) and bool(settings.prev_state_ind_key):
            before=kw['prev_frames'];n=kw['mdm_model']._num_prev_states
            for key in ('root_pos','root_rot','joint_rot','contacts'):
                x,y=getattr(result,key)[:,:n],getattr(before,key)[:,:n]
                error=torch.minimum((x-y).abs().amax(-1),(x+y).abs().amax(-1)).max() if key.endswith('rot') else (x-y).abs().max()
                audit['max_prefix_error']=max(audit['max_prefix_error'],float(error))
                if error>3e-5:raise ValueError('Native PARC continuation changed the supplied history: '+key)
            audit['checked_history_calls']+=1
        return result
    gen_util.gen_mdm_motion=checked_generate
    args.kind='stairs';args.device='cuda:0';args.batch_size=8;args.max_seconds=30.
    args.threads=4;args.score_frame_chunk=4;args.course_spec=None;args.course_sections=1
    args.variant=0;args.sampling_profile='official_data';args.expected_terrain_sha256=None
    try:
        generation.run(args)
    finally:
        if args.output.exists():
            audit.update(wrapper_sha256=generation.pilot.sha256(Path(__file__)),
                builder_sha256=generation.pilot.sha256(Path(__file__).with_name('build_parc_stair64_course.py')))
            (args.output/'prefix-contract.json').write_text(json.dumps(audit,indent=2))
            if schedule is not None:
                (args.output/'command-schedule.json').write_text(json.dumps(dict(calls=schedule.calls,
                    command_semantics='unit movement direction or exact zero request; no metric speed control',
                    initialization='eight distinct flat-ground seeds, changed before first generation only'),indent=2))
    report=json.loads((args.output/'generation.json').read_text())
    if len(report['candidates'])!=8:
        raise ValueError('Expected all eight candidates, including failures')
    report.update(layout_index=args.layout,split='validation' if args.layout==6 else 'test' if args.layout==7 else 'train',
        prefix_contract=audit,course_wrapper='regular_stair64_v1',quality_approved=False)
    import numpy as np
    from parc.util import file_io_helper
    for row in report['candidates']:
        stored=file_io_helper.load_ms_file(row['motion'],'cpu')
        with np.load(row['intent'],allow_pickle=False) as archive:
            commands=archive['command_world_xy'].copy()
        row['stop_response']=stop_response_report(stored.motion_data,commands,None,schedule.nodes.cpu(),schedule.course,
            float(stored.motion_data.fps))
        row['stop_schedule_recorded']=True
    report['kinematic_stop_pass_candidates']=sum(r['stop_response']['kinematic_stops_pass'] for r in report['candidates'])
    report['stop_screen_pending']=False
    (args.output/'generation.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--layout',type=int,choices=range(8),required=True)
    parser.add_argument('--seed',type=int,required=True)
    main(parser.parse_args())
