"""Matched move/stop PARC calls from factual pre-stop histories; no training."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.helpers));sys.path.insert(0,str(args.repo))
    sys.modules.setdefault('parc',importlib.import_module('PARC'))
    import run_parc_local_terrain_pilot as pilot
    from generate_parc_stair64 import preserve_flags
    from parc.motion_generator import gen_util
    from parc.motion_generator.diffusion_util import MDMKeyType
    from parc.util import file_io_helper,terrain_util
    from parc.util.motion_util import MotionFrames
    torch.set_num_threads(4)
    if pilot.sha256(args.checkpoint)!='a03a9d4cc7b3a43b217d67116429d241c92e31e7de93e02849832b0371860e2e':
        raise ValueError('Expert identity mismatch')
    model=pilot.load_official_model(args.repo,args.checkpoint,'cuda:0','ema_model')
    model.predict_x0=preserve_flags(model.predict_x0,(MDMKeyType.PREV_STATE_NOISE_IND_KEY,MDMKeyType.PREV_STATE_FLAG_KEY))
    captured=[];embed=model._denoise_model.embed_conds
    def capture(conds,*a,**kw):
        if len(captured)<2:
            captured.append(dict(target=conds[MDMKeyType.TARGET_KEY].detach().cpu().tolist(),
                enabled=conds[MDMKeyType.TARGET_FLAG_KEY].detach().cpu().tolist()))
        return embed(conds,*a,**kw)
    model._denoise_model.embed_conds=capture
    rows=[]
    for layout in ((3,6) if args.rollout_stop else (0,6)):
        folder=args.batch/f'layout_{layout:02d}'/'generation'
        report=json.loads((folder/'generation.json').read_text())
        eligible=[r for r in report['candidates'] if r['stop_response']['stops']]
        if not eligible:raise ValueError('No factual pre-stop history in this layout')
        wanted=0 if args.rollout_stop else 7
        row=next((r for r in eligible if r['candidate_id']==wanted),eligible[0])
        source=file_io_helper.load_ms_file(row['motion'],'cuda:0')
        stop=row['stop_response']['stops'][0]['start_frame']
        m=source.motion_data;td=source.terrain_data
        previous=MotionFrames(root_pos=m.root_pos[None,stop-2:stop].repeat(4,1,1),
            root_rot=m.root_rot[None,stop-2:stop].repeat(4,1,1),
            joint_rot=m.joint_rot[None,stop-2:stop].repeat(4,1,1,1),
            contacts=m.body_contacts[None,stop-2:stop].repeat(4,1,1))
        terrain=terrain_util.SubTerrain('matched',x_dim=td.hf.shape[0],y_dim=td.hf.shape[1],
            dx=float(td.dx),dy=float(td.dx),min_x=float(td.min_point[0]),min_y=float(td.min_point[1]),device='cuda:0')
        terrain.hf.copy_(td.hf);terrain.hf_maxmin.copy_(td.hf_maxmin)
        with np.load(row['intent'],allow_pickle=False) as a:
            direction=torch.from_numpy(a['command_world_xy'][stop-1].copy()).float().cuda()
        if args.rollout_stop:
            previous=previous.get_idx(0).unsqueeze(0)
            history=previous;parts=[];stop_blocks=0;settled=False
            settings=gen_util.MDMGenSettings();settings.ddim_stride=5;settings.cfg_scale=.65
            torch.manual_seed(20261013)
            for block in range(26):
                with torch.no_grad():
                    generated=gen_util.gen_mdm_motion(target_world_pos=history.root_pos[:,-1].clone(),prev_frames=history,
                        terrain=terrain,mdm_model=model,char_model=model._kin_char_model,mdm_settings=settings)
                torch.testing.assert_close(generated.root_pos[:,:2],history.root_pos,atol=3e-5,rtol=0)
                parts.append(generated.get_slice(slice(2,9)))
                history=generated.get_slice(slice(7,9))
                stop_blocks+=1
                roots=torch.cat([p.root_pos for p in parts],1)
                if roots.shape[1]>=30:
                    speed=torch.diff(roots[0,-15:,:2],dim=0).norm(dim=-1)*30
                    settled=bool(torch.quantile(speed,.9)<=.15 and (roots[0,-1,:2]-roots[0,-15,:2]).norm()<=.08)
                    if settled:break
            stopped_root=history.root_pos[0,-1].clone()
            for block in range(4):
                target=history.root_pos[:,-1].clone();target[:,:2]+=direction*3.
                with torch.no_grad():
                    generated=gen_util.gen_mdm_motion(target_world_pos=target,prev_frames=history,terrain=terrain,
                        mdm_model=model,char_model=model._kin_char_model,mdm_settings=settings)
                torch.testing.assert_close(generated.root_pos[:,:2],history.root_pos,atol=3e-5,rtol=0)
                parts.append(generated.get_slice(slice(2,9)));history=generated.get_slice(slice(7,9))
            course=json.loads((folder/'course/course_manifest.json').read_text())
            phase=next(p for p in course['route_segments'] if p['stage']=='ascent')
            a=torch.tensor(phase['start_xyz_m'][:2],device='cuda:0');b=torch.tensor(phase['end_xyz_m'][:2],device='cuda:0')
            axis=(b-a)/(b-a).norm();progress=float((stopped_root[:2]-a) @ axis)
            value=dict(layout=layout,condition='held_stop_until_settled_or6s_then_resume',settled=settled,
                stop_seconds=stop_blocks*7/30,stop_displacement_m=float((stopped_root[:2]-previous.root_pos[0,-1,:2]).norm()),
                stopped_inside_flight=.15<progress<float((b-a).norm())-.15,
                stop_tail_speed_p90_mps=float(torch.quantile(speed,.9)),
                resume_progress_m=float((history.root_pos[0,-1,:2]-stopped_root[:2]) @ axis),updates=0)
            rows.append(value)
            np.savez_compressed(args.output/f'layout{layout}-held-stop.npz',
                **{key:torch.cat([getattr(p,key) for p in parts],1).cpu().numpy() for key in ('root_pos','root_rot','joint_rot','contacts')})
            (args.output/'progress.json').write_text(json.dumps(dict(rows=rows),indent=2))
            print(json.dumps(value),flush=True)
            continue
        for label,stopping,cfg in [('move',False,True),('stop',True,True),('stop_without_cfg',True,False)]:
            captured.clear()
            target=previous.root_pos[:,-1].clone()
            if not stopping:target[:,:2]+=direction*3.
            settings=gen_util.MDMGenSettings();settings.ddim_stride=5;settings.cfg_scale=.65;settings.use_cfg=cfg
            torch.manual_seed(20261013)
            begin=time.perf_counter()
            with torch.no_grad():
                generated=gen_util.gen_mdm_motion(target_world_pos=target,prev_frames=previous,terrain=terrain,
                    mdm_model=model,char_model=model._kin_char_model,mdm_settings=settings)
            torch.cuda.synchronize()
            norms=[float(torch.tensor(v['target']).norm(dim=-1).max()) for v in captured]
            if not captured or not all(all(v['enabled']) for v in captured) or (stopping and any(norm!=0 for norm in norms)):
                raise ValueError('Stop condition was not actually supplied to the network')
            delta=generated.root_pos[:,-1,:2]-previous.root_pos[:,-1,:2]
            speed=(generated.root_pos[:,-1,:2]-generated.root_pos[:,-2,:2]).norm(dim=-1)*30
            value=dict(layout=layout,condition=label,future_frames=generated.root_pos.shape[1]-2,
                input_speed_mps=float((previous.root_pos[0,-1,:2]-previous.root_pos[0,-2,:2]).norm()*30),
                displacements_m=delta.norm(dim=-1).cpu().tolist(),end_speeds_mps=speed.cpu().tolist(),
                captured_conditions=captured.copy(),seconds=time.perf_counter()-begin,updates=0)
            rows.append(value)
            np.savez_compressed(args.output/f'layout{layout}-{label}.npz',root_pos=generated.root_pos.cpu().numpy(),
                root_rot=generated.root_rot.cpu().numpy(),joint_rot=generated.joint_rot.cpu().numpy(),contacts=generated.contacts.cpu().numpy())
            (args.output/'progress.json').write_text(json.dumps(dict(rows=rows),indent=2))
            print(json.dumps(value),flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(rows=rows,updates=0,postprocessing=None,
        source_target_semantics='PARC training maps small future planar displacement and small heading change to zero direction; this does not certify braking from a moving history'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('helpers','repo','checkpoint','batch','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--rollout-stop',action='store_true')
    main(parser.parse_args())
