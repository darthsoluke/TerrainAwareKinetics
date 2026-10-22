"""Read-only sole-material and gait audit; no generated motion is changed."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from scripts.audit_joint_sdf_stair_footfalls import true_runs
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(args):
    torch.set_num_threads(4); args.output.mkdir(parents=True,exist_ok=False)
    rep=build_native_representation(); geometry=JointGeometryObjective(rep,'cpu')
    surfaces,*_=official_sole_surfaces(rep.skeleton)
    left_count=len(sole_region_indices(surfaces,'Left'))
    axis=torch.tensor(args.direction,dtype=torch.float32);axis/=axis.norm()
    figure,axes=plt.subplots(len(args.recordings),3,figsize=(14,3.2*len(args.recordings)),squeeze=False,constrained_layout=True)
    rows={}; traces={}
    for plots,entry in zip(axes,args.recordings):
        label,filename=entry.split('=',1);path=Path(filename)
        with np.load(path,allow_pickle=False) as data:
            recorded=torch.from_numpy(data['world'].copy())
            scene={k:torch.from_numpy(data[k].copy()) for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
        if args.begin<1 or args.begin>=len(recorded):raise ValueError('Need a real previous played frame')
        world=recorded[None,args.begin-1:]
        pieces=[geometry.geometry(world[:,start:start+32]) for start in range(0,world.shape[1],32)]
        joints=torch.cat([p[0] for p in pieces],1)[0]
        soles=torch.cat([p[1] for p in pieces],1)[0,:,:geometry.sole_count]
        gap,known=heightfield_gap(soles,scene)
        near=known & (gap.abs()<=.02)
        speed=torch.diff(soles,dim=0).norm(dim=-1)*30
        xz_speed=torch.diff(soles[:,:,[0,2]],dim=0).norm(dim=-1)*30
        near_intervals=near[1:] & near[:-1]
        stationary=near_intervals & (speed<=.1)
        root_v=torch.diff(world[0,:,[0,2]],dim=0)*30
        forward=root_v@axis
        ids=[rep.skeleton.bone_order_names.index(side+'Foot') for side in ('Left','Right')]
        feet=joints[:,ids]
        relative=feet-world[0,:,:3][:,None]
        foreaft=relative[:,:,[0,2]]@axis
        separation=foreaft[1:,0]-foreaft[1:,1]
        contact=world[0,1:,rep.indices['foot_contacts']]>.5
        foot_rows={}; support=[]
        for side,subset,color in [('left',slice(0,left_count),'#2171a7'),('right',slice(left_count,geometry.sole_count),'#bf6a29')]:
            close=near_intervals[:,subset]
            stable=stationary[:,subset].sum(-1)>=3
            foot_rows[side]=dict(near_surface_interval_fraction=float((close.sum(-1)>=3).float().mean()),
                stationary_material_interval_fraction=float(stable.float().mean()),
                stationary_runs_min3frames=[dict(start_frame=args.begin+a,end_frame=args.begin+b-1,seconds=(b-a)/30)
                    for a,b in true_runs(stable.tolist(),minimum=3)],
                near_surface_material_speed_median=float(speed[:,subset][close].median()) if close.any() else None,
                near_surface_material_speed_p90=float(torch.quantile(speed[:,subset][close],.9)) if close.any() else None)
            support.append(stable)
            plots[2].plot(np.arange(len(stable))/30,stationary[:,subset].sum(-1).numpy(),color=color,label=side)
        both_no_support=~torch.stack(support,1).any(1)
        runs=[]
        for first,last in true_runs((separation.abs()<.1).tolist(),minimum=3):
            runs.append(dict(start_frame=args.begin+first,end_frame=args.begin+last-1,seconds=(last-first)/30,
                forward_m=float(forward[first:last].sum()/30)))
        temporal=[]
        for begin in range(0,len(forward),60):
            stop=min(begin+60,len(forward));mask=near_intervals[begin:stop]
            temporal.append(dict(seconds=[begin/30,stop/30],forward_m=float(forward[begin:stop].sum()/30),
                close_feet_fraction=float((separation[begin:stop].abs()<.1).float().mean()),
                no_stationary_material_fraction=float(both_no_support[begin:stop].float().mean()),
                near_surface_material_speed_p90=float(torch.quantile(speed[begin:stop][mask],.9)) if mask.any() else None))
        rows[label]=dict(recording_sha256=sha256(path),frames=len(recorded),assessed_frames=len(forward),begin=args.begin,
            forward_m=float(forward.sum()/30),close_feet_fraction=float((separation.abs()<.1).float().mean()),
            close_runs=runs,longest_close_run_seconds=max((r['seconds'] for r in runs),default=0),
            no_stationary_material_fraction=float(both_no_support.float().mean()),
            predicted_contact_fraction=float(contact.float().mean()),
            any_predicted_contact_frame_fraction=float(contact.any(1).float().mean()),
            feet=foot_rows,windows=temporal,
            proxy_scope='Near surface means abs sole gap<=2cm;stationary means same material point<=.1m/s with>=3points. Not measured force contact.')
        traces[label]=dict(forward_speed_mps=forward.tolist(),foreaft_separation_m=separation.tolist(),
            left_stationary_points=stationary[:,:left_count].sum(-1).tolist(),
            right_stationary_points=stationary[:,left_count:].sum(-1).tolist())
        t=np.arange(len(forward))/30
        plots[0].plot(t,forward.numpy(),color='#276c65');plots[0].axhline(1.6,color='#8c8c8c',ls='--',lw=.8)
        plots[1].plot(t,separation.numpy(),color='#754d9e');plots[1].axhspan(-.1,.1,color='#d8dce0',alpha=.6)
        for plot,title in zip(plots,('Root forward speed (m/s)','Left-right fore-aft gap (m)','Stationary sole material points')):
            plot.set_title(label+' | '+title,fontsize=10);plot.set_xlabel('Seconds after forward input');plot.grid(alpha=.2)
        plots[2].legend()
    coefficients=None
    if args.checkpoint:
        saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
        coefficients=saved['coefficients']
    native=None
    if args.bundle:
        bundle=torch.load(args.bundle,map_location='cpu',mmap=True,weights_only=False)
        config=bundle['tokenizer_config']['model']['args']
        native={k:config.get(k) for k in ('skate_contact_loss_coeff','joint_vel_loss_coeff','commit_loss_coeff')}
    write_json(args.output/'complete.json',dict(rows=rows,physical_coefficients=coefficients,native_coefficients=native,
        optimizer_updates=0,postprocessing=None,quality_approved=False))
    write_json(args.output/'series.json',traces)
    figure.savefig(args.output/'mechanics.png',dpi=160);plt.close(figure)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recordings',nargs='+',required=True);p.add_argument('--begin',type=int,default=480)
    p.add_argument('--direction',nargs=2,type=float,default=[0.,-1.])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--bundle',type=Path)
    main(p.parse_args())
