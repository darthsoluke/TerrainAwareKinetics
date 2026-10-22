"""Inspect recorded live motion using official SOMA skinning, without correction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.soma_skin_reference import load_official_skinner
from motionbricks.data.bones_seed import official_sole_surfaces
from motionbricks.data.sole_region_contact import sole_region_indices
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


@torch.inference_mode()
def analyze(directory, scene_path, selected=None):
    torch.set_num_threads(4)
    skeleton=SOMASkeleton30()
    skin=load_official_skinner(skeleton)
    surfaces,*_=official_sole_surfaces(skeleton)
    left=sole_region_indices(surfaces,'Left');right=sole_region_indices(surfaces,'Right')
    soles=np.concatenate((left,right))
    with np.load(scene_path) as data:
        hf=data['heightfield_y_m'];origin=data['min_point_world_xyz_m'];dx=float(data['dx_m'])
    reports=[]
    for name in selected or ('flat_default','stairs_default','stairs_long'):
        with np.load(directory/(name+'.npz')) as data:
            frames=data['frames'];joints=data['joints']
        rig=torch.from_numpy(frames[:,3:].reshape(-1,77,4,4))
        body=torch.cat([skin.lbs(chunk) for chunk in rig.split(8)]).numpy()
        vertices=body[:,soles]
        body_xz=np.rint((body[:,:,[0,2]]-origin[[0,2]])/dx).astype(int)
        body_gap=body[...,1]-hf[body_xz[...,1],body_xz[...,0]]
        inside_fraction=(body_gap<-.02).mean(axis=1)
        xz=np.rint((vertices[:,:,[0,2]]-origin[[0,2]])/dx).astype(int)
        if (xz<0).any() or (xz[...,0]>=hf.shape[1]).any() or (xz[...,1]>=hf.shape[0]).any():
            raise ValueError('Unknown terrain cannot be scored as flat')
        gap=vertices[...,1]-hf[xz[...,1],xz[...,0]]
        penetration=np.maximum(-gap,0)
        row={'case':name,'official_sole_vertex_count':len(soles),
            'vertical_penetration_p99_m':float(np.quantile(penetration,.99)),
            'vertical_penetration_max_m':float(penetration.max()),
            'frames_with_sole_below_surface_by_2cm':int((penetration.max(axis=1)>.02).sum()),
            'frames':len(frames)}
        row['max_body_vertex_fraction_inside_terrain_by_vertical_2cm']=float(inside_fraction.max())
        row['frames_over_5percent_body_inside_terrain']=int((inside_fraction>.05).sum())
        row['first_over_5percent_body_inside_s']=float(np.flatnonzero(inside_fraction>.05)[0]/30) if (inside_fraction>.05).any() else None
        bad=np.flatnonzero(penetration.max(axis=1)>.02)
        row['first_penetration_over_2cm_s']=float(bad[0]/30) if len(bad) else None
        recording=json.loads((directory/(name+'.json')).read_text())
        row['first_observed_raised_terrain_s']=None
        for i,plan in enumerate(recording['plans']):
            debug=plan['debug'];root=np.asarray(debug['observation_root_world_xyz_m'])
            cell=np.rint((root[[0,2]]-origin[[0,2]])/dx).astype(int)
            ground=hf[cell[1],cell[0]]
            far=next(patch for patch in debug['terrain_patches'] if patch['name']=='intent_far')
            points=np.asarray(far['world_xyz_m']);valid=np.asarray(far['valid'],dtype=bool)
            if np.any(points[valid,1]>ground+.05):
                row['first_observed_raised_terrain_s']=i*8/30
                row['observed_relief_at_that_time_m']=float(points[valid,1].max()-ground)
                break
        if name!='flat_default' and (directory/'flat_default.npz').exists():
            flat=np.load(directory/'flat_default.npz')
            relative=joints-frames[:,:3,None].transpose(0,2,1)
            flat_relative=flat['joints']-flat['frames'][:,:3,None].transpose(0,2,1)
            row['relative_pose_difference_from_flat_m']=float(np.linalg.norm(relative-flat_relative,axis=-1).mean())
        np.savez_compressed(directory/(name+'-sole.npz'),vertices=vertices,gap=gap)
        reports.append(row)
    (directory/'skin_report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    if selected:
        print(json.dumps(reports,indent=2))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),layout='constrained')
    for name,label,color in [('flat_default','Flat, default','#278678'),
                             ('stairs_default','16 cm stairs, default','#c44940'),
                             ('stairs_long','16 cm stairs, long horizon','#74549c')]:
        with np.load(directory/(name+'-sole.npz')) as data:
            gap=data['gap']
        axes[0].plot(np.arange(len(gap))/30,np.minimum(gap.min(axis=1),0)*100,label=label,color=color,lw=1.3)
    axes[0].set(xlabel='Generated time (s)',ylabel='Lowest sole point below surface (cm)',
        title='Longer prediction does not resolve penetration')
    axes[0].axhline(-2,color='#606060',ls=':',lw=1)
    axes[0].legend(fontsize=8,loc='lower left');axes[0].grid(alpha=.2)
    data=np.load(directory/'stairs_default.npz');joints=data['joints'];frames=data['frames']
    gaps=np.load(directory/'stairs_default-sole.npz')['gap']
    chosen=int(np.argmin(gaps.min(axis=1)))
    xyz=joints[chosen];center=frames[chosen,:3]
    z0=max(0,int((center[2]-2-origin[2])/dx));z1=min(hf.shape[0],int((center[2]+2-origin[2])/dx)+1)
    x=int(np.rint((center[0]-origin[0])/dx));z=origin[2]+np.arange(z0,z1)*dx
    axes[1].step(z-center[2],hf[z0:z1,x],where='mid',color='#303d42',label='Terrain surface')
    parents=skeleton.joint_parents.tolist()
    for i,parent in enumerate(parents):
        if parent>=0 and parent!=i:
            axes[1].plot(xyz[[parent,i],2]-center[2],xyz[[parent,i],1],color='#397fa1',lw=2)
    vertices=np.load(directory/'stairs_default-sole.npz')['vertices'][chosen]
    axes[1].scatter(vertices[:,2]-center[2],vertices[:,1],s=7,color='#c44940',label='Official skinned soles')
    axes[1].set(xlabel='Forward distance from pelvis (m)',ylabel='World height (m)',
        title=f'Actual generated pose at {chosen/30:.2f} s')
    axes[1].set_aspect('equal',adjustable='datalim');axes[1].grid(alpha=.2);axes[1].legend(fontsize=8)
    fig.savefig(directory/'terrain_response.png',dpi=150)
    plt.close(fig)
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--cases',nargs='+')
    args=parser.parse_args()
    analyze(args.directory,args.scene,args.cases)
