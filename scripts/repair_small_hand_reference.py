"""Offline SOMA wrist-orientation diagnostic; never applied to model output."""

import argparse
import json
from pathlib import Path

import torch

from terrain_small_repair import initialize,source_screen
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_exact_interior import ExactTerrainInterior
from motionbricks.data.sole_geometry_losses import region_geometry_from_motion
from motionbricks.motionlib.core.utils.rotations import matrix_to_cont6d,cont6d_to_matrix
from probe_terrain_overfit import save_motion


def main(args):
    if args.manifest_only:
        report=json.loads((args.output/'report.json').read_text())
        if not all(row['accepted'] for row in report):raise ValueError('Hand surface screening failed')
        contract=json.loads((args.cases/'cases.json').read_text())
        contract['offline_hand_surface_retarget']=report
        contract['source_motion_unchanged']=False
        contract['contact_force_and_grasp_validated']=False
        (args.output/'cases.json').write_text(json.dumps(contract,indent=2))
        return
    args.output.mkdir(parents=True,exist_ok=False)
    model,_=initialize(args.checkpoint);model.requires_grad_(False)
    rep=model.inference.global_motion_rep;rig=rep.skeleton
    geometry=JointGeometryObjective(rep,'cuda:0');skin=geometry.skin
    cases=torch.load(args.cases/'cases.pt',weights_only=False);report=[];changed=[]
    names=rig.bone_order_names;hands=[names.index(side+'Hand') for side in ('Left','Right')]
    dominant=skin.lbs_indices.gather(1,skin.lbs_weights.argmax(1)[:,None]).squeeze(1)
    hand_bones=[i for i,n in enumerate(skin.skeleton_skin.bone_order_names) if 'Hand' in n]
    vertices=torch.nonzero(torch.isin(dominant,dominant.new_tensor(hand_bones))).flatten()
    for case in cases:
        if case['name'] not in ('climbing_up','climbing_down') or case['split']!='train':
            changed.append(case);continue
        world=case['world'][None].to('cuda:0')
        with torch.no_grad():
            original=rep.inverse(world,False,return_all=True,joint_positions_from='global_rot_data')
            local=original['local_joint_rots'][0].detach();roots=world[0,:,:3].detach()
        value=torch.nn.Parameter(matrix_to_cont6d(local[:,hands]).clone())
        initial=value.detach().clone();optimizer=torch.optim.Adam([value],lr=.01)
        query=ExactTerrainInterior(case['scene'])
        for step in range(200):
            rotations=local.clone();rotations[:,hands]=cont6d_to_matrix(value)
            y,xz=region_geometry_from_motion(rotations,roots,skin,vertices)
            points=torch.stack((xz[...,0],y,xz[...,1]),-1)
            distances,known=query.query(points)
            if not known.all():raise ValueError('Source hand left measured terrain')
            penetration=(-distances).clamp_min(0)
            correction=value-initial
            # Source hand spheres have no observed finger orientation. Fit
            # surface feasibility while retaining the original wrist path,
            # minimal orientation change and temporal coherence of that change.
            loss=penetration.square().sum()/len(roots)+.001*correction.square().mean()+.01*(correction[1:]-correction[:-1]).square().mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if step%40==0:print(json.dumps({'name':case['name'],'step':step,'max_penetration_m':float(penetration.max())}),flush=True)
        with torch.no_grad():
            rotations=local.clone();rotations[:,hands]=cont6d_to_matrix(value)
            global_rots,joints,_=rig.fk(rotations[None],roots[None])
            motion={'posed_joints':joints,'global_joint_rots':global_rots,'local_joint_rots':rotations[None],
                'translation':roots[None],'foot_contacts':world[...,rep.indices['foot_contacts']]}
            canonical,origin=rep(motion,to_normalize=False,return_init_heading_info=True,
                lengths=torch.tensor([len(roots)],device=roots.device))
            quat=origin['init_heading_quat'].reshape(1,4);yaw=2*torch.atan2(quat[:,2],quat[:,0])
            corrected=rep.change_first_heading(canonical,yaw,False,False)
            corrected[...,rep.indices['global_root_pos_2d']]+=roots[:1,[0,2]][None]
            check=rep.inverse(corrected,False,return_all=True,joint_positions_from='global_rot_data')
            torch.testing.assert_close(check['posed_joints'],joints,atol=2e-5,rtol=0)
            torch.testing.assert_close(corrected[...,:3],world[...,:3],atol=2e-5,rtol=0)
            untouched=[i for i in range(30) if i not in hands]
            torch.testing.assert_close(rotations[:,untouched],local[:,untouched],atol=0,rtol=0)
            fixed=dict(case,world=corrected[0].cpu(),offline_hand_surface_retarget=True)
            assessment=source_screen(fixed,geometry)
            record={'name':case['name'],'source_id':case['row']['source_id'],**assessment,
                'root_and_non_wrist_rotations_unchanged':True,'runtime_correction':False,
                'contact_force_and_grasp_validated':False}
            save_motion(args.output/(case['name']+'.npz'),corrected,case['scene'],rep,record)
            save_motion(args.output/(case['name']+'-source.npz'),world,case['scene'],rep,dict(record,role='unchanged_reference'))
            changed.append(fixed);report.append(record);print(json.dumps(record),flush=True)
    torch.save(changed,args.output/'cases.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','cases','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--manifest-only',action='store_true')
    main(parser.parse_args())
