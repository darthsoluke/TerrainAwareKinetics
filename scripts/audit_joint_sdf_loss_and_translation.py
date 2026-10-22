"""Read-only objective gradients and terrain-fixed lateral augmentation audit."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.joint_sdf_training import ExactPenetrationBatch
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap
from motionbricks.data.terrain_control import SoftCommandState
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256


def main(args):
    torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    case=torch.load(args.cases,map_location='cpu',weights_only=False)[0];rows=[];translations=[]
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    source=Path(saved['contract']['source'])
    if sha256(source)!=saved['contract']['source_bundle_sha256']:raise ValueError('Source mismatch')
    inference,bundle=load_terrain_bundle(source,'cuda:0');model=JointSDFController(inference,bundle)
    model.load_state_dict(saved['model_state'],strict=True);model.freeze_base();model.eval()
    rep=inference.global_motion_rep;geometry=JointGeometryObjective(rep,'cuda:0')
    world=case['world'][None].to('cuda:0');scene=case['scene'];field=HeightfieldPrismSDF(scene)
    with torch.no_grad():
        _,points=geometry.geometry(world)
        baseline_gap,baseline_known=heightfield_gap(points,scene)
        baseline_d,_,baseline_valid=field.query(points)
        for displacement in (-.3,-.2,-.1,.1,.2,.3):
            # This real staircase is extruded across world Z. Do not assume
            # arbitrary motion/terrain pairs share this symmetry.
            translated=world.clone();translated[:,:,2]+=displacement
            _,moved=geometry.geometry(translated);gap,known=heightfield_gap(moved,scene)
            distance,_,valid=field.query(moved)
            both=valid & baseline_valid
            changed=(gap-baseline_gap)[known & baseline_known].abs()
            differences=(distance.clamp_max(0)-baseline_d.clamp_max(0))[both].abs()
            row={'lateral_world_z_m':displacement,'all_occupancy_known':bool(known.all()),
                'surface_height_max_change_m':float(changed.max()),
                'certified_penetration_max_change_m':float(differences.max()) if both.any() else None,
                'sdf_known_fraction':float(valid.float().mean())}
            row['reference_correspondence_preserved']=bool(known.all() and baseline_known.all()
                and row['surface_height_max_change_m']<=1e-5 and both.any() and row['certified_penetration_max_change_m']<=1e-5)
            translations.append(row)
    parameters=[p for p in model.parameters() if p.requires_grad]
    for start in (24,40,48):
        window=world[:,start:start+33];history=window[:,:4]
        task=SoftCommandState().intent(history,case['command_velocity'],case['command_heading'],8)
        result=model(history,task,[scene],source=window,straight_through=True)
        native=model.native_losses(result,window)
        with torch.no_grad():joints,points=geometry.geometry(window)
        exact=ExactPenetrationBatch([scene],window.device,[field])
        physical=geometry.prepared_losses(result['world'],history,
            {'joints':joints,'points':points,'heights':exact.heights,'volumes':exact},offset=0,diagnostics=False)
        losses={'native':native['native_total'],**{key:value*physical[key] for key,value in saved['coefficients'].items()}}
        def gradient(value):
            values=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
            return torch.cat([v.detach().flatten() if v is not None else torch.zeros_like(p).flatten() for v,p in zip(values,parameters)])
        reference=gradient(losses['native']);norm=reference.norm()
        for name,value in losses.items():
            grad=reference if name=='native' else gradient(value)
            row={'start_frame':start,'term':name,'weighted_loss':float(value.detach()),
                'gradient_norm':float(grad.norm()),'gradient_ratio_to_native':float(grad.norm()/norm.clamp_min(1e-12)),
                'cosine_with_native':float(grad.dot(reference)/(grad.norm()*norm).clamp_min(1e-12))}
            rows.append(row)
    report={'checkpoint_sha256':sha256(args.checkpoint),'source_id':case['row']['source_id'],
        'translations':translations,'gradients':rows,'intervention':'none; no training or runtime output changes',
        'quality_approved':False}
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
