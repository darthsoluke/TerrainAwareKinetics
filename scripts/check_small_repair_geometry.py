"""Check source intersections against exact union-of-air-box geometry."""

import argparse
import json
from pathlib import Path

import torch

from terrain_small_repair import initialize
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.terrain_response_loss import heightfield_gap,terrain_clearance


@torch.no_grad()
def main(args):
    model,_=initialize(args.checkpoint);rep=model.inference.global_motion_rep
    geometry=JointGeometryObjective(rep,'cuda:0')
    cases=torch.load(args.cases/'cases.pt',weights_only=False);rows=[]
    for case in cases:
        if case['split']!='train' or case['name'] in ('idle','walk'):continue
        source=case['world'][None].to('cuda:0');scene=case['scene']
        volume=geometry.volume(scene,source,case['row']['source_id'])
        vertices=[]
        for part in source[:,4:108].split(8,dim=1):
            pose=rep.inverse(part,False,return_all=True,joint_positions_from='global_rot_data')
            vertices.append(geometry.skin.skin(pose['local_joint_rots'][0],part[0,:,:3,None].transpose(1,2).expand(-1,30,-1),rot_is_global=False))
        points=torch.cat(vertices)
        gap,known=heightfield_gap(points,scene)
        dist,valid=volume.query(points)
        selected=torch.topk((-dist).reshape(-1),32).indices
        probe=points.reshape(-1,3)[selected]
        crop={'heightfield_y_m':volume.heights,'heightfield_valid':volume.known,
            'min_point_world_xyz_m':volume.origin,'dx_m':volume.dx}
        exact,known_exact=terrain_clearance(probe,crop)
        skin=geometry.skin
        dominant=skin.lbs_indices.gather(1,skin.lbs_weights.argmax(1)[:,None]).squeeze(1)
        parts=[skin.skeleton_skin.bone_order_names[i] for i in dominant[selected%len(skin.bind_vertices)].tolist()]
        row={'name':case['name'],'source_id':case['row']['source_id'],
            'vertex_fraction_inside_vertical_2cm':float(((gap<-.02)&known).float().mean()),
            'sdf_p99_penetration_m':float(torch.quantile((-dist[valid]).clamp_min(0),.99)),
            'sdf_max_penetration_m':float((-dist[valid]).clamp_min(0).max()),
            'worst32_exact_penetration_m':(-exact).clamp_min(0).tolist(),
            'worst32_approx_penetration_m':(-dist.reshape(-1)[selected]).clamp_min(0).tolist(),
            'worst32_known':known_exact.tolist()}
        row['worst32_dominant_bones']=parts
        rows.append(row);print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
