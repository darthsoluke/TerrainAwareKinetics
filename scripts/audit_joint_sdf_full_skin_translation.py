"""Verify lateral augmentation against every official SOMA skin vertex."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.soma_skin_reference import load_official_skinner,move_skinner
from motionbricks.data.joint_sdf_training import verified_lateral_offsets

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cases',type=Path,required=True);parser.add_argument('--stats',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
torch.set_num_threads(4)
case=torch.load(args.cases,map_location='cpu',weights_only=False)[0]
rep=build_native_representation(args.stats)
skin=move_skinner(load_official_skinner(rep.skeleton),torch.device('cuda:0'));skin.skeleton_skin.float().to('cuda:0')
with torch.no_grad():
    motion=case['world'];pose=rep.inverse(motion[None],False,return_all=True,joint_positions_from='global_rot_data')
    rotations=pose['local_joint_rots'][0].cuda();root=motion[:,:3].cuda()
    vertices=[]
    for first in range(0,len(motion),16):
        part=root[first:first+16]
        vertices.append(skin.skin(rotations[first:first+16],part[:,None].expand(-1,30,-1),rot_is_global=False))
    points=torch.cat(vertices)
    accepted,rows=verified_lateral_offsets(points,case['scene'],[-.3,-.2,-.1,0.,.1,.2,.3])
report={'source_id':case['row']['source_id'],'frames':len(motion),'vertices_per_frame':len(skin.bind_vertices),
    'accepted_offsets_world_z_m':accepted,'rows':rows,'postprocessing':None}
args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
