"""Offline Root/code factorial diagnosis of the trained causal controller."""

import argparse
import importlib.util
from pathlib import Path
import torch

from scripts.pilot_joint_sdf_contact_preserve import load_model, prepare_cases


@torch.no_grad()
def main(args):
    spec=importlib.util.spec_from_file_location('trained_gait_oracle',args.gait_script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model,_=load_model(saved)
    case=next(c for c in prepare_cases(args.cases,Path(saved['contract']['dataset']))
              if c['row']['source_id']=='generated_8030ae7a2a7ad640')
    case['command_velocity']=case['exact_constant_control'][0,:2]
    case['command_heading']=torch.atan2(case['exact_constant_control'][0,3],case['exact_constant_control'][0,2])
    case['source_start']=case['start_frame']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    module.review_case(model,case,args.output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','cases','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gait-script',type=Path,default=Path(__file__).with_name('audit_joint_sdf_gait_chain.py'))
    main(parser.parse_args())
