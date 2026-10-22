"""Inspect replay differences without turning a failed comparison into a pass."""
import json
from pathlib import Path
import torch

root=Path.cwd().parent/'joint_sdf_20261012'
a=torch.load(root/'ddp_whole_v2/last.pt',map_location='cpu',weights_only=False)
b=torch.load(root/'ddp_split_v2/last.pt',map_location='cpu',weights_only=False)
errors=sorted([(k,float((v-b['model_state'][k]).abs().max())) for k,v in a['model_state'].items() if v.is_floating_point()],key=lambda x:-x[1])
print(json.dumps({'max_model_differences':errors[:12],
    'rng':[{k:torch.equal(x[k],y[k]) for k in ('torch','cuda')} for x,y in zip(a['rank_state'],b['rank_state'])],
    'counts_equal':[x['episodes']['counts']==y['episodes']['counts'] for x,y in zip(a['rank_state'],b['rank_state'])],
    'histories_max_difference':max(float((u['history']-v['history']).abs().max()) for x,y in zip(a['rank_state'],b['rank_state'])
        for u,v in zip(x['episodes']['slots'],y['episodes']['slots']))},indent=2))
