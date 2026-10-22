"""Real two-GPU uninterrupted versus interrupted optimizer/history replay."""
import json
from pathlib import Path
import subprocess
import sys
import torch

task=Path.cwd().parent;root=task/'joint_sdf_20261012'
base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m','scripts.train_joint_sdf',
    '--checkpoint',str(task/'control_sample_v8_live/weights-005709.pt'),'--batch','4','--deterministic']
for name,stop,resume in ([] if '--compare-only' in sys.argv else
    [('ddp_whole_v3',6,None),('ddp_split_v3',3,None),('ddp_split_v3',6,root/'ddp_split_v3/last.pt')]):
    command=base+['--output',str(root/name),'--stop-at',str(stop)]
    if resume is not None:command+=['--resume',str(resume)]
    subprocess.run(command,check=True)
suffix='' if '--compare-only' in sys.argv else '_v3'
a=torch.load(root/('ddp_whole'+suffix)/'last.pt',map_location='cpu',weights_only=False)
b=torch.load(root/('ddp_split'+suffix)/'last.pt',map_location='cpu',weights_only=False)
differences={}
def compare(x,y,path=''):
    if torch.is_tensor(x):
        if x.is_floating_point():
            differences[path]=float((x-y).abs().max()) if x.numel() else 0.
            torch.testing.assert_close(x,y,atol=2e-6,rtol=1e-6,msg=path)
        elif not torch.equal(x,y):raise ValueError('Tensor mismatch: '+path)
    elif isinstance(x,dict):
        if x.keys()!=y.keys():raise ValueError('Keys mismatch: '+path)
        for k in x:compare(x[k],y[k],path+'.'+str(k))
    elif isinstance(x,(tuple,list)):
        if len(x)!=len(y):raise ValueError('Length mismatch: '+path)
        for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'.'+str(i))
    elif hasattr(x,'shape'):
        import numpy as np
        if not np.array_equal(x,y):raise ValueError('Array mismatch: '+path)
    elif x!=y:raise ValueError('Value mismatch: '+path)
compare(a,b)
result={'passed':True,'updates':6,'split':[3,3],'world_size':2,'per_rank_batch':4,
    'comparison':'entire checkpoint; integer/RNG/cursors exact; CUDA floating tensors atol2e-6 rtol1e-6',
    'bitwise':all(value==0 for value in differences.values()),'max_tensor_difference':max(differences.values()),
    'largest_differences':sorted(differences.items(),key=lambda pair:-pair[1])[:8]}
(root/'ddp_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
