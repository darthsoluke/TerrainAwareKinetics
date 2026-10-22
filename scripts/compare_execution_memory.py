"""Compare saved memory-policy probes with the existing loss/gradient gates."""

import argparse
import json
import os
from pathlib import Path


def main(a):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from scripts.train_native_mb import sha256
    left_meta = json.loads((a.reference/'complete.json').read_text())
    right_meta = json.loads((a.candidate/'complete.json').read_text())
    if left_meta['checkpoint_sha256'] != right_meta['checkpoint_sha256']:
        raise ValueError('Checkpoint identities differ')
    reports = []
    for index in range(len(left_meta['rows'])):
        left = torch.load(a.reference/f'batch_{index}.pt',map_location='cpu',mmap=True,weights_only=False)
        right = torch.load(a.candidate/f'batch_{index}.pt',map_location='cpu',mmap=True,weights_only=False)
        errors, differences = [], []
        def compare(x,y,path=''):
            if torch.is_tensor(x):
                if torch.equal(x,y):return
                maximum = float((x-y).abs().max()) if x.is_floating_point() else None
                differences.append(dict(path=path,max_abs=maximum))
                if path.startswith('.gradients.'):
                    passed = torch.allclose(x,y,atol=5e-6,rtol=1e-3)
                elif path.startswith('.outputs.'):
                    passed = torch.allclose(x,y,atol=1e-6,rtol=1e-5)
                else:passed=False
                if not passed:errors.append(path)
            elif isinstance(x,dict):
                if x.keys()!=y.keys():raise ValueError('Tensor inventory differs')
                for key in x:compare(x[key],y[key],path+'.'+str(key))
            elif isinstance(x,(tuple,list)):
                if len(x)!=len(y):raise ValueError('Sequence length differs')
                for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'.'+str(i))
            elif x!=y:
                differences.append(dict(path=path,reference=x,candidate=y))
                if not (path.startswith('.losses.') and abs(x-y)<=1e-6+1e-5*abs(x)):
                    errors.append(path)
        compare(left,right)
        reports.append(dict(batch_index=index,passed=not errors,errors=errors,
            bitwise_equal=not differences,differences=differences))
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=all(row['passed'] for row in reports),rows=reports,
        actual_optimizer_updates=0,quality_approved=False,production_approved=False,
        reference_sha256=sha256(a.reference/'complete.json'),candidate_sha256=sha256(a.candidate/'complete.json'),
        checkpoint_sha256=left_meta['checkpoint_sha256'],tolerances_unchanged=True)
    (a.output/'complete.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
