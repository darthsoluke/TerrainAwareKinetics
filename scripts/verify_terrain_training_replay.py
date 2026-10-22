"""Compare complete real-training checkpoints across a stop/resume boundary."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare(left,right,path,failures):
    if isinstance(left,torch.Tensor):
        equal=isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left,right)
    elif isinstance(left,np.ndarray):
        equal=isinstance(right,np.ndarray) and left.dtype==right.dtype and np.array_equal(left,right)
    elif isinstance(left,dict):
        equal=isinstance(right,dict) and left.keys()==right.keys()
        if equal:
            for key in left:compare(left[key],right[key],path+'/'+str(key),failures)
            return
    elif isinstance(left,(list,tuple)):
        equal=isinstance(right,type(left)) and len(left)==len(right)
        if equal:
            for index,(a,b) in enumerate(zip(left,right)):compare(a,b,path+'/'+str(index),failures)
            return
    else:equal=left==right
    if not equal:failures.append(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resumed',type=Path,required=True)
    parser.add_argument('--continuous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    a=torch.load(args.resumed,map_location='cpu',mmap=True,weights_only=False)
    b=torch.load(args.continuous,map_location='cpu',mmap=True,weights_only=False)
    failures=[];compare(a,b,'checkpoint',failures)
    report={'exact_match':not failures,'updates':a['step'],'world_size':a['contract']['world_size'],
        'device':a['contract']['device_type'],'differences':failures,
        'ranks':[{'data_draw_step':r['data_step'],'episode_positions':[e['position'] for e in r['episodes']]}
            for r in a['rank_rng']],
        'scope':'entire model, optimizer, contract, per-rank RNG, actual generated history, requests and counters',
        'quality_approved':False}
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
