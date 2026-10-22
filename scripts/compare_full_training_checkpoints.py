"""Read-only comparison of full training continuation state."""

import argparse
import json

import torch


def main(a):
    first, second = [torch.load(path, map_location='cpu', mmap=True, weights_only=False) for path in (a.first,a.second)]
    changed = []
    for key, left in first['model_state'].items():
        right = second['model_state'][key]
        if not torch.equal(left,right):
            changed.append(dict(key=key,max=float((left-right).abs().max()),count=int((left!=right).sum())))
    print(json.dumps(dict(updates=[value['actual_updates'] for value in (first,second)],
        samples_equal=first['sample_digest']==second['sample_digest'],
        cpu_rng_equal=torch.equal(first['rng_state']['torch'],second['rng_state']['torch']),
        cuda_rng_equal=all(torch.equal(left,right) for left,right in zip(first['rng_state']['cuda'],second['rng_state']['cuda'])),
        changed_tensors=len(changed), largest_changes=sorted(changed,key=lambda row:row['max'],reverse=True)[:12]),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first')
    parser.add_argument('second')
    main(parser.parse_args())
