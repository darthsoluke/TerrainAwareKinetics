"""Bounded real-batch full-model comparison on our temporarily paused GPU."""

import argparse
from pathlib import Path
from types import SimpleNamespace

from scripts.check_dilated_conv_layout import main, save


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--paused-owner-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path('${PRIVATE_STORAGE}/motionbricks-20261010')
    run = base/'pfnn_mixed_20261015'
    args.output.mkdir(parents=True, exist_ok=False)
    options = SimpleNamespace(**vars(args), after_probe=None, batches=1, layout='batched_bias',
        bundle=base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt',
        initial=run/'full_training_v3/root/latest.pt',
        cache=[run/'cache_mixed_v1', run/'cache_pfnn_world_v2'])
    try:
        main(options)
    except Exception as error:
        save(args.output/'failure.json', dict(error=repr(error), actual_optimizer_updates=0))
        raise
