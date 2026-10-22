"""Run the author's terrain-patch sampler with an explicit augmentation budget."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


def main(args):
    if args.sparsity < 15:
        raise ValueError('Patch budget cannot exceed the released sampling density')
    source = args.source/'official_source/generate_patches.py'
    text = source.read_text(encoding='utf-8-sig')
    tree = ast.parse(text)
    changed = removed = 0
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'sparsity' for target in node.targets):
            node.value = ast.Constant(args.sparsity)
            changed += 1
        elif isinstance(node, ast.Import) and all(alias.name == 'scipy.misc' for alias in node.names):
            tree.body[index] = ast.Pass()
            removed += 1
    if changed != 1 or removed != 1:
        raise ValueError('Official patch script layout changed')
    destination = args.source/'patches.npz'
    if destination.exists():
        raise ValueError('Do not overwrite an existing patch bank')
    os.chdir(args.source)
    namespace = {'__name__': '__main__', '__file__': str(source)}
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    import numpy as np
    with np.load(destination, allow_pickle=False) as bank:
        if set(bank.files) != {'X', 'C'} or bank['X'].shape[1:] != (128, 128) or not np.isfinite(bank['X']).all():
            raise ValueError('Invalid generated PFNN patch bank')
        count = len(bank['X'])
    receipt = dict(author_script_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        patches=count, seed=2, sparsity=args.sparsity, author_sparsity=15,
        modifications=['explicit lower augmentation density', 'remove unused removed scipy.misc import'],
        motion_recordings_subsampled=False, output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (args.source/'patches.receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument('--sparsity', type=int, default=75)
    main(parser.parse_args())
