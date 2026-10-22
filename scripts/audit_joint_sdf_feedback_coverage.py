"""Replay the bounded pilot's source-window sampling without running a model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main(args):
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    rng = np.random.default_rng(2333)
    observed = np.zeros(len(cases[0]['world']), dtype=np.int64)
    supervised = np.zeros_like(observed)
    feedback = np.zeros_like(observed)
    depths = {}
    examples = 0
    for step in range(args.updates):
        ids = [0, 0, 1 + step % 2]
        items = [cases[i] for i in ids]
        depth = 1 if step < args.updates // 2 else min(
            13, min(1 + (len(item['world']) - 33) // 8 for item in items))
        starts = [int(rng.integers(0, len(item['world']) - (33 + (depth - 1) * 8) + 1))
                  for item in items]
        depths[str(depth)] = depths.get(str(depth), 0) + 1
        examples += len(items) * depth
        for index, start in zip(ids, starts):
            if index != 0:
                continue
            for segment in range(depth):
                offset = start + segment * 8
                observed[offset:offset + 4] += 1
                if segment:
                    feedback[offset:offset + 4] += 1
                # The actual geometry objective uses prediction[:, 4:frames-4].
                supervised[offset + 4:offset + 28] += 1

    stair = cases[0]
    frames = len(stair['world'])
    rows = []
    for block in range((frames - 4) // 8):
        begin = block * 8
        rows.append({'evaluation_block': block,
                     'history_frames': [begin, begin + 3],
                     'published_frames': [begin + 4, begin + 11],
                     'source_time_history_exposures': observed[begin:begin + 4].tolist(),
                     'generated_history_exposures': feedback[begin:begin + 4].tolist(),
                     'geometry_target_exposures': supervised[begin + 4:begin + 12].tolist()})

    def span(counts):
        nonzero = np.flatnonzero(counts)
        return [int(nonzero[0]), int(nonzero[-1])] if len(nonzero) else None

    result = {
        'source_id': stair['row']['source_id'], 'frames': frames,
        'stair_edge_frame': stair['edge_frame'], 'updates': args.updates,
        'primitive_examples': examples, 'update_depths': depths,
        'source_time_history_span': span(observed),
        'generated_history_span': span(feedback),
        'geometry_target_span': span(supervised), 'evaluation_blocks': rows,
        'interpretation': 'Source-clock coverage only. Generated poses can differ from source at the same '
                          'time. Missing source-time history does not prove the cause of failures. '
                          'Native Root/token objectives have different future masks from geometry.',
        'scope': 'pilot_joint_sdf.py 300-update trials only; not the completed main campaign',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=300)
    main(parser.parse_args())
