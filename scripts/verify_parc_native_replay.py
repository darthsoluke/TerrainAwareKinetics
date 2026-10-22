"""Verify saved native probe arrays without another generator call."""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
with np.load(args.directory / 'released_repeat.npz', allow_pickle=False) as repeat:
    with np.load(args.directory / 'preserved_condition_flags.npz', allow_pickle=False) as preserved:
        differences = {key: float(np.max(np.abs(repeat[key][:, 2:] - preserved[key][:, 2:])))
                       for key in ('root_pos', 'root_rot', 'joint_rot', 'contacts')}
report = {'future_difference_vs_unmodified_repeat': differences,
          'future_array_equal_to_repeat': all(value == 0 for value in differences.values()),
          'additional_generation_calls': 0, 'quality_approved': False}
with (args.directory / 'repeat_verification.json').open('x') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report), flush=True)
