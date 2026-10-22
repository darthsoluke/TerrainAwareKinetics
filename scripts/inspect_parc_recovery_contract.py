"""Read only the saved native generation timing metadata."""
import argparse
import json
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint')
args = parser.parse_args()
saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
keys = ('sequence_duration', 'sequence_fps', 'num_prev_states', 'seq_len',
        '_sequence_duration', '_sequence_fps', '_num_prev_states', '_seq_len', '_diffusion_timesteps')
result = {}
for section in ('cfg', 'extra'):
    result[section] = {key: (value.item() if torch.is_tensor(value) and value.numel() == 1 else value)
                       for key, value in saved.get(section, {}).items() if key in keys}
print(json.dumps(result), flush=True)
