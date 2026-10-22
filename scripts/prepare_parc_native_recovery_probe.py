"""Select one real PARC prefix for an isolated native continuation capability check."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.parc_soma_retarget import load_native_pair, official_parc_character
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cases = torch.load(args.cases, map_location='cpu', weights_only=False)
    case = next(case for case in cases if case['row']['source_id'] == 'generated_8030ae7a2a7ad640')
    row = case['row']
    path = Path(row['source_original_path'])
    expected = row['source_original_sha256']
    report = {'native_source_path': str(path), 'source_sha256_expected': expected,
              'source_id': row['source_id'], 'purpose': 'Original PARC history only, not SOMA policy recovery',
              'quality_approved': False}
    if not path.exists():
        write_json(args.output / 'unavailable.json', dict(report, reason='Original native source path does not exist'))
        raise ValueError('Missing original native source: ' + str(path))
    if sha256(path) != expected:
        raise ValueError('Native source identity changed')
    character = official_parc_character(args.parc)
    source = load_native_pair(path, character)
    contacts_path = args.dataset / row['source_contact_sidecar']
    intent_path = args.dataset / row['source_intent_sidecar']
    if sha256(contacts_path) != row['source_contact_sidecar_sha256'] or sha256(intent_path) != row['source_intent_sidecar_sha256']:
        raise ValueError('Source clock or intent sidecar changed')
    with np.load(contacts_path, allow_pickle=False) as archive:
        frame_indices = archive['source_frame_indices'].copy()
    with np.load(intent_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(frame_indices, archive['source_frame_indices'])
        commands = archive['command_world_xy'].copy()
    current = case['start_frame'] + 3 * 8 + 3
    native_end = int(frame_indices[current])
    if frame_indices[current] - frame_indices[current - 1] != 1 or native_end + 14 >= len(source['root_pos']):
        raise ValueError('Native prefix and future must have real contiguous frame support')
    if source['fps'] != 30:
        raise ValueError('Released generator/source FPS mismatch')
    start, end = native_end - 1, native_end + 15
    np.savez_compressed(args.output / 'input.npz',
                        **{key: source[key][start:end] for key in ('root_pos', 'root_rot', 'joint_rot', 'body_contacts')},
                        hf=source['hf'], min_point=source['min_point'], dx=source['dx'], fps=source['fps'],
                        command_world_xy=commands[current], native_frame_indices=np.arange(start, end))
    report.update(native_history_indices=[start, start + 1], native_reference_indices=[start, end],
                  soma_current_source_index=current, command_world_xy=commands[current].tolist(),
                  input_sha256=sha256(args.output / 'input.npz'), history_source='two unchanged original PARC frames',
                  contacts_source='original PARC body contacts, not generated SOMA predictions',
                  future_reference_is_diagnostic_only=True)
    write_json(args.output / 'complete.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--parc', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
