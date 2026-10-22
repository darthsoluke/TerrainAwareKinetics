"""Read-only decoder response to terrain, holding tokens and Root conditions fixed."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.terrain_decoder_pilot import DecoderExperiment
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_feedback_ablation import evaluation_goal
from scripts.run_mixed_control_workflow import available_devices


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    try:
        available_devices([0]); device = 'cuda:0'
    except RuntimeError:
        device = 'cpu'
    data = torch.load(args.base/'input/data/windows.pt', map_location='cpu', weights_only=False)
    validation = [w for w in data['windows'] if w['split']=='validation']
    report = {'scope': 'decoder only; fixed token/Root/history; not a traversal or physical validity test',
              'height_perturbation_m': .1, 'runtime_changed': False, 'arms': {}}
    for arm in ('terrain', 'both'):
        inference, bundle = load_terrain_bundle(args.base/'input/source.pt', device)
        pilot = DecoderExperiment(inference, bundle, terrain=True)
        weights = torch.load(args.base/'runs'/arm/'decoder.pt', map_location=device, weights_only=False)
        pilot.codec.decoder.load_state_dict(weights['decoder'], strict=True)
        pilot.adapter.load_state_dict(weights['adapter'], strict=True)
        rows = []
        for index in (0, 3, 12, 15, 27, 30):
            item = validation[index]; world = item['world'].to(device)
            history = world[:, :4]
            goal = evaluation_goal(inference, world, 0, bundle['protocol']['version'])
            _, call = pilot.request(history, goal, data['scenes'][item['scene']])
            reference = pilot.restore_world(pilot.replay(call), history, call['observation'])
            changed = dict(call, observation={k:v.clone() for k,v in call['observation'].items()})
            changed['observation']['points'][..., 1] += .1*changed['observation']['valid']
            # Keep the coordinate origin fixed: only the decoder's terrain input changes.
            shifted = pilot.restore_world(pilot.replay(changed), history, call['observation'])
            gate = pilot.adapter.attention.gate.clone()
            pilot.adapter.attention.gate.zero_()
            disabled = pilot.restore_world(pilot.replay(call), history, call['observation'])
            pilot.adapter.attention.gate.copy_(gate)
            rows.append({'window': index,
                'height_shift_response_mean_m': float((shifted[:, 4:32, 1]-reference[:, 4:32, 1]).abs().mean()),
                'gate_disable_response_mean_m': float((disabled[:, 4:32, 1]-reference[:, 4:32, 1]).abs().mean())})
        report['arms'][arm] = {'gate': float(pilot.adapter.attention.gate), 'rows': rows,
            'height_shift_response_mean_m': sum(r['height_shift_response_mean_m'] for r in rows)/len(rows),
            'gate_disable_response_mean_m': sum(r['gate_disable_response_mean_m'] for r in rows)/len(rows)}
        pilot.close();del pilot, inference, bundle, weights
        if device.startswith('cuda'):torch.cuda.empty_cache()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
