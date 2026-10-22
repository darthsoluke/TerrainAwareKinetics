"""Preserved Oct11 flat controller, isolated from later terrain candidates."""

import argparse
import json

from aiohttp import web

from scripts.soma_a100_trial_server import A100TrialRuntime
from scripts.soma_lab_server import ROOT, create_app


class RecoveredFlatRuntime(A100TrialRuntime):
    def config(self):
        value = super().config()
        value['runtime_controls']['defaults'] = dict(value['runtime_controls']['trained_defaults'])
        value['flat_recovery_profile'] = 'native-original-buffer'
        return value


def main(args):
    folder = ROOT/'out/a100_live_20261010'
    bundle_hash = '0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d'
    scene = folder/'flat_recovery_20261013/flat100m.npz'
    receipt = json.loads(scene.with_suffix('.json').read_text())
    if (receipt.get('source_scene_sha256') != '6823d99cd4a68e35874242978d898c6ba68a577bb473538f5acb3855d2344406'
            or receipt.get('source_history_modified') is not False or not receipt.get('authored_test_geometry')):
        raise ValueError('Recovered flat test-scene provenance differs')
    runtime = RecoveredFlatRuntime(folder/'a100-terrain.pt', bundle_hash, 'cuda:0', scene,
        'Oct11 flat baseline / terrain unqualified', args.execution, interactive=True, start_paused=True)
    runtime.terrain_quality['quality_note'] = 'Recovered flat baseline; stopping drift and terrain traversal remain unqualified'
    print(json.dumps(dict(ready=True, port=args.port, model=runtime.terrain_quality,
        defaults=runtime.config()['runtime_controls']['defaults'])), flush=True)
    web.run_app(create_app(runtime, ROOT/'web/soma_lab'), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=28940)
    parser.add_argument('--execution', choices=('eager', 'cuda-graph'), default='cuda-graph')
    main(parser.parse_args())
