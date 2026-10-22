"""Explicitly experimental WASD endpoint for the unified scene-conditioned pilot."""

import argparse
import json
from pathlib import Path

import torch
from aiohttp import web

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE, SCHEMA
from scripts.soma_a100_trial_server import A100TrialRuntime
from scripts.soma_lab_server import ROOT, create_app


BASELINE_SHA256 = '0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d'
CANDIDATE_SHA256 = 'ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979'


def validate_candidate(checkpoint, source_hash):
    if checkpoint.get('schema') != SCHEMA:
        raise ValueError('Unified terrain checkpoint schema mismatch')
    contract = checkpoint.get('contract', {})
    if contract.get('bundle_sha256') != source_hash or contract.get('profile') != PROFILE:
        raise ValueError('Candidate source or trained playback profile mismatch')
    if checkpoint.get('steps') != contract.get('updates') or checkpoint['steps'] != 200:
        raise ValueError('Expected the completed 200-update capacity pilot')
    if not isinstance(checkpoint.get('adaptation'), dict):
        raise ValueError('Candidate lacks adaptation weights')


class UnifiedTrialRuntime(A100TrialRuntime):
    def __init__(self, bundle_path, candidate_path, scene_path, device, execution='eager'):
        if file_sha256(candidate_path) != CANDIDATE_SHA256:
            raise ValueError('Downloaded candidate hash differs from the A100 result')
        checkpoint = torch.load(candidate_path, map_location='cpu', mmap=True, weights_only=False)
        validate_candidate(checkpoint, BASELINE_SHA256)
        super().__init__(bundle_path, BASELINE_SHA256, device, scene_path,
            'Unified SDF / 200 updates / unqualified', 'eager', interactive=True, start_paused=True)
        bundle = torch.load(bundle_path, map_location='cpu', mmap=True, weights_only=False)
        self.unified_controller = UnifiedTerrainController(self.inference, bundle)
        self.unified_controller.adaptation.load_state_dict(checkpoint['adaptation'], strict=True)
        self.unified_controller.eval().requires_grad_(False)

        def build(_inference, history, scene):
            backend = UnifiedTerrainBackend(self.unified_controller, history, scene)
            backend.configure_runtime(dict(PROFILE))
            return backend

        self.terrain_backend_builder = build
        self.terrain_quality.update(
            approval_status='experimental_not_approved', candidate_sha256=CANDIDATE_SHA256,
            unified_schema=SCHEMA, fitting_updates=checkpoint['steps'], primitive_exposures=800,
            quality_note='Experimental pilot: stair penetration and flat speed regression remain. No IK or output correction.',
            trained_profile=dict(PROFILE), execution='native-eager-unified-v1',
            arbitrary_terrain_quality_approved=False)
        # Use the same official skin weights, with compact rig transport rather than full vertex packets.
        self.rig_transport = True
        self.prefer_default_cuda_stream = True
        with torch.inference_mode():
            sample = self.terrain_initial_history
            rig = self.skin_rig_geometry(sample, self.inference)['rig']
            _, expected, _ = self.skin_features_with_joints(sample, self.inference)
            actual = self.skinner.lbs(rig)
            expected = torch.as_tensor(expected, device=actual.device)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            self.terrain_quality['rig_transport_max_vertex_error_m'] = float((actual-expected).abs().max())
        if execution == 'verified-cuda':
            from motionbricks.data.soma_terrain_execution import TerrainExecution
            print('Verifying unchanged unified motion, SDF observations and official skin before CUDA replay', flush=True)
            runner = TerrainExecution(self.inference, self.terrain_scene, self.skinner,
                                      self.terrain_initial_history, controller=self.unified_controller)
            object.__setattr__(self.inference, '_terrain_execution', runner)
            self.terrain_keep_ready = True
            self.terrain_quality.update(execution='verified-static-unified-cuda-v1',
                                        execution_parity=runner.validation,
                                        active_viewer_keep_ready=True)

    @torch.inference_mode()
    def skin_rig_geometry(self, features, inference):
        geometry = super().skin_rig_geometry(features, inference)
        if hasattr(self, 'unified_controller'):
            field = self.unified_controller.scene_field(self.terrain_scene)
            distance, _, valid = field.query(geometry['joints'])
            geometry.update(bone_sdf_distance=distance, bone_sdf_valid=valid)
        return geometry

    def config(self):
        result = super().config()
        result['runtime_controls']['defaults'] = dict(PROFILE)
        result['runtime_controls']['trained_defaults'] = dict(PROFILE)
        result['live_model'] = dict(source='Unified SDF capacity pilot, 200 updates', terrain_conditioned=True,
                                    candidate_sha256=CANDIDATE_SHA256, quality_approved=False)
        return result


def main(args):
    runtime = UnifiedTrialRuntime(args.bundle, args.candidate, args.scene, args.device, args.execution)
    print(json.dumps(dict(ready=True, port=args.port, model=runtime.terrain_quality,
                          defaults=runtime.config()['runtime_controls']['defaults'])), flush=True)
    web.run_app(create_app(runtime, ROOT/'web/soma_lab'), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'out/a100_live_20261010/a100-terrain.pt')
    parser.add_argument('--candidate', type=Path, default=ROOT/'out/unified_terrain_20261014/live_v1/candidate.pt')
    parser.add_argument('--scene', type=Path, default=ROOT/'out/a100_live_20261010/terrain-world.npz')
    parser.add_argument('--port', type=int, default=28941)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execution', choices=('eager', 'verified-cuda'), default='eager')
    main(parser.parse_args())
