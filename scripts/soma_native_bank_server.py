"""Full native Root/Pose/codec WASD trial, independent of the preserved28941server."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from aiohttp import web

from motionbricks.data.bones_contract import file_sha256
from motionbricks.data.unified_terrain import UnifiedTerrainController, UnifiedTerrainBackend, PROFILE, SCHEMA
from motionbricks.data.unified_committed_playback import CommittedUnifiedBackend
from scripts.package_approved_parc_live import state_hash
from scripts.soma_a100_trial_server import A100TrialRuntime
from scripts.soma_lab_server import ROOT, create_app


BASELINE_SHA256 = '0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d'
TRAINED_SHA256 = '03580db6f19c407b605fb43a6dd23514f8d5f17c5ad7b869c7ec95059ca68ae1'


def validate_native_export(saved, receipt):
    if (receipt.get('schema') != 'motionbricks.native_bank_live.v1'
            or receipt.get('original_checkpoint_sha256') != TRAINED_SHA256
            or receipt.get('source_bundle_sha256') != BASELINE_SHA256
            or receipt.get('runtime_profile') != PROFILE
            or receipt.get('model_tensors_unchanged') is not True
            or saved.get('schema') != 'motionbricks.unified.native_route_curriculum.v1'
            or saved.get('contract', {}).get('stage') != 'pose'
            or saved['contract'].get('bundle_sha256') != BASELINE_SHA256
            or saved.get('actual_updates') != 50):
        raise ValueError('Native live export identity or inference contract differs')
    state = saved.get('model_state', {})
    if not all(any(key.startswith(prefix) for key in state) for prefix in ('root.', 'pose.', 'codec.', 'adaptation.')):
        raise ValueError('Full Root/Pose/codec/adaptation tensors are required')
    if state_hash(state) != receipt.get('model_state_sha256'):
        raise ValueError('Native model tensor digest differs')


class NativeBankRuntime(A100TrialRuntime):
    def __init__(self, bundle_path, artifact, device='cuda:0', entry='up', execution='eager', world_path=None):
        if execution not in ('eager', 'verified-cuda'):
            raise ValueError('Unknown native execution mode')
        self.native_receipt = json.loads((artifact / 'export.json').read_text())
        receipt = self.native_receipt
        checkpoint = artifact / 'weights.pt'
        if file_sha256(checkpoint) != receipt['checkpoint_sha256']:
            raise ValueError('Downloaded native model file hash differs')
        saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        validate_native_export(saved, receipt)
        scene_path = world_path
        if world_path is None:
            selected = next(value for value in receipt['entries'] if value['entry'] == entry)
            scene_path = artifact / selected['scene']
            for path in (scene_path, scene_path.with_suffix('.json')):
                if file_sha256(path) != receipt['files'][path.name]:
                    raise ValueError('Live source scene changed')
        # Base initialization supplies the native inference/skin machinery. New
        # scenes are bound below to the full trained model, not its older dataset.
        super().__init__(bundle_path, BASELINE_SHA256, device, world_path,
            'Native terrain / full world' if world_path else 'Native terrain / stairs trial',
            'eager', interactive=True, start_paused=True,
            committed_prefetch=True)
        bundle = torch.load(bundle_path, map_location='cpu', mmap=True, weights_only=False)
        self.unified_controller = UnifiedTerrainController(self.inference, bundle)
        self.unified_controller.load_state_dict(saved['model_state'], strict=True)
        self.unified_controller.eval().requires_grad_(False)
        if state_hash(self.unified_controller.state_dict()) != receipt['model_state_sha256']:
            raise ValueError('Runtime did not load the complete trained model')

        def build(_inference, history, scene):
            backend = CommittedUnifiedBackend(self.unified_controller, history, scene)
            backend.configure_runtime(dict(PROFILE))
            return backend

        self.terrain_backend_builder = build
        terrain_view = None
        if world_path is None:
            with np.load(scene_path, allow_pickle=False) as archive:
                keys = {'heightfield_y_m', 'min_point_world_xyz_m', 'dx_m', 'initial_history', 'metadata_json'}
                if set(archive.files) != keys:
                    raise ValueError('Live scene may contain only static terrain and four history frames')
                metadata = json.loads(str(archive['metadata_json']))
                labels = {window['label'] for window in saved['contract']['capacity_anchors']}
                if (metadata.get('history_frames') != 4 or metadata.get('future_motion_included') is not False
                        or metadata.get('source_id', '') + '-' + entry not in labels):
                    raise ValueError('Scene history does not belong to the trained model source contract')
                history = torch.from_numpy(archive['initial_history'].copy())
                scene = {key: torch.from_numpy(archive[key].copy()) for key in
                         ('heightfield_y_m', 'min_point_world_xyz_m', 'dx_m')}
            self.entry_command = json.loads(scene_path.with_suffix('.json').read_text())['entry_command']
        else:
            if self.world_metadata is None:
                raise ValueError('Full-world mode requires a verified authored map and spawn manifest')
            scene, history, terrain_view = self.terrain_scene, self.terrain_initial_history, self.terrain_view
        if history.shape != self.terrain_initial_history.shape or not torch.isfinite(history).all():
            raise ValueError('Exactly four finite native SOMA history frames required')
        self.configure_terrain_live(build, scene, history, self.inference,
            terrain_quality=dict(self.terrain_quality, scene_file=scene_path.name,
                model_file=checkpoint.name, model_sha256=receipt['checkpoint_sha256'],
                source_bundle_sha256=BASELINE_SHA256, cases_sha256=saved['contract']['cases_sha256']),
            terrain_view=terrain_view)
        self.initial_root, self.initial_vertices, self.initial_joints = (
            self.terrain_initial_root, self.terrain_initial_vertices, self.terrain_initial_joints)
        self.rig_transport = True; self.prefer_default_cuda_stream = True
        self.terrain_quality.update(candidate_sha256=receipt['checkpoint_sha256'],
            trained_checkpoint_sha256=TRAINED_SHA256, model_state_sha256=receipt['model_state_sha256'],
            full_native_model_loaded=True, visual_baseline_approved=True,
            unified_schema=SCHEMA,
            approval_status='interactive_trial', arbitrary_terrain_quality_approved=False,
            trained_profile=dict(PROFILE), execution='native-eager-full-bank-v1',
            quality_note='Visual motion accepted; constant-command and flat-regression limitations remain.')
        with torch.inference_mode():
            rig = self.skin_rig_geometry(self.terrain_initial_history, self.inference)['rig']
            _, expected, _ = self.skin_features_with_joints(self.terrain_initial_history, self.inference)
            actual = self.skinner.lbs(rig); expected = torch.as_tensor(expected, device=actual.device)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            self.terrain_quality['rig_transport_max_vertex_error_m'] = float((actual - expected).abs().max())
        if execution == 'verified-cuda':
            from motionbricks.data.soma_terrain_execution import TerrainExecution
            runner = TerrainExecution(self.inference, self.terrain_scene, self.skinner,
                self.terrain_initial_history, controller=self.unified_controller,
                independent_validation_seeds=True, fk_tolerance_m=3e-5)
            object.__setattr__(self.inference, '_terrain_execution', runner)
            self.terrain_keep_ready = True
            self.terrain_quality.update(execution='verified-static-full-native-cuda-v1', execution_parity=runner.validation)

    @torch.inference_mode()
    def warm_live_entry(self):
        if self.entry_command is None:
            return
        started = time.perf_counter()
        backend = self.create_backend('terrain')
        result = backend.predict_plan(self.entry_command['velocity_xz_mps'], self.entry_command['heading_rad'])
        self.skin_rig_geometry(result['planned_motion'], self.inference)
        if self.terrain_initial_history.is_cuda:
            torch.cuda.synchronize(self.terrain_initial_history.device)
        self.terrain_quality['warmup_ms'] = (time.perf_counter() - started) * 1000

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
        result['live_model'] = dict(source='Native terrain full world' if self.world_metadata else 'Native terrain stair trial',
            terrain_conditioned=True,
            candidate_sha256=self.native_receipt['checkpoint_sha256'], trained_checkpoint_sha256=TRAINED_SHA256,
            visual_approved=True, quality_approved=False, full_model=True)
        return result


def main(a):
    runtime = NativeBankRuntime(a.bundle, a.artifact, a.device, a.entry, a.execution, a.world)
    print(json.dumps(dict(ready=True, port=a.port, model=runtime.terrain_quality)), flush=True)
    web.run_app(create_app(runtime, ROOT / 'web/soma_lab'), host='127.0.0.1', port=a.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'out/a100_live_20261010/a100-terrain.pt')
    parser.add_argument('--artifact', type=Path, default=ROOT / 'out/parc_rugged32_20261013_v2/native_bank_live_20261015_v1')
    parser.add_argument('--entry', choices=('up', 'down'), default='up')
    parser.add_argument('--world', type=Path, help='Verified authored full-world NPZ with its mesh/spawn receipt')
    parser.add_argument('--port', type=int, default=28942)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--execution', choices=('eager', 'verified-cuda'), default='eager')
    main(parser.parse_args())
