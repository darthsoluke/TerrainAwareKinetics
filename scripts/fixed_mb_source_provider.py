"""Diagnostic provider: retarget an immutable saved MB forecast, with no model calls."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.mb_reference import MBProposal
from motionbricks.mb_sonic.reference import file_sha256, load_soma_reference
from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter
from motionbricks.mb_sonic.runtime import FrozenMBProvider, consumer_horizon_contract
from motionbricks.mb_sonic.scene import sha256
from motionbricks.mb_sonic.trajectory_reference import TrajectoryReferenceSolver


class FixedNativeForecast:
    def __init__(self, path):
        self.source = load_soma_reference(path)
        self.digest = file_sha256(path)
        self.identity = dict(checkpoint_sha256=self.digest, kind='saved_MB_forecast;no_model_loaded')
        self.scene_query_receipt = dict(kind='saved_forecast;no_new_scene_query')
        self.controller = SimpleNamespace(inference=SimpleNamespace(device='cpu'))
        self.accounted_bytes = 0

    def generate(self, history, *, velocity_world_xz_mps, heading_world_rad, command_id, **kwargs):
        if not np.allclose(velocity_world_xz_mps, [0.,1.2], atol=1e-8):
            raise ValueError('The fixed-source experiment only permits its original forward command')
        time = history.timestamps_s[-1]
        select = (self.source.timestamps_s >= time-.14-1e-8)&(self.source.timestamps_s <= time+1.0+1e-8)
        indices = np.flatnonzero(select)
        names = ('timestamps_s','raw_features','root_positions_m','local_rot_mats','global_rot_mats','world_positions_m')
        source = replace(self.source, **{name:getattr(self.source,name)[indices] for name in names},
            trusted_until_s=float(self.source.timestamps_s[indices[-1]]),
            provenance=dict(self.source.provenance, fixed_source_sha256=self.digest,
                original_sample_indices=indices.tolist(), diagnostic='saved_source_dynamic_retarget;zero_MB_calls'))
        return MBProposal(source, (), history, dict(total=0, model_calls=0))


def load_provider(settings, *, canonical_scene, source_path):
    adapter = CalibratedSomaG1Adapter.load(settings['calibration_path'],
        robot_model_path=settings['robot_model'], representation=build_native_representation())
    options = settings['reference_constraints']
    solver = TrajectoryReferenceSolver(adapter.kinematics, **{key:options[key] for key in
        ('iterations','acceleration_time_s','jerk_time_s','overlap_time_s','overlap_weight')},
        pose_task_frame=options.get('pose_task_frame','world'))
    contract = consumer_horizon_contract(settings)
    return FrozenMBProvider(FixedNativeForecast(source_path), adapter, conditioning_mode='B1',
        estimate_actual_diagnostics=False, robot_model_sha256=settings['robot_model_sha256'],
        scene_manifest_sha256=sha256(canonical_scene.manifest_path), rig_id='g1',
        reference_history_builder=None, consumer_horizon_s=contract['consumer_horizon_s'],
        constraint_solver=solver, maximum_adoption_age_s=settings['max_generation_state_age_s'],
        planning_lead_s=options['planning_lead_s'], native_blocks=1, required_future_s=.16)
