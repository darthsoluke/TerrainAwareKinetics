"""One scene-bound replay of the existing post-Root terrain pipeline.

prepare() performs CUDA admission on the caller's serialized inference worker.
__call__(raw_history[1,4,366], controls[1,3]) takes VX,VZ,heading and returns
motion[1,32,366]; only motion[:,4:28] is playable. rig[24,77,4,4] and
joints[24,30,3] describe that future. history/position_error/rotation_error
describe the re-encoded INPUT history, never a predicted playback advance.
All coarse and post-Root query fields remain tensor outputs for admission.
goal/query_controls/root retain the exact captured query inputs. Query admission
holds these inputs fixed; independent neural-trajectory differences are recorded
separately. observation_known preserves each sampler's per-batch any-known rule.
"""

import torch

from .local_terrain_observation import trained_query_controls
from .native_mb_online_conditions import velocity_direction_local
from .native_mb_online_live import online_command_goal
from .native_mb_terrain_protocol import terrain_observation_config
from .root_path_split_execution import VOLUME_KEYS
from .soma_cuda_graph import StaticCudaCall
from .soma_static_execution import trace_fixed_call


SCHEMA = 'motionbricks.reactive_path_execution.v1'
COARSE_NAMES = ('points', 'valid', 'world_query_xz', 'query_envelope_m',
    'survey_points', 'survey_valid', 'survey_world_query_xz', 'survey_query_envelope_m',
    'sdf_points', 'sdf_distance', 'sdf_normal', 'sdf_valid', 'sdf_features')
PATH_NAMES = ('points', 'valid', 'world_query_xz', 'query_envelope_m',
    'sdf_points', 'sdf_distance', 'sdf_normal', 'sdf_valid', 'sdf_features', 'path_world_xz_m')
QUERY_NAMES = (*('coarse_' + key for key in COARSE_NAMES), *PATH_NAMES, 'actor_query_controls')


def _assert_queries(actual, expected):
    errors = {}
    for name in QUERY_NAMES:
        left, right = actual[name], expected[name]
        torch.testing.assert_close(left, right, atol=0 if left.dtype == torch.bool else 1e-5, rtol=0)
        if left.dtype != torch.bool:
            errors[name] = float((left - right).abs().max())
    return errors


def _query_differences(actual, expected):
    return {name: ({'different_count': int((actual[name] != expected[name]).sum())}
                  if actual[name].dtype == torch.bool else
                  {'max_abs': float((actual[name] - expected[name]).abs().max())})
            for name in QUERY_NAMES}


def _assert_output_contract(result):
    shapes = dict(motion=(1, 32, 366), history=(1, 4, 366),
        rig=(24, 77, 4, 4), joints=(24, 30, 3))
    if any(tuple(result[key].shape) != shape for key, shape in shapes.items()):
        raise ValueError('Reactive path requires GLOBAL366 and24-frame official SOMA rig/joints')
    if not bool(result['finite']) or not bool((result['count'] == 8).all()):
        raise ValueError('Reactive path produced nonfinite values or changed its8-token horizon')
    if not bool(result['observation_known']):
        raise ValueError('All scene observations are unknown')
    if any(not torch.isfinite(result[key]).all() or (result[key] > 2e-5).any()
           for key in ('position_error', 'rotation_error')):
        raise ValueError('Reactive path changed the displayed input-history FK')


class ReactivePathExecution:
    def __init__(self, runtime, geometry_function, *, comparison_field=None):
        self.runtime = runtime
        self.runner = runtime.inference._terrain_execution
        self.geometry_function = geometry_function
        self.comparison_field = comparison_field
        self.call = self.proof = self.binding = None
        self.root_function, self.pose_function = (call.function for call in self.runner.graphs[8])
        self.observe_function = self.runner._observe.function
        self.history_shape = (1, 4, 366)

    def _scene_binding(self):
        from motionbricks.mb_sonic.geometry_execution import storage_binding

        runtime, runner = self.runtime, self.runner
        field_entry = runner.controller._fields.get(id(runner.scene['heightfield_y_m']))
        if (runtime.inference._terrain_execution is not runner
                or runtime.terrain_scene is not runner.scene
                or runner.inference is not runtime.inference or field_entry is None
                or field_entry[0] is not runner.scene or field_entry[1] is not runner.field
                or any(call.function is not expected for call, expected in
                       zip(runner.graphs[8], (self.root_function, self.pose_function)))
                or runner._observe.function is not self.observe_function):
            raise ValueError('Reactive path scene binding changed; prepare a new executor')
        receipt = getattr(runtime, 'acceptance_receipt', {})
        return storage_binding(dict(scene_id=getattr(runtime, 'current_map', None),
            revision=getattr(runtime, 'scene_revision', None),
            source=receipt.get('scene_sha256'), checkpoint=receipt.get('checkpoint_sha256'),
            inference=id(runtime.inference), scene_id_object=id(runner.scene), scene=runner.scene,
            field_id=id(runner.field), field=vars(runner.field),
            sampler_id=id(runner.controller.sampler), sampler=vars(runner.controller.sampler),
            functions=tuple(id(call.function) for call in runner.graphs[8]),
            observer=id(runner._observe.function)), compact=True)

    def _goal(self, history, controls):
        return online_command_goal(history[..., :5], controls[:, :2], controls[:, 2],
            8, self.runner.protocol_version)

    def _coarse_native(self, history, goal, query):
        return self.runner._scene_inputs(history, goal, 8, query)

    def _actor_query(self, history, goal, query):
        config = terrain_observation_config(self.runner.inference._root_model.args['observation_range_policy'])
        return trained_query_controls(query, velocity_direction_local(history[..., :5], goal, 28 / 30), config)

    def _path_native(self, history, root, *coarse):
        volume, height, path = self.runner._path_inputs(history, root, coarse, 8)
        return (height.points, height.valid, height.world_query_xz, height.query_envelope_m,
            *(volume[key] for key in VOLUME_KEYS), path)

    def _assemble(self, raw, controls, *, observe, goal, coarse, path, root, pose):
        observed = observe(raw)
        history = observed['history']
        target = goal(history, controls)
        query = self.query
        original = coarse(history, target, query)
        root_result = root(history, target, query, *original)
        fine = path(history, root_result['root'], *original)
        result = dict(pose(history, target, query, root_result['count'], root_result['root'],
            root_result['local'], *original[:8], *fine[:9]))
        return self._outputs(raw, controls, observed, history, target, query, original, fine, result)

    def _outputs(self, raw, controls, observed, history, target, query, coarse, fine, prediction):
        result = dict(prediction, motion=prediction['motion'][:, :32])
        result.update(dict(zip(('coarse_' + name for name in COARSE_NAMES), coarse)))
        result.update(dict(zip(PATH_NAMES, fine)))
        result.update(survey_points=coarse[4], survey_valid=coarse[5], survey_world_query_xz=coarse[6])
        result.update(goal=target, query_controls=query)
        result['actor_query_controls'] = self.actor_query_function(history, target, query)
        result.update(self.geometry_function(result['motion'][:, 4:28].contiguous()))
        result.update(observed)
        result['observation_known'] = (coarse[11].flatten(1).any(-1).all()
            & result['sdf_valid'].flatten(1).any(-1).all())
        result['finite'] = (torch.isfinite(raw).all() & torch.isfinite(controls).all()
            & torch.isfinite(result['motion']).all() & torch.isfinite(result['rig']).all()
            & torch.isfinite(result['joints']).all())
        return result

    def _fused(self, raw, controls):
        return self._assemble(raw, controls, observe=self.observe_function, goal=self.goal_function,
            coarse=self.coarse_function, path=self.path_function,
            root=self.root_function, pose=self.pose_function)

    def _trace_queries(self, raw, controls):
        history = self.observe_function(raw)['history']
        target = self._goal(history, controls)
        self.query = query = history.new_tensor([[-1., .6, 1., .4]])
        coarse = self._coarse_native(history, target, query)
        root = self.root_function(history, target, query, *coarse)
        self.goal_function = trace_fixed_call(self._goal, (history, controls), shared_cuda_constants=True)
        self.actor_query_function = trace_fixed_call(self._actor_query,
            (history, target, query), shared_cuda_constants=True)
        self.coarse_function = trace_fixed_call(self._coarse_native,
            (history, target, query), shared_cuda_constants=True)
        self.path_function = trace_fixed_call(self._path_native,
            (history, root['root'], *coarse), shared_cuda_constants=True)

    def _native_reference(self, raw, controls):
        from .native_mb_observed_history import reencode_played_history
        from .root_path_terrain import RootPathTerrainExecution

        runner, checks = self.runner, {}
        history = reencode_played_history(raw, runner.inference.global_motion_rep, tensor_checks=checks)
        target = self._goal(history, controls)
        query = self.query
        coarse = self._coarse_native(history, target, query)
        tokenizer = runner.inference._vqvae_pose_model
        decode, tokens = tokenizer.forward_decoder, {}

        def capture(ids, *args, **kwargs):
            tokens['value'] = ids.clone()
            return decode(ids, *args, **kwargs)

        tokenizer.forward_decoder = capture
        try:
            with runner.controller.context(history, runner.scene):
                prediction = RootPathTerrainExecution._prediction(runner, history, target,
                    num_tokens=8, query_controls=query)
        finally:
            tokenizer.forward_decoder = decode
        prediction['tokens'] = tokens['value']
        height, volume = runner.observer.height, runner.controller.volume_observation
        fine = (height.points, height.valid, height.world_query_xz, height.query_envelope_m,
            *(volume[key] for key in VOLUME_KEYS), runner.observer.path)
        result = self._outputs(raw, controls, dict(history=history, **checks),
            history, target, query, coarse, fine, prediction)
        result['actor_query_controls'] = self._actor_query(history, target, query)
        return result

    def _admit_queries(self, actual, expected):
        history, target, query = actual['history'], actual['goal'], actual['query_controls']
        captured_coarse = tuple(actual['coarse_' + name] for name in COARSE_NAMES)
        native_coarse = self._coarse_native(history, target, query)
        native_path = self._path_native(history, actual['root'], *captured_coarse)
        same_input = dict(zip(('coarse_' + name for name in COARSE_NAMES), native_coarse))
        same_input.update(dict(zip(PATH_NAMES, native_path)))
        same_input['actor_query_controls'] = self._actor_query(history, target, query)
        errors = _assert_queries(actual, same_input)
        backend_points = 0
        if self.comparison_field is not None:
            for points in (actual['coarse_sdf_points'],actual['sdf_points']):
                reference = self.comparison_field.query(points)
                current = self.runner.field.query(points)
                for left,right in zip(current,reference):
                    torch.testing.assert_close(left,right,atol=0,rtol=0)
                backend_points += points.numel()//3
        return dict(same_input_query_max_abs=errors, masks_equal=True,
            backend_identical_query_points=backend_points,
            cross_trajectory_query_errors=_query_differences(actual, expected))

    def _admit(self, actual, expected):
        _assert_output_contract(actual)
        _assert_output_contract(expected)
        if not torch.equal(actual['tokens'][:, :8], expected['tokens'][:, :8]):
            raise ValueError('Reactive path changed active native pose tokens')
        torch.testing.assert_close(actual['history'], expected['history'], atol=1e-5, rtol=0)
        rep = self.runner.inference.global_motion_rep
        left = rep.inverse(actual['motion'], False, return_all=False, joint_positions_from='global_rot_data')
        right = rep.inverse(expected['motion'], False, return_all=False, joint_positions_from='global_rot_data')
        fk = float((left['posed_joints'] - right['posed_joints']).norm(dim=-1).max())
        if not torch.isfinite(torch.tensor(fk)) or fk > 3e-5:
            raise ValueError(f'Reactive path native FK mismatch: {fk}')
        query_proof = self._admit_queries(actual, expected)
        same_motion_geometry = self.geometry_function(actual['motion'][:, 4:28].contiguous())
        for name in ('rig', 'joints'):
            torch.testing.assert_close(actual[name], same_motion_geometry[name], atol=1e-5, rtol=1e-5)
        future = rep.inverse(actual['motion'][:, 4:28], False,
            return_all=False, joint_positions_from='global_rot_data')
        native_skin = self.runtime.skinner.skin(future['global_joint_rots'][0],
            future['posed_joints'][0], rot_is_global=True)
        rig_skin = self.runtime.skinner.lbs(actual['rig'])
        torch.testing.assert_close(rig_skin, native_skin, atol=1e-5, rtol=1e-5)
        return dict(active_tokens_equal=True, fk_max_error_m=fk, **query_proof,
            cross_trajectory_rig_max_abs=float((actual['rig'] - expected['rig']).abs().max()),
            cross_trajectory_joint_max_error_m=float((actual['joints'] - expected['joints']).norm(dim=-1).max()),
            skin_max_abs_m=float((rig_skin - native_skin).abs().max()),
            skin_vertices=int(native_skin.shape[-2]), observation_known=True)

    @torch.inference_mode()
    def prepare(self):
        from .full_corpus_inference import tensor_digest
        from .native_mb_evaluation import full_precision_evaluation

        if self.proof is not None:
            self._assert_binding()
            return self.proof
        seed = self.runtime.terrain_initial_history
        if tuple(seed.shape) != self.history_shape or seed.dtype != torch.float32 or not seed.is_cuda:
            raise ValueError('Reactive path preparation requires allocated CUDA FP32[1,4,366] history')
        self._scene_binding()
        controller = self.runner.controller
        before = tensor_digest(controller.state_dict())
        cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state(seed.device)
        call, cases = None, []
        try:
            with full_precision_evaluation(), torch.jit.optimized_execution(False):
                controls = seed.new_tensor([[0., 0., 0.]])
                self._trace_queries(seed, controls)
                call = StaticCudaCall(self._fused)
                for shift, vx, vz, heading in (
                    ((0., 0., 0.), 0., 0., 0.), ((0., 0., 0.), 0., 3., 0.),
                    ((0., 0., 0.), 1.6, 0., .7), ((0., 0., 0.), 0., -1.6, -1.2),
                    ((0., 0., 0.), -1.6, .4, 1.57),
                    ((.05, .02, -.04), .6, -.2, .35), ((-.06, -.015, .03), -.4, .8, -.6)):
                    raw = seed.clone()
                    raw[..., :3] += raw.new_tensor(shift)
                    controls = raw.new_tensor([[vx, vz, heading]])
                    expected = self._native_reference(raw, controls)
                    actual = call(raw, controls)
                    cases.append(dict(translation_m=list(shift), controls=[vx, vz, heading],
                        **self._admit(actual, expected)))
        finally:
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, seed.device)
        after = tensor_digest(controller.state_dict())
        if before != after:
            raise ValueError('Reactive path preparation changed frozen checkpoint tensors')
        self.binding = self._scene_binding()
        self.call = call
        self.proof = dict(schema=SCHEMA, passed=True, cases=cases, checkpoint_unchanged=True,
            checkpoint_tensor_sha256=after, rng_restored=True, optimizer_updates=0,
            reference='eager native post-Root pipeline with runtime Transformer mode',
            query_comparison_scope='native_query_at_captured_history_goal_root_and_coarse_inputs',
            cross_trajectory_query_comparison_scope='diagnostic_only_independent_eager_fused_trajectories',
            query_atol=1e-5, query_valid_masks_exact=True, fk_atol_m=3e-5,
            motion_shape=[1, 32, 366], future_slice=[4, 28], rig_shape=[24, 77, 4, 4],
            history_source='input_raw_four_played_frames_reencoded', tokens=8, pose_passes=4)
        return self.proof

    def _assert_binding(self):
        if self.binding != self._scene_binding():
            raise ValueError('Reactive path scene binding changed; prepare a new executor')

    @torch.inference_mode()
    def __call__(self, raw_history_tensor, controls_tensor):
        if self.proof is None:
            raise ValueError('Reactive path requires successful prepare() admission')
        self._assert_binding()
        seed = self.runtime.terrain_initial_history
        if (tuple(raw_history_tensor.shape) != self.history_shape or tuple(controls_tensor.shape) != (1, 3)
                or any(value.dtype != seed.dtype or value.device != seed.device
                       for value in (raw_history_tensor, controls_tensor))):
            raise ValueError('Reactive path expects same-device FP32 history[1,4,366] and controls[1,3]')
        return self.call(raw_history_tensor, controls_tensor)
