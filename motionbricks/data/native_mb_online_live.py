"""Online planar intent -> current local terrain -> native MB generated motion.

No future reference clip, terrain height target, IK, or output root replacement
is used. A planar endpoint is a command goal, not a predicted root trajectory.
"""

import time

import torch

from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from .local_terrain_observation import extract_local_terrain_observation,trained_query_controls
from .native_mb_online_conditions import (ONLINE_PROTOCOL, VELOCITY_PROTOCOL, online_protocol,
    sparse_planar_conditions, goal_direction_local, sparse_velocity_conditions, velocity_direction_local)
from .native_mb_terrain_protocol import terrain_observation_config


_DIAGNOSTICS_VERSION = 'soma_motion_debug_v1'
_HISTORY_FRAMES = 4
_DROP_END_FRAMES = 4


def planar_command_goal(history_root, velocity_world_xz, heading_world_rad, num_tokens, fps=30):
    if history_root.ndim != 3 or history_root.shape[1:] != (4, 5):
        raise ValueError('Expected four observed Root states')
    if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or not 6 <= num_tokens <= 16:
        raise ValueError('Online command horizon must be 6..16 native tokens')
    velocity = torch.as_tensor(velocity_world_xz, device=history_root.device, dtype=history_root.dtype)
    if velocity.shape == (2,):
        velocity = velocity[None].expand(history_root.shape[0], -1)
    heading = torch.as_tensor(heading_world_rad, device=history_root.device,
                              dtype=history_root.dtype).reshape(-1)
    if heading.numel() == 1:
        heading = heading.expand(history_root.shape[0])
    if velocity.shape != (history_root.shape[0], 2) or heading.shape != (history_root.shape[0],):
        raise ValueError('Command batch must match observed history')
    if not torch.isfinite(velocity).all() or not torch.isfinite(heading).all() or fps != 30:
        raise ValueError('Online command must be finite at native 30 Hz')
    # Endpoint time is measured from history frame 3, not frame 0.
    seconds = (4 * num_tokens - 4) / fps
    target_xz = history_root[:, -1, [0, 2]] + velocity * seconds
    return torch.cat((target_xz, torch.cos(heading)[:, None], torch.sin(heading)[:, None]), -1)


def online_command_goal(history_root, velocity_world_xz, heading_world_rad, num_tokens, version):
    goal = planar_command_goal(history_root, velocity_world_xz, heading_world_rad, num_tokens)
    if online_protocol(version)['version'] == VELOCITY_PROTOCOL['version']:
        goal[:, :2] = torch.as_tensor(velocity_world_xz, device=goal.device, dtype=goal.dtype).reshape(-1, 2)
    return goal


def intent_diagnostics(observation, root, goal, tokens, version):
    display_goal = goal
    if version == VELOCITY_PROTOCOL['version']:
        display_goal = goal.clone()
        display_goal[:, :2] = root[:, -1, [0, 2]] + goal[:, :2] * ((4*tokens-4)/30)
    result = _terrain_diagnostics(observation, root, display_goal)
    result['input_intent']['horizon_seconds'] = (4*tokens-4)/30
    if version == VELOCITY_PROTOCOL['version']:
        result['input_intent'].update(condition_kind='terminal_planar_velocity',
            velocity_world_xz_mps=goal[0, :2].detach().cpu().tolist(), endpoint_is_visual_guide=True)
    return result


def add_survey_diagnostics(diagnostics,survey,root,goal,tokens,version,actor_controls):
    patch=intent_diagnostics(survey,root,goal,tokens,version)['terrain_patches'][1]
    patch.update(name='survey',role='survey_only_not_model_input')
    diagnostics['terrain_patches'].append(patch)
    diagnostics['motion_observation_extent_m']=float(actor_controls[0,0])
    diagnostics['motion_observation_halfwidth_m']=float(actor_controls[0,1])
    diagnostics['survey_role']='untrained_far_range_preview'


def _terrain_diagnostics(observation, root, goal):
    """Describe the exact current observation supplied to the model."""
    points = observation.points.detach()
    valid = observation.valid.detach().bool()
    world_query_xz = observation.world_query_xz.detach()
    patches = []
    for patch_index, name in enumerate(('near', 'intent')):
        patch_valid = valid[0, patch_index]
        query_xz = world_query_xz[0, patch_index]
        world_y = root[0, -1, 1] + points[0, patch_index, ..., 1]
        # Invalid heights use a finite current-root-height sentinel and are masked.
        world_y = torch.where(patch_valid, world_y, root[0, -1, 1])
        world_xyz = torch.stack((query_xz[..., 0], world_y, query_xz[..., 1]), dim=-1)
        patches.append({
            'name': name,
            'shape': [9, 9],
            'world_xyz_m': world_xyz.reshape(-1, 3).cpu().tolist(),
            'valid': patch_valid.reshape(-1).cpu().tolist(),
        })
    return {
        'version': _DIAGNOSTICS_VERSION,
        'input_intent': {
            'start_world_xyz_m': root[0, -1, :3].detach().cpu().tolist(),
            'endpoint_world_xz_m': goal[0, :2].detach().cpu().tolist(),
            'heading_world_rad': float(torch.atan2(goal[0, 3], goal[0, 2]).item()),
            'horizon_seconds': None,
            'height_source': 'not_conditioned',
        },
        'root_stage_world_xyz_m': [],
        'decoded_world_xyz_m': [],
        'terrain_patches': patches,
        'observation_root_world_xyz_m': root[0, -1, :3].detach().cpu().tolist(),
        'observation_source': 'current_played_history',
        'observation_snapshot_note': (
            'Generation-time snapshot; terrain is not newly observed or resampled on cached blocks.'),
    }


def _future_xyz(values, start, stop):
    if (not isinstance(values, torch.Tensor) or values.ndim != 3 or values.shape[0] != 1
            or values.shape[1] < stop or values.shape[2] < 3 or stop <= start):
        return []
    future = values[:, start:stop, :3]
    if not future.numel() or not torch.isfinite(future).all():
        return []
    return future[0].detach().cpu().tolist()


@torch.no_grad()
def predict_planar_terrain(inference, history, goal, scene, *, num_tokens=8, config=None,
                           info=None, diagnostics=None, tensor_diagnostics=None, query_controls=None,
                           query_path_world_xz=None,query_path_valid=None, prepared_observations=None):
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise TypeError('diagnostics must be a dict or None')
    version = inference._root_model.args.get('online_conditioning')
    online_protocol(version)
    if inference._pose_model.args.get('online_conditioning') != version:
        raise ValueError('Online terrain inference requires matched, trained online Root and Pose checkpoints')
    if history.shape != (1, _HISTORY_FRAMES, len(inference.global_motion_rep.indices['all'])):
        raise ValueError('Online backend requires exactly four observed full-motion frames')
    if not torch.isfinite(history).all():
        raise ValueError('History must contain finite played motion')
    root = extract_feature_from_motion_rep(history, inference.global_motion_rep, 'root')
    local = inference.motion_rep.dual_rep.global_to_local(
        history, False, False, lengths=torch.tensor([_HISTORY_FRAMES], device=history.device))
    velocity = extract_feature_from_motion_rep(local, inference.local_motion_rep, 'root')
    poses = extract_feature_from_motion_rep(local, inference.local_motion_rep,
                                            inference.EXTERNAL_POSE_FEATURE_MODE)
    velocity_mode = version == VELOCITY_PROTOCOL['version']
    if velocity_mode:
        if inference.local_motion_rep.compute_kwargs.get('removing_heading') is not False:
            raise ValueError('Terminal velocity goals require native world-XZ velocity features')
        conditions = sparse_velocity_conditions(root, velocity, poses, goal)
        direction = velocity_direction_local(root, goal, (4*num_tokens-4)/30)
    else:
        conditions = sparse_planar_conditions(root, velocity, poses, goal)
        direction = goal_direction_local(root, goal)
    range_policy = inference._root_model.args.get('observation_range_policy')
    if range_policy != inference._pose_model.args.get('observation_range_policy'):
        raise ValueError('Root and Pose observation range policies differ')
    observation_config=terrain_observation_config(range_policy)
    actor_controls=(trained_query_controls(query_controls,direction,observation_config)
                    if query_controls is not None else None)
    if prepared_observations is not None and not torch.is_inference_mode_enabled():
        raise ValueError('Prepared scene observations are inference-only')
    observation_snapshot = prepared_observations[0] if prepared_observations is not None else extract_local_terrain_observation(
        torch.as_tensor(scene['heightfield_y_m'], device=history.device),
        scene['min_point_world_xyz_m'], scene['dx_m'], root[:, -1, :3],
        torch.atan2(root[:, -1, 4], root[:, -1, 3]), direction,
        heightfield_valid=scene.get('heightfield_valid'),config=observation_config,
        query_controls=actor_controls,query_path_world_xz=query_path_world_xz,query_path_valid=query_path_valid)
    survey_snapshot=None
    if query_controls is not None:
        survey_snapshot=prepared_observations[1] if prepared_observations is not None else extract_local_terrain_observation(
            torch.as_tensor(scene['heightfield_y_m'],device=history.device),
            scene['min_point_world_xyz_m'],scene['dx_m'],root[:,-1,:3],
            torch.atan2(root[:,-1,4],root[:,-1,3]),direction,
            heightfield_valid=scene.get('heightfield_valid'),config=observation_config,
            query_controls=query_controls,query_path_world_xz=query_path_world_xz,query_path_valid=query_path_valid)
    observation = observation_snapshot.as_model_inputs()
    settings = dict(config or {})
    settings.setdefault('allow_pred_out_of_reach_num_tokens', False)
    predict_duration = settings.pop('online_predict_duration', False)
    if velocity_mode and predict_duration:
        raise ValueError('Velocity control requires its explicit prediction horizon')
    allowed_duration = None
    if predict_duration:
        minimum = int(inference._root_model.args['min_tokens'])
        maximum = int(inference._root_model.args['max_tokens'])
        lengths = torch.arange(minimum, maximum + 1, device=history.device)
        # Keep the existing publication/context limits while using MB's learned
        # duration classifier rather than asserting that every task takes 32 frames.
        allowed_duration = (lengths >= 6).to(history.dtype)[None]
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(intent_diagnostics(observation_snapshot, root, goal, num_tokens, version))
        if survey_snapshot is not None:
            add_survey_diagnostics(diagnostics,survey_snapshot,root,goal,num_tokens,version,actor_controls)
        if query_path_world_xz is not None:
            diagnostics['terrain_sampling']='predicted_decoded_path_corridor'
            diagnostics['query_proposal_world_xz_m']=query_path_world_xz[0][query_path_valid[0]].detach().cpu().tolist()

    traced_root = {}
    root_model = getattr(inference, '_root_model', None)
    root_backbone = getattr(root_model, 'backbone_net', None)
    global_rep = getattr(inference, 'global_motion_rep', None)
    can_capture_root = (callable(getattr(root_backbone, 'register_forward_hook', None))
                        and callable(getattr(global_rep, 'unnormalize', None)))
    hook = None
    if (diagnostics is not None or tensor_diagnostics is not None) and can_capture_root:
        def capture_root(_module, _inputs, output):
            if isinstance(output, dict):
                values = output.get('pred_global_root_values')
                if isinstance(values, torch.Tensor):
                    traced_root['values'] = values.detach().clone()

        hook = root_backbone.register_forward_hook(capture_root)
    try:
        result, count = inference.predict(
            conditions['global_root_values'], conditions['has_global_root_values'],
            conditions['local_root_values'], conditions['has_local_root_values'],
            conditions['local_poses'], conditions['has_local_poses'], None if predict_duration else num_tokens,
            allowed_pred_num_tokens=allowed_duration,
            global_root_component_mask=conditions['global_root_component_mask'],
            **({'local_root_component_mask': conditions['local_root_component_mask']} if velocity_mode else {}),
            terrain_observation=observation, config=settings, info=info if info is not None else {})
    finally:
        if hook is not None:
            hook.remove()

    if tensor_diagnostics is not None:
        tensor_diagnostics.update(root=traced_root['values'], points=observation_snapshot.points,
                                  valid=observation_snapshot.valid,
                                  world_query_xz=observation_snapshot.world_query_xz)
        if survey_snapshot is not None:
            tensor_diagnostics.update(survey_points=survey_snapshot.points,survey_valid=survey_snapshot.valid,
                survey_world_query_xz=survey_snapshot.world_query_xz,actor_query_controls=actor_controls)

    if diagnostics is not None:
        frames = int(count.item()) * 4
        start, stop = _HISTORY_FRAMES, frames - _DROP_END_FRAMES
        try:
            decoded_root = extract_feature_from_motion_rep(result, global_rep, 'root')
            diagnostics['decoded_world_xyz_m'] = _future_xyz(decoded_root, start, stop)
        except (AttributeError, IndexError, TypeError, ValueError):
            diagnostics['decoded_world_xyz_m'] = []

        normalized_root = traced_root.get('values')
        if normalized_root is not None:
            try:
                predicted_root = global_rep.unnormalize(normalized_root)
                predicted_root = predicted_root.clone()
                if inference._pose_model.args.get('vertical_frame'):
                    from .native_mb_vertical_frame import center_motion_window, observed_terrain_origin, TERRAIN_VERTICAL_FRAME
                    if inference._pose_model.args['vertical_frame'] == TERRAIN_VERTICAL_FRAME:
                        vertical_origin = observed_terrain_origin(root[:, -1, 1], observation)
                    else:
                        _, vertical_origin = center_motion_window(history, global_rep, normalized=False)
                    predicted_root[..., 1] += vertical_origin[:, None]
                # Official online evaluation restores XZ by translation only; yaw is not applied.
                predicted_root[..., [0, 2]] += root[:, :1, [0, 2]]
                diagnostics['root_stage_world_xyz_m'] = _future_xyz(predicted_root, start, stop)
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                diagnostics['root_stage_world_xyz_m'] = []
    return result, count


class NativeTerrainOnlineBackend:
    """Eight generated frames per plan, with history owned by the playback clock."""

    PUBLISH_FRAMES = 8
    DROP_END_FRAMES = 4

    def __init__(self, inference, initial_history, scene, *, num_tokens=8, inference_config=None,
                 reencode_observed_history=True):
        if not 6 <= num_tokens <= 16:
            raise ValueError('Online horizon must be in the trained native range')
        self.inference = inference
        self.history = initial_history.to(inference.device).clone()
        self.scene = scene
        self.num_tokens = num_tokens
        self.inference_config = dict(inference_config or {
            'num_inference_step': 4, 'pose_token_sampling_use_argmax': True,
            'cache_terrain_encoder': True})
        self.last_generation = None
        version = getattr(getattr(inference, '_root_model', None), 'args', {}).get(
            'online_conditioning', ONLINE_PROTOCOL['version'])
        self.condition_contract = online_protocol(version)
        self.condition_contract['buffer_policy'] = (
            'Prefetch existing generation only; fresh generation starts at the fully played boundary')
        self._future_buffer = None
        self._last_published_history = None
        self._last_command = None
        self._debug_future = None
        self._history_observer = None
        self.execution = getattr(inference, '_terrain_execution', None)
        self.runtime_controls = None
        if reencode_observed_history and callable(getattr(inference.global_motion_rep, 'inverse', None)):
            from .native_mb_observed_history import reencode_played_history
            self._history_observer = reencode_played_history
        self.condition_contract['played_history_features'] = (
            'official FK and MotionRep re-encoding' if self._history_observer else 'raw feature diagnostic')

    def observe_history(self, frames, info=None):
        if self.execution is not None and info is None:
            return self.execution.observe(frames)
        if self._history_observer is None:
            return frames.detach().clone()
        return self._history_observer(frames, self.inference.global_motion_rep, info)

    def _publication(self, future):
        if self.execution is not None:
            started = time.perf_counter()
            geometry = self.execution.geometry(future)
            torch.cuda.synchronize(future.device)
            return {'geometry': geometry, 'published_history': geometry['history'],
                    'geometry_ms': (time.perf_counter() - started) * 1000}
        return {'published_history': self.observe_history(future[:, -4:])}

    @staticmethod
    def _advance_debug(debug, frames):
        if not isinstance(debug, dict):
            return debug
        updated = dict(debug)
        for key in ('root_stage_world_xyz_m', 'decoded_world_xyz_m'):
            path = debug.get(key)
            if isinstance(path, list):
                updated[key] = path[frames:]
        # Terrain coordinates intentionally remain the generation-time snapshot.
        updated['observation_lag_frames'] = int(debug.get('observation_lag_frames', 0)) + frames
        return updated

    def configure_runtime(self, values):
        from .terrain_runtime_controls import validate_runtime_controls
        values = validate_runtime_controls(values)
        if values != self.runtime_controls:
            self.num_tokens = values['num_tokens']
            self.inference_config['num_inference_step'] = values['inference_steps']
            self.runtime_controls = values
            self._future_buffer = None
            self._debug_future = None
            self._last_published_history = None
            self._last_command = None

    @torch.inference_mode()
    def predict_plan(self, velocity_xz, heading):
        velocity = torch.as_tensor(velocity_xz).reshape(-1)
        direction = torch.as_tensor(heading).reshape(-1)
        if velocity.shape != (2,) or direction.shape != (1,) or not (
                torch.isfinite(velocity).all() and torch.isfinite(direction).all()):
            raise ValueError('Expected one finite planar command')
        command = (tuple(velocity.tolist()), float(direction[0]))
        compatible_history = (self._last_published_history is not None
            and torch.allclose(self.history, self._last_published_history, atol=1e-5, rtol=0))
        if (command == self._last_command and compatible_history
                and self._future_buffer.shape[1] >= self.PUBLISH_FRAMES):
            future = self._future_buffer[:, :self.PUBLISH_FRAMES].clone()
            self._future_buffer = self._future_buffer[:, self.PUBLISH_FRAMES:].clone()
            self._debug_future = self._advance_debug(self._debug_future, self.PUBLISH_FRAMES)
            publication = self._publication(future)
            self._last_published_history = publication['published_history']
            self.last_generation = {'neural_generation': False, 'reason': 'model_buffer_sample',
                'future_height_source': 'model', 'future_pose_source': 'model',
                'remaining_frames': self._future_buffer.shape[1],
                'inference_config': dict(self.inference_config)}
            if self.runtime_controls:
                self.last_generation['runtime_controls'] = dict(self.runtime_controls)
            return {'planned_motion': future, 'generation_seconds': 0.0,
                    'debug': self._debug_future, **publication}

        # A changed command or history invalidates every trace from the old plan.
        self._future_buffer = None
        self._debug_future = None
        self._last_published_history = None
        self._last_command = None
        self.last_generation = None
        root = extract_feature_from_motion_rep(self.history, self.inference.global_motion_rep, 'root')
        goal = online_command_goal(root, velocity_xz, heading, self.num_tokens, self.condition_contract['version'])
        if self.history.is_cuda:
            torch.cuda.synchronize(self.history.device)
        started = time.perf_counter()
        diagnostics = {}
        query_controls = (self.history.new_tensor([[self.runtime_controls['lookahead_m'],
            self.runtime_controls['lateral_halfwidth_m'],self.runtime_controls['lookahead_gain'],
            self.runtime_controls['lookahead_base_m']]]) if self.runtime_controls else None)
        if self.execution is not None and (self.runtime_controls is None or self.runtime_controls['inference_steps']==4):
            from .soma_terrain_execution import SETTINGS
            if self.runtime_controls is None and (self.num_tokens != 8 or self.inference_config != SETTINGS):
                raise ValueError('Compiled execution configuration differs from the verified native contract')
            if self.runtime_controls is None:
                prediction, count = self.execution.predict(self.history, goal, diagnostics)
            else:
                prediction, count = self.execution.predict(self.history, goal, diagnostics,
                    num_tokens=self.num_tokens,query_controls=query_controls,
                    path_refinements=self.runtime_controls['path_refinements'])
        else:
            prediction, count = predict_planar_terrain(self.inference, self.history, goal, self.scene,
                                                       num_tokens=self.num_tokens, config=self.inference_config,
                                                       diagnostics=diagnostics, query_controls=query_controls)
            if self.runtime_controls:
                for _ in range(self.runtime_controls['path_refinements']):
                    proposal=prediction[:,4:self.num_tokens*4-4,[0,2]].detach()
                    prediction,count=predict_planar_terrain(self.inference,self.history,goal,self.scene,
                        num_tokens=self.num_tokens,config=self.inference_config,diagnostics=diagnostics,
                        query_controls=query_controls,query_path_world_xz=proposal,
                        query_path_valid=torch.ones(proposal.shape[:-1],dtype=torch.bool,device=proposal.device))
        if self.history.is_cuda:
            torch.cuda.synchronize(self.history.device)
        elapsed = time.perf_counter() - started
        frames = int(count.item()) * 4
        if (frames < _HISTORY_FRAMES + self.PUBLISH_FRAMES + _DROP_END_FRAMES
                or not torch.isfinite(prediction[:, :frames]).all()):
            raise ValueError('Native MB produced an invalid online plan')
        self._future_buffer = prediction[:, _HISTORY_FRAMES:frames - _DROP_END_FRAMES].clone()
        if self.runtime_controls:
            self._future_buffer = self._future_buffer[:, :self.runtime_controls['refresh_frames']].clone()
            diagnostics['runtime_controls'] = dict(self.runtime_controls)
        self._debug_future = diagnostics
        future = self._future_buffer[:, :self.PUBLISH_FRAMES].clone()
        self._future_buffer = self._future_buffer[:, self.PUBLISH_FRAMES:].clone()
        publication = self._publication(future)
        self._last_published_history = publication['published_history']
        self._last_command = command
        self.last_generation = {'neural_generation': True, 'predicted_frames': frames,
                                'future_height_source': 'model', 'future_pose_source': 'model',
                                'remaining_frames': self._future_buffer.shape[1],
                                'inference_config': dict(self.inference_config)}
        if self.runtime_controls:
            self.last_generation['runtime_controls'] = dict(self.runtime_controls)
        return {'planned_motion': future, 'generation_seconds': elapsed,
                'debug': self._debug_future, **publication}

    @torch.inference_mode()
    def prepare_next_plan(self, velocity_xz, heading, observed_history, expected_published_history,
                          overlap_frames=4):
        """Prefetch an existing continuation, or defer inference to the played boundary."""
        if overlap_frames not in (4, 6):
            raise ValueError('Prefetch overlap must match a two/four-frame played prefix')
        velocity = torch.as_tensor(velocity_xz).reshape(-1)
        direction = torch.as_tensor(heading).reshape(-1)
        if velocity.shape != (2,) or direction.shape != (1,) or not (
                torch.isfinite(velocity).all() and torch.isfinite(direction).all()):
            raise ValueError('Expected one finite planar command')
        command = (tuple(float(value) for value in velocity.tolist()), float(direction[0]))

        observed = torch.as_tensor(observed_history, device=self.history.device, dtype=self.history.dtype)
        expected = torch.as_tensor(expected_published_history, device=self.history.device,
                                   dtype=self.history.dtype)
        if (observed.shape != self.history.shape or observed.shape[1] != _HISTORY_FRAMES
                or expected.shape != observed.shape or not torch.isfinite(observed).all()
                or not torch.isfinite(expected).all()):
            raise ValueError('Midpoint preparation requires finite four-frame histories')

        can_reuse = (
            command == self._last_command
            and isinstance(self._future_buffer, torch.Tensor)
            and self._future_buffer.ndim == 3
            and self._future_buffer.shape[0] == 1
            and self._future_buffer.shape[1] >= self.PUBLISH_FRAMES
            and self._last_published_history is not None
            and torch.allclose(self._last_published_history, expected, atol=1e-5, rtol=0)
        )
        if can_reuse:
            future = self._future_buffer[:, :self.PUBLISH_FRAMES].clone()
            publication = self._publication(future)
            remaining = self._future_buffer[:, self.PUBLISH_FRAMES:].clone()
            generation = {
                'neural_generation': False,
                'reason': 'model_buffer_sample',
                'future_height_source': 'model',
                'future_pose_source': 'model',
                'remaining_frames': remaining.shape[1],
                'inference_config': dict(self.inference_config),
            }
            return {
                'planned_motion': future,
                'remaining_motion': remaining,
                'generation_seconds': 0.0,
                'debug': self._advance_debug(self._debug_future, self.PUBLISH_FRAMES),
                'generation': generation,
                'command': command,
                'expected_prior_history': expected.clone(),
                **publication,
            }

        # A new forecast from this earlier prefix can disagree with the six/four
        # immutable frames still playing. Dropping overlap aligns time, not pose.
        return {
            'deferred': True,
            'reason': 'fresh_inference_requires_played_boundary',
            'command': command,
            'expected_prior_history': expected.clone(),
        }

    @torch.inference_mode()
    def commit_prepared_plan(self, candidate):
        """Install prepared future buffers after its predecessor was fully played."""
        if not isinstance(candidate, dict):
            raise ValueError('Prepared plan must be a candidate mapping')
        future = candidate.get('planned_motion')
        remaining = candidate.get('remaining_motion')
        expected = candidate.get('expected_prior_history')
        if (not isinstance(future, torch.Tensor) or future.ndim != 3
                or future.shape != (1, self.PUBLISH_FRAMES, self.history.shape[2])
                or not isinstance(remaining, torch.Tensor) or remaining.ndim != 3
                or remaining.shape[0] != 1 or remaining.shape[2] != self.history.shape[2]
                or not isinstance(expected, torch.Tensor) or expected.shape != self.history.shape
                or not torch.isfinite(future).all() or not torch.isfinite(remaining).all()
                or not torch.isfinite(expected).all()):
            raise ValueError('Prepared plan violated the native publication contract')
        if not torch.allclose(self.history, expected, atol=1e-5, rtol=0):
            raise ValueError('Prepared plan no longer follows the actual published history')
        generation = candidate.get('generation')
        if not isinstance(generation, dict):
            raise ValueError('Prepared plan has no generation metadata')
        if generation.get('neural_generation') is not False or generation.get('reason') != 'model_buffer_sample':
            raise ValueError('Only a continuation from the existing generation may be prefetched')

        self._future_buffer = remaining.clone()
        self._last_published_history = candidate.get('published_history')
        if self._last_published_history is None:
            self._last_published_history = self.observe_history(future[:, -_HISTORY_FRAMES:])
        self._last_command = candidate.get('command')
        self._debug_future = candidate.get('debug')
        self.last_generation = dict(generation)
