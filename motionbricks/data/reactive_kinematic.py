"""Played-frame feedback and replaceable native motion forecasts."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import numpy as np


SCHEMA = 'motionbricks.reactive_kinematic.v4'
FUTURE_FRAMES = 24
STEP_CHOICES = (4, 2, 1)
MAX_PREFIX_FRAMES = 6


@dataclass(frozen=True)
class GenerationRequest:
    episode_id: int
    origin_frame: int
    history: np.ndarray
    input_seq: int
    velocity: tuple
    heading: float
    client_input_ms: float
    input_received_ns: int
    requested_ns: int
    step_frames: int
    played_frame: int
    played_history: np.ndarray
    prefix_plan_ids: tuple


class ReactivePlayback:
    """Only acknowledged rendered frames enter the four-frame motion history."""

    def __init__(self, seed, *, episode_id=1, step_frames=4, max_speed=12.):
        seed = np.asarray(seed, dtype=np.float32)
        if seed.ndim != 2 or len(seed) != 4 or not np.isfinite(seed).all():
            raise ValueError('Four finite native seed frames required')
        self.history = seed.copy()
        self.episode_id, self.frame, self.started = episode_id, 0, False
        self.max_speed = max_speed
        self.configure(step_frames)
        self.command = None
        self.last_dispatch_frame = self.last_dispatch_input = None
        self.plans = {}
        self.plan_id = 0
        self.accepted = set()
        self.pending_plan = None
        self.committed_until = 0
        self.lead_frames = 3
        self.delivery_floor = 0
        self.latency_ms = []
        self.pending_lead = 0

    def configure(self, step_frames):
        if type(step_frames) is not int or step_frames not in STEP_CHOICES:
            raise ValueError('Execution interval must be4,2or1 native frames')
        self.step_frames = step_frames

    def observe_latency(self, milliseconds):
        if not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError('Measured generation latency must be finite and nonnegative')
        self.latency_ms = (self.latency_ms+[milliseconds])[-12:]
        # Include browser scheduling margin independently of request frequency.
        measured = math.ceil((max(self.latency_ms)+18.)*30/1000)-1
        self.lead_frames = min(MAX_PREFIX_FRAMES,max(0,measured,self.delivery_floor))

    def receive_intent(self, value, received_ns=None):
        seq, velocity, heading = value['seq'], np.asarray(value['velocity_xz'], float), value['heading']
        client_ms = value['client_input_ms']
        if (type(seq) is not int or seq < 1 or
                (self.command is not None and seq <= self.command['seq'])):
            raise ValueError('Intent sequence must increase')
        if (velocity.shape != (2,) or not np.isfinite(velocity).all()
                or np.linalg.norm(velocity) > self.max_speed+1e-6
                or type(heading) not in (int, float) or not np.isfinite(heading)
                or type(client_ms) not in (int, float) or not np.isfinite(client_ms)):
            raise ValueError('Finite bounded movement and heading required')
        self.command = dict(seq=seq, velocity=tuple(velocity), heading=float(heading),
            client_input_ms=float(client_ms), received_ns=received_ns or time.perf_counter_ns())

    def due(self):
        return self.started and self.command is not None and self.pending_plan is None and (
            self.last_dispatch_frame is None
            or self.frame-self.last_dispatch_frame >= self.step_frames)

    def capture(self):
        if not self.due():
            return None
        command = self.command
        origin = max(self.frame,self.committed_until)
        if self.accepted:
            origin = max(origin,self.frame+self.lead_frames)
            # Input arrivals and a shorter compute lead must not truncate the
            # selected execution interval of an already accepted forecast.
            origin = max(origin, max(self.plans[key][0] for key in self.accepted) + self.step_frames)
        prefix, sources = [], []
        for frame in range(self.frame+1,origin+1):
            plan_id, sample = self.scheduled_frame(frame)
            prefix.append(sample)
            sources.append(plan_id)
        history = (np.concatenate((self.history,np.asarray(prefix)))[-4:]
            if prefix else self.history.copy())
        request = GenerationRequest(self.episode_id, origin, history, command['seq'],
            command['velocity'], command['heading'], command['client_input_ms'], command['received_ns'],
            time.perf_counter_ns(), self.step_frames,self.frame,self.history.copy(),tuple(sources))
        request.history.flags.writeable = False
        request.played_history.flags.writeable = False
        self.committed_until = origin
        self.last_dispatch_frame, self.last_dispatch_input = self.frame, command['seq']
        return request

    def scheduled_frame(self, frame):
        choices = [key for key in self.accepted if self.plans[key][0]<frame<=self.plans[key][0]+FUTURE_FRAMES]
        if not choices:
            raise ValueError('No accepted native forecast covers the committed prefix')
        plan_id = max(choices,key=lambda key:(self.plans[key][0],key))
        origin,motion = self.plans[plan_id]
        return plan_id,motion[frame-origin-1]

    def publish(self, request, motion):
        if request.episode_id != self.episode_id:
            raise ValueError('Forecast belongs to another episode')
        motion = np.asarray(motion, dtype=np.float32)
        if motion.shape != (FUTURE_FRAMES, self.history.shape[1]) or not np.isfinite(motion).all():
            raise ValueError('Forecast requires24 genuine native future frames')
        if request.origin_frame < self.frame:
            self.last_dispatch_frame = None
            return None
        self.plan_id += 1
        saved = motion.copy()
        saved.flags.writeable = False
        self.plans[self.plan_id] = (request.origin_frame, saved)
        self.pending_plan = self.plan_id
        self.pending_lead = request.origin_frame-request.played_frame
        return self.plan_id

    def accept(self, episode_id, plan_id, first_frame):
        if episode_id != self.episode_id or plan_id != self.pending_plan:
            raise ValueError('Acceptance must confirm the pending forecast of this episode')
        origin,_ = self.plans[plan_id]
        if first_frame != origin+1 or first_frame <= self.frame:
            raise ValueError('Accepted forecast must start at its unplayed handoff frame')
        self.accepted.add(plan_id)
        self.pending_plan = None

    def reject(self, episode_id, plan_id):
        if episode_id != self.episode_id or plan_id != self.pending_plan:
            raise ValueError('Rejection must identify the pending forecast')
        del self.plans[plan_id]
        self.pending_plan = None
        self.delivery_floor = min(MAX_PREFIX_FRAMES,max(self.delivery_floor,self.pending_lead+1))
        self.lead_frames = max(self.lead_frames,self.delivery_floor)
        self.last_dispatch_frame = self.frame if self.accepted else None
        self.last_dispatch_input = self.command['seq']

    def acknowledge(self, episode_id, frame, plan_id):
        if episode_id != self.episode_id or type(frame) is not int or frame != self.frame+1:
            raise ValueError('Playback must acknowledge the next frame of this episode')
        if type(plan_id) is not int or plan_id not in self.plans:
            raise ValueError('Playback refers to an unknown forecast')
        origin, motion = self.plans[plan_id]
        index = frame-origin-1
        if not 0 <= index < len(motion):
            raise ValueError('Played frame lies outside its generated forecast')
        expected,_ = self.scheduled_frame(frame)
        if plan_id != expected:
            raise ValueError('Played frame differs from the accepted handoff schedule')
        self.history = np.concatenate((self.history[1:], motion[index:index+1]))
        self.frame = frame
        self.plans = {key:value for key,value in self.plans.items() if value[0]+len(value[1]) > frame}
        self.accepted.intersection_update(self.plans)


class ReactiveNativeEngine:
    """One unchanged8-token Root/Pose prediction, with official SOMA geometry."""

    def __init__(self, runtime, query_proofs):
        self.runtime = runtime
        self.query_proofs = query_proofs
        self.geometry = None
        self.geometry_proof = None
        self.path = None
        self.scene_binding = None

    def _install_scene_query(self):
        from motionbricks.mb_sonic.geometry_execution import install_scene_field
        from motionbricks.mb_sonic.reference import file_sha256

        runtime = self.runtime
        entry = self.query_proofs[runtime.current_map]
        path = Path(entry['path'])
        if file_sha256(path) != entry['sha256']:
            raise ValueError('Bound sphere query proof changed')
        runner = runtime.inference._terrain_execution
        previous = runner.field
        field, proof = install_scene_field(runner.controller,runner.scene,
            backend='warp_bvh_sphere_exact',source_sha256=runtime.acceptance_receipt['scene_sha256'],
            proof_path=path,scene_id=runtime.current_map)
        runner.field = field
        return previous,proof

    def _prepare_geometry(self, sample):
        import torch
        from scripts.soma_a100_trial_server import A100TrialRuntime
        from .soma_static_execution import trace_fixed_call
        from .soma_cuda_graph import StaticCudaCall

        runtime, inference = self.runtime, self.runtime.inference
        def native(frames):
            return A100TrialRuntime.skin_rig_geometry(runtime, frames, inference)

        self.geometry = StaticCudaCall(trace_fixed_call(native, (sample,), shared_cuda_constants=True))
        checks = []
        for offset in (0., .13):
            frames = sample.clone()
            frames[..., 0] += offset
            expected, actual = native(frames), self.geometry(frames)
            for key in ('rig', 'joints'):
                torch.testing.assert_close(actual[key], expected[key], atol=1e-5, rtol=1e-5)
            legacy_rig = torch.cat([inference._terrain_execution.geometry(
                frames[:,start:start+8].contiguous())['rig'] for start in (0,8,16)])
            torch.testing.assert_close(actual['rig'],legacy_rig,atol=1e-5,rtol=1e-5)
            pose = inference.global_motion_rep.inverse(frames, False, return_all=False,
                joint_positions_from='global_rot_data')
            vertices = runtime.skinner.skin(pose['global_joint_rots'][0], pose['posed_joints'][0], rot_is_global=True)
            actual_vertices = runtime.skinner.lbs(actual['rig'])
            torch.testing.assert_close(actual_vertices, vertices, atol=1e-5, rtol=1e-5)
            checks.append(dict(offset_m=offset, vertices=int(vertices.shape[-2]),
                skin_max_abs_m=float((actual_vertices-vertices).abs().max()),
                legacy_rig_max_abs=float((actual['rig']-legacy_rig).abs().max()),
                joint_max_abs_m=float((actual['joints']-expected['joints']).abs().max())))
        self.geometry_proof = dict(passed=True, frames=24, checks=checks)

    def prepare(self):
        import torch
        from .native_mb_online_live import online_command_goal
        from .reactive_path_execution import ReactivePathExecution

        with torch.inference_mode():
            runtime = self.runtime
            binding = (runtime.current_map,runtime.scene_revision)
            if self.path is not None and binding == self.scene_binding:
                return dict(geometry=self.geometry_proof,path=self.path.proof)
            runner = runtime.inference._terrain_execution
            comparison_field,query_proof = self._install_scene_query()
            seed = runtime.terrain_initial_history
            rng = torch.cuda.get_rng_state(seed.device)
            goal = online_command_goal(seed[..., :5], [0.,0.], 0., 8, runner.protocol_version)
            try:
                if self.geometry is None:
                    result = runner._prediction(seed, goal)
                    self._prepare_geometry(result['motion'][:, 4:28].contiguous())
                candidate = ReactivePathExecution(runtime,self.geometry.function,comparison_field=comparison_field)
                candidate.prepare()
                candidate.proof['scene_query_backend'] = query_proof
                self.path,self.scene_binding = candidate,binding
            finally:
                torch.cuda.set_rng_state(rng, seed.device)
        return dict(geometry=self.geometry_proof,path=self.path.proof)

    def generate(self, request):
        import torch
        from .soma_lab import pack_rig_frames

        started = time.perf_counter_ns()
        runtime = self.runtime
        if self.scene_binding != (runtime.current_map,runtime.scene_revision):
            raise ValueError('Current scene requires reactive execution preparation')
        with torch.inference_mode():
            raw = torch.from_numpy(request.history.copy())[None].to(runtime.inference.device)
            controls = raw.new_tensor([[*request.velocity,request.heading]])
            begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            result = self.path(raw,controls)
            packed = torch.cat((result['motion'][:,4:28].reshape(-1),result['rig'].reshape(-1),
                result['joints'].reshape(-1),(result['motion'][0,3,:3]-result['history'][0,3,:3]).reshape(3),
                result['count'].to(raw.dtype).reshape(1),
                result['finite'].to(raw.dtype).reshape(1),result['observation_known'].to(raw.dtype).reshape(1),
                result['position_error'].reshape(1),result['rotation_error'].reshape(1)))
            end.record()
            payload = packed.cpu().numpy()
            finished = time.perf_counter_ns()
            if (not np.isfinite(payload).all() or not np.array_equal(payload[-5:-2],[8.,1.,1.])
                    or np.any(payload[-2:] > 2e-5)):
                raise ValueError('Reactive forecast checks failed: '
                    f'count={payload[-5]}, finite={payload[-4]}, known={payload[-3]}, '
                    f'history_position_error={payload[-2]}, history_rotation_error={payload[-1]}')
            width = request.history.shape[1]
            motion = payload[:24*width].reshape(24,width)
            rig = payload[24*width:24*width+24*77*16].reshape(24,77,4,4)
            joints = payload[24*width+24*77*16:-8].reshape(24,30,3)
        timing = dict(history_ms=0.,prediction_ms=(finished-started)/1e6,
            geometry_ms=0.,compute_ms=(finished-started)/1e6,
            component_scope='combined_history_queries_Root_Pose_codec_official_rig_and_single_CPU_transfer',
            fused_cuda_ms=begin.elapsed_time(end),geometry_included_in_inference=True,
            decoded_context_root_delta_m=payload[-8:-5].tolist(),
            queue_ms=(started-request.requested_ns)/1e6,
            history_source='played_frames_and_committed_native_prefix', history_frame=request.origin_frame,
            played_history_frame=request.played_frame,committed_prefix_frames=request.origin_frame-request.played_frame,
            prefix_plan_ids=list(request.prefix_plan_ids),
            played_history_sha256=hashlib.sha256(request.played_history.tobytes()).hexdigest(),
            history_sha256=hashlib.sha256(request.history.tobytes()).hexdigest(),
            input_seq=request.input_seq, input_received_ns=request.input_received_ns,
            requested_ns=request.requested_ns, completed_ns=finished,
            requested_step_frames=request.step_frames)
        return dict(motion=motion, joints=joints, rig=rig, timing=timing,
            payload=pack_rig_frames(motion[:,:3],rig))
