"""Native SOMA plans and lossless float32 geometry transport for a browser viewer."""

import math
import time

import numpy as np
import torch


def validate_command(message, previous_request_id, max_speed_mps=3.):
    request_id = message.get("request_id")
    if (not isinstance(request_id, int) or isinstance(request_id, bool)
            or request_id != previous_request_id + 1):
        raise ValueError("Expected the next sequential request_id")
    velocity = np.asarray(message.get("velocity_xz"), dtype=np.float64)
    if velocity.shape != (2,) or not np.isfinite(velocity).all():
        raise ValueError("velocity_xz must be two finite values")
    if np.linalg.norm(velocity) > max_speed_mps+1e-6:
        raise ValueError('Movement intent exceeds endpoint speed range')
    heading = message.get("heading")
    if (not isinstance(heading, (int, float)) or isinstance(heading, bool)
            or not math.isfinite(heading)):
        raise ValueError("heading must be finite radians")
    return request_id, velocity.tolist(), float(heading)


def pack_geometry_frames(roots, vertices):
    """Each frame is root XYZ followed by all official skinned vertex XYZ values."""
    roots = np.asarray(roots)
    vertices = np.asarray(vertices)
    if (roots.ndim != 2 or roots.shape[1] != 3 or vertices.ndim != 3
            or vertices.shape[0] != len(roots) or vertices.shape[2] != 3
            or len(roots) < 1 or vertices.shape[1] < 1):
        raise ValueError("Expected matching [T,3] roots and [T,V,3] vertices")
    if not np.isfinite(roots).all() or not np.isfinite(vertices).all():
        raise ValueError("Geometry contains non-finite values")
    packed = np.concatenate((roots, vertices.reshape(len(roots), -1)), axis=1).astype("<f4")
    if not np.isfinite(packed).all():
        raise ValueError("Geometry exceeds float32 range")
    return packed.tobytes(order="C")


def pack_rig_frames(roots, rig):
    roots, rig = np.asarray(roots), np.asarray(rig)
    if (roots.ndim != 2 or roots.shape[1] != 3 or rig.shape != (len(roots), 77, 4, 4)
            or not np.isfinite(roots).all() or not np.isfinite(rig).all()):
        raise ValueError('Expected finite SOMA77 world rig matrices aligned to roots')
    packed = np.concatenate((roots, rig.reshape(len(roots), -1)), axis=1).astype('<f4')
    if not np.isfinite(packed).all():
        raise ValueError('Rig exceeds float32 range')
    return packed.tobytes(order='C')


class SomaLabSession:
    """An independent native backend with playback-acknowledged history."""

    def __init__(self, runtime, mode="live", spawn_id=None):
        self.runtime = runtime
        if mode not in {"live", "terrain"}:
            raise ValueError("Unknown live backend mode")
        self.mode = mode
        self.spawn_id = spawn_id
        self.backend = runtime.create_backend(mode,spawn_id=spawn_id) if spawn_id is not None else runtime.create_backend(mode)
        self.request_id = 0
        self.pending_playback = None
        self.scene_revision = getattr(runtime,'scene_revision',None)
        track=getattr(runtime,'track_session',None)
        if callable(track):track(self)

    def _check_scene(self):
        if self.scene_revision!=getattr(self.runtime,'scene_revision',None):
            raise ValueError('Map changed in another session; reload this view')

    def _observed_history(self, frames):
        observer = getattr(self.backend, 'observe_history', None)
        return observer(frames) if callable(observer) else frames.detach().clone()

    def _plan_response(self, request_id, velocity, heading, result, *, prefetched=False,
                       source_request_id=None, forecast_alignment_lag_frames=None,
                       generation=None):
        future = result["planned_motion"]
        expected = self.backend.PUBLISH_FRAMES
        if (future.ndim != 3 or future.shape[0] != 1 or future.shape[1] != expected
                or not torch.isfinite(future).all()):
            raise ValueError("Native backend violated its published-frame contract")
        started = time.perf_counter()
        inference = getattr(self.backend, "inference", None)
        if inference is None:
            inference = getattr(getattr(self.backend, "controller", None), "inference", None)
        joints = None
        debug_skin = getattr(self.runtime, "skin_features_with_joints", None)
        geometry = result.get('geometry')
        if geometry is None and getattr(self.runtime,'rig_transport',False):
            geometry=self.runtime.skin_rig_geometry(future,inference)
        if geometry is not None:
            roots = future[0, :, :3].cpu().numpy()
            rig = geometry['rig'].cpu().numpy()
            joints = geometry['joints'].cpu().numpy()
        elif callable(debug_skin):
            roots, vertices, joints = debug_skin(future, inference)
        elif inference is None:
            roots, vertices = self.runtime.skin_features(future)
        else:
            roots, vertices = self.runtime.skin_features(future, inference)
        if geometry is not None:
            payload = pack_rig_frames(roots, rig)
            vertex_count, stride = self.runtime.vertex_count, 3 + 77 * 16
        else:
            payload = pack_geometry_frames(roots, vertices)
            vertex_count, stride = vertices.shape[1], 3 + vertices.shape[1] * 3
        skin_ms = (time.perf_counter() - started) * 1000 + result.get('geometry_ms', 0.)
        metadata = {
            "type": "plan", "request_id": request_id, "fps": self.runtime.fps,
            "frame_count": len(roots), "vertex_count": vertex_count,
            "frame_stride_floats": stride,
            "encoding": "float32-le", "inference_ms": result["generation_seconds"] * 1000,
            "skin_ms": skin_ms, "style": getattr(self.backend, "last_style", None),
            "generation": (generation if generation is not None else
                           getattr(self.backend, "last_generation", None)),
            "mode": self.mode,
        }
        if geometry is not None:
            metadata.update(geometry_format='soma77-rig-v1', rig_count=77,
                            influences_per_vertex=self.runtime.skinner.lbs_indices.shape[-1])
        # Diagnostics are a separate read-only channel; binary geometry stays identical.
        debug = dict(result.get("debug") or {})
        debug.setdefault("version", "soma_motion_debug_v1")
        debug.setdefault("decoded_world_xyz_m", np.asarray(roots).tolist())
        debug["published_frame_count"] = expected
        debug["played_command"] = {"velocity_xz": velocity, "heading_world_rad": heading}
        if joints is not None:
            debug["joints_world_xyz_m"] = np.asarray(joints).tolist()
        if geometry is not None and 'bone_sdf_distance' in geometry:
            debug['bone_sdf'] = {
                'role': 'diagnostic_only_not_model_input',
                'signed_distance_m': geometry['bone_sdf_distance'].detach().cpu().tolist(),
                'valid': geometry['bone_sdf_valid'].detach().cpu().tolist(),
                'joint_centers_not_skin': True,
            }
        metadata["debug"] = debug
        if self.mode == "terrain":
            metadata["quality_status"] = "experimental_not_approved"
            metadata["display_only"] = True
            metadata["actuating"] = False
        if prefetched:
            metadata["prefetched"] = True
            metadata["source_request_id"] = source_request_id
            metadata["history_source"] = ('immutable_published_model_frames'
                if (generation or {}).get('prefetch_policy')=='committed-model-history-v1' else 'played_midpoint')
            if forecast_alignment_lag_frames is not None:
                metadata["forecast_alignment_lag_frames"] = forecast_alignment_lag_frames
        return metadata, payload, future

    @torch.inference_mode()
    def plan(self, message):
        self._check_scene()
        if self.mode == "terrain" and self.pending_playback is not None:
            raise ValueError("The previous plan must be fully played before requesting another")
        if message.get("mode", self.mode) != self.mode:
            raise ValueError("Plan mode differs from the active backend; reset the session first")
        request_id, velocity, heading = validate_command(message,self.request_id,
            getattr(self.runtime,'max_speed_mps',3.))
        if 'runtime_controls' in message:
            if not getattr(self.runtime,'allow_runtime_controls',False):
                raise ValueError('This endpoint does not enable runtime controls')
            self.backend.configure_runtime(message['runtime_controls'])
        if 'speed_governor' in message:
            self.backend.configure_governor(message['speed_governor'])
        if 'planning_policy' in message:
            configure=getattr(self.backend,'configure_policy',None)
            if not callable(configure):raise ValueError('This backend does not expose planning policies')
            configure(message['planning_policy'])
        prior_history = self.backend.history.detach().clone() if self.mode == 'terrain' else None
        result = self.backend.predict_plan([velocity], [heading])
        future = result["planned_motion"]
        expected = self.backend.PUBLISH_FRAMES
        metadata, payload, future = self._plan_response(request_id, velocity, heading, result)
        if 'client_timing' in message:
            metadata['client_timing'] = message['client_timing']
        if self.mode == "terrain":
            # The playback clock, not plan generation, advances terrain history.
            self.pending_playback = {
                "request_id": request_id,
                "frame_count": expected,
                "history": (result['published_history'] if 'published_history' in result
                            else self._observed_history(future[:, -4:])),
                "motion": future.detach().clone(),
                "prior_history": prior_history,
                "governor_state": result.get('governor_state_after'),
            }
        else:
            self.backend.history = future[:, -4:].detach().clone()
        self.request_id = request_id
        return metadata, payload

    @torch.inference_mode()
    def prefetch(self, message):
        self._check_scene()
        if self.mode != "terrain":
            raise ValueError("Midpoint prefetch is only used by Terrain Live")
        if message.get("type") != "prefetch" or message.get("mode") != "terrain":
            raise ValueError("Midpoint prefetch requires the Terrain Live prefetch message")
        pending = self.pending_playback
        if pending is None:
            raise ValueError("There is no published plan available for midpoint prefetch")
        if pending.get("prepared") is not None:
            raise ValueError("A prefetched future is already awaiting playback")
        source_request_id = message.get("source_request_id")
        if (not isinstance(source_request_id, int) or isinstance(source_request_id, bool)
                or source_request_id != pending["request_id"]):
            raise ValueError("Midpoint prefetch source does not match the pending plan")
        played_frame_count = message.get("played_frame_count")
        if (not isinstance(played_frame_count, int) or isinstance(played_frame_count, bool)
                or played_frame_count not in ((0,2,4) if getattr(self.backend,'committed_history_prefetch',False) else (2,4))):
            raise ValueError("Prefetch frame count differs from the backend policy")
        request_id, velocity, heading = validate_command(message,self.request_id,
            getattr(self.runtime,'max_speed_mps',3.))
        future_settings=({'speed_governor':message['speed_governor']} if 'speed_governor' in message else {})
        if 'planning_policy' in message:
            if not callable(getattr(self.backend,'configure_policy',None)):
                raise ValueError('This backend does not expose planning policies')
            future_settings['planning_policy']=message['planning_policy']
        if 'runtime_controls' in message:
            from .terrain_runtime_controls import validate_runtime_controls
            validator=getattr(self.backend,'validate_runtime_controls',validate_runtime_controls)
            requested = validator(message['runtime_controls'])
            if requested != getattr(self.backend,'runtime_controls',None):
                accepts=getattr(self.backend,'can_prefetch_runtime',None)
                if callable(accepts) and accepts(requested):
                    future_settings['runtime_controls']=requested
                else:
                    self.request_id = request_id
                    return {'type':'prefetch_deferred','request_id':request_id,
                        'source_request_id':source_request_id,'reason':'controls_apply_at_played_boundary'},None
        motion = pending.get("motion")
        if (not isinstance(motion, torch.Tensor) or motion.ndim != 3 or motion.shape[0] != 1
                or motion.shape[1] != pending["frame_count"] or motion.shape[1] != 8):
            raise ValueError("Pending plan does not contain the required eight published frames")
        prepare = getattr(self.backend, "prepare_next_plan", None)
        commit = getattr(self.backend, "commit_prepared_plan", None)
        if not callable(prepare) or not callable(commit):
            raise ValueError("Terrain backend does not support midpoint prefetch")

        # This snapshot is factual playback history, even if subsequent inference fails.
        prior_history = pending['prior_history']
        observed_history = (prior_history.detach().clone() if played_frame_count==0 else self._observed_history(
            torch.cat((prior_history, motion[:, :played_frame_count]), dim=1)[:, -4:]))
        self.backend.history = observed_history.clone()
        try:
            candidate = prepare(velocity, heading, observed_history, pending["history"],
                                overlap_frames=8 - played_frame_count,**future_settings)
            if not isinstance(candidate, dict):
                raise ValueError("Native backend returned an invalid prefetched candidate")
            if candidate.get('deferred') is True:
                self.request_id = request_id
                return {'type': 'prefetch_deferred', 'request_id': request_id,
                        'source_request_id': source_request_id,
                        'reason': 'fresh_inference_requires_played_boundary'}, None
            remaining = candidate.get("remaining_motion")
            prior_history = candidate.get("expected_prior_history")
            if (not isinstance(remaining, torch.Tensor) or remaining.ndim != 3
                    or remaining.shape[0] != 1
                    or remaining.shape[2] != motion.shape[2]
                    or not torch.isfinite(remaining).all()
                    or not isinstance(prior_history, torch.Tensor)
                    or prior_history.shape != pending["history"].shape
                    or not torch.allclose(prior_history, pending["history"], atol=1e-5, rtol=0)
                    or not isinstance(candidate.get("generation"), dict)):
                raise ValueError("Native backend returned an incomplete prefetched candidate")
            result = {
                "planned_motion": candidate.get("planned_motion"),
                "generation_seconds": candidate.get("generation_seconds"),
                "debug": candidate.get("debug"),
            }
            if 'geometry' in candidate:
                result['geometry'] = candidate['geometry']
                result['geometry_ms'] = candidate.get('geometry_ms', 0.)
            metadata, payload, future = self._plan_response(
                request_id, velocity, heading, result, prefetched=True,
                source_request_id=source_request_id,
                forecast_alignment_lag_frames=candidate.get("forecast_alignment_lag_frames"),
                generation=candidate.get("generation"))
            if 'client_timing' in message:
                metadata['client_timing'] = message['client_timing']
        except Exception:
            pending.pop("prepared", None)
            raise

        pending["prepared"] = {
            "request_id": request_id,
            "frame_count": self.backend.PUBLISH_FRAMES,
            "history": (candidate['published_history'] if 'published_history' in candidate
                        else self._observed_history(future[:, -4:])),
            "motion": future.detach().clone(),
            "candidate": candidate,
            "governor_state": candidate.get('governor_state_after'),
        }
        self.request_id = request_id
        return metadata, payload

    def acknowledge_played(self, message):
        self._check_scene()
        if self.mode != "terrain":
            raise ValueError("Playback acknowledgements are only used by Terrain Live")
        pending = self.pending_playback
        if pending is None:
            raise ValueError("There is no generated plan awaiting playback acknowledgement")
        request_id = message.get("request_id")
        frame_count = message.get("frame_count")
        if (not isinstance(request_id, int) or isinstance(request_id, bool)
                or request_id != pending["request_id"]):
            raise ValueError("Playback acknowledgement does not match the pending plan")
        if (not isinstance(frame_count, int) or isinstance(frame_count, bool)
                or frame_count != pending["frame_count"]):
            raise ValueError("Playback acknowledgement must cover the entire published block")
        self.backend.history = pending["history"]
        if pending.get('governor_state') is not None:
            self.backend.commit_governor_state(pending['governor_state'])
        prepared = pending.get("prepared")
        if prepared is None:
            self.pending_playback = None
        else:
            commit = getattr(self.backend, "commit_prepared_plan", None)
            if not callable(commit):
                raise ValueError("Terrain backend cannot commit the prefetched future")
            commit(prepared["candidate"])
            self.pending_playback = {
                "request_id": prepared["request_id"],
                "frame_count": prepared["frame_count"],
                "history": prepared["history"],
                "motion": prepared["motion"],
                "prior_history": self.backend.history.detach().clone(),
                "governor_state": prepared.get('governor_state'),
            }
        return {"type": "played", "request_id": request_id, "frame_count": frame_count}
