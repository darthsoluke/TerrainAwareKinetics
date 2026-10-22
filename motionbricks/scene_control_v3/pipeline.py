"""Shared eager capacity path with one pending, display-ACK-committed frame."""

from dataclasses import dataclass

import torch

from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix

from .geometry import encode_scene_features
from .state import ContactLifecycleState, contact_features, observe_contacts


QUERY_POLICY = "committed_root_preview_256_128_128_v2"


def root_preview_to_world(history, preview_ego):
    yaw = torch.atan2(history[:, -1, 4], history[:, -1, 3])
    rotation = angle_to_Y_rotation_matrix(yaw)
    return history[:, -1:, :3] + preview_ego @ rotation.transpose(-1, -2)


@dataclass(frozen=True)
class CapacityState:
    history: torch.Tensor
    hidden: torch.Tensor
    joints: torch.Tensor
    body_points: torch.Tensor
    object_pose: torch.Tensor
    contact_lifecycle: ContactLifecycleState
    previous_contact_logits: torch.Tensor
    root_plan_world_m: torch.Tensor
    root_plan_origin_frame_id: int
    root_plan_kind: str
    frame_id: int = 0


@dataclass(frozen=True)
class CapacityFrame:
    frame_id: int
    payload: bytes
    native_frame: dict
    next_state: CapacityState
    prediction: dict
    query: dict
    scene_valid: torch.Tensor
    query_metadata: dict


class CapacityPipeline:
    """Per-session dynamic state; supplied actor/native/geometry resources are shared.

    Contacts are synthetic capacity observations, never source truth or actor
    proposals. A proposal cannot advance physical history or recurrent memory.
    Only the matching display acknowledgement commits its candidate once.
    """

    @torch.inference_mode()
    def __init__(self, adapter, actor, scene, initial_history, *, object_velocity):
        self.adapter, self.actor, self.scene = adapter, actor, scene
        self.device = initial_history.device
        history = initial_history.clone()
        initial = adapter.source_frame(history[:, -1:])
        horizon = actor.config.preview_frames
        offsets = torch.arange(1, horizon + 1, device=self.device, dtype=history.dtype)[None, :, None]
        seed_plan = history[:, -1:, :3] + (history[:, -1:, :3] - history[:, -2:-1, :3]) * offsets
        self.state = CapacityState(history, actor.init_hidden(1),
            initial["joints"][:, 0], initial["body_points"][:, 0], scene.initial_object_pose.clone(),
            ContactLifecycleState.init(1), torch.zeros(2, 1, 15, device=self.device),
            seed_plan, 0, "history_root_velocity_extrapolation")
        self.object_velocity = object_velocity
        self.object_known = scene.initial_object_known.clone()
        self.object_enabled = scene.initial_object_enabled.clone()
        self.query_valid = torch.ones(1, 512, device=self.device, dtype=torch.bool)
        self.surface_ids = torch.cat((torch.arange(0, 512, 2, device=self.device),
            torch.arange(1, 512, 4, device=self.device), torch.arange(3, 512, 4, device=self.device)))
        self.body_part_ids = self.surface_ids.remainder(15)[None]
        self.plan_indices = (torch.linspace(0, horizon - 1, 128, device=self.device).round().long(),
                             torch.linspace(1, horizon - 1, 128, device=self.device).round().long())
        self.intent_time = torch.linspace(1, horizon, 128, device=self.device, dtype=history.dtype)[None] / 30
        self.query_sources = torch.cat((torch.zeros(256, device=self.device, dtype=torch.long),
            torch.ones(128, device=self.device, dtype=torch.long),
            torch.full((128,), 2, device=self.device, dtype=torch.long)))[None]
        self.pending = None

    def contract(self):
        return dict(mode="capacity", classification=self.actor.capacity_scope,
            actor_status=self.actor.probe_status, quality_approved=False, learned_wasd_usable=False,
            optimizer_updates=0, preview_frames=self.actor.config.preview_frames,
            preview_units="30Hz_frames_not_VQ_tokens", hidden_dim=self.actor.config.hidden_dim,
            scene_dim=self.actor.config.scene_dim, frame_bytes=4940,
            contact_observations="synthetic_CPU_capacity_fixture_not_GT_quality",
            body_point_source="512_selected_official_surface_vertices_all_skin_influences",
            query_policy=QUERY_POLICY, query_source_counts=dict(current_body=256, committed_root_plan=128, latest_intent=128),
            root_plan_commit="matching_rendered_ACK_once", root_plan_seed="history_root_velocity_extrapolation_no_future_GT",
            history_commit="matching_rendered_ACK_once", pending_limit=1)

    def scene_asset(self):
        """Cold export of the exact query triangles and initial object transforms."""
        import warp as wp
        poses = self.state.object_pose[0].detach().cpu().tolist()
        known = self.object_known[0].detach().cpu().tolist()
        enabled = self.object_enabled[0].detach().cpu().tolist()
        objects = [dict(id=index,
            vertices=wp.to_torch(mesh.points).detach().cpu().flatten().tolist(),
            faces=wp.to_torch(mesh.indices).detach().cpu().flatten().tolist(),
            pose=poses[index], known=known[index], enabled=enabled[index])
            for index, mesh in enumerate(self.scene._meshes)]
        return dict(kind="triangle_meshes", objects=objects, geometry=self.scene.storage_summary)

    def _contact_inputs(self, state):
        slots = torch.arange(15)[None]
        active = (slots + state.frame_id // 10).remainder(3) == 0
        normal = torch.zeros(1, 15, 3)
        normal[..., 1] = 1
        factual = observe_contacts(state.contact_lifecycle,
            observation_time_s=torch.tensor([state.frame_id / 30], dtype=torch.float64),
            active=active, valid=torch.ones(1, 15, dtype=torch.bool),
            object_id=slots.remainder(self.scene.object_count),
            material_id=slots + (state.frame_id // 20) * 15,
            anchor_ego_m=torch.zeros(1, 15, 3), normal_ego=normal,
            surface_velocity_ego_mps=torch.zeros(1, 15, 3))
        probabilities = state.previous_contact_logits.detach().cpu().sigmoid()
        contacts, valid = contact_features(factual, probabilities[0], probabilities[1])
        return contacts.to(self.device), valid.to(self.device), factual

    def _query_inputs(self, state, command):
        root = state.history[:, -1:, :3]
        body = state.body_points[:, self.surface_ids]
        age = state.frame_id - state.root_plan_origin_frame_id
        indices = self.plan_indices[0 if age == 0 else 1]
        plan_time = (indices.to(body.dtype)[None] + 1 - age) / 30
        planned = state.root_plan_world_m[:, indices] + body[:, 256:384] - root
        intent_offset = torch.stack((command[:, 0], torch.zeros_like(command[:, 0]), command[:, 1]), -1)
        intended = body[:, 384:] + self.intent_time[..., None] * intent_offset[:, None]
        points = torch.cat((body[:, :256], planned, intended), dim=1)
        query_time = torch.cat((body.new_zeros(1, 256), plan_time, self.intent_time), dim=1)
        metadata = dict(policy=QUERY_POLICY, source_counts=dict(current_body=256, committed_root_plan=128, latest_intent=128),
            query_source_ids={"0": "current_official_body", "1": "committed_predicted_root_plan", "2": "latest_intent"},
            time_units="seconds_relative_to_current_committed_frame", current_frame_id=state.frame_id,
            committed_plan_origin_frame_id=state.root_plan_origin_frame_id, committed_plan_age_frames=age,
            committed_plan_kind=state.root_plan_kind,
            time_ranges_s=dict(current_body=[0.0, 0.0],
                committed_root_plan=[1/30, (self.actor.config.preview_frames-age)/30],
                latest_intent=[1/30, self.actor.config.preview_frames/30]))
        return points, query_time, metadata

    @torch.inference_mode()
    def propose(self, command, *, stage_callback=None):
        if self.pending is not None:
            raise RuntimeError("A capacity frame is awaiting its rendered ACK")
        state = self.state
        command_cpu = torch.as_tensor(command, dtype=torch.float32, device="cpu").reshape(1, 3)
        if not torch.isfinite(command_cpu).all():
            raise ValueError("Capacity command must be finite world vx, vz and heading")
        command_tensor = command_cpu.to(self.device)
        contacts, contact_valid, factual = self._contact_inputs(state)
        if stage_callback is not None:
            stage_callback("contact_cpu_admit_pack_and_transfers")
        points, query_time, query_metadata = self._query_inputs(state, command_tensor)
        pose = state.object_pose.clone()
        pose[..., :3] += self.object_velocity[..., :3] / 30
        query = self.scene.query(points, pose, self.object_known, self.object_enabled, self.query_valid)
        query.update(points_world_m=points, query_time_offset_s=query_time, query_source=self.query_sources)
        if stage_callback is not None:
            stage_callback("query_and_point_construction")
        features, valid = encode_scene_features(points, query, state.history[:, -1, :3],
            torch.atan2(state.history[:, -1, 4], state.history[:, -1, 3]), self.body_part_ids,
            query_time, pose, self.object_velocity)
        if stage_callback is not None:
            stage_callback("encode")
        prediction = self.actor(state.history, command_tensor, contacts, contact_valid, features, valid, state.hidden)
        if stage_callback is not None:
            stage_callback("actor")
        native = self.adapter.step(state.history, prediction)
        next_root_plan = root_preview_to_world(state.history, prediction["root_preview"])
        if stage_callback is not None:
            stage_callback("native_rig_and_history")
        payload = self.adapter.pack_actor_frame(native, prediction["input_valid"] & query["scene_known"])
        next_state = CapacityState(native["history"], prediction["hidden"], native["joints"][:, 0],
            native["body_points"][:, 0], pose,
            factual, torch.stack((prediction["contact_logits"], prediction["release_logits"])),
            next_root_plan, state.frame_id, "actor_root_preview", state.frame_id + 1)
        frame = CapacityFrame(next_state.frame_id, payload, native, next_state, prediction, query, valid, query_metadata)
        self.pending = frame
        return frame

    def commit(self, frame_id):
        if type(frame_id) is not int:
            raise ValueError("Capacity ACK frame_id must be an integer")
        if frame_id == self.state.frame_id:
            return False
        if self.pending is None or frame_id != self.pending.frame_id:
            raise ValueError("Capacity ACK does not match the pending frame")
        self.state = self.pending.next_state
        self.pending = None
        return True
