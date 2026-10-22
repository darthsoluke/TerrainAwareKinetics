"""Fixed tensor contract for the untrained scene-control capacity probe."""

from dataclasses import dataclass
from enum import IntEnum


PROBE_STATUS = "untrained_capacity_probe"
CAPACITY_SCOPE = "architecture_capacity_only"
ACTOR_SCHEMA = "motionbricks.scene_control_v3.actor.v1"


@dataclass(frozen=True)
class SceneActorConfig:
    history_frames: int = 4
    history_dim: int = 366
    command_dim: int = 3
    contact_slots: int = 15
    contact_dim: int = 16
    scene_slots: int = 512
    scene_dim: int = 16
    hidden_dim: int = 256
    scene_token_count: int = 8
    scene_token_dim: int = 64
    preview_frames: int = 16
    preview_basis_dim: int = 8
    joint_count: int = 30
    fps: float = 30.0


SCENE_FEATURE_LAYOUT = (
    "ego_x_m", "ego_y_m", "ego_z_m", "distance_m",
    "ego_normal_x", "ego_normal_y", "ego_normal_z", "body_part_id",
    "query_time_offset_s", "ego_closest_x_m", "ego_closest_y_m", "ego_closest_z_m",
    "ego_surface_vx_mps", "ego_surface_vy_mps", "ego_surface_vz_mps",
    "distance_signed_flag",
)

CONTACT_FEATURE_LAYOUT = (
    "actual_active", "proposed_contact_probability", "predicted_release_probability",
    "observed_active_age_s", "observed_inactive_age_s", "seconds_since_event",
    "ego_anchor_x_m", "ego_anchor_y_m", "ego_anchor_z_m",
    "ego_normal_x", "ego_normal_y", "ego_normal_z",
    "ego_surface_vx_mps", "ego_surface_vy_mps", "ego_surface_vz_mps",
    "observation_known",
)


class ContactEvent(IntEnum):
    NONE = 0
    ACQUIRED = 1
    RELEASED = 2
    REBOUND = 3
    OBSERVATION_LOST = 4
    OBSERVED_FREE = 5
