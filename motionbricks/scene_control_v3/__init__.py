"""Independent, untrained scene-control-v3 performance prototype."""

from .contracts import (
    ACTOR_SCHEMA, CAPACITY_SCOPE, CONTACT_FEATURE_LAYOUT, PROBE_STATUS, SCENE_FEATURE_LAYOUT,
    ContactEvent, SceneActorConfig,
)
from .model import CausalSceneActor
from .state import ContactLifecycleState, contact_features, observe_contacts

__all__ = [
    "ACTOR_SCHEMA", "CAPACITY_SCOPE", "CONTACT_FEATURE_LAYOUT", "PROBE_STATUS", "SCENE_FEATURE_LAYOUT",
    "ContactEvent", "SceneActorConfig", "CausalSceneActor", "ContactLifecycleState",
    "contact_features", "observe_contacts",
]
