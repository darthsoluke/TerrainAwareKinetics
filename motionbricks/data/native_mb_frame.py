"""Explicit rigid frame transforms for native, unnormalized MB features.

This is a coordinate transform, not a floor estimator or output grounding rule.
The caller must supply an origin independent of the predicted pose.
"""

import math
import json
from pathlib import Path
import torch

from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints


def load_authored_idle_reference(path, representation, device):
    from motionbricks.data.soma_replay import skeleton_manifest
    artifact = json.loads(Path(path).read_text())
    if (artifact.get('schema') != 'motionbricks.authored_idle.v1'
            or artifact.get('source_id') != skeleton_manifest()['source_id']
            or artifact.get('fps') != representation.fps
            or artifact.get('provenance', {}).get('split') != 'train'):
        raise ValueError('Authored idle reference identity mismatch')
    features = torch.tensor(artifact['features'], dtype=torch.float32, device=device)[None]
    if features.shape != (1, 4, len(representation.indices['all'])) or not torch.isfinite(features).all():
        raise ValueError('Invalid authored idle features')
    return features


def translate_global_height(features, representation, offset_m):
    """Translate hip and global-height joint positions by one shared constant."""
    if not isinstance(representation, GlobalRootGlobalJoints) or representation.removed_heading:
        raise ValueError('Expected the native global-root/global-joints representation')
    if not math.isfinite(offset_m) or features.shape[-1] != len(representation.indices['all']):
        raise ValueError('Need finite offset and full unnormalized features')
    result = features.clone()
    result[..., representation.indices['global_root_pos'][1]] += offset_m
    result[..., representation.indices['ric_data'][1::3]] += offset_m
    return result


class NativePlatformLocomotion:
    """Run native locomotion in a fixed, explicitly authored platform frame.

    The platform height is input scene metadata, never inferred from generated
    feet. The inverse uses the exact same origin; it is not an output correction.
    """

    def __init__(self, backend, representation, platform_y_m):
        if not math.isfinite(platform_y_m):
            raise ValueError('Platform origin must be finite')
        self.backend = backend
        self.representation = representation
        self.platform_y_m = float(platform_y_m)

    def predict(self, world_history, velocity_xz, heading, *, target_offset_xz=(0., 0.)):
        if any(value != 0 for value in target_offset_xz):
            raise ValueError('Free locomotion does not implement an anchored input offset')
        self.backend.history = translate_global_height(
            world_history, self.representation, -self.platform_y_m)
        result = self.backend.predict_plan(velocity_xz, heading)
        return dict(result, neural_generation=self.backend.last_generation['neural_generation'],
                    planned_motion=translate_global_height(
            result['planned_motion'], self.representation, self.platform_y_m))


class NativeAnchoredIdle:
    """A real idle reference pose anchored to an authored platform, as MB inputs."""

    def __init__(self, primitive, representation, reference, root_xz, heading, platform_y_m):
        if not math.isfinite(platform_y_m) or not math.isfinite(heading):
            raise ValueError('Authored platform frame must be finite')
        if (reference.shape != (1, 4, len(representation.indices['all']))
                or not torch.isfinite(reference).all()):
            raise ValueError('Need four real idle reference frames')
        if len(root_xz) != 2 or not all(math.isfinite(value) for value in root_xz):
            raise ValueError('Authored idle anchor must contain two finite coordinates')
        self.primitive, self.representation = primitive, representation
        self.platform_y_m = float(platform_y_m)
        self.goal = representation.change_first_heading(reference, heading, False, False)
        self.goal = self.goal.clone()
        for axis, value in zip((0, 2), root_xz):
            if not math.isfinite(value):
                raise ValueError('Authored idle anchor must be finite')
            self.goal[..., axis] += float(value) - self.goal[0, -1, axis]

    def predict(self, world_history, velocity_xz, heading, *, target_offset_xz=(0., 0.)):
        if any(value != 0 for value in velocity_xz[0]):
            raise ValueError('An anchored idle does not implement free navigation')
        if len(target_offset_xz) != 2 or not all(math.isfinite(value) for value in target_offset_xz):
            raise ValueError('Need two finite authored target offsets')
        history = translate_global_height(world_history, self.representation, -self.platform_y_m)
        goal = self.goal.clone()
        goal[..., 0] += float(target_offset_xz[0])
        goal[..., 2] += float(target_offset_xz[1])
        prediction, frames = self.primitive.predict(history, goal)
        if frames < 12:
            raise ValueError('Native idle must provide an eight-frame future')
        return dict(planned_motion=translate_global_height(
            prediction[:, 4:12], self.representation, self.platform_y_m), neural_generation=True)
