"""Causal error-state G1 feedback with an explicit native-pose prior."""

import numpy as np
from scipy.spatial.transform import Rotation

from .reference import NativeSomaHistory, ReferenceCoverageError
from .retarget import SEGMENTS, _matrix
from .scene import SOMA_TO_MUJOCO


class FixedPairStateObserver:
    """One time-zero frame calibration for every factual state in an episode."""

    schema = 'motionbricks.mb_sonic.fixed_initial_pair_observer.v1'

    def __init__(self, seed, initial_qpos, adapter, *, limb_source='measured'):
        if limb_source not in ('measured', 'reference'):
            raise ValueError('Choose measured limbs or the continuous consumed reference explicitly')
        self.limb_source = limb_source
        self.adapter = adapter
        self.seed_features = np.array(seed.features[-1], copy=True)
        self.seed_global = _matrix_from_features(self.seed_features, adapter.rep)
        self.seed_local = self.seed_global.copy()
        self.seed_local[1:] = self.seed_global[adapter.parents[1:]].swapaxes(-1, -2)@self.seed_global[1:]
        initial = np.asarray(initial_qpos, float)
        if initial.shape != (36,) or not np.isfinite(initial).all():
            raise ValueError('Fixed state observation requires the exact time-zero G1 pose')
        robot_global = adapter.kinematics.fk(initial[None])[1][0, adapter.kinematics.body_ids]
        self.initial_qpos = initial.copy()
        self.frame_offsets = robot_global.swapaxes(-1, -2)@SOMA_TO_MUJOCO@self.seed_global[adapter.segment_ids]
        self.root_offset = initial[:3]@SOMA_TO_MUJOCO-self.seed_features[adapter.rep.indices['global_root_pos']]
        self.segment_lookup = dict(zip(adapter.segment_ids, range(len(SEGMENTS))))
        lengths = np.linalg.norm(np.diff(adapter.neutral[:4], axis=0), axis=-1)
        self.spine_weights = dict(zip((adapter.names.index('Spine1'), adapter.names.index('Spine2')),
            (np.cumsum(lengths)/lengths.sum())[:2]))

    def history(self, executed, *, end_time_s, observer):
        import torch

        a = self.adapter
        samples = executed.samples
        sample_times = np.array([row['timestamp_s'] for row in samples])
        times = float(end_time_s)+(np.arange(4)-3)/30
        if len(samples) < 2 or (np.diff(sample_times) <= 0).any():
            raise ReferenceCoverageError('Fixed state observation requires increasing factual history')
        ids = np.array([executed.body_names.index(name) for name in a.kinematics.body_names])
        features, provenance = [], []
        brackets = [_bracket(sample_times, time) for time in times]
        planned_global = None
        if self.limb_source == 'reference':
            poses = []
            for left, right, weight in brackets:
                pair = []
                for sample in (samples[left], samples[right]):
                    pose = sample['reference_qpos']
                    if pose is None:
                        if sample['timestamp_s'] > sample['bootstrap_until_s']+1e-9:
                            raise ReferenceCoverageError('Causal limb prior is missing after measured warmup')
                        pose = self.initial_qpos
                    pair.append(pose)
                pose = (1-weight)*pair[0]+weight*pair[1]
                pose[3:7] = Rotation.from_matrix(_rotation_between(_matrix(pair[0][3:7]),
                    _matrix(pair[1][3:7]), weight)).as_quat()[[3, 0, 1, 2]]
                poses.append(pose)
            planned_global = a.kinematics.fk(np.asarray(poses))[1][:, a.kinematics.body_ids]
        for frame_index, (time, (left, right, weight)) in enumerate(zip(times, brackets)):
            first, second = samples[left], samples[right]
            actual = _rotation_between(_matrix(first['body_quat_w'])[ids],
                _matrix(second['body_quat_w'])[ids], weight)
            measured = SOMA_TO_MUJOCO.T@actual[a.kinematics.body_ids]@self.frame_offsets
            prior = measured
            if self.limb_source == 'reference':
                prior = SOMA_TO_MUJOCO.T@planned_global[frame_index]@self.frame_offsets
            root = (1-weight)*first['qpos'][:3]+weight*second['qpos'][:3]
            relative = measured[0].T@measured[1]
            seed_relative = self.seed_global[0].T@self.seed_global[a.names.index('Chest')]
            trunk_delta = Rotation.from_matrix(relative@seed_relative.T).as_rotvec()
            prior_global = self.seed_global.copy()
            prior_global[a.segment_ids] = prior
            for bone, parent in enumerate(a.parents):
                if bone not in self.segment_lookup:
                    prior_global[bone] = prior_global[parent]@self.seed_local[bone]
            prior_local = prior_global.copy()
            prior_local[1:] = prior_global[a.parents[1:]].swapaxes(-1, -2)@prior_global[1:]
            native = np.empty_like(self.seed_global)
            for bone, parent in enumerate(a.parents):
                if bone in self.segment_lookup and (self.limb_source == 'measured'
                        or self.segment_lookup[bone] in (0, 1)):
                    native[bone] = measured[self.segment_lookup[bone]]
                elif bone in self.spine_weights:
                    native[bone] = (measured[0]@Rotation.from_rotvec(trunk_delta*self.spine_weights[bone]).as_matrix()
                        @self.seed_global[0].T@self.seed_global[bone])
                else:
                    native[bone] = native[parent]@prior_local[bone]
            frame = self.seed_features.copy()
            frame[a.rep.indices['global_rot_data']] = np.concatenate((native[..., 0], native[..., 1]), -1).reshape(-1)
            frame[a.rep.indices['global_root_pos']] = root@SOMA_TO_MUJOCO-self.root_offset
            selected = samples[max(0, int(np.searchsorted(sample_times, time+1e-9, side='right')-1))]
            frame[a.rep.indices['foot_contacts']] = measured_contact_channels(selected, executed.body_names, a)
            features.append(frame)
            provenance.append(dict(actual_brackets_s=[float(sample_times[left]), float(sample_times[right])],
                weight=weight, actual_state_versions=[first['state_version'], second['state_version']],
                prior_plan_id='fixed_initial_pair', execution_reference_plan_id=selected.get('reference_plan_id')))
        encoded = observer(torch.from_numpy(np.asarray(features, dtype=np.float32))[None])['history'][0].numpy()
        return NativeSomaHistory(times, encoded, executed.scene_id,
            dict(kind='actual_g1_error_state', conditioning_mode='B2', observer=self.schema,
                measured='Root XYZ and body orientations; actual foot forces',
                measured_correction_segments=list(SEGMENTS if self.limb_source == 'measured' else SEGMENTS[:2]),
                robot_pose_prior='fixed_initial_pair', limb_source=self.limb_source,
                pose_prior='time-zero factual native local rotations for unobserved bones',
                unobserved_dofs='retain explicit seed local pose; spine distributes measured trunk rotation',
                samples=provenance, command_dependent=False,
                raw_native_plan_dependent=False, reference_dependent=self.limb_source == 'reference',
                optimizer_updates=0, physical_state_writes=0))


def _bracket(times, requested):
    times = np.asarray(times)
    if requested < times[0]-1e-9 or requested > times[-1]+1e-9:
        raise ReferenceCoverageError("Feedback interpolation requires factual covered timestamps")
    right = min(int(np.searchsorted(times, requested, side="left")), len(times)-1)
    if abs(times[right]-requested) <= 1e-9:
        return right, right, 0.
    left = max(right-1, 0)
    return left, right, float((requested-times[left])/(times[right]-times[left]))


def _rotation_between(first, second, weight):
    shape = first.shape
    delta = Rotation.from_matrix((first.swapaxes(-1, -2) @ second).reshape(-1, 3, 3)).as_rotvec()
    return first @ Rotation.from_rotvec(delta*weight).as_matrix().reshape(shape)


def _native_prior(source, requested):
    if requested > source.trusted_until_s+1e-9:
        raise ReferenceCoverageError("Feedback pose prior exceeds its trusted coverage")
    a, b, w = _bracket(source.timestamps_s, requested)
    features = ((1-w)*source.raw_features[a]+w*source.raw_features[b]).copy()
    rotations = _rotation_between(source.global_rot_mats[a], source.global_rot_mats[b], w)
    return features, rotations


def error_state_history(seed, executed, *, end_time_s, adapter, observer, correction_segments=None,
                        robot_prior='consumed_reference'):
    """Measured Root/body errors correct the paired prior, never the physical state."""
    samples = executed.samples
    sample_times = np.array([row["timestamp_s"] for row in samples])
    requested = float(end_time_s)+(np.arange(4)-3)/30
    if len(samples) < 2 or (np.diff(sample_times) <= 0).any():
        raise ReferenceCoverageError("Actual feedback requires increasing measured G1 history")
    kin, rep = adapter.kinematics, adapter.rep
    if robot_prior not in ('consumed_reference', 'native_anatomical'):
        raise ValueError('Unknown error-state robot-pose prior')
    ids = np.array([executed.body_names.index(name) for name in kin.body_names])
    selected_segments = SEGMENTS if correction_segments is None else tuple(correction_segments)
    if "Hips" not in selected_segments or set(selected_segments)-set(SEGMENTS):
        raise ValueError("Feedback must include measured Root and known named correction segments")
    lookup = {adapter.names.index(name): i for i, name in enumerate(SEGMENTS) if name in selected_segments}
    features, provenance = [], []
    for time_s in requested:
        a, b, weight = _bracket(sample_times, time_s)
        first, second = samples[a], samples[b]
        root = (1-weight)*first["qpos"][:3]+weight*second["qpos"][:3]
        actual = _rotation_between(_matrix(first["body_quat_w"])[ids],
                                   _matrix(second["body_quat_w"])[ids], weight)
        selected = samples[max(0, int(np.searchsorted(sample_times, time_s+1e-9, side="right")-1))]
        native, robot = selected["native_reference_source"], selected["robot_reference_source"]
        if native is None:
            if time_s > samples[0]["bootstrap_until_s"]+1e-9:
                raise ReferenceCoverageError("Initial pose prior cannot replace an executed reference")
            frame = seed.features[-1].copy()
            prior = _matrix_from_features(frame, rep)
            expected = _matrix(samples[0]["body_quat_w"])[ids]
            prior_id = "explicit_time_zero_seed_pair"
        else:
            frame, prior = _native_prior(native, time_s)
            if time_s > robot.trusted_until_s+1e-9:
                raise ReferenceCoverageError("Feedback robot prior exceeds its trusted coverage")
            prior_times, prior_poses = robot.timestamps_s, robot.qpos
            if robot_prior == 'native_anatomical':
                prior_times = robot.adaptation['native_pose_prior']['timestamps_s']
                prior_poses = robot.adaptation['native_pose_prior']['qpos']
            ra, rb, rw = _bracket(prior_times, time_s)
            pose = (1-rw)*prior_poses[ra]+rw*prior_poses[rb]
            pose[3:7] = Rotation.from_matrix(_rotation_between(_matrix(prior_poses[ra, 3:7]),
                _matrix(prior_poses[rb, 3:7]), rw)).as_quat()[[3, 0, 1, 2]]
            expected = kin.fk(pose[None])[1][0]
            prior_id = selected["reference_plan_id"]
        error = SOMA_TO_MUJOCO.T @ (actual[kin.body_ids] @ expected[kin.body_ids].swapaxes(-1, -2)) @ SOMA_TO_MUJOCO
        corrected = np.empty_like(prior)
        locals_ = prior.copy()
        for bone, parent in enumerate(adapter.parents):
            if parent >= 0:
                locals_[bone] = prior[parent].T @ prior[bone]
            corrected[bone] = (error[lookup[bone]] @ prior[bone] if bone in lookup
                               else locals_[bone] if parent < 0 else corrected[parent] @ locals_[bone])
        frame[rep.indices["global_rot_data"]] = np.concatenate((corrected[..., 0], corrected[..., 1]), -1).reshape(-1)
        frame[rep.indices["global_root_pos"]] = (root-[0., 0., adapter.root_z_offset_m]) @ SOMA_TO_MUJOCO
        frame[rep.indices["foot_contacts"]] = measured_contact_channels(selected, executed.body_names, adapter)
        features.append(frame)
        provenance.append(dict(actual_brackets_s=[float(sample_times[a]), float(sample_times[b])],
            weight=weight, prior_plan_id=prior_id, actual_state_versions=[first["state_version"], second["state_version"]]))
    import torch

    result = observer(torch.from_numpy(np.asarray(features, dtype=np.float32))[None])
    return NativeSomaHistory(requested, result["history"][0].numpy(), executed.scene_id,
        dict(kind="actual_g1_error_state", conditioning_mode="B2", calibration_identity=adapter.calibration["identity"],
            measured="Root XYZ,body orientation errors,foot force contacts", measured_correction_segments=list(selected_segments),
            robot_pose_prior=robot_prior,
            pose_prior="paired native reference local pose",
            unobserved_dofs="retain reference local rotations; not zero-filled or called measured", samples=provenance,
            optimizer_updates=0, physical_state_writes=0))


def _matrix_from_features(frame, representation):
    six = frame[representation.indices["global_rot_data"]].reshape(30, 6)
    first = six[:, :3]/np.linalg.norm(six[:, :3], axis=-1, keepdims=True)
    second = six[:, 3:]-(first*six[:, 3:]).sum(-1, keepdims=True)*first
    second /= np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), -1)


def measured_contact_channels(sample, body_names, adapter):
    columns = sample["contact_columns"]
    bodies = columns["body_ids"]
    channels = np.zeros(4, np.float32)
    for side_index, side in enumerate(("left", "right")):
        body = body_names.index(side+"_ankle_roll_link")
        mask = ((bodies == body).any(1) & (bodies == 0).any(1)
                & (columns["wrench_contact"][:, 0] > adapter.force_threshold_n))
        rotation = _matrix(sample["body_quat_w"][body])
        for position in columns["position_world"][mask]:
            local = rotation.T @ (position-sample["body_pos_w"][body])
            channels[2*side_index+int(local[0] >= adapter.heel_toe_split_x_m[side])] = 1.
    return channels
