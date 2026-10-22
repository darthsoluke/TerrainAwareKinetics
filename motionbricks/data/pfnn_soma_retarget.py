"""Offline PFNN CMU skeleton conversion using pinned GMR and official SOMA FK."""

import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .lafan_soma_retarget import SomaGMRKinematics, Y_TO_Z, anatomical_frame, load_gmr_core, soma_mjcf


# The author scales BVH units by5.6444 before centimetre-valued terrain fitting.
PFNN_UNIT_METRES = .056444
MAPPING = {'Hips': 'Hips', 'Spine1': 'Spine', 'Spine2': 'Spine1', 'Chest': 'Spine1',
    'Neck1': 'Neck', 'Neck2': 'Neck1', 'Head': 'Head',
    **{side+target: side+source for side in ('Left', 'Right') for target, source in
       (('Shoulder', 'Shoulder'), ('Arm', 'Arm'), ('ForeArm', 'ForeArm'), ('Hand', 'Hand'),
        ('HandMiddleEnd', 'HandIndex1'), ('Leg', 'UpLeg'), ('Shin', 'Leg'), ('Foot', 'Foot'), ('ToeBase', 'ToeBase'))}}
CHILDREN = {'Hips': ('Spine', 'Spine1'), 'Spine1': ('Spine1', 'Spine2'),
    'Spine2': ('Neck1', 'Chest'), 'Chest': ('Neck1', 'Neck1'),
    'Neck1': ('Neck1', 'Neck2'), 'Neck2': ('Head', 'Head'),
    **{side+target: (side+source, side+child) for side in ('Left', 'Right')
       for target, source, child in (('Shoulder', 'Arm', 'Arm'), ('Arm', 'ForeArm', 'ForeArm'),
           ('ForeArm', 'Hand', 'Hand'), ('Hand', 'HandIndex1', 'HandMiddleEnd'),
           ('Leg', 'Leg', 'Shin'), ('Shin', 'Foot', 'Foot'), ('Foot', 'ToeBase', 'ToeBase'))}}


def official_motion_modules(folder):
    folder = Path(folder).resolve()
    if not all((folder/name).is_file() for name in ('BVH.py', 'Animation.py', 'Quaternions.py')):
        raise ValueError('Verified author BVH/Animation/Quaternion sources required')
    # NumPy removed this historical matmul test module; its sole used operation
    # is the same broadcast matrix product now supplied by public np.matmul.
    if 'numpy.core.umath_tests' not in sys.modules:
        compatibility = types.ModuleType('numpy.core.umath_tests')
        compatibility.matrix_multiply = np.matmul
        sys.modules['numpy.core.umath_tests'] = compatibility
    sys.path.insert(0, str(folder))
    for name in ('BVH', 'Animation', 'AnimationStructure', 'Quaternions', 'Pivots'):
        previous = sys.modules.get(name)
        if previous is not None and Path(previous.__file__).resolve().parent != folder:
            raise ValueError('Another animation library already owns the PFNN module names')
    return importlib.import_module('BVH'), importlib.import_module('Animation')


def source_fk(local, roots, offsets, parents):
    rotations = np.empty_like(local)
    positions = np.empty((*local.shape[:2], 3), dtype=local.dtype)
    for index, parent in enumerate(parents):
        if parent < 0:
            rotations[:, index], positions[:, index] = local[:, index], roots
        else:
            rotations[:, index] = rotations[:, parent]@local[:, index]
            positions[:, index] = positions[:, parent]+(rotations[:, parent]@offsets[index, :, None])[..., 0]
    return rotations, positions


def sample_source_motion(local, roots, source_fps, target_fps=30.):
    source_times = np.arange(len(local), dtype=np.float64)/source_fps
    target_times = np.arange(int(np.floor(source_times[-1]*target_fps+1e-8))+1)/target_fps
    sampled = np.stack([Slerp(source_times, Rotation.from_matrix(local[:, joint]))(target_times).as_matrix()
        for joint in range(local.shape[1])], 1)
    positions = np.stack([np.interp(target_times, source_times, roots[:, axis]) for axis in range(3)], -1)
    return sampled.astype(np.float32), positions.astype(np.float32)


def load_source(path, official_folder, target_fps=30.):
    bvh, animation = official_motion_modules(official_folder)
    source, names, dt = bvh.load(str(path))
    if len(names) != 31 or len(set(names)) != 31 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Expected the verified31-joint PFNN BVH and finite frame clock')
    local = source.rotations.transforms()
    roots = source.positions[:, 0]*PFNN_UNIT_METRES
    offsets = source.offsets*PFNN_UNIT_METRES
    global_rot, positions = source_fk(local, roots, offsets, source.parents)
    official = animation.positions_global(source)*PFNN_UNIT_METRES
    source_error = float(np.abs(positions-official).max())
    if source_error > 1e-6 or not np.isfinite(positions).all():
        raise ValueError('Independent source FK differs from author implementation')
    original_frames = len(roots)
    if original_frames > 1:
        local, roots = sample_source_motion(local, roots, 1/dt, target_fps)
        global_rot, positions = source_fk(local, roots, offsets, source.parents)
    return dict(local_rot_mats=local, root_positions=roots, global_rot_mats=global_rot,
        posed_joints=positions, source_skeleton=dict(names=names, parents=np.asarray(source.parents)),
        source_fps=1/dt, fps=target_fps, original_frames=original_frames,
        source_fk_max_error_m=source_error, unit_metres=PFNN_UNIT_METRES, offsets=offsets,
        temporal_sampling='Exact timestamp SLERP and linear Root interpolation; no filter or padding')


def calibration(bind, skeleton):
    names = bind['source_skeleton']['names']
    source = bind['posed_joints'][0]
    source_rot = bind['global_rot_mats'][0]
    target_names = list(skeleton.bone_order_names)
    neutral = skeleton.neutral_joints.detach().cpu().numpy().astype(np.float64)
    neutral -= neutral[0]
    parents = skeleton.joint_parents.detach().cpu().numpy()
    source_lengths, target_lengths = [], []
    for side in ('Left', 'Right'):
        a = source[[names.index(side+name) for name in ('UpLeg', 'Leg', 'Foot')]]
        b = neutral[[target_names.index(side+name) for name in ('Leg', 'Shin', 'Foot')]]
        source_lengths.append(float(np.linalg.norm(np.diff(a, axis=0), axis=-1).sum()))
        target_lengths.append(float(np.linalg.norm(np.diff(b, axis=0), axis=-1).sum()))
    if not all(.5 < length < 1.3 for length in source_lengths):
        raise ValueError('PFNN physical units or reference leg lengths are invalid')
    scale = float(np.mean(target_lengths)/np.mean(source_lengths))
    mapping, aligned, offsets = {}, np.empty((len(target_names), 3, 3)), {}
    for index, name in enumerate(target_names):
        source_name = MAPPING.get(name, mapping.get(target_names[int(parents[index])]))
        if source_name not in names:
            raise ValueError('Missing PFNN anatomical mapping: '+name)
        mapping[name] = source_name
        si = names.index(source_name)
        if name in CHILDREN:
            source_child, target_child = CHILDREN[name]
            primary = source_rot[si].T@(source[names.index(source_child)]-source[si])
            desired = neutral[target_names.index(target_child)]-neutral[index]
            forward = np.array([0., 1., 0.]) if name.endswith('Foot') else np.array([0., 0., 1.])
            offset = anatomical_frame(primary, source_rot[si].T@forward)@anatomical_frame(desired, forward).T
        elif parents[index] >= 0:
            offset = source_rot[si].T@aligned[int(parents[index])]
        else:
            offset = np.eye(3)
        offsets[name] = offset
        aligned[index] = source_rot[si]@offset
    reference_positions = np.zeros_like(neutral)
    for index, parent in enumerate(parents):
        if parent >= 0:
            reference_positions[index] = reference_positions[parent]+aligned[parent]@(neutral[index]-neutral[parent])
    table = {}
    for index, name in enumerate(target_names):
        si = names.index(mapping[name])
        position_offset = aligned[index].T@(reference_positions[index]-scale*(source[si]-source[0]))
        rotation_offset = Y_TO_Z@offsets[name]@Y_TO_Z.T
        position_cost, rotation_cost = (1000., 100.) if name == 'Hips' else \
            (50., 10.) if name.endswith(('Foot', 'ToeBase')) else \
            (10., 10.) if name.endswith(('Leg', 'Shin')) else (0., 100.)
        # Unique task names allow multiple target spine joints to follow a
        # sparse source torso without silently overwriting GMR task entries.
        table[name] = [name, position_cost, rotation_cost, (Y_TO_Z@position_offset).tolist(),
                       Rotation.from_matrix(rotation_offset).as_quat(scalar_first=True).tolist()]
    config = dict(human_root_name='Hips', robot_root_name='Hips', ground_height=0., human_height_assumption=1.,
        human_scale_table={name: scale for name in target_names}, use_ik_match_table1=True,
        use_ik_match_table2=False, ik_match_table1=table, ik_match_table2={})
    local = aligned.copy()
    local[1:] = aligned[parents[1:]].swapaxes(-1, -2)@aligned[1:]
    return config, dict(uniform_scale=scale, mapping=mapping, reference_local_rot_mats=local.tolist(),
        reference_positions_relative_root_m=reference_positions.tolist(), source_leg_lengths_m=source_lengths,
        target_leg_lengths_m=target_lengths, source_unit_metres=PFNN_UNIT_METRES,
        solver_role='offline skeleton conversion only', runtime_ik=False, per_frame_ground_projection=False)


class PFNNSomaRetargeter(SomaGMRKinematics):
    def __init__(self, bind, skeleton, repository, work_directory):
        self.skeleton = skeleton
        self.source_names = bind['source_skeleton']['names']
        config, self.calibration = calibration(bind, skeleton)
        directory = Path(work_directory)
        directory.mkdir(parents=True, exist_ok=True)
        xml, settings = directory/'soma30.xml', directory/'pfnn_to_soma30.json'
        xml.write_text(soma_mjcf(skeleton, fps=30.), encoding='utf-8')
        settings.write_text(json.dumps(config, indent=2), encoding='utf-8')
        implementation, params, hashes = load_gmr_core(repository)
        params.ROBOT_XML_DICT['motionbricks_soma30'] = xml
        params.IK_CONFIG_DICT['motionbricks_pfnn'] = {'motionbricks_soma30': settings}
        self.engine = implementation('motionbricks_pfnn', 'motionbricks_soma30', verbose=False)
        import mujoco
        self.addresses = [int(self.engine.model.jnt_qposadr[mujoco.mj_name2id(
            self.engine.model, mujoco.mjtObj.mjOBJ_JOINT, name+'_joint')]) for name in skeleton.bone_order_names]
        self.calibration['gmr_core_sha256'] = hashes

    def solve(self, positions, global_rotations):
        human = {}
        for target, source in self.calibration['mapping'].items():
            index = self.source_names.index(source)
            rotation = Y_TO_Z@global_rotations[index]@Y_TO_Z.T
            human[target] = [Y_TO_Z@positions[index], Rotation.from_matrix(rotation).as_quat(scalar_first=True)]
        pose = self.engine.retarget(human, offset_to_ground=False).copy()
        local, root = self.motion_from_qpos(pose)
        if not np.isfinite(local).all() or not np.isfinite(root).all():
            raise ValueError('PFNN retarget produced a non-finite pose')
        return local, root
