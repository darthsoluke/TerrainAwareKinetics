"""Offline PARC-to-SOMA adapter using pinned upstream GMR and official FK/LBS.

The adapter calibrates body frames and foot surface anchors, not a new solver.
It neither fits terrain to each frame nor asserts measured physical contacts.
Unobserved SOMA articulations inherit the corresponding source body orientation.
"""

from pathlib import Path
import importlib
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .lafan_soma_retarget import (
    GMR_COMMIT, SomaGMRKinematics, Y_TO_Z, anatomical_frame, load_gmr_core, soma_mjcf,
)
from .parc_paired_source import _decode_section, _restricted_load
from .bones_seed import official_sole_surfaces
from .soma_skin_reference import load_official_skinner


MAPPING = {
    'Hips': 'pelvis', 'Chest': 'torso', 'Head': 'head',
    **{side + target: source_side + source
       for side, source_side in (('Left', 'left_'), ('Right', 'right_'))
       for target, source in (('Arm', 'upper_arm'), ('ForeArm', 'lower_arm'),
                              ('Hand', 'hand'), ('Leg', 'thigh'),
                              ('Shin', 'shin'), ('Foot', 'foot'))},
}
# SOMA +Z forward/+X left/+Y up expressed in PARC +X forward/+Y left/+Z up.
SOMA_LOCAL_TO_PARC = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def official_parc_character(repository):
    sys.path.insert(0, str(Path(repository).resolve()))
    sys.modules.setdefault('parc', importlib.import_module('PARC'))
    from parc.anim.kin_char_model import KinCharModel
    char = KinCharModel(torch.device('cpu'))
    char.load_char_file(str(Path(repository) / 'data/assets/humanoid.xml'))
    return char


def load_native_pair(path, character):
    container = _restricted_load(Path(path).read_bytes())
    motion = _decode_section(container, 'motion_data')
    terrain = _decode_section(container, 'terrain_data')
    arrays = {key: np.asarray(motion[key], dtype=np.float32) for key in
              ('root_pos', 'root_rot', 'joint_rot', 'body_contacts')}
    count = len(arrays['root_pos'])
    expected = {'root_pos': (count, 3), 'root_rot': (count, 4),
                'joint_rot': (count, 14, 4), 'body_contacts': (count, 15)}
    if count < 2 or any(value.shape != expected[key] or not np.isfinite(value).all()
                        for key, value in arrays.items()):
        raise ValueError('Invalid source motion frame/body contract')
    for key in ('root_rot', 'joint_rot'):
        if np.max(np.abs(np.linalg.norm(arrays[key], axis=-1) - 1)) > 1e-4:
            raise ValueError('Source quaternion must be unit length; no silent repair')
    fps = float(motion['fps'])
    hf, origin, dx = np.asarray(terrain['hf']), np.asarray(terrain['min_point']), float(terrain['dx'])
    if (not np.isfinite(fps) or fps <= 0 or hf.ndim != 2 or origin.shape != (2,)
            or not np.isfinite(hf).all() or not np.isfinite(origin).all()
            or not np.isfinite(dx) or dx <= 0):
        raise ValueError('Invalid paired terrain or FPS')
    with torch.no_grad():
        pos, rot = character.forward_kinematics(*[
            torch.from_numpy(arrays[key]) for key in ('root_pos', 'root_rot', 'joint_rot')])
    arrays.update(positions_zup=pos.numpy(), rotations_xyzw=rot.numpy(),
                  global_rot_mats=Rotation.from_quat(rot.reshape(-1, 4).numpy()).as_matrix().reshape(count, 15, 3, 3),
                  fps=fps, hf=hf, min_point=origin, dx=dx,
                  source_names=character.get_body_names())
    return arrays


def scale_scene(hf, min_point, dx, scale):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scene scale must be positive and finite')
    return np.asarray(hf) * scale, np.asarray(min_point) * scale, float(dx) * scale


def source_foot_geometry(character, side):
    from parc.anim.kin_char_model import GeomType
    geoms = character.get_geoms(character.get_body_names().index(side.lower() + '_foot'))
    if len(geoms) != 1 or geoms[0]._shape_type != GeomType.BOX:
        raise ValueError('Expected one official foot box; recalibrate changed geometry')
    geom = geoms[0]
    size, origin = np.asarray(geom._dims), np.asarray(geom._offset)
    rotation = np.eye(3) if geom._quat is None else Rotation.from_quat(np.asarray(geom._quat)).as_matrix()
    corners = np.asarray([[x, y, -1.] for x in (-1., 1.) for y in (-1., 1.)]) * size
    return corners @ rotation.T + origin, rotation @ [0., 0., -size[2]] + origin


def terrain_gaps_zup(points, hf, origin, dx):
    """Signed vertical gaps at official nearest-cell coordinates; unknown stays NaN.

    This measures top-surface clearance, not Euclidean distance to vertical walls.
    It must not be called a complete collision or hand-contact certificate.
    """
    indices = np.rint((points[..., :2] - origin) / dx).astype(np.int64)
    known = ((indices >= 0) & (indices < np.array(hf.shape))).all(axis=-1)
    heights = np.full(points.shape[:-1], np.nan)
    heights[known] = hf[indices[..., 0][known], indices[..., 1][known]]
    return points[..., 2] - heights, known


def terrain_surface_distance_zup(points, hf, origin, dx, chunk_size=1024, device='cpu'):
    """Official PARC box-union distance, including exposed vertical step faces.

    PARC uses normal boxes for exterior distance and inverted boxes for proper
    interior distance. Combining their respective domains avoids artificial
    interior seams. Map exterior is unknown, not padded/clamped ground.
    """
    from parc.util.terrain_util import points_hf_sdf
    points = np.asarray(points)
    if chunk_size < 1 or not np.isfinite(points).all():
        raise ValueError('SDF needs finite points and a positive chunk size')
    _, known = terrain_gaps_zup(points, hf, origin, dx)
    flat = torch.as_tensor(points.reshape(-1, 3), dtype=torch.float32, device=device)
    floor = min(float(np.min(hf)) - 10., -float(np.max(hf)) - 10., float(flat[:, 2].min()) - 10.)
    arguments = (torch.as_tensor(hf, dtype=torch.float32, device=device)[None],
                 torch.as_tensor(origin, dtype=torch.float32, device=device)[None],
                 torch.tensor([dx, dx], device=device))
    rows = []
    with torch.no_grad():
        for start in range(0, len(flat), chunk_size):
            chunk = flat[start:start + chunk_size][None]
            interior = points_hf_sdf(chunk, *arguments, base_z=floor, inverted=True)
            exterior = points_hf_sdf(chunk, *arguments, base_z=floor, inverted=False)
            rows.append(torch.where(interior < 0, interior, exterior)[0].cpu().numpy())
    result = np.concatenate(rows).reshape(points.shape[:-1])
    result[~known] = np.nan
    return result, known


def _local_from_global(global_rotations, parents):
    result = global_rotations.copy()
    result[1:] = global_rotations[parents[1:]].swapaxes(-1, -2) @ global_rotations[1:]
    return result


def reference_calibration(character, skeleton):
    names = list(skeleton.bone_order_names)
    source_names = character.get_body_names()
    neutral = skeleton.neutral_joints.numpy().astype(np.float64)
    parents = skeleton.joint_parents.numpy()
    identity = torch.tensor([0., 0., 0., 1.]).expand(1, 15, 4)
    with torch.no_grad():
        source_pos, source_rot = character.forward_kinematics(
            torch.zeros(1, 3), identity[:, 0], identity[:, 1:])
    source_pos = source_pos[0].numpy()
    source_rot = Rotation.from_quat(source_rot[0].numpy()).as_matrix()
    lengths = []
    for side, prefix in (('Left', 'left_'), ('Right', 'right_')):
        src = source_pos[[source_names.index(prefix + suffix) for suffix in ('thigh', 'shin', 'foot')]]
        dst = neutral[[names.index(side + suffix) for suffix in ('Leg', 'Shin', 'Foot')]]
        lengths.append((np.linalg.norm(np.diff(src, axis=0), axis=-1).sum(),
                        np.linalg.norm(np.diff(dst, axis=0), axis=-1).sum()))
    scale = float(np.mean([row[1] for row in lengths]) / np.mean([row[0] for row in lengths]))
    assignments, offsets, reference_global = {}, {}, np.empty((len(names), 3, 3))
    child_pairs = {'Chest': ('head', 'Neck1')}
    for side, prefix in (('Left', 'left_'), ('Right', 'right_')):
        child_pairs.update({side + target: (prefix + source, side + child) for target, source, child in
                            (('Arm', 'lower_arm', 'ForeArm'), ('ForeArm', 'hand', 'Hand'),
                             ('Leg', 'shin', 'Shin'), ('Shin', 'foot', 'Foot'))})
    for ti, name in enumerate(names):
        if name in MAPPING:
            source_name = MAPPING[name]
            si = source_names.index(source_name)
            if name in child_pairs:
                sc, tc = child_pairs[name]
                source_axis = source_rot[si].T @ (source_pos[source_names.index(sc)] - source_pos[si])
                target_axis = neutral[names.index(tc)] - neutral[ti]
                offset = anatomical_frame(source_axis, source_rot[si].T @ [1., 0., 0.]) @ anatomical_frame(
                    target_axis, np.array([0., 0., 1.])).T
            else:
                # Foot frames come from sole axes, not the sloping ankle-to-toe bone.
                offset = SOMA_LOCAL_TO_PARC.copy()
            offsets[name] = offset
            reference_global[ti] = Y_TO_Z.T @ source_rot[si] @ offset
        else:
            parent_name = names[int(parents[ti])]
            source_name = assignments[parent_name]
            offsets[name] = offsets[parent_name].copy()
            reference_global[ti] = reference_global[int(parents[ti])]
        assignments[name] = source_name
    reference_local = _local_from_global(reference_global, parents)
    with torch.no_grad():
        _, target_pos, _ = skeleton.fk(torch.from_numpy(reference_local[None]).float(), torch.zeros(1, 3))
        skinner = load_official_skinner(skeleton)
        vertices = skinner.skin(torch.from_numpy(reference_local[None]).float(), target_pos, rot_is_global=False)[0].numpy()
    target_pos = target_pos[0].numpy()
    surfaces, _, _, _ = official_sole_surfaces(skeleton)
    sole_indices, anchors, root_lifts = {}, {}, []
    for side, prefix in (('Left', 'left_'), ('Right', 'right_')):
        name = side + 'Foot'
        ti, si = names.index(name), source_names.index(prefix + 'foot')
        indices = np.unique(np.concatenate([s['selected_indices'] for s in surfaces if s['channel'].startswith(side)]))
        sole_indices[side] = indices
        region = vertices[indices]
        center = region.mean(axis=0)
        center[1] = region[:, 1].min()
        anchors[name] = reference_global[ti].T @ (center - target_pos[ti])
        _, source_anchor = source_foot_geometry(character, side)
        root_lifts.append(scale * (source_pos[si, 2] + (source_rot[si] @ source_anchor)[2]) - center[1])
    lift = float(np.mean(root_lifts))
    table = {}
    for ti, name in enumerate(names):
        si = source_names.index(assignments[name])
        desired_rot_z = source_rot[si] @ offsets[name] @ Y_TO_Z.T
        rotation_offset = source_rot[si].T @ desired_rot_z
        if name.endswith('Foot'):
            _, source_anchor = source_foot_geometry(character, name.removesuffix('Foot'))
            offset_y = offsets[name].T @ (scale * source_anchor) - anchors[name]
            position_offset = Y_TO_Z @ offset_y
            position_cost, orientation_cost = 500., 100.
        else:
            target_reference_z = Y_TO_Z @ (target_pos[ti] + [0., lift, 0.])
            position_offset = desired_rot_z.T @ (target_reference_z - scale * source_pos[si])
            position_cost = (1000. if name == 'Hips' else 500. if name.endswith('Hand')
                             else 10. if name.endswith(('Leg', 'Shin')) else 0.)
            orientation_cost = 100.
        # Unique human aliases avoid GMR's dictionary collapsing shared source bodies.
        table[name] = [name, position_cost, orientation_cost, position_offset.tolist(),
                       Rotation.from_matrix(rotation_offset).as_quat(scalar_first=True).tolist()]
    config = {'human_root_name': 'Hips', 'robot_root_name': 'Hips', 'ground_height': 0.,
              'human_height_assumption': 1., 'human_scale_table': {name: scale for name in names},
              'use_ik_match_table1': True, 'use_ik_match_table2': False,
              'ik_match_table1': table, 'ik_match_table2': {}}
    metadata = {'gmr_commit': GMR_COMMIT, 'uniform_scale': scale, 'joint_mapping': assignments,
                'observed_mapping': MAPPING, 'unobserved_joints': [n for n in names if n not in MAPPING],
                'unobserved_policy': 'inherit source parent orientation; no invented toe/finger motion',
                'reference_global_rot_mats': reference_global.tolist(),
                'reference_local_rot_mats': reference_local.tolist(),
                'reference_root_local_height_offset_m': lift,
                'root_offset_source': 'fixed reference morphology/sole geometry, not per-frame terrain fitting',
                'sole_indices': {k: v.tolist() for k, v in sole_indices.items()},
                'sole_anchor_local_yup': {k: v.tolist() for k, v in anchors.items()},
                'foot_anchor_limit': 'reference bottom-center; a target patch is not a rigid PARC box',
                'hand_anchor_limit': 'calibrated wrist task only; hand surface contact remains unverified',
                'runtime_ik': False, 'per_frame_ground_fit': False, 'training_approved': False}
    return config, metadata


class ParcSomaRetargeter(SomaGMRKinematics):
    def __init__(self, character, skeleton, repository, directory, fps):
        self.skeleton = skeleton
        self.source_names = character.get_body_names()
        config, self.calibration = reference_calibration(character, skeleton)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        xml, settings = directory / 'soma30.xml', directory / 'parc_to_soma.json'
        xml.write_text(soma_mjcf(skeleton, fps), encoding='utf-8')
        settings.write_text(json.dumps(config, indent=2), encoding='utf-8')
        implementation, params, hashes = load_gmr_core(repository)
        params.ROBOT_XML_DICT['motionbricks_parc_soma'] = xml
        params.IK_CONFIG_DICT['motionbricks_parc'] = {'motionbricks_parc_soma': settings}
        self.engine = implementation('motionbricks_parc', 'motionbricks_parc_soma', verbose=False)
        self.calibration['gmr_core_sha256'] = hashes
        import mujoco
        self.addresses = [int(self.engine.model.jnt_qposadr[mujoco.mj_name2id(
            self.engine.model, mujoco.mjtObj.mjOBJ_JOINT, name + '_joint')])
                          for name in skeleton.bone_order_names]

    def solve(self, positions_zup, global_rotations_zup):
        human = {name: [positions_zup[self.source_names.index(source)].copy(),
                       Rotation.from_matrix(global_rotations_zup[self.source_names.index(source)]).as_quat(scalar_first=True)]
                 for name, source in self.calibration['joint_mapping'].items()}
        qpos = self.engine.retarget(human, offset_to_ground=False)
        local, root = self.motion_from_qpos(qpos)
        if not np.isfinite(local).all() or not np.isfinite(root).all():
            raise ValueError('Nonfinite GMR output')
        return local, root
