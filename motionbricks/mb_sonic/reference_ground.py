"""Original G1 foot collision constraints on a private canonical scene model."""

import copy

import mujoco
import numpy as np

from .constrained_reference import ReferenceConstraintSolveError


class ReferenceFootGround:
    def __init__(self, scene_model, *, clearance_m=.001, search_margin_m=.05,
                 scene=None, support_times=None, support_confidence=None):
        self.model = copy.copy(scene_model)
        self.data = mujoco.MjData(self.model)
        self.clearance_m = float(clearance_m)
        self.search_margin_m = float(search_margin_m)
        if not 0 <= clearance_m < search_margin_m:
            raise ValueError('Foot clearance must be inside the collision search margin')
        m = self.model
        bodies = [m.body(side+'_ankle_roll_link').id for side in ('left', 'right')]
        self.foot_geoms = np.flatnonzero(np.isin(m.geom_bodyid, bodies)
            & ((m.geom_contype != 0) | (m.geom_conaffinity != 0)))
        self.world_geoms = np.flatnonzero(m.geom_bodyid == 0)
        self.scene = scene
        self.support_times = support_times
        self.support_confidence = support_confidence
        if any(value is not None for value in (scene, support_times, support_confidence)):
            if scene is None or support_times is None or np.shape(support_confidence) != (len(support_times), 4):
                raise ValueError('Support tasks require canonical scene and four source contact channels')
        self.support_points, self.support_bodies = [], []
        for body in bodies:
            candidates = []
            for geom in self.foot_geoms[m.geom_bodyid[self.foot_geoms] == body]:
                if m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                    raise ValueError('G1 sole support task requires the original capsule feet')
                axis = np.zeros(9)
                mujoco.mju_quat2Mat(axis, m.geom_quat[geom])
                endpoint = axis.reshape(3, 3)[:, 2]*m.geom_size[geom, 1]
                candidates.extend((m.geom_pos[geom]-endpoint-[0., 0., m.geom_size[geom, 0]],
                                   m.geom_pos[geom]+endpoint-[0., 0., m.geom_size[geom, 0]]))
            candidates = np.asarray(candidates)
            self.support_points.extend(candidates[[np.argmin(candidates[:, 0]), np.argmax(candidates[:, 0])]])
            self.support_bodies.extend([body, body])
        if not len(self.foot_geoms) or not len(self.world_geoms):
            raise ValueError('Ground constraints need original feet and canonical scene collision geometry')
        for pair, (g1, g2) in enumerate(zip(m.pair_geom1, m.pair_geom2)):
            if self._foot_world(g1, g2):
                m.pair_margin[pair] = search_margin_m
                m.pair_gap[pair] = 0.
        m.geom_margin[self.foot_geoms] = search_margin_m
        self.identity = dict(kind='canonical_MuJoCo_foot_scene_signed_distance',
            clearance_m=clearance_m, search_margin_m=search_margin_m,
            original_foot_geom_ids=self.foot_geoms.tolist(),
            private_collision_search_only=True, physical_model_modified=False)
        if scene is not None:
            self.identity.update(support='source_MB_heel_toe_predictions; bounded_confidence',
                support_height_weight_per_m=200., support_velocity_weight_per_mps=3.,
                support_points_local_m=np.asarray(self.support_points).tolist())

    def support_weights(self, times):
        if self.scene is None:
            return None
        return np.clip(np.stack([np.interp(times, self.support_times, self.support_confidence[:, i])
            for i in range(4)], axis=-1), 0., 1.)

    def support_linearize(self, selected):
        m, d = self.model, self.data
        positions, jacobians = [], []
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for body, local in zip(self.support_bodies, self.support_points):
            point = d.xpos[body]+d.xmat[body].reshape(3, 3)@local
            mujoco.mj_jac(m, d, jp, jr, point, body)
            positions.append(point)
            jacobians.append(jp[:, selected].copy())
        points = np.asarray(positions)
        ground = self.scene.query_height(points[:, :2])
        if not ground.known.all():
            raise ReferenceConstraintSolveError('Reference sole support point left known ground')
        normal = ground.normal_xyz
        gap = (points[:, 2]-ground.height_m)*normal[:, 2]-self.clearance_m
        return points, np.asarray(jacobians), normal, gap

    def _foot_world(self, g1, g2):
        return ((g1 in self.foot_geoms and g2 in self.world_geoms)
                or (g2 in self.foot_geoms and g1 in self.world_geoms))

    def linearize(self, pose, qadr, selected):
        m, d = self.model, self.data
        d.qpos[:] = m.qpos0
        d.qpos[:7], d.qpos[qadr] = pose[:7], pose[7:]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)
        rows, distances = [], []
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for contact in d.contact:
            if not self._foot_world(contact.geom1, contact.geom2):
                continue
            first = contact.geom1 in self.foot_geoms
            geom = contact.geom1 if first else contact.geom2
            normal = (-1. if first else 1.)*contact.frame[:3]
            mujoco.mj_jac(m, d, jp, jr, contact.pos, int(m.geom_bodyid[geom]))
            rows.append(normal@jp[:, selected])
            distances.append(contact.dist)
        return np.asarray(rows).reshape(-1, len(selected)), np.asarray(distances)

    def audit(self, poses, qadr, selected, *, tolerance_m=1e-5):
        minimum, count = self.search_margin_m, 0
        for pose in poses:
            _, distance = self.linearize(pose, qadr, selected)
            if len(distance):
                minimum = min(minimum, float(distance.min()))
                count += int((distance < self.clearance_m-tolerance_m).sum())
        return dict(self.identity, minimum_distance_m=minimum, violation_count=count,
            status='PASS' if count == 0 else 'FAIL', tolerance_m=tolerance_m)

    def require_feasible(self, poses, qadr, selected):
        report = self.audit(poses, qadr, selected)
        if report['status'] != 'PASS':
            raise ReferenceConstraintSolveError(f'Reference foot-ground infeasible: {report}')
        return report
