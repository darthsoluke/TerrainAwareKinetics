"""Six-row floating-base friction-cone feasibility on cached PFNN windows.

Per stance frame the linear and angular momentum equations are stacked into an
equality system over the active contact points, with a square-pyramid friction
cone and unilateral normal. The smallest friction coefficient admitting a feasible
force distribution is found by bisection, so the metric is a property of the motion
and the contact set rather than of any particular force solution.

Normals are assumed vertical. On tilted terrain this underestimates the admissible
set, so the reported coefficient is an upper bound on the true requirement.
"""
import argparse, json, pathlib, sys
import numpy as np, torch
from scipy.optimize import linprog

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from scripts.audit_pfnn_physical_consistency import (FOOT_ORDER, FPS, G, foot_indices,
                                                     segment_com_world)


def cone_rows(count, mu):
    A = np.zeros((5 * count, 3 * count))
    for k in range(count):
        base = 5 * k
        x, y, z = 3 * k, 3 * k + 1, 3 * k + 2
        A[base + 0, y] = -1.0
        A[base + 1, x] = 1.0;  A[base + 1, y] = -mu
        A[base + 2, x] = -1.0; A[base + 2, y] = -mu
        A[base + 3, z] = 1.0;  A[base + 3, y] = -mu
        A[base + 4, z] = -1.0; A[base + 4, y] = -mu
    return A


def feasible(points, linear, angular, mu):
    """Is there a force distribution satisfying both momentum rows inside the cone?"""
    count = len(points)
    A_eq = np.zeros((6, 3 * count))
    for k, r in enumerate(points):
        A_eq[0:3, 3 * k:3 * k + 3] = np.eye(3)
        skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
        A_eq[3:6, 3 * k:3 * k + 3] = skew
    b_eq = np.concatenate((linear, angular))
    result = linprog(np.zeros(3 * count), A_ub=cone_rows(count, mu),
                     b_ub=np.zeros(5 * count), A_eq=A_eq, b_eq=b_eq,
                     bounds=[(None, None)] * (3 * count), method='highs')
    return bool(result.success)


def minimum_mu(points, linear, angular, ceiling=2.0, iterations=14):
    if not feasible(points, linear, angular, ceiling):
        return None
    low, high = 0.0, ceiling
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if feasible(points, linear, angular, mid):
            high = mid
        else:
            low = mid
    return high


def analyse(row, rep, proxy, neutral, foot_joints):
    world = torch.from_numpy(np.array(row, copy=True))[None].to(rep.skeleton.joint_parents.device)
    pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
    joints = pose['posed_joints'][0]
    rots = pose['global_joint_rots'][0]
    contacts = (world[0][:, rep.indices['foot_contacts']] > 0.5).detach().cpu().numpy()
    frames = min(joints.shape[0], rots.shape[0], contacts.shape[0])
    joints, rots, contacts = joints[:frames], rots[:frames], contacts[:frames]
    coms, mass = segment_com_world(joints, rots, proxy, neutral)
    coms = coms.detach().cpu().numpy(); mass = mass.detach().cpu().numpy()
    total = mass.sum()
    com = (coms * mass[:, None]).sum(1) / total
    feet = joints.detach().cpu().numpy()[:, foot_joints, :]
    # Per-segment velocities first; the centre-of-mass velocity is their mass-weighted
    # mean. An earlier version weighted the centre-of-mass velocity by segment mass.
    segment_velocity = np.zeros_like(coms)
    segment_velocity[1:-1] = (coms[2:] - coms[:-2]) * FPS / 2.0
    centre_velocity = (segment_velocity * mass[None, :, None]).sum(1) / total
    accel = np.zeros_like(com)
    accel[1:-1] = (centre_velocity[2:] - centre_velocity[:-2]) * FPS / 2.0
    relative = coms - com[:, None]
    momentum = (mass[None, :, None] *
                np.cross(relative, segment_velocity - centre_velocity[:, None])).sum(1) / total
    torque = np.zeros_like(momentum)
    torque[1:-1] = (momentum[2:] - momentum[:-2]) * FPS / 2.0
    # A foot is a patch, not a point: a single contact point leaves three unknowns
    # against six equations, which is infeasible by construction. Whenever a foot is
    # in contact its ankle and toe points are both admitted, giving the sole a lever
    # arm and recovering the centre-of-pressure degree of freedom.
    # Two point contacts span only a five-dimensional wrench space, so a foot is
    # modelled as a patch: ankle, toe and one lateral toe corner. Half the foot width
    # is a declared constant, not a fitted quantity.
    half_width = 0.045
    up = np.array([0.0, 1.0, 0.0])
    mus, infeasible, skipped, single_point, norms = [], 0, 0, 0, []
    for t in range(1, frames - 1):
        active = set()
        for side in (0, 1):
            pair = (2 * side, 2 * side + 1)
            if contacts[t, list(pair)].any():
                active.update(pair)
        active = sorted(active)
        if not active:
            skipped += 1; continue
        if len(active) < 2:
            single_point += 1
        base = feet[t, active] - com[t]
        axis = base[1] - base[0]
        lateral = np.cross(axis, up)
        norm = np.linalg.norm(lateral)
        lateral = lateral / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        # Four corners of the foot patch: three points left two of them nearly
        # coincident, which ill-conditions the moment rows and drives enormous
        # opposing forces.
        points = np.stack([base[0] + half_width * lateral, base[0] - half_width * lateral,
                           base[1] + half_width * lateral, base[1] - half_width * lateral])
        linear = accel[t] - G
        angular = torque[t]
        norms.append((float(np.linalg.norm(linear)), float(np.linalg.norm(angular)),
                      float(np.abs(points).max())))
        value = minimum_mu(points, linear, angular)
        if value is None:
            infeasible += 1
        else:
            mus.append(value)
    return dict(mu_6row=mus, infeasible=infeasible, skipped=skipped,
                single_point=single_point, frames=int(frames), norms=norms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=pathlib.Path, required=True)
    parser.add_argument('--pfnn', type=pathlib.Path, required=True)
    parser.add_argument('--stats', type=pathlib.Path, required=True)
    parser.add_argument('--proxy', type=pathlib.Path, required=True)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    rep = build_native_representation(args.stats)
    proxy = dict(np.load(args.proxy, allow_pickle=True))
    mass = np.array(proxy['mass'], dtype=np.float64)
    com = np.array(proxy['com'], dtype=np.float64)
    neutral = rep.skeleton.neutral_joints.detach().cpu().numpy()
    negligible = (mass <= 0.0) | (np.linalg.norm(com - neutral, axis=-1) > 1.0)
    com[negligible] = neutral[negligible]; mass[negligible] = 0.0
    proxy['mass'] = mass; proxy['com'] = com
    foot_joints = foot_indices(proxy)
    dataset = PFNNFrameDataset(args.cache, args.pfnn, rep, args.split)
    indices = list(range(args.skip, min(args.skip + args.limit, len(dataset))))
    results = [analyse(dataset[i]['world'].numpy(), rep, proxy, neutral, foot_joints) for i in indices]
    mus = np.concatenate([r['mu_6row'] for r in results]) if results else np.zeros(0)
    summary = dict(rows=len(results),
                   evaluated_frames=int(mus.size),
                   infeasible_frames=sum(r['infeasible'] for r in results),
                   skipped_frames=sum(r['skipped'] for r in results),
                   single_point_frames=sum(r['single_point'] for r in results))
    flat = [v for r in results for v in r['norms']]
    if flat:
        arr = np.array(flat)
        summary.update(linear_p50=float(np.median(arr[:, 0])), angular_p50=float(np.median(arr[:, 1])),
                       lever_p90=float(np.percentile(arr[:, 2], 90)),
                       torque_capacity_p50=float(np.median(arr[:, 2] * arr[:, 0])))
    if mus.size:
        summary.update(mu_p50=float(np.median(mus)), mu_p90=float(np.percentile(mus, 90)),
                       mu_over_0_5=float((mus > 0.5).mean()), mu_over_1_0=float((mus > 1.0).mean()))
    print(json.dumps(summary, indent=2))
    if args.output:
        np.savez(args.output, mu_6row=mus)


if __name__ == '__main__':
    main()
