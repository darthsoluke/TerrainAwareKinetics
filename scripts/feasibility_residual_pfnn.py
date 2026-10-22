"""Cone-constrained minimum wrench residual per frame.

Binary feasibility is too brittle here: the reference motion sits at or beyond the
edge of what a point-patch foot can supply, so a per-frame yes/no mostly reports
numerical luck. The residual is the size of the momentum imbalance that no
admissible force distribution can remove, in m/s^2 of linear and m/s^2 of torque
per unit mass; it is continuous and comparable across conditions.
"""
import argparse, json, pathlib, sys
import numpy as np, torch
from scipy.optimize import minimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from scripts.audit_pfnn_physical_consistency import FPS, G, foot_indices, segment_com_world

HALF_WIDTH = 0.045


def wrench_matrix(points):
    A = np.zeros((6, 3 * len(points)))
    for k, r in enumerate(points):
        A[0:3, 3 * k:3 * k + 3] = np.eye(3)
        A[3:6, 3 * k:3 * k + 3] = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return A


def residual(A, b, mu):
    n = A.shape[1]
    def objective(x):
        r = A @ x - b
        return float(r @ r)
    def gradient(x):
        return 2.0 * A.T @ (A @ x - b)
    constraints = []
    for k in range(n // 3):
        x, y, z = 3 * k, 3 * k + 1, 3 * k + 2
        constraints += [
            {'type': 'ineq', 'fun': (lambda t, y=y: t[y])},
            {'type': 'ineq', 'fun': (lambda t, x=x, y=y: mu * t[y] - t[x])},
            {'type': 'ineq', 'fun': (lambda t, x=x, y=y: mu * t[y] + t[x])},
            {'type': 'ineq', 'fun': (lambda t, z=z, y=y: mu * t[y] - t[z])},
            {'type': 'ineq', 'fun': (lambda t, z=z, y=y: mu * t[y] + t[z])}]
    start = np.zeros(n)
    vertical = max(b[1], 0.0)
    for k in range(n // 3):
        start[3 * k + 1] = vertical / (n // 3)
    result = minimize(objective, start, jac=gradient, constraints=constraints,
                      method='SLSQP', options=dict(maxiter=300, ftol=1e-10))
    r = A @ result.x - b
    return float(np.linalg.norm(r[:3])), float(np.linalg.norm(r[3:]))


def analyse(row, rep, proxy, neutral, foot_joints, mu):
    world = torch.from_numpy(np.array(row, copy=True))[None].to(rep.skeleton.joint_parents.device)
    pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
    joints = pose['posed_joints'][0]; rots = pose['global_joint_rots'][0]
    contacts = (world[0][:, rep.indices['foot_contacts']] > 0.5).detach().cpu().numpy()
    n = min(joints.shape[0], rots.shape[0], contacts.shape[0])
    joints, rots, contacts = joints[:n], rots[:n], contacts[:n]
    coms, mass = segment_com_world(joints, rots, proxy, neutral)
    coms = coms.detach().cpu().numpy(); mass = mass.detach().cpu().numpy()
    total = mass.sum(); centre = (coms * mass[:, None]).sum(1) / total
    v = np.zeros_like(coms); v[1:-1] = (coms[2:] - coms[:-2]) * FPS / 2.0
    cv = (v * mass[None, :, None]).sum(1) / total
    a = np.zeros_like(centre); a[1:-1] = (cv[2:] - cv[:-2]) * FPS / 2.0
    h = (mass[None, :, None] * np.cross(coms - centre[:, None], v - cv[:, None])).sum(1) / total
    tau = np.zeros_like(h); tau[1:-1] = (h[2:] - h[:-2]) * FPS / 2.0
    feet = joints.detach().cpu().numpy()[:, foot_joints, :]
    up = np.array([0.0, 1.0, 0.0]); linear, angular = [], []
    for t in range(1, n - 1):
        active = sorted({i for side in (0, 1) for i in (2*side, 2*side+1)
                         if contacts[t, 2*side:2*side+2].any()})
        if not active:
            continue
        base = feet[t, active] - centre[t]
        axis = base[1] - base[0]; lat = np.cross(axis, up)
        lat = lat / (np.linalg.norm(lat) + 1e-12)
        points = np.stack([base[0] + HALF_WIDTH * lat, base[0] - HALF_WIDTH * lat,
                           base[1] + HALF_WIDTH * lat, base[1] - HALF_WIDTH * lat])
        lin, ang = residual(wrench_matrix(points), np.concatenate((a[t] - G, tau[t])), mu)
        linear.append(lin); angular.append(ang)
    return np.array(linear), np.array(angular)


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'pfnn', 'stats', 'proxy'):
        parser.add_argument('--' + name, type=pathlib.Path, required=True)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--mu', type=float, default=1.0)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    rep = build_native_representation(args.stats)
    proxy = dict(np.load(args.proxy, allow_pickle=True))
    mass = np.array(proxy['mass'], dtype=np.float64); com = np.array(proxy['com'], dtype=np.float64)
    neutral = rep.skeleton.neutral_joints.detach().cpu().numpy()
    bad = (mass <= 0.0) | (np.linalg.norm(com - neutral, axis=-1) > 1.0)
    com[bad] = neutral[bad]; mass[bad] = 0.0
    proxy['mass'] = mass; proxy['com'] = com
    joints_idx = foot_indices(proxy)
    dataset = PFNNFrameDataset(args.cache, args.pfnn, rep, args.split)
    indices = list(range(args.skip, min(args.skip + args.limit, len(dataset))))
    lin, ang = [], []
    for i in indices:
        a, b = analyse(dataset[i]['world'].numpy(), rep, proxy, neutral, joints_idx, args.mu)
        lin.append(a); ang.append(b)
    lin = np.concatenate(lin) if lin else np.zeros(0)
    ang = np.concatenate(ang) if ang else np.zeros(0)
    summary = dict(mu=args.mu, rows=len(indices), frames=int(lin.size))
    if lin.size:
        summary.update(linear_p50=float(np.median(lin)), linear_p90=float(np.percentile(lin, 90)),
                       angular_p50=float(np.median(ang)), angular_p90=float(np.percentile(ang, 90)),
                       feasible_fraction=float(((lin < 1e-3) & (ang < 1e-3)).mean()))
    print(json.dumps(summary, indent=2))
    if args.output:
        np.savez(args.output, linear=lin, angular=ang)


if __name__ == '__main__':
    main()
