"""Diagnose whether the six-row system is violated by the equations or by the cone."""
import argparse, json, pathlib, sys
import numpy as np, torch
from scipy.optimize import minimize
from scipy.signal import savgol_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from scripts.audit_pfnn_physical_consistency import FPS, G, foot_indices, segment_com_world


def wrench_matrix(points):
    A = np.zeros((6, 3 * len(points)))
    for k, r in enumerate(points):
        A[0:3, 3 * k:3 * k + 3] = np.eye(3)
        A[3:6, 3 * k:3 * k + 3] = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    return A


def cone_residual(A, b, mu):
    """Smallest residual achievable with every force inside the friction cone."""
    n = A.shape[1]
    def objective(f):
        r = A @ f - b
        return float(r @ r)
    def gradient(f):
        return 2.0 * A.T @ (A @ f - b)
    constraints = []
    for k in range(n // 3):
        x, y, z = 3 * k, 3 * k + 1, 3 * k + 2
        constraints.append({'type': 'ineq', 'fun': lambda f, y=y: f[y]})
        constraints.append({'type': 'ineq', 'fun': lambda f, x=x, y=y: mu * f[y] - f[x]})
        constraints.append({'type': 'ineq', 'fun': lambda f, x=x, y=y: mu * f[y] + f[x]})
        constraints.append({'type': 'ineq', 'fun': lambda f, z=z, y=y: mu * f[y] - f[z]})
        constraints.append({'type': 'ineq', 'fun': lambda f, z=z, y=y: mu * f[y] + f[z]})
    result = minimize(objective, np.zeros(n), jac=gradient, constraints=constraints,
                      method='SLSQP', options=dict(maxiter=200, ftol=1e-10))
    return float(np.sqrt(max(result.fun, 0.0))), result.x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=pathlib.Path, required=True)
    parser.add_argument('--pfnn', type=pathlib.Path, required=True)
    parser.add_argument('--stats', type=pathlib.Path, required=True)
    parser.add_argument('--proxy', type=pathlib.Path, required=True)
    parser.add_argument('--row', type=int, default=0)
    parser.add_argument('--frames', type=int, default=6)
    args = parser.parse_args()
    rep = build_native_representation(args.stats)
    proxy = dict(np.load(args.proxy, allow_pickle=True))
    mass = np.array(proxy['mass'], dtype=np.float64); com = np.array(proxy['com'], dtype=np.float64)
    neutral = rep.skeleton.neutral_joints.detach().cpu().numpy()
    bad = (mass <= 0.0) | (np.linalg.norm(com - neutral, axis=-1) > 1.0)
    com[bad] = neutral[bad]; mass[bad] = 0.0
    proxy['mass'] = mass; proxy['com'] = com
    feet_idx = foot_indices(proxy)
    dataset = PFNNFrameDataset(args.cache, args.pfnn, rep, 'validation')
    world = torch.from_numpy(dataset[args.row]['world'].numpy())[None]
    pose = rep.inverse(world, False, return_all=True, joint_positions_from='global_rot_data')
    joints = pose['posed_joints'][0]; rots = pose['global_joint_rots'][0]
    contacts = (world[0][:, rep.indices['foot_contacts']] > 0.5).numpy()
    n = min(joints.shape[0], rots.shape[0], contacts.shape[0])
    joints, rots, contacts = joints[:n], rots[:n], contacts[:n]
    coms, m = segment_com_world(joints, rots, proxy, neutral)
    # Second differences at 30 Hz amplify noise roughly a thousandfold. A local
    # quadratic fit over a five-frame window is the standard remedy and does not
    # change the contact set; the raw trajectory is still what the slip test measures.
    coms = torch.from_numpy(savgol_filter(coms.detach().cpu().numpy(), 5, 2, axis=0))
    coms = coms.detach().cpu().numpy(); m = m.detach().cpu().numpy()
    total = m.sum(); centre = (coms * m[:, None]).sum(1) / total
    seg_v = np.zeros_like(coms); seg_v[1:-1] = (coms[2:] - coms[:-2]) * FPS / 2.0
    centre_v = (seg_v * m[None, :, None]).sum(1) / total
    accel = np.zeros_like(centre); accel[1:-1] = (centre_v[2:] - centre_v[:-2]) * FPS / 2.0
    rel = coms - centre[:, None]
    mom = (m[None, :, None] * np.cross(rel, seg_v - centre_v[:, None])).sum(1) / total
    torque = np.zeros_like(mom); torque[1:-1] = (mom[2:] - mom[:-2]) * FPS / 2.0
    feet = joints.detach().cpu().numpy()[:, feet_idx, :]
    up = np.array([0.0, 1.0, 0.0]); shown = 0
    for t in range(1, n - 1):
        active = sorted({i for side in (0, 1) for i in (2 * side, 2 * side + 1) if contacts[t, 2 * side:2 * side + 2].any()})
        if not active or shown >= args.frames:
            continue
        base = feet[t, active] - centre[t]
        axis = base[1] - base[0]; lat = np.cross(axis, up)
        lat = lat / (np.linalg.norm(lat) + 1e-12)
        points = np.stack([base[0], base[1], base[1] + 0.045 * lat])
        A = wrench_matrix(points); b = np.concatenate((accel[t] - G, torque[t]))
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        free_res = float(np.linalg.norm(A @ sol - b))
        for mu in (0.5, 1.0, 2.0):
            res, f = cone_residual(A, b, mu)
            print(f'frame {t:3d} mu={mu:<4} unconstrained={free_res:.2e} cone_residual={res:.3f} '
                  f'min_fy={min(f[1], f[4], f[7]):+.2f} max|fh|={max(np.hypot(f[0], f[2]), np.hypot(f[3], f[5]), np.hypot(f[6], f[8])):.2f}')
        print(f'  linear={np.linalg.norm(accel[t]-G):.2f}  angular={np.linalg.norm(torque[t]):.3f}  points={len(points)}')
        shown += 1


if __name__ == '__main__':
    main()
