import argparse, pathlib, sys
import numpy as np, torch
from scipy.optimize import minimize
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from scripts.audit_pfnn_physical_consistency import FPS, G, foot_indices, segment_com_world

def matrix(points):
    A = np.zeros((6, 3 * len(points)))
    for k, r in enumerate(points):
        A[0:3, 3*k:3*k+3] = np.eye(3)
        A[3:6, 3*k:3*k+3] = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
    return A

def constrained(A, b, mu, rows):
    As, bs = A[rows], b[rows]
    n = A.shape[1]
    def f(x):
        r = As @ x - bs
        return float(r @ r)
    def g(x):
        return 2.0 * As.T @ (As @ x - bs)
    cons = []
    for k in range(n // 3):
        x, y, z = 3*k, 3*k+1, 3*k+2
        cons += [{'type':'ineq','fun':(lambda t,y=y: t[y])},
                 {'type':'ineq','fun':(lambda t,x=x,y=y: mu*t[y]-t[x])},
                 {'type':'ineq','fun':(lambda t,x=x,y=y: mu*t[y]+t[x])},
                 {'type':'ineq','fun':(lambda t,z=z,y=y: mu*t[y]-t[z])},
                 {'type':'ineq','fun':(lambda t,z=z,y=y: mu*t[y]+t[z])}]
    start = np.zeros(n)
    for k in range(n // 3):
        start[3*k+1] = max(bs[1], 0.0) / max(n // 3, 1)
    res = minimize(f, start, jac=g, constraints=cons, method='SLSQP',
                   options=dict(maxiter=500, ftol=1e-12))
    return float(np.sqrt(max(res.fun, 0.0)))

parser = argparse.ArgumentParser()
parser.add_argument('--cache', type=pathlib.Path, required=True)
parser.add_argument('--pfnn', type=pathlib.Path, required=True)
parser.add_argument('--stats', type=pathlib.Path, required=True)
parser.add_argument('--proxy', type=pathlib.Path, required=True)
parser.add_argument('--row', type=int, default=0)
parser.add_argument('--frames', type=int, default=4)
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
coms = coms.detach().cpu().numpy(); m = m.detach().cpu().numpy()
total = m.sum(); centre = (coms * m[:, None]).sum(1) / total
v = np.zeros_like(coms); v[1:-1] = (coms[2:] - coms[:-2]) * FPS / 2.0
cv = (v * m[None, :, None]).sum(1) / total
a = np.zeros_like(centre); a[1:-1] = (cv[2:] - cv[:-2]) * FPS / 2.0
rel = coms - centre[:, None]
h = (m[None, :, None] * np.cross(rel, v - cv[:, None])).sum(1) / total
tau = np.zeros_like(h); tau[1:-1] = (h[2:] - h[:-2]) * FPS / 2.0
feet = joints.detach().cpu().numpy()[:, feet_idx, :]
up = np.array([0.0, 1.0, 0.0]); shown = 0
print('frame  linear  |tau_x| |tau_y| |tau_z|   res_lin  res+pitch/roll  res_all(mu=1)  res_all(mu=2)')
for t in range(1, n - 1):
    if shown >= args.frames:
        break
    active = sorted({i for side in (0,1) for i in (2*side, 2*side+1) if contacts[t, 2*side:2*side+2].any()})
    if not active:
        continue
    base = feet[t, active] - centre[t]
    axis = base[1] - base[0]; lat = np.cross(axis, up); lat = lat / (np.linalg.norm(lat) + 1e-12)
    points = np.stack([base[0], base[1], base[1] + 0.045 * lat])
    A = matrix(points); b = np.concatenate((a[t] - G, tau[t]))
    r_lin = constrained(A, b, 1.0, [0,1,2])
    r_pr  = constrained(A, b, 1.0, [0,1,2,3,5])
    r_all1 = constrained(A, b, 1.0, [0,1,2,3,4,5])
    r_all2 = constrained(A, b, 2.0, [0,1,2,3,4,5])
    print('%5d  %6.2f  %6.3f %6.3f %6.3f   %7.3f  %13.3f  %13.3f  %13.3f' % (
          t, np.linalg.norm(a[t]-G), tau[t][0], tau[t][1], tau[t][2], r_lin, r_pr, r_all1, r_all2))
    shown += 1
