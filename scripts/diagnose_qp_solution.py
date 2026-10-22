import argparse, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from scripts.audit_pfnn_physical_consistency import FPS, G, foot_indices, segment_com_world

parser = argparse.ArgumentParser()
for name in ('cache','pfnn','stats','proxy'):
    parser.add_argument('--' + name, type=pathlib.Path, required=True)
parser.add_argument('--row', type=int, default=0)
parser.add_argument('--frames', type=int, default=3)
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
for t in range(1, n - 1):
    active = sorted({i for side in (0,1) for i in (2*side, 2*side+1) if contacts[t, 2*side:2*side+2].any()})
    if not active or shown >= args.frames:
        continue
    base = feet[t, active] - centre[t]
    axis = base[1] - base[0]; lat = np.cross(axis, up); lat = lat / (np.linalg.norm(lat) + 1e-12)
    # Four corners of the foot patch. Three points with two of them nearly coincident
    # make the moment rows ill-conditioned and drive enormous opposing forces.
    points = np.stack([base[0] + 0.045 * lat, base[0] - 0.045 * lat,
                       base[1] + 0.045 * lat, base[1] - 0.045 * lat])
    A = np.zeros((6, 12))
    for k, r in enumerate(points):
        A[0:3, 3*k:3*k+3] = np.eye(3)
        A[3:6, 3*k:3*k+3] = np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
    b = np.concatenate((a[t] - G, tau[t]))
    f, *_ = np.linalg.lstsq(A, b, rcond=None)
    print('frame %d  residual %.2e' % (t, float(np.linalg.norm(A @ f - b))))
    for k in range(len(points)):
        fx, fy, fz = f[3*k:3*k+3]
        need = np.hypot(fx, fz)
        print('   point %d  r=(%+.3f,%+.3f,%+.3f)  f=(%+.2f,%+.2f,%+.2f)  |fh|=%.2f  fy=%.2f  need mu=%.2f' % (
              k, points[k][0], points[k][1], points[k][2], fx, fy, fz, need, fy, (need/fy if fy > 1e-9 else float('inf'))))
    shown += 1
