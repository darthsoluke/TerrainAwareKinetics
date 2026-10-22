import argparse, json, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.pfnn_full_training_data import PFNNFrameDataset
from scripts.audit_pfnn_physical_consistency import foot_indices
p = argparse.ArgumentParser()
for n in ('cache','pfnn','stats','proxy'):
    p.add_argument('--' + n, type=pathlib.Path, required=True)
p.add_argument('--per-category', type=int, default=8)
p.add_argument('--start', type=int, default=16)
p.add_argument('--steps', type=int, default=17)
a = p.parse_args()
rep = build_native_representation(a.stats)
proxy = dict(np.load(a.proxy, allow_pickle=True))
fi = foot_indices(proxy)
ds = PFNNFrameDataset(a.cache, a.pfnn, rep, 'validation')
picked = {}
idx = []
for i, r in enumerate(ds.records):
    c = r['category']
    if picked.get(c, 0) < a.per_category:
        picked[c] = picked.get(c, 0) + 1
        idx.append(i)
st, sw, body = [], [], []
for i in idx:
    world = torch.from_numpy(ds[i]['world'].numpy())[None]
    pose = rep.inverse(world, False, return_all=False, joint_positions_from='global_rot_data')
    j = pose['posed_joints'][0]
    stop = min(a.start + a.steps, j.shape[0])
    step = j[a.start:stop] - j[a.start - 1:stop - 1]
    mask = (world[0][a.start:stop][:, rep.indices['foot_contacts']] > 0.5).numpy()
    f = step[:, fi].norm(dim=-1).numpy()
    st.append(f[mask]); sw.append(f[~mask]); body.append(step.norm(dim=-1).numpy().ravel())
st = np.concatenate(st); sw = np.concatenate(sw); body = np.concatenate(body)
out = dict(rows=len(idx), categories=dict(picked),
    persistence_stance_p50_m=float(np.median(st)), persistence_stance_values=int(st.size),
    persistence_swing_p50_m=float(np.median(sw)), persistence_swing_values=int(sw.size),
    persistence_body_p50_m=float(np.median(body)))
print(json.dumps(out, indent=2))
