"""Revalidate an immutable codec checkpoint using the actual publication contract."""
import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.played_motion_review import PUBLICATION_GEOMETRY
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.probe_unified_native_codec import evaluate
from scripts.probe_unified_native_stages import prepare
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(a.weights, map_location='cpu', weights_only=False)
    if saved['contract']['cases_sha256'] != sha256(a.cases) or saved['contract']['bundle_sha256'] != sha256(a.bundle):
        raise ValueError('Checkpoint provenance differs')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    controller.load_state_dict(saved['model_state'], strict=True); controller.eval().requires_grad_(False)
    prepared = prepare(controller, torch.load(a.cases, map_location='cpu', weights_only=False),
                       saved['contract']['source_windows'], a.device)
    records = []
    for item in prepared:
        case = dict(item['case'], label=item['window']['label'] + f'-f{item["window"]["start"]}',
                    probe_start=item['window']['start'])
        records.append((case, item['source'], item['example']))
    rows = evaluate(controller, records, JointGeometryObjective(inference.global_motion_rep, a.device), a.output / 'recordings')
    write_json(a.output / 'complete.json', dict(schema='motionbricks.codec_publication_gate.v1',
        geometry_contract=PUBLICATION_GEOMETRY, source_checkpoint_sha256=sha256(a.weights),
        legacy_report_sha256=sha256(a.weights.with_name('complete.json')), cases_sha256=sha256(a.cases),
        source_windows=saved['contract']['source_windows'], rows=rows,
        training_bank_gate_pass=all(r['capacity_gate_pass'] for r in rows),
        optimizer_updates=0, runtime_changed=False, future_frames_modified=False,
        thresholds=dict(leg_mpjpe_m=.015, root_rms_m=.02, skin_peak_m=.02), quality_approved=False))
    controller.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'weights', 'cases', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    main(p.parse_args())
