"""Package the exact three-panel G1 reference for the real-time application."""

import argparse
import json
from pathlib import Path

import numpy as np

from motionbricks.mb_sonic.reference import G1_ARRAY_FIELDS, file_sha256, load_g1_reference, save_g1_reference
from scripts.isolate_mb_continuous_reference import load_fixed_reference, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference, receipt = load_fixed_reference(args.case)
    artifact = (args.output/'reference.npz').resolve()
    save_g1_reference(artifact, reference)
    digest = file_sha256(artifact)
    loaded = load_g1_reference(artifact, expected_sha256=digest, scene_id=reference.scene_id)
    for name in G1_ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(reference, name))
    config = json.loads(args.config.read_text())
    config.update(mode='fixed_reference', mb_planner_backend='thread', run_mode='finite', duration_s=16.,
        scene_root=str(args.case.resolve()), initial_scene_id=reference.scene_id,
        enabled_scene_ids=[reference.scene_id],
        scene_manifest_sha256={reference.scene_id: file_sha256(args.case/'scene/scene_manifest.json')},
        fixed_reference=dict(path=str(artifact), sha256=digest),
        recording_dir=str((args.output/'recording').resolve()))
    for name in ('scene_catalog', 'scene_query_proofs', 'scene_seeds', 'mb_conditioning_mode',
                 'state_feedback_segments', 'state_feedback_stop_segments'):
        config.pop(name, None)
    save_json(args.output/'config.json', config)
    save_json(args.output/'package.json', dict(original_reference=receipt,
        artifact_sha256=digest, all_arrays_bitwise_equal=True,
        initialization='exact_original_first_qpos;zero_qvel;100_PD_warmup_substeps',
        clock='original_absolute_reference_times;no_warmup_rebase',
        optimizer_updates=0, interpolation=False, filtering=False))
    print(json.dumps(dict(config=str((args.output/'config.json').resolve()), sha256=digest)))


if __name__ == '__main__':
    main()
