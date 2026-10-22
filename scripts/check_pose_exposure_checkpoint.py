"""Read one stable post-migration Pose checkpoint without replaying training."""

import argparse
import hashlib
import json
from pathlib import Path


def main(a):
    import numpy as np
    import torch
    from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches
    from motionbricks.data.reference_exposure_budget import ExposureBudget
    from scripts.run_pfnn_bias_workflow import digest, read, save

    if a.output.exists():
        raise ValueError('This one-time checkpoint review already has a result')
    path = a.directory/'pose/latest.pt'
    before = path.stat()
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    old = torch.load(a.previous/'pose'/a.handoff_name, map_location='cpu', mmap=True, weights_only=False)
    contract = saved['contract']
    if a.require_label_graphs and contract.get('frozen_pose_label_graphs') is not True:
        raise ValueError('This review requires the frozen-label graph contract')
    policy = contract['exposure_budget']
    budget = ExposureBudget(policy['initial_updates'], policy['initial_exposures'], 6144,
        saved['actual_updates'], saved['actual_exposures'])
    if a.same_contract_recovery:
        recovery = read(a.directory/'recovery.json')
        binding_ok = (contract == old['contract'] and recovery['no_optimizer_updates_lost']
            and recovery['failed_graph_optimizer_updates'] == 0
            and recovery['resumed_actual_updates'] == old['actual_updates']
            and recovery['resumed_actual_exposures'] == old['actual_exposures'])
    else:
        migration = read(a.directory/'migration.json')
        binding_ok = (contract['implementation_migration_sha256'] == digest(a.directory/'migration.json')
            and migration['actual_updates'] == old['actual_updates'] and migration['no_optimizer_updates_lost'])
    codec_directory = a.previous/'codec'
    if not codec_directory.is_dir():
        codec_directory = Path(read(a.previous/'codec_completion.json')['directory'])
    steps = {int(value['step']) for value in saved['optimizer_state']['state'].values()}
    cuda = saved['rng_state']['cuda']
    if (saved['actual_updates'] <= old['actual_updates'] or steps != {saved['actual_updates']}
            or saved['scheduler_state']['last_epoch'] != budget.equivalent_steps
            or len(cuda) != 2 or not torch.equal(cuda[0], cuda[1])
            or contract['pose_parallel_devices'] != 2 or contract['batch'] != 6144
            or contract['micro_batch'] != 3072 or contract['cuda_memory_fraction'] != .9
            or not contract['isolated_pose_teacher_rng']
            or contract['protected_sha256'] != old['contract']['protected_sha256']
            or contract['parent_sha256'] != read(codec_directory/'complete.json')['checkpoint_sha256']
            or not binding_ok):
        raise ValueError('Pose checkpoint optimizer, exposure, RNG, parent or migration binding differs')
    run = a.directory.parent
    train = PrimitiveDataset([run/'cache_mixed_v1/data', run/'cache_pfnn_world_v2/data'], 'train')
    schedule = BalancedPrimitiveBatches(train, 256, 50000, seed=contract['seed'])
    sample_digest = old['sample_digest']
    for step in range(old['actual_exposures']//256, budget.equivalent_steps):
        ids = np.asarray([train.ids[index] for index in schedule.indices(step)], dtype='<i8').tobytes()
        sample_digest = hashlib.sha256(bytes.fromhex(sample_digest)+ids).hexdigest()
    after = path.stat()
    identity = lambda value: (value.st_ino, value.st_size, value.st_mtime_ns)
    if sample_digest != saved['sample_digest'] or identity(before) != identity(after):
        raise ValueError('Reference sample chain or checkpoint file identity changed')
    save(a.output, dict(passed=True, actual_updates=saved['actual_updates'], actual_exposures=saved['actual_exposures'],
        equivalent_reference_updates=budget.equivalent_steps, adam_state_count=len(saved['optimizer_state']['state']),
        adam_steps=sorted(steps), scheduler_epoch=saved['scheduler_state']['last_epoch'], cuda_rng_count=len(cuda),
        cuda_rng_equal=True, reference_chain_recomputed_steps=budget.equivalent_steps-old['actual_exposures']//256,
        sample_digest=sample_digest, stable_checkpoint_identity=identity(after), parent_sha256=contract['parent_sha256'],
        protected_sha256=contract['protected_sha256'], metadata_review_only=True, optimizer_replayed=False,
        model_tensors_rehashed=False, same_contract_recovery=a.same_contract_recovery, quality_approved=False))
    print(a.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('directory', 'previous', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--handoff-name', choices=('exposure_handoff.pt', 'graph_handoff.pt'), default='exposure_handoff.pt')
    parser.add_argument('--require-label-graphs', action='store_true')
    parser.add_argument('--same-contract-recovery', action='store_true')
    main(parser.parse_args())
