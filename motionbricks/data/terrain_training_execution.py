"""Explicit execution migration and bounded asynchronous checkpoint persistence."""

from concurrent.futures import ThreadPoolExecutor

import torch


def validate_execution_migration(previous,current,*,sample_budget_authorized=False):
    allowed={'global_batch','micro_batch','world_size','workers','validate_every','checkpoint_every',
        'code_sha256','execution_options','execution_migration','sampler_draw_offset'}
    if sample_budget_authorized:allowed.update(('steps','sample_budget'))
    changed={key for key in previous.keys()|current.keys() if previous.get(key)!=current.get(key)}
    if changed-allowed:raise ValueError('Execution migration changed learning contract: '+str(sorted(changed-allowed)))
    permitted_code={'scripts/train_joint_terrain_v3.py','motionbricks/data/terrain_joint_objective.py',
        'motionbricks/data/terrain_batched_geometry.py','motionbricks/data/terrain_training_execution.py',
        'motionbricks/data/terrain_model_v3.py','motionbricks/data/terrain_sample_budget.py'}
    old=previous['code_sha256'];new=current['code_sha256']
    code_changes={key for key in old.keys()|new.keys() if old.get(key)!=new.get(key)}
    if code_changes-permitted_code:raise ValueError('Execution migration changed unapproved model/data code: '+str(sorted(code_changes-permitted_code)))
    return sorted(changed)


def cpu_snapshot(value):
    if torch.is_tensor(value):return value.detach().to('cpu',copy=True)
    if isinstance(value,dict):return {k:cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_snapshot(v) for v in value)
    return value


class AsyncCheckpointWriter:
    def __init__(self):
        self.executor=ThreadPoolExecutor(max_workers=1);self.pending=None

    @staticmethod
    def write(path,value):
        temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)

    def submit(self,path,value):
        self.flush()
        snapshot=cpu_snapshot(value)
        self.pending=self.executor.submit(self.write,path,snapshot)

    def flush(self):
        if self.pending is not None:self.pending.result();self.pending=None

    def close(self):
        self.flush();self.executor.shutdown()
