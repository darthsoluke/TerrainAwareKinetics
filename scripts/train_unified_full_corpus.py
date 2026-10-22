"""Source-balanced full-corpus native MB stages, with exact optimizer-step resume."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import DataLoader

from motionbricks.data.full_terrain_cache import PrimitiveDataset, BalancedPrimitiveBatches, collate_primitives
from motionbricks.data.unified_full_training import NativeTrainingStage
from motionbricks.data.unified_terrain import UnifiedTerrainController
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest
from scripts.train_joint_terrain_v3 import write_json


SCHEMA = 'motionbricks.unified.full_corpus.v1'


def validate_implementation_migration(saved, contract, receipt, checkpoint_hash):
    if receipt['from_checkpoint_sha256'] != checkpoint_hash:
        raise ValueError('Implementation migration checkpoint changed')
    if receipt.get('transition') == 'pose-frozen-label-loader-v1':
        from motionbricks.data.pose_loader_graph_migration import validate_pose_loader_graph_migration
        return validate_pose_loader_graph_migration(saved, contract, receipt)
    if receipt.get('transition') == 'pose-frozen-label-graph-v1':
        from motionbricks.data.pose_graph_migration import validate_pose_graph_migration
        return validate_pose_graph_migration(saved, contract, receipt)
    if receipt.get('transition') == 'pose-reference-exposure-v1':
        from motionbricks.data.pose_exposure_migration import validate_pose_exposure_migration
        return validate_pose_exposure_migration(saved, contract, receipt)
    if receipt.get('transition') in ('codec-reference-exposure-v1','codec-reference-exposure-four-v1'):
        from motionbricks.data.codec_exposure_migration import validate_codec_exposure_migration
        return validate_codec_exposure_migration(saved,contract,receipt)
    if receipt.get('transition') == 'codec-dilated-graph-v1':
        return validate_codec_graph_migration(saved, contract, receipt)
    if receipt.get('transition') in ('root-parallel-memory-v1','root-parallel-four-v1'):
        return validate_execution_migration(saved, contract, receipt)
    old = saved['contract']
    if (old.get('coalesced_dilated_bias', False) or not contract['coalesced_dilated_bias']
            or old.get('pose_token_memoization', False) or contract['pose_token_memoization']
            or old['stage'] != 'root' or contract['stage'] != 'root'):
        raise ValueError('Only the validated native Root to coalesced-bias transition is allowed')
    changed = {'source_code_sha256', 'coalesced_dilated_bias', 'pose_token_memoization',
        'implementation_migration_sha256'}
    if {k:v for k,v in old.items() if k not in changed} != {k:v for k,v in contract.items() if k not in changed}:
        raise ValueError('Migration changed the data, model scope, optimizer or budget')
    if (receipt['from_source_code_sha256'] != old['source_code_sha256']
            or receipt['to_source_code_sha256'] != contract['source_code_sha256']
            or receipt['actual_updates'] != saved['actual_updates']
            or saved['actual_exposures'] != saved['actual_updates']*contract['batch']):
        raise ValueError('Migration source identity or accounting mismatch')
    if receipt['validated_implementation_sha256'] != contract['source_code_sha256'].get(
            'motionbricks/data/dilated_bias_convolution.py'):
        raise ValueError('Validated accelerated implementation differs')
    for proof in receipt['proofs']:
        if sha256(proof['path']) != proof['sha256'] or not json.loads(Path(proof['path']).read_text())['passed']:
            raise ValueError('Migration numerical or resume proof changed')
    if {p['role'] for p in receipt['proofs']} != {'full_model', 'optimizer_resume'}:
        raise ValueError('Both full-model and optimizer-resume proof are required')


def validate_codec_graph_migration(saved, contract, receipt):
    old = saved['contract']
    if (old['stage'] != 'codec' or contract['stage'] != 'codec'
            or not old.get('coalesced_dilated_bias') or not contract['coalesced_dilated_bias']
            or old.get('graph_dilated_convolutions',False) or not contract['graph_dilated_convolutions']
            or old['cuda_memory_fraction'] != .1 or contract['cuda_memory_fraction'] != .125):
        raise ValueError('Unexpected codec graph or memory configuration')
    changed = {'source_code_sha256','implementation_migration_sha256',
        'cuda_memory_fraction','graph_dilated_convolutions'}
    if {k:v for k,v in old.items() if k not in changed} != {k:v for k,v in contract.items() if k not in changed}:
        raise ValueError('Codec graph transition changed learning or data contract')
    if (receipt['from_source_code_sha256'] != old['source_code_sha256']
            or receipt['to_source_code_sha256'] != contract['source_code_sha256']
            or receipt['actual_updates'] != saved['actual_updates']
            or saved['actual_exposures'] != saved['actual_updates']*contract['batch']):
        raise ValueError('Codec migration source or accounting mismatch')
    implementation = contract['source_code_sha256']['motionbricks/data/dilated_graph_convolution.py']
    if receipt['graph_implementation_sha256'] != implementation:
        raise ValueError('Graph implementation differs from proof')
    roles = set()
    for proof in receipt['proofs']:
        if sha256(proof['path']) != proof['sha256']: raise ValueError('Codec execution proof changed')
        result = json.loads(Path(proof['path']).read_text())
        if proof['role'] == 'cold_process':
            integration = next((item for item in receipt['proofs'] if item['role']=='stage_integration'), None)
            if (not result['passed'] or result.get('cold_process') is not True
                    or result['actual_optimizer_updates']!=2 or result['endpoint_changes']
                    or not result['stage_integration'] or result['graph_memory_fraction']!=.125
                    or result['graph_mode']!='both' or result['implementation_sha256']!=implementation
                    or not all(row['passed'] and row['bitwise_equal'] for row in result['computation_gates'])
                    or integration is None or result['reference_proof_sha256']!=integration['sha256']):
                raise ValueError('Fresh-process codec graph recovery differs')
            roles.add(proof['role'])
            continue
        if (not result['passed'] or result['implementation_sha256'] != implementation
                or result['graph_mode'] != 'both' or result['graph_memory_fraction'] != .125
                or result['native_memory_fraction'] != .1 or not result['tolerances_unchanged']
                or result['actual_optimizer_updates'] != 12
                or result['native_graph_endpoint_changes'] or result['cold_resume_endpoint_changes']
                or result['native_to_graph_endpoint_changes']
                or not all(result['restoration_bitwise_checks'])
                or not all(row['passed'] and row['bitwise_equal'] for row in result['computation_gates'])):
            raise ValueError('Codec graph requires exact Adam/EMA/RNG trajectory and resume')
        if proof['role'] == 'stage_integration':
            if result.get('stage_integration') is not True: raise ValueError('Stage integration untested')
            for name, expected in result['integration_source_sha256'].items():
                if contract['source_code_sha256'].get(name) != expected:
                    raise ValueError('Integrated training stage differs from proof')
        roles.add(proof['role'])
    if roles != {'optimizer_resume','stage_integration','cold_process'}: raise ValueError('Codec graph proofs incomplete')


def validate_execution_migration(saved, contract, receipt):
    old = saved['contract']
    four = receipt.get('transition') == 'root-parallel-four-v1'
    if (old['stage'] != 'root' or contract['stage'] != 'root'
            or not old.get('coalesced_dilated_bias') or not contract['coalesced_dilated_bias']
            or old.get('cuda_memory_fraction', 1.) != (.1 if four else 1.) or contract['cuda_memory_fraction'] != .1
            or old.get('root_parallel_devices', 1) != (2 if four else 1)
            or contract['root_parallel_devices'] != (4 if four else 2)):
        raise ValueError('Unexpected parallel/memory execution transition')
    changed = {'source_code_sha256', 'implementation_migration_sha256',
        'cuda_memory_fraction', 'root_parallel_devices'}
    if {k:v for k,v in old.items() if k not in changed} != {k:v for k,v in contract.items() if k not in changed}:
        raise ValueError('Execution transition changed data, model, optimizer or budget')
    if (receipt['from_source_code_sha256'] != old['source_code_sha256']
            or receipt['to_source_code_sha256'] != contract['source_code_sha256']
            or receipt['actual_updates'] != saved['actual_updates']
            or saved['actual_exposures'] != saved['actual_updates']*contract['batch']):
        raise ValueError('Execution migration identity/accounting mismatch')
    if receipt['parallel_implementation_sha256'] != contract['source_code_sha256'].get(
            'motionbricks/data/parallel_root_training.py'):
        raise ValueError('Parallel implementation differs from the validated source')
    roles = set()
    for proof in receipt['proofs']:
        if sha256(proof['path']) != proof['sha256']:
            raise ValueError('Execution proof file changed')
        result = json.loads(Path(proof['path']).read_text())
        if not result['passed']:
            raise ValueError('Execution proof failed')
        if proof['role'] == 'memory_equivalence' and not result['tolerances_unchanged']:
            raise ValueError('Memory policy proof must preserve numerical gates')
        if proof['role'] == 'parallel_optimizer_resume' and (
                result['rng_schedule_from_reference'] or result['implementation_sha256'] != receipt['parallel_implementation_sha256']):
            raise ValueError('Independent parallel RNG and optimizer continuity are required')
        if four and (result.get('reference_devices')!=2 or result.get('target_devices')!=4
                or result.get('resume_from_reference_devices')!=2
                or result.get('serial_errors')!=[] or result.get('resume_errors')!=[]
                or result.get('logical_cuda_rng_canonicalized') is not True
                or result.get('reference_implementation_sha256')!=old['source_code_sha256'].get(
                    'motionbricks/data/parallel_root_training.py')):
            raise ValueError('Four-device proof must bind the deployed two-device optimizer and resume')
        roles.add(proof['role'])
    expected_roles = {'parallel_optimizer_resume'} if four else {'memory_equivalence', 'parallel_optimizer_resume'}
    if roles != expected_roles:
        raise ValueError('Both memory and independent parallel optimizer proofs are required')


def device_batch(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: device_batch(value, device) for key, value in batch.items()}
    return batch


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def load_full_model(controller, saved, bundle_hash):
    if saved.get('schema') != SCHEMA or saved['contract']['bundle_sha256'] != bundle_hash:
        raise ValueError('Full-corpus model schema or bundle identity differs')
    controller.load_state_dict(saved['model_state'], strict=True)
    # This nonpersistent lookup belongs to the loaded codec, not the old bundle.
    controller.codebook.copy_(controller.codec.get_codebook())


def dataset_contract(paths, bundle_hash, mechanics_only=False):
    receipts = []
    families = defaultdict(set)
    for folder in paths:
        complete = json.loads((folder/'complete.json').read_text())
        if (complete['pilot'] and not mechanics_only) or complete['bundle_sha256'] != bundle_hash:
            raise ValueError('Pilot data or different normalization/bundle cannot enter full training')
        for line in (folder/'data/records.jsonl').read_text().splitlines():
            row = json.loads(line)
            families[(row['domain'], row['source_group'])].add(row['split'])
        receipts.append(dict(path=str(folder.resolve()), complete_sha256=sha256(folder/'complete.json'),
            records_sha256=sha256(folder/'data/records.jsonl')))
    leaked = [str(key) for key, values in families.items() if len(values) != 1]
    if leaked and not mechanics_only:
        raise ValueError('Recording families cross splits: '+str(leaked[:20]))
    return receipts


def split_micro(batch, first, last):
    if torch.is_tensor(batch):
        return batch[first:last]
    if isinstance(batch, dict):
        return {key: split_micro(value, first, last) for key, value in batch.items()}
    return batch[first:last]


@torch.no_grad()
def validation(stage, dataset, batch_size, per_domain=128):
    state = rng_state()
    stage.eval()
    result = {}
    try:
        for domain in ('pfnn', 'parc', 'bones'):
            ids = [i for i, actual in enumerate(dataset.ids) if dataset.records[actual]['domain'] == domain]
            if not ids:
                raise ValueError('No held-out '+domain+' examples')
            ids = np.random.default_rng(123).permutation(ids)[:per_domain].tolist()
            total = 0.
            for start in range(0, len(ids), batch_size):
                selected = ids[start:start+batch_size]
                batch = device_batch(collate_primitives([dataset[i] for i in selected]),
                                     stage.controller.inference.device)
                value = stage(batch, 8123+start)
                if not torch.isfinite(value):
                    raise ValueError('Nonfinite validation objective')
                total += float(value)*len(selected)
            result[domain] = dict(examples=len(ids), native_loss=total/len(ids))
    finally:
        restore_rng(state)
        stage.train(True)
    return result


def main(a):
    torch.set_num_threads(4)
    codec_exposure=getattr(a,'codec_exposure_pipeline',False)
    pose_exposure=getattr(a,'pose_exposure_parallel',False)
    pose_label_graphs=getattr(a,'graph_frozen_pose_labels',False)
    if pose_label_graphs and not pose_exposure:
        raise ValueError('Frozen label graphs require the verified two-device Pose exposure policy')
    exposure_mode=codec_exposure or pose_exposure
    codec_devices=getattr(a,'codec_pipeline_devices',2)
    if pose_exposure and (codec_exposure or a.stage!='pose' or a.batch!=6144 or a.micro_batch!=3072
            or a.cuda_memory_fraction!=.9 or a.graph_dilated_convolutions
            or not a.coalesce_dilated_bias or not a.cache_pose_tokens or a.root_parallel_devices!=1
            or a.steps!=50000 or a.resume is None):
        raise ValueError('Exposure Pose requires the verified6144/3072/two-GPU continuation')
    if codec_exposure and (a.stage!='codec' or a.batch!=512*codec_devices or a.micro_batch!=512
            or a.cuda_memory_fraction!=.9 or not a.graph_dilated_convolutions
            or not a.coalesce_dilated_bias or a.root_parallel_devices!=1 or a.steps!=50000
            or a.resume is None):
        raise ValueError('Exposure codec requires the approved1024/512/two-GPU continuation')
    execution_devices=2 if pose_exposure else codec_devices if codec_exposure else a.root_parallel_devices
    preload_workspaces=[]
    if not 0 < a.cuda_memory_fraction <= 1 or a.root_parallel_devices not in (1, 2, 4):
        raise ValueError('Invalid CUDA execution configuration')
    if a.root_parallel_devices > 1 and (a.stage != 'root' or a.micro_batch != 64 or a.batch != 256):
        raise ValueError('Parallel execution requires Root batch256/micro64')
    if a.graph_dilated_convolutions and not exposure_mode and (a.stage != 'codec' or not a.coalesce_dilated_bias
            or a.cuda_memory_fraction != .125 or a.batch != 256 or a.micro_batch != 64
            or a.root_parallel_devices != 1):
        raise ValueError('Codec convolution graphs require validated256/64 and0.125memory policy')
    if a.device.startswith('cuda'):
        if torch.cuda.device_count() < execution_devices:
            raise ValueError('Not enough visible CUDA devices')
        for index in range(execution_devices):
            torch.cuda.set_per_process_memory_fraction(a.cuda_memory_fraction, index)
        torch.cuda.set_device(a.device)
        if exposure_mode and a.await_resume:
            # Hold the measured imminent working set while CPU checks and the
            # old checkpoint finish; release it before graph construction.
            preload_workspaces=[torch.empty(50*1024**3,dtype=torch.uint8,device=f'cuda:{index}')
                                for index in range(execution_devices)]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Native codec FK uses CUDA cumsum, whose deterministic mode is unsupported.
    # Keep that original operation and record the narrower reproducibility limit.
    torch.use_deterministic_algorithms(True, warn_only=a.stage == 'codec')
    if a.batch % 4 or a.micro_batch % 4 or a.batch % a.micro_batch:
        raise ValueError('Balanced batches and microbatches must divide by4 exactly')
    if a.steps < 1 or not 0 <= a.stop_after <= a.steps:
        raise ValueError('Invalid optimizer update budget')
    preflight = json.loads(a.preflight.read_text())
    bundle_hash = sha256(a.bundle)
    if not preflight.get('passed') or preflight['bundle_sha256'] != bundle_hash:
        raise ValueError('Actual-model causal input and gradient preflight must pass')
    if a.mechanics_only and (a.steps > 4 or a.batch > 16):
        raise ValueError('Mechanics fixture is limited to4updates/batch16')
    receipts = dataset_contract(a.cache, bundle_hash, a.mechanics_only)
    audit_hash = None
    if not a.mechanics_only:
        if a.corpus_audit is None:
            raise ValueError('Full training requires the complete corpus/hash audit')
        audit = json.loads(a.corpus_audit.read_text())
        if not audit['passed'] or audit['receipts'] != receipts:
            raise ValueError('Corpus audit does not identify the selected full caches')
        for field in audit['fields']:
            if sha256(field['path']) != field['sha256']:
                raise ValueError('Cached primitive tensor changed: '+field['path'])
        audit_hash = sha256(a.corpus_audit)
    train = PrimitiveDataset([path/'data' for path in a.cache], 'train')
    heldout = PrimitiveDataset([path/'data' for path in a.cache], 'validation')
    inference, bundle = load_terrain_bundle(a.bundle, a.device)
    controller = UnifiedTerrainController(inference, bundle)
    parent = torch.load(a.parent, map_location='cpu', weights_only=False)
    if parent.get('schema') == SCHEMA:
        if parent['contract']['mechanics_only'] != a.mechanics_only:
            raise ValueError('Mechanics checkpoints cannot initialize a full-corpus run')
        load_full_model(controller, parent, bundle_hash)
        expected = dict(root=None, codec='root', pose='codec')[a.stage]
        if a.resume is None and parent['contract']['stage'] != expected:
            raise ValueError('Full stage dependency order must be Root -> codec -> Pose')
    elif a.stage == 'root':
        if parent['contract']['bundle_sha256'] != bundle_hash:
            raise ValueError('Prototype warmstart bundle mismatch')
        if sha256(a.parent) != preflight['model_sha256']:
            raise ValueError('Prototype differs from the actual-model preflight')
        controller.load_state_dict(parent['model_state'], strict=True)
    else:
        raise ValueError('Only Root may warmstart from the preserved prototype')
    if a.stage == 'pose' and a.resume is None:
        controller.pose.init_embedding_from_codebooks(controller.codec.get_codebook(), None)
    controller.codebook.copy_(controller.codec.get_codebook())
    stage = NativeTrainingStage(controller, a.stage, cache_pose_tokens=a.cache_pose_tokens,
        coalesce_dilated_bias=a.coalesce_dilated_bias,
        graph_dilated_convolutions=a.graph_dilated_convolutions,
        isolate_pose_teacher_rng=pose_exposure).train(True)
    replicas, replica_stages, executor = [], [], None
    if execution_devices > 1:
        before_replica_rng = rng_state()
        for index in range(1,execution_devices):
            replica_inference, replica_bundle = load_terrain_bundle(a.bundle, f'cuda:{index}')
            replica = UnifiedTerrainController(replica_inference, replica_bundle)
            replica.load_state_dict(controller.state_dict(), strict=True)
            replica.codebook.copy_(replica.codec.get_codebook())
            replicas.append(replica)
            replica_stages.append(NativeTrainingStage(replica, a.stage,
                cache_pose_tokens=a.cache_pose_tokens,
                coalesce_dilated_bias=a.coalesce_dilated_bias,
                graph_dilated_convolutions=a.graph_dilated_convolutions,
                isolate_pose_teacher_rng=pose_exposure).train(True))
        restore_rng(before_replica_rng)
    if a.await_resume:
        if a.resume is None or a.resume_migration is None:
            raise ValueError('A staged handoff requires explicit checkpoint and migration paths')
        write_json(a.output.parent/(a.stage+'-await-resume.json'), dict(pid=os.getpid(),
            status='models_ready_waiting_checkpoint', resume=str(a.resume), actual_optimizer_updates=0,
            preload_reserved_bytes_per_device=50*1024**3 if preload_workspaces else 0))
        while not a.resume.is_file() or not a.resume_migration.is_file():
            time.sleep(5)
    if preload_workspaces:
        preload_workspaces.clear()
        torch.cuda.empty_cache()
    names = [name for name, parameter in controller.named_parameters() if parameter.requires_grad]
    parameters = [parameter for parameter in controller.parameters() if parameter.requires_grad]
    protected = [name for name in controller.state_dict() if name not in names
                 and not (a.stage == 'codec' and name.startswith('codec.quantizer.'))]
    protected_hash = state_digest(controller, protected)
    counts = Counter(train.records[index]['domain'] for index in train.ids)
    code = Path(__file__).resolve().parents[1]
    source_hashes = {path.relative_to(code).as_posix(): sha256(path)
        for folder in ('motionbricks','scripts') for path in sorted((code/folder).rglob('*.py'))}
    contract = dict(stage=a.stage, mechanics_only=a.mechanics_only,
        bundle_sha256=bundle_hash, parent_sha256=sha256(a.parent),
        cache_receipts=receipts, corpus_audit_sha256=audit_hash,
        preflight_sha256=sha256(a.preflight), seed=a.seed,
        train_examples=dict(counts), steps=a.steps, batch=a.batch, micro_batch=a.micro_batch,
        max_exposures=a.steps*(256 if exposure_mode else a.batch), source_quota=dict(pfnn=.5, parc=.25, bones=.25),
        learning_rate=a.lr, eta_min=5e-7, optimizer='Adam', gradient_clip=.5,
        deterministic_algorithms=True, deterministic_warn_only=a.stage == 'codec',
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        source_history_only=True, future_actor_height_pose=False, native_losses_unchanged=True,
        pose_token_memoization=a.cache_pose_tokens,
        coalesced_dilated_bias=a.coalesce_dilated_bias,
        graph_dilated_convolutions=a.graph_dilated_convolutions,
        cuda_memory_fraction=a.cuda_memory_fraction, root_parallel_devices=a.root_parallel_devices,
        quantizer_ema_updates_per_optimizer_step=a.batch//(64 if exposure_mode else a.micro_batch) if a.stage == 'codec' else 0,
        trainable=names, protected_sha256=protected_hash, source_code_sha256=source_hashes,
        runtime=dict(tokens=8, pose_passes=4, history_frames=4, played_frames=24, packet_frames=8,
            fps=30, refinement=0, postprocessing=None), quality_approved=False)
    optimizer = torch.optim.Adam(parameters, lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.steps, eta_min=5e-7)
    step, sample_digest = 0, hashlib.sha256(b'full-corpus-sample-chain-v1').hexdigest()
    exposure_budget=None
    if a.resume_migration:
        contract['implementation_migration_sha256'] = sha256(a.resume_migration)
    if pose_label_graphs:
        contract['frozen_pose_label_graphs'] = True
        contract['frozen_pose_capture_error_mode'] = 'thread_local'
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        if exposure_mode:
            from motionbricks.data.reference_exposure_budget import ExposureBudget
            prior=saved['contract'].get('exposure_budget')
            same_batch=prior and prior['global_batch']==a.batch
            initial_updates=prior['initial_updates'] if same_batch else saved['actual_updates']
            initial_exposures=prior['initial_exposures'] if same_batch else saved['actual_exposures']
            exposure_budget=ExposureBudget(initial_updates,initial_exposures,a.batch,saved['actual_updates'],saved['actual_exposures'])
            contract.update(exposure_budget=exposure_budget.contract())
            if pose_exposure:
                contract.update(pose_parallel_devices=2,isolated_pose_teacher_rng=True)
            else:contract.update(codec_pipeline_devices=execution_devices)
        if saved['contract'] != contract:
            if a.resume_migration is None or a.output.resolve() == a.resume.parent.resolve():
                raise ValueError('Resume data, source, scope, budget or parent identity differs')
            validate_implementation_migration(saved, contract,
                json.loads(a.resume_migration.read_text()), sha256(a.resume))
        load_full_model(controller, saved, bundle_hash)
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        step, sample_digest = saved['actual_updates'], saved['sample_digest']
        restore_rng(saved['rng_state'])
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        write_json(a.output/'contract.json', contract)
    a.output.mkdir(parents=True, exist_ok=True)
    if not (a.output/'contract.json').exists():
        write_json(a.output/'contract.json', contract)
    if replicas:
        from motionbricks.data.parallel_root_training import ParallelRootMicrobatches
        for replica in replicas:replica.load_state_dict(controller.state_dict(), strict=True)
        if pose_exposure:
            if pose_label_graphs:
                from motionbricks.data.frozen_graph_pose_training import FrozenGraphPoseMicrobatches
                executor=FrozenGraphPoseMicrobatches([stage,*replica_stages])
            else:
                from motionbricks.data.parallel_pose_training import ParallelPoseMicrobatches
                executor=ParallelPoseMicrobatches([stage,*replica_stages])
        elif codec_exposure:
            from motionbricks.data.ordered_codec_pipeline import OrderedCodecPipeline
            executor=OrderedCodecPipeline([stage,*replica_stages],a.micro_batch,a.batch//a.micro_batch)
        else:executor = ParallelRootMicrobatches([stage, *replica_stages])
    start_step, start_time = step, time.monotonic()
    start_exposures=exposure_budget.actual_exposures if exposure_budget else step*a.batch
    schedule = BalancedPrimitiveBatches(train, 256 if exposure_mode else a.batch, a.steps, seed=a.seed,
        start_step=exposure_budget.equivalent_steps if exposure_budget else step)
    sampler=schedule
    if exposure_mode:
        from motionbricks.data.reference_exposure_budget import ReferenceExposureBatches
        sampler=ReferenceExposureBatches(schedule,train,a.batch,exposure_budget.equivalent_steps,a.steps)
    loader = DataLoader(train, batch_sampler=sampler, num_workers=a.workers,
        collate_fn=collate_primitives, pin_memory=True,
        generator=torch.Generator().manual_seed(a.seed), persistent_workers=a.workers > 0)

    def checkpoint():
        if state_digest(controller, protected) != protected_hash:
            raise ValueError('A frozen module or buffer changed')
        value = dict(schema=SCHEMA, contract=contract, model_state=controller.state_dict(),
            optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
            rng_state=rng_state(), actual_updates=step,
            actual_exposures=exposure_budget.actual_exposures if exposure_budget else step*a.batch,
            sample_digest=sample_digest, quality_approved=False)
        temporary = a.output/'latest.tmp.pt'
        torch.save(value, temporary)
        temporary.replace(a.output/'latest.pt')

    baseline = validation(stage, heldout, a.micro_batch) if step == 0 else None
    if baseline is not None:
        write_json(a.output/'baseline.json', baseline)
    for batch in loader:
        examples=len(batch['sample_ids'])
        previous_reference=exposure_budget.equivalent_steps if exposure_budget else step
        if exposure_mode:
            for reference_step in range(exposure_budget.equivalent_steps,
                    exposure_budget.equivalent_steps+len(batch['sample_ids'])//256):
                ids=np.asarray([train.ids[i] for i in schedule.indices(reference_step)],dtype='<i8').tobytes()
                sample_digest=hashlib.sha256(bytes.fromhex(sample_digest)+ids).hexdigest()
        else:
            ids = batch['sample_ids'].numpy().astype('<i8').tobytes()
            sample_digest = hashlib.sha256(bytes.fromhex(sample_digest)+ids).hexdigest()
        # Reorder each batch into balanced microbatches for equal native loss scale.
        order = []
        for offset in range(a.batch//a.micro_batch):
            for domain, quota in (('pfnn',a.micro_batch//2),('parc',a.micro_batch//4),('bones',a.micro_batch//4)):
                selected = [i for i, name in enumerate(batch['domains']) if name == domain]
                order.extend(selected[offset*quota:(offset+1)*quota])
        def reorder(value):
            if torch.is_tensor(value):
                return value[order]
            if isinstance(value, dict):
                return {key: reorder(item) for key, item in value.items()}
            return [value[index] for index in order]
        if not pose_exposure:
            batch = device_batch(batch if exposure_mode else reorder(batch), a.device)
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        if executor is not None:
            offsets = list(range(0, examples, a.micro_batch))
            losses = executor.backward([split_micro(batch, offset, offset+a.micro_batch) for offset in offsets],
                [a.seed+(exposure_budget.equivalent_steps if exposure_budget else step)*103+offset for offset in offsets])
            if not all(math.isfinite(value) for value in losses):
                raise ValueError('Nonfinite parallel Root loss')
            for offset,value in zip(offsets,losses):
                loss_value += value*(min(a.micro_batch,examples-offset)/examples)
        else:
            for offset in range(0, a.batch, a.micro_batch):
                loss = stage(split_micro(batch, offset, offset+a.micro_batch), a.seed+step*103+offset)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                (loss*(a.micro_batch/a.batch)).backward()
                loss_value += float(loss.detach())*(a.micro_batch/a.batch)
        norm = torch.nn.utils.clip_grad_norm_(parameters, .5)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError('Nonfinite or zero full-stage gradient')
        if any(parameter.grad is not None for parameter in controller.parameters() if not parameter.requires_grad):
            raise ValueError('Frozen parameter acquired gradient')
        optimizer.step()
        if exposure_budget:
            exposure_budget.advance(examples)
            scheduler.step(exposure_budget.equivalent_steps)
        else:scheduler.step()
        step += 1
        if exposure_budget and step!=exposure_budget.actual_updates:
            raise ValueError('Optimizer and exposure-ledger update counts diverged')
        actual_exposures=exposure_budget.actual_exposures if exposure_budget else step*a.batch
        reference_step=exposure_budget.equivalent_steps if exposure_budget else step
        progress = dict(stage=a.stage, actual_updates=step, actual_exposures=actual_exposures,
            budget_updates=a.steps, loss=loss_value, gradient_norm=float(norm),
            lr=optimizer.param_groups[0]['lr'], sample_digest=sample_digest,
            seconds_per_update=(time.monotonic()-start_time)/(step-start_step), quality_approved=False,
            token_cache_hits=stage.token_cache_hits, token_cache_misses=stage.token_cache_misses,
            graph_dilated_convolutions=a.graph_dilated_convolutions,
            convolution_graph_signatures=len(stage._dilated_graph_execution.calls) if stage._dilated_graph_execution else 0,
            root_parallel_devices=a.root_parallel_devices,
            parallel_calibration_microbatches=executor.calibration_microbatches if executor else 0)
        if exposure_mode:
            progress.update(equivalent_reference_updates=reference_step,budget_reference_updates=50000,
                budget_updates=exposure_budget.initial_updates+math.ceil((12800000-exposure_budget.initial_exposures)/a.batch),
                budget_exposures=12800000,global_batch=a.batch,compute_micro=a.micro_batch,
                cuda_memory_fraction=a.cuda_memory_fraction,
                seconds_per_reference_update=(time.monotonic()-start_time)/((actual_exposures-start_exposures)/256))
            if pose_exposure:
                progress.update(pose_parallel_devices=2,isolated_pose_teacher_rng=True,
                    current_batch_exposures=examples,dropout_rng_increments=executor.rng_increments)
                if pose_label_graphs:
                    progress.update(frozen_pose_label_graphs=True,label_graph_warmups=executor.label_graph_warmups,
                        label_graph_signatures=[len(graph.calls) for graph in executor.label_graphs])
            else:progress.update(codec_pipeline_devices=execution_devices,ema_chunk_size=64)
        checkpoint_due=(previous_reference//a.checkpoint_every<reference_step//a.checkpoint_every
                        if pose_exposure else reference_step%a.checkpoint_every==0)
        validation_due=(previous_reference//a.validation_every<reference_step//a.validation_every
                        if pose_exposure else reference_step%a.validation_every==0)
        if step == 1 or step % 10 == 0 or checkpoint_due or reference_step==a.steps:
            write_json(a.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        if validation_due or reference_step == a.steps:
            values = validation(stage, heldout, 64 if exposure_mode else a.micro_batch)
            with (a.output/'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step,reference_step=reference_step,domains=values))+'\n')
        if checkpoint_due or reference_step == a.steps or step == a.stop_after:
            checkpoint()
        if a.stop_after and step >= a.stop_after:
            write_json(a.output/'stopped.json', progress)
            break
    if (exposure_budget.equivalent_steps if exposure_budget else step) == a.steps:
        (a.output/'latest.pt').replace(a.output/'weights.pt')
        write_json(a.output/'complete.json', dict(progress, checkpoint_sha256=sha256(a.output/'weights.pt'),
            protected_hash_passed=True, source_history_training_is_not_closed_loop_approval=True))
    if executor is not None:
        executor.close()
        for replica in replicas:replica.close()
    controller.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'parent', 'preflight', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache', nargs='+', type=Path, required=True)
    parser.add_argument('--stage', choices=['root','codec','pose'], required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--corpus-audit', type=Path)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--micro-batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=2333)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--validation-every', type=int, default=1000)
    parser.add_argument('--stop-after', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mechanics-only', action='store_true')
    parser.add_argument('--cache-pose-tokens', action='store_true')
    parser.add_argument('--coalesce-dilated-bias', action='store_true')
    parser.add_argument('--graph-dilated-convolutions', action='store_true')
    parser.add_argument('--resume-migration', type=Path)
    parser.add_argument('--cuda-memory-fraction', type=float, default=1.)
    parser.add_argument('--root-parallel-devices', type=int, default=1)
    parser.add_argument('--codec-exposure-pipeline',action='store_true')
    parser.add_argument('--pose-exposure-parallel',action='store_true')
    parser.add_argument('--graph-frozen-pose-labels',action='store_true')
    parser.add_argument('--codec-pipeline-devices',type=int,choices=(2,4),default=2)
    parser.add_argument('--await-resume', action='store_true')
    options = parser.parse_args()
    try:
        main(options)
    except Exception as error:
        if options.output.is_dir():
            write_json(options.output/'failure.json', dict(error=repr(error)))
        raise
