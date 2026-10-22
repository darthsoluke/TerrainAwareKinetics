"""Same-host admitted PARC + BONES dataset -> matched two-GPU MB experiments."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.train_native_mb import sha256, verify_dataset
from scripts.train_mixed_control_stage import write_json


def verify_ready_dataset(dataset, expected_sha256):
    """Admit an already built native corpus without the obsolete campaign quota."""
    from motionbricks.data.parc_training_sources import verify_parc_dataset_sources
    if not expected_sha256 or sha256(dataset / 'native_manifest.json') != expected_sha256:
        raise ValueError('Native dataset identity mismatch')
    manifest = json.loads((dataset / 'native_manifest.json').read_text())
    if manifest.get('training_usable') is not True:
        raise ValueError('Native corpus has not passed its data admission')
    verify_dataset(dataset, manifest)
    records = json.loads((dataset / 'accepted.json').read_text())
    sources = verify_parc_dataset_sources(dataset, manifest, records)
    return {'manifest_sha256': expected_sha256, 'source_verification': sources,
            'records': len(records), 'motion_quality_approved': False}


def verify_completed_arm(run, state, frame):
    """Reuse a finished arm only when its inputs, budget and artifacts still match."""
    bundle = json.loads((run/'pipeline.json').read_text())
    if (bundle['sha256'] != sha256(run/'pipeline.pt') or
            bundle['dataset_manifest_sha256'] != state['dataset_manifest_sha256']):
        raise ValueError('Completed arm bundle identity changed')
    for name in ('tokenizer', 'root', 'pose'):
        directory = run/name
        contract = json.loads((directory/'run_contract.json').read_text())
        result = json.loads((directory/'training_result.json').read_text())
        expected = {'stage': name, 'source_bundle_sha256': state['source_sha256'],
                    'dataset_manifest_sha256': state['dataset_manifest_sha256'],
                    'steps': state['steps_per_stage_per_arm'], 'vertical_frame': frame,
                    'sampling_policy': state['sampling_policy'],
                    'world_size': 2 if name == 'pose' else 1,
                    'precision': 'bf16-mixed' if name == 'pose' else '32'}
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError('Completed arm training contract changed: '+name)
        if (result['optimizer_updates'] != expected['steps'] or
                contract['batch']['global'] != (2048 if name == 'pose' else 256)):
            raise ValueError('Completed arm training budget changed: '+name)
        checkpoint = (directory/result['checkpoint_relative']).resolve()
        checkpoint.relative_to(directory.resolve())
        if sha256(checkpoint) != result['checkpoint_sha256']:
            raise ValueError('Completed arm checkpoint changed: '+name)
    flat = json.loads((run/'closed_loop_review/metrics.json').read_text())
    if (flat['bundle_sha256'] != bundle['sha256'] or
            flat['dataset_manifest_sha256'] != state['dataset_manifest_sha256']):
        raise ValueError('Completed arm flat review identity changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','source','output','drive-config'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=10000)
    dataset_input = parser.add_mutually_exclusive_group(required=True)
    dataset_input.add_argument('--local-dataset-receipt',type=Path)
    dataset_input.add_argument('--ready-dataset',type=Path)
    parser.add_argument('--dataset-sha256')
    parser.add_argument('--source-sha256')
    parser.add_argument('--mixed-dataset',type=Path,help='Reuse a verified same-source mixed corpus')
    parser.add_argument('--sampling-policy', choices=('clips-v1', 'parc-windows-v1'), default='clips-v1')
    parser.add_argument('--gpus',default='1,2')
    parser.add_argument('--resume',action='store_true',help='Recover a failed supervisor using completed exported arms')
    args=parser.parse_args()
    cards=[int(value) for value in args.gpus.split(',')]
    if len(cards)!=2 or len(set(cards))!=2 or not set(cards)<=set(range(1,7)):
        parser.error('Use two available GPUs from 1..6; GPUs 0 and 7 are reserved')
    if args.ready_dataset and not args.dataset_sha256:
        parser.error('A ready dataset requires its immutable manifest hash')
    if args.source_sha256 and sha256(args.source) != args.source_sha256:
        parser.error('Warmstart bundle identity mismatch')
    previous = None
    if args.resume:
        previous=json.loads((args.output/'state.json').read_text())
        if previous.get('status') != 'failed':
            parser.error('Only a failed supervisor can be recovered')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    state={'status':'preparing_training','gpus':cards, 'reserved_gpus':[0,7],
           'pid':os.getpid(), 'sampling_policy':args.sampling_policy,
           'source_sha256':sha256(args.source), 'steps_per_stage_per_arm':args.steps,
           'runtime_deployment':'local PC only; no automatic live replacement','training_approved':False}
    if previous:
        for key in ('gpus','source_sha256','steps_per_stage_per_arm','sampling_policy'):
            if previous.get(key)!=state[key]:
                parser.error('Recovery configuration changed: '+key)
        state['previous_failure']=previous['error']
    state['code']=str(Path(__file__).resolve().parents[1])
    status_path=args.output/'state.json'
    code=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'),
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TMPDIR=str(args.base/'tmp'))
    def stage(name,command):
        state.update(status=name,updated_at=time.time()); write_json(status_path,state)
        with (args.output/(name+'.log')).open('a' if args.resume else 'w') as log:
            subprocess.run(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        write_json(status_path,state)
        if args.ready_dataset:
            parc=args.ready_dataset.resolve()
            parc.relative_to(args.base.resolve())
            state['data_admission']=verify_ready_dataset(parc,args.dataset_sha256)
        else:
            while not args.local_dataset_receipt.exists():
                state.update(status='waiting_for_local_generated_dataset',updated_at=time.time())
                write_json(status_path,state);time.sleep(30)
            transfer=json.loads(args.local_dataset_receipt.read_text())
            from scripts.parc_collection_budget import verify_training_gate
            verify_training_gate(transfer)
            parc=Path(transfer['dataset']).resolve()
            parc.relative_to(args.base.resolve())
            if sha256(parc/'native_manifest.json')!=transfer['dataset_manifest_sha256']:
                raise ValueError('Native dataset identity mismatch')
            verify_dataset(parc,json.loads((parc/'native_manifest.json').read_text()))
        from scripts.run_mixed_control_workflow import available_devices
        def wait_for_cards():
            while True:
                try:
                    available_devices(cards)
                    return
                except RuntimeError:
                    state.update(status='waiting_for_two_free_gpus',updated_at=time.time())
                    write_json(status_path,state)
                    time.sleep(30)
        wait_for_cards()
        if args.mixed_dataset:
            from motionbricks.data.mixed_control_training import validate_source_policy
            mixed=args.mixed_dataset.resolve()
            mixed.relative_to(args.base.resolve())
            manifest=json.loads((mixed/'native_manifest.json').read_text())
            verify_dataset(mixed,manifest)
            policy=manifest['source_policy']
            validate_source_policy(policy)
            if (policy['parc_manifest_sha256']!=sha256(parc/'native_manifest.json') or
                    policy['bones_selection_sha256']!=sha256(args.base/'bones_all_v2/selection.json')):
                raise ValueError('Reused mixed corpus has different source data')
        else:
            mixed=args.output/'mixed_dataset'
            stage('mixing_bones_parc',[sys.executable,'-u','-m','scripts.build_mixed_control_dataset',
                '--parc',str(parc),'--bones',str(args.base/'bones_all_v2'),'--output',str(mixed),'--device','cuda:'+str(cards[0])])
        state['mixed_dataset']=str(mixed)
        state['dataset_manifest_sha256']=sha256(mixed/'native_manifest.json')
        if previous and previous.get('dataset_manifest_sha256')!=state['dataset_manifest_sha256']:
            raise ValueError('Recovery mixed dataset changed')
        arms={}
        for frame,label in [('observed-pelvis-y-v1','pelvis_baseline'),('observed-terrain-y-v1','terrain_reference')]:
            wait_for_cards()
            run=args.output/label
            if args.resume and run.exists():
                verify_completed_arm(run,state,frame)
            else:
                stage('training_'+label,[sys.executable,'-u','-m','scripts.run_mixed_control_workflow',
                    '--source',str(args.source),'--dataset',str(mixed),'--output',str(run),
                    '--gpus',args.gpus,'--steps',str(args.steps),'--batch-per-gpu','256',
                    '--pose-batch-per-gpu','1024','--precision','32','--pose-precision','bf16-mixed',
                    '--validation-crops','4','--validate-every','2000','--vertical-frame',frame,
                    '--sampling-policy',args.sampling_policy])
            review=run/'terrain_review'
            if review.exists():
                review.rename(run/('terrain_review.previous-'+str(time.time_ns())))
            stage('terrain_review_'+label,[sys.executable,'-u','-m','scripts.review_parc_feedback_ablation',
                '--bundle',str(run/'pipeline.pt'),'--dataset',str(mixed),'--output',str(run/'terrain_review'),
                '--max-windows','24','--gpu',str(cards[0])])
            arms[label]={'run':str(run),
                'flat_review':json.loads((run/'closed_loop_review/metrics.json').read_text()),
                'terrain_summary':json.loads((run/'terrain_review/metrics.json').read_text())['summary']}
            write_json(args.output/'comparison.json',arms)
        state.update(status='matched_training_complete_review_required',arms={key:{
            'run':value['run'],'flat_stop_gate_passed':value['flat_review']['flat_stop_gate_passed']}
            for key,value in arms.items()})
        write_json(status_path,state)
        stage('uploading_reports',['rclone','--config',str(args.drive_config),'copy',str(args.output),
            'mb-transfer:runs/'+args.output.name,'--include','/comparison.json',
            '--include','/state.json','--include','/*/pipeline.pt','--include','/*/pipeline.json',
            '--include','/*/closed_loop_review/**','--include','/*/terrain_review/**','--immutable'])
        state['status']='matched_training_complete_review_required'; write_json(status_path,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error)); write_json(status_path,state)
        raise


if __name__=='__main__':
    main()
