"""Launch the authorized two-A100 run and publish independent numerical/visual evidence."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def available_cards(indices):
    result=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    rows={int(parts[0]):parts for line in result.stdout.splitlines() if (parts:=line.split(', '))}
    if len(indices)!=2 or len(set(indices))!=2 or any(index in (0,7) for index in indices):
        raise ValueError('Use two distinct GPUs and preserve reserved cards 0/7')
    selected=[rows[index] for index in indices]
    if any(float(row[2])>512 or float(row[3])>10 for row in selected):
        raise RuntimeError(f'Physical GPUs {indices} are not free; reserved cards and other jobs stay untouched')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--run-name',default='guarded-v3')
    p.add_argument('--gpus',type=int,nargs=2,default=[2,3])
    args=p.parse_args();base=args.base.resolve();task=base/'model_terrain_v3_20261011'
    if not re.fullmatch('[a-z][a-z0-9-]*',args.run_name):p.error('Use a simple task-specific run name')
    suffix=args.run_name.replace('-','_')
    output=task/('long_'+suffix);code=task/('code_'+suffix)
    receipt=task/(args.run_name+'-launch.json')
    if not args.worker:
        if output.exists() or receipt.exists():raise ValueError('This run was already launched; inspect or explicitly resume it')
        available_cards(args.gpus)
        with (task/(args.run_name+'-campaign.log')).open('x') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--base',str(base),
                '--run-name',args.run_name,'--gpus',*[str(index) for index in args.gpus],'--worker'],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=code)
        save(receipt,{'pid':process.pid,'output':str(output),'status':'worker_launched',
            'requested_optimizer_updates':200000,'gpu_indices':args.gpus,'reserved_gpu_indices':[0,7]})
        print(json.dumps({'campaign_pid':process.pid,'output':str(output)}),flush=True)
        return
    state_path=task/(args.run_name+'-state.json')
    state={'status':'checking_free_cards','pid':os.getpid(),'output':str(output),'quality_approved':False}
    save(state_path,state)
    try:
        cards=available_cards(args.gpus);state['gpu_snapshot']=cards
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus)),PYTHONPATH=str(code)+':'+str(code/'out/vendor/kimodo'))
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            '-m','scripts.train_joint_terrain_v3','--source',str(base/'long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt'),
            '--dataset',str(base/'long_route_mixed_10k_ab_v1/mixed_dataset'),'--pairs',str(task/'weak_pairs_v1'),
            '--output',str(output),'--steps','200000','--global-batch','32','--micro-batch','16','--workers','2',
            '--adapter-warmup','2000','--lr-warmup','2000','--feedback-start','2000','--feedback-ramp','8000',
            '--validate-every','1000','--checkpoint-every','250','--evaluation-seconds','10','--evaluate-initial','--adaptive-unfreeze',
            '--max-adapter-updates','8000','--max-stale-reviews','4']
        with (task/(args.run_name+'-training.log')).open('x') as log:
            process=subprocess.Popen(command,env=env,cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='training',training_pid=process.pid,command=command);save(state_path,state)
            handled=set()
            def publish_reviews():
                for review in sorted(output.glob('review-*')):
                    if review.name in handled or not (review/'metrics.json').exists():continue
                    metrics=json.loads((review/'metrics.json').read_text())
                    if 'absolute_gates_passed' not in metrics:continue
                    step=int(review.name.split('-')[1]);handled.add(review.name)
                    state.update(last_review=step,last_absolute_gates=metrics['absolute_gates_passed'])
                    if step and step%5000==0:
                        try:
                            destination=output/'visualizations'/review.name
                            subprocess.run([sys.executable,'-m','scripts.export_joint_terrain_visuals','--stats',
                                str(base/'long_route_mixed_10k_ab_v1/mixed_dataset/stats'),'--reviews',
                                'initial='+str(output/'review-000000'),'learned='+str(review),'--output',str(destination)],
                                env=env,cwd=code,check=True,timeout=600)
                        except Exception as error:
                            state['visual_export_error']=type(error).__name__+': '+str(error)
                    try:
                        proxy=dict(os.environ,HTTP_PROXY='http://127.0.0.1:27890',HTTPS_PROXY='http://127.0.0.1:27890')
                        subprocess.run(['rclone','--config',str(base/'drive-auth/rclone.conf'),'copy',str(output),
                            'mb-transfer:runs/model-terrain-v3-20261011/'+args.run_name,'--include','/contract.json',
                            '--include','/progress.json','--include','/selection.json','--include','/best_candidate.json',
                            '--include','/complete.json','--include','/review-*/metrics.json','--include','/visualizations/**',
                            '--contimeout','15s','--timeout','60s'],env=proxy,check=True,timeout=600)
                    except Exception as error:
                        state['private_publish_error']=type(error).__name__+': '+str(error)
                    save(state_path,state)
            while process.poll() is None:
                publish_reviews();time.sleep(30)
            publish_reviews()
            if process.returncode:raise RuntimeError(f'Training exited {process.returncode}; see the task training log')
        result=json.loads((output/'complete.json').read_text())
        state.update(status='completed_review_required',training_result=result)
        # Checkpoints remain private on the server. Only a selected model is
        # eligible for a later private Drive export and local inference review.
        save(state_path,state)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));save(state_path,state)
        raise


if __name__=='__main__':main()
