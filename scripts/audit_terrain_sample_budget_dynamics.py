"""Read-only CPU audit of loss logging, source lengths and optimizer time scales."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import numpy as np
import torch

from motionbricks.data.terrain_learning_data import learning_sampling_weights


class Records:
    def __init__(self,records):self.records=records
    def __len__(self):return len(self.records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    run=args.task/'control_sample_v8'
    checkpoint=torch.load(run/'last.pt',map_location='cpu',mmap=True,weights_only=False)
    contract=checkpoint['contract'];data_path=Path(contract['dataset'])
    base=[r for r in json.loads((data_path/'accepted.json').read_text()) if r.get('split')=='train' and r.get('frames_30hz',0)>=65]
    pairs=[r for r in json.loads((Path(contract['pairs'])/'manifest.json').read_text())['items'] if r['split']=='train']
    data=Records(base+pairs);data.base=Records(base)
    weights=learning_sampling_weights(data,contract['weak_fraction'])
    lengths=np.asarray([min(contract['sequence_frames'],r['frames_30hz']) for r in base]+[0]*len(pairs))
    offset=contract['sampler_draw_offset'];episodes=min(r['data_step'] for r in checkpoint['rank_rng'])
    batch=contract['global_batch']
    ids=torch.multinomial(weights,offset+episodes*batch,replacement=True,generator=torch.Generator().manual_seed(2333))
    selected=ids[offset:].numpy().reshape(episodes,batch)
    minima=lengths[selected].min(1);depths=[];frames=[]
    for index,minimum in enumerate(minima):
        rng=np.random.default_rng(2333+index)
        tokens=8 if rng.random()<.6 else int(rng.integers(6,17))
        tokens=min(tokens,(int(minimum)-1)//4);count=tokens*4
        frames.append(count);depths.append(min(contract['feedback_depth'],1+(int(minimum)-count-1)//8))
    ids_to_check=sorted(set(int(row[lengths[row].argmin()]) for row in selected))[:8]
    lengths_checked=[]
    for index in ids_to_check:
        record=base[index]
        with np.load(data_path/record['shard'],allow_pickle=False) as file:
            actual=len(file['global_features'])
        if actual!=record['frames_30hz']:raise ValueError('Manifest/source length mismatch')
        lengths_checked.append({'source_id':record['source_id'],'actual_frames':actual,'family':record.get('control_category',record.get('terrain_family'))})
    rows=[]
    for line in (args.task/'control-sample-v8-training.log').read_text(errors='replace').splitlines():
        try:row=json.loads(line)
        except (TypeError,ValueError):continue
        if 'step' in row and 'loss' in row:rows.append(row)
    breakdown=[]
    for lower,upper in ((4000,4200),(4200,4400),(4400,4600),(4600,4800),(4800,5000),(5000,5200),(5200,5400),(5400,5710)):
        values=[r for r in rows if lower<r['step']<=upper]
        if not values:continue
        means={k:statistics.mean(r[k] for r in values) for k in ('loss','token_ce','native_motion','native_root',*checkpoint['coefficients'])}
        weighted={k:means[k]*v for k,v in checkpoint['coefficients'].items()}
        breakdown.append({'logged_steps':[values[0]['step'],values[-1]['step']],'logged_batches':len(values),
            'means':means,'weighted_physical':weighted,'ce_fraction_of_total':means['token_ce']/means['loss']})
    latest=rows[-1];categories=Counter()
    for k,v in latest['counts_per_rank'].items():
        if k.startswith('primitive_exposure:'):categories[k.split(':')[1]]+=v
    original_p=float(weights[lengths>=129].sum())
    matched={}
    for row in rows:
        episode_index=row['data_draw_step']-1
        if episode_index>=episodes or row.get('feedback_probability')!=1.:continue
        position=(row['episode_position'] or depths[episode_index])-1
        key=f'frames{frames[episode_index]}-position{position}'
        bucket=matched.setdefault(key,{'early':[],'late':[]})
        if row['step']<=4200:bucket['early'].append(row)
        elif row['step']>=4400:bucket['late'].append(row)
    matched_rows=[]
    for key,value in matched.items():
        if all(value.values()):
            matched_rows.append({'stratum':key,**{side:{'logged_batches':len(part),'mean_loss':statistics.mean(r['loss'] for r in part),
                'mean_fk_m':statistics.mean(r['fk_m'] for r in part)} for side,part in value.items()}})
    report={'schema':'motionbricks.sample_budget_dynamics_audit.v1','checkpoint_step':checkpoint['step'],
        'scope':'CPU metadata, recorded logs and checkpoint audit only; no model inference, optimization or quality evaluation',
        'source_lengths_verified':lengths_checked,'recorded_global_batch':batch,'observed_episodes':episodes,
        'batch_minimum_frame_counts':dict(Counter(map(int,minima))),'realized_episode_depth_counts':dict(Counter(depths)),
        'realized_depth_mean':float(np.mean(depths)),'frames_per_primitive_counts':dict(Counter(frames)),
        'full_13_primitive_episode_fraction':float(np.mean(np.asarray(depths)==13)),
        'source_clip_probability_at_least129_frames':original_p,
        'probability_all_clips_at_least129_frames':{str(b):original_p**b for b in (32,768)},
        'emitted_seconds_per_episode_range':[min(depths)*8/30,max(depths)*8/30],
        'domain_exposure_rank0':dict(categories),'loss_blocks':breakdown,
        'same_horizon_same_feedback_position_comparisons':matched_rows,
        'optimizer_betas':checkpoint['optimizer']['param_groups'][0]['betas'],
        'batch_ratio_to_reference':batch/32,'new_updates_to_full_feedback':int(np.ceil((5000-4000)/(batch/32))),
        'learning_rate_changed_clock_not_peak':True,'max_update_lr':5e-5,'gradient_clip':.5,
        'gradient_norm_minmax_logged':[min(r['gradient_norm'] for r in rows),max(r['gradient_norm'] for r in rows)],
        'rows_over_clip':sum(r['gradient_norm']>.5 for r in rows),'logged_rows':len(rows),
        'loss_logging':'one rank and one update every25; not an all-step or all-rank epoch mean',
        'quality_conclusion':'Not evaluated; no claim of convergence equivalence from equal sample exposure'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
