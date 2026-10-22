"""Review actual render acknowledgements and their saved native forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values):
    array=np.asarray(values,dtype=float)
    if not len(array): return None
    return dict(count=len(array),mean=float(array.mean()),
        **{name:float(np.percentile(array,p)) for name,p in (('p50',50),('p95',95),('p99',99))},
        maximum=float(array.max()))


def review(folder):
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
    episodes=sorted({row['episode_id'] for row in events if row['kind']=='played'})
    reports=[]
    for episode in episodes:
        played=[row for row in events if row['kind']=='played' and row['episode_id']==episode]
        forecasts={row['metadata']['plan_id']:row for row in events if row['kind']=='forecast'
            and row['metadata']['episode_id']==episode}
        numbers=np.array([row['frame'] for row in played])
        if not np.array_equal(numbers,np.arange(1,len(numbers)+1)):
            raise ValueError('Recorded playback must contain each actual frame exactly once')
        cache={}
        roots,joints,motions,requests,steps,handoffs=[],[],[],[],[],[]
        previous=None
        for row in played:
            plan_id=row['plan_id']
            forecast=forecasts[plan_id]
            if plan_id not in cache:
                with np.load(forecast['path'],allow_pickle=False) as data:
                    cache[plan_id]={key:data[key] for key in ('motion','joints')}
            data=cache[plan_id]
            index=row['frame']-forecast['metadata']['first_frame']
            if not 0<=index<24: raise ValueError('Frame is outside recorded native prediction')
            roots.append(data['motion'][index,:3])
            motions.append(data['motion'][index])
            joints.append(data['joints'][index])
            requests.append(forecast['metadata']['input_seq'])
            steps.append(forecast['metadata']['step_frames'])
            if previous is not None and previous!=plan_id:
                old_index=row['frame']-forecasts[previous]['metadata']['first_frame']
                if 0<=old_index<24:
                    old=cache[previous]
                    handoffs.append(dict(frame=row['frame'],old_plan=previous,new_plan=plan_id,
                        step_frames=forecast['metadata']['step_frames'],
                        root_disagreement_m=float(np.linalg.norm(data['motion'][index,:3]-old['motion'][old_index,:3])),
                        joint_disagreement_max_m=float(np.linalg.norm(data['joints'][index]-old['joints'][old_index],axis=-1).max())))
            previous=plan_id
        times=np.array([row['client_render_ms'] for row in played])
        intervals=np.diff(times)
        if (intervals<=0).any(): raise ValueError('Rendered frame times must strictly increase')
        roots,joints=np.asarray(roots),np.asarray(joints)
        history_error=[]
        seed=np.asarray(next(row['seed'] for row in events if row['kind']=='session'),np.float32)
        actual=np.concatenate((seed,np.asarray(motions)))
        for record in forecasts.values():
            meta=record['metadata'];timing=meta['timing']
            if 'played_history_frame' not in timing:continue
            factual=timing['played_history_frame']
            expected=actual[factual:factual+4]
            prefix=[]
            for i,plan in enumerate(timing['prefix_plan_ids']):
                saved=forecasts[plan]
                with np.load(saved['path'],allow_pickle=False) as values:
                    prefix.append(values['motion'][factual+i+1-saved['metadata']['first_frame']])
            conditioned=np.concatenate((expected,np.asarray(prefix)))[-4:] if prefix else expected
            with np.load(record['path'],allow_pickle=False) as values:
                error=max(float(np.abs(values['played_history']-expected).max()),
                    float(np.abs(values['history']-conditioned).max()))
            history_error.append(error)
        if history_error and max(history_error)!=0:
            raise ValueError('Recorded generation history differs from rendered frames and committed prefix')
        relative=joints-roots[:,None]
        generation=[row['metadata'] for row in forecasts.values()]
        source_intervals=np.diff([row['origin_frame'] for row in generation])
        actual_speed=np.linalg.norm(np.diff(roots[:,[0,2]],axis=0),axis=-1)*30
        intent_events={row['seq']:row for row in events if row['kind']=='intent' and row['episode_id']==episode}
        first_visible={}
        for row,seq in zip(played,requests): first_visible.setdefault(seq,row['client_render_ms'])
        latency=[at-intent_events[seq]['client_input_ms'] for seq,at in first_visible.items() if seq in intent_events]
        final_idle=None
        if intent_events:
            last_intent=list(intent_events.values())[-1]
            if np.linalg.norm(last_intent['velocity'])<1e-8:
                idle=times>=last_intent['client_input_ms']+1000
                if idle.sum()>1:
                    xy=roots[idle][:,[0,2]]
                    duration=(times[idle][-1]-times[idle][0])/1000
                    final_idle=dict(frames=int(idle.sum()),duration_s=float(duration),
                        displacement_xz_m=(xy[-1]-xy[0]).tolist(),
                        net_drift_speed_mps=float(np.linalg.norm(xy[-1]-xy[0])/duration),
                        instantaneous_planar_speed_mps=distribution(np.linalg.norm(np.diff(xy,axis=0),axis=-1)*30))
        by_step=[]
        steps=np.asarray(steps)
        for step in sorted(set(steps),reverse=True):
            selected=[row for row in generation if row['step_frames']==step]
            mask=(steps[:-1]==step)&(steps[1:]==step)
            delta=intervals[mask]
            used={seq for seq,mode in zip(requests,steps) if mode==step}
            command_latency=[first_visible[seq]-intent_events[seq]['client_input_ms']
                for seq in used if seq in intent_events]
            by_step.append(dict(step_frames=int(step),played_frames=int(np.sum(steps==step)),
                display_hz=float(len(delta)*1000/delta.sum()) if len(delta) else None,
                compute_ms=distribution([row['timing']['compute_ms'] for row in selected]),
                requested_interval_met_fraction=float(np.mean([row['timing']['interval_target_met'] for row in selected])),
                origin_interval_frames=distribution(np.diff([row['origin_frame'] for row in selected])),
                consumed_input_count=len(used),consumed_input_first_visible_ms=distribution(command_latency),
                handoff_root_disagreement_m=distribution([row['root_disagreement_m'] for row in handoffs if row['step_frames']==step]),
                handoff_joint_disagreement_m=distribution([row['joint_disagreement_max_m'] for row in handoffs if row['step_frames']==step])))
        reports.append(dict(episode_id=episode,played_frames=len(played),
            conditioned_histories_checked=len(history_error),
            played_and_committed_history_max_abs=max(history_error) if history_error else None,
            final_idle=final_idle,
            by_step=by_step,input_count=len(intent_events),consumed_input_count=len(first_visible),
            actual_display_hz=float((len(times)-1)*1000/(times[-1]-times[0])) if len(times)>1 else None,
            rendered_interval_ms=distribution(intervals),
            intervals_over_50ms=int(np.sum(intervals>50)),
            requested_steps=sorted({row['step_frames'] for row in generation}),
            generation_origin_interval_frames=distribution(source_intervals),
            generation_compute_ms=distribution([row['timing']['compute_ms'] for row in generation]),
            requested_interval_met_fraction=float(np.mean([row['timing']['interval_target_met'] for row in generation])),
            input_to_first_visible_ms=distribution(latency),
            actual_planar_speed_mps=distribution(actual_speed),
            root_height_m=distribution(roots[:,1]),
            maximum_joint_step_m=distribution(np.linalg.norm(np.diff(relative,axis=0),axis=-1).max(axis=-1)),
            handoff_root_disagreement_m=distribution([row['root_disagreement_m'] for row in handoffs]),
            handoff_joint_disagreement_m=distribution([row['joint_disagreement_max_m'] for row in handoffs]),
            handoffs=handoffs,
            failures=[row for row in events if row['kind']=='failure' and row['episode_id']==episode]))
    return dict(schema='motionbricks.reactive_kinematic_review.v1',session=str(folder),episodes=reports,
        optimizer_updates=0,new_model_calls=0,physical_simulation=False,
        geometry_scope='native_root_and_joint_seams; full_skin_terrain_quality_not_evaluated')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=review(args.session)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps([{key:row[key] for key in ('episode_id','played_frames','actual_display_hz',
        'requested_steps','generation_compute_ms','input_to_first_visible_ms','handoff_root_disagreement_m')}
        for row in result['episodes']],indent=2))
