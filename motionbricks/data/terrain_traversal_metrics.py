"""Source-defined traversal review. Stopping or detouring is not task completion."""

import numpy as np


TRAVERSAL_PROTOCOL='real-paired-terrain-traversal-v1'


def traversal_window(roots,seconds=10):
    """Choose the most vertically active complete interval, using only source data.

    Leave enough real frames for every command-label horizon, including the
    last primitive. Never pad the reference or choose a window from predictions.
    """
    roots=np.asarray(roots,dtype=np.float64)
    blocks=min(int(seconds*30)//8,(len(roots)-25)//8)
    if blocks<4:raise ValueError('Traversal reference needs at least 57 real frames')
    emitted=blocks*8;available=len(roots)-(emitted+25)
    steps=np.diff(roots,axis=0)
    activity=np.abs(steps[:,1])+.05*np.linalg.norm(steps[:,[0,2]],axis=-1)
    prefix=np.r_[0.,np.cumsum(activity)]
    starts=np.arange(available+1)
    scores=prefix[starts+emitted+3]-prefix[starts+3]
    start=int(starts[np.argmax(scores)])
    return start,blocks,float(scores[start])


def traversal_metrics(predicted,reference):
    """Score ordered route coverage and source-clock body root error in meters.

    Ordered 3-D milestones reject a stop, a lateral bypass, and traversing the
    correct XZ route at the wrong elevation. They do not enforce a runtime path.
    """
    predicted=np.asarray(predicted,dtype=np.float64)
    reference=np.asarray(reference,dtype=np.float64)
    if (predicted.ndim!=2 or reference.ndim!=2 or predicted.shape[1:]!=(3,) or reference.shape[1:]!=(3,)
            or not 2<=len(predicted)<=len(reference)):
        raise ValueError('Prediction must be a real consecutive prefix of the source clock')
    if not np.isfinite(predicted).all() or not np.isfinite(reference).all():raise ValueError('Nonfinite traversal')
    source_step=np.linalg.norm(np.diff(reference,axis=0),axis=-1)
    distance=np.r_[0.,np.cumsum(source_step)]
    total=float(distance[-1])
    if total<.5:raise ValueError('A stationary reference does not demonstrate traversal')
    indices=np.searchsorted(distance,np.linspace(0,total,17))
    milestones=reference[indices]
    reached=0;cursor=0;errors=[]
    for target in milestones:
        distances=np.linalg.norm(predicted[cursor:]-target,axis=-1)
        candidates=np.flatnonzero(distances<=.35)
        errors.append(float(distances.min()))
        if not len(candidates):break
        cursor+=int(candidates[0]);reached+=1
    delta=predicted-reference[:len(predicted)]
    return {'ordered_route_fraction':reached/len(milestones),'milestone_radius_m':.35,
        'milestones_reached':reached,'milestones_total':len(milestones),
        'source_path_length_m':total,'source_vertical_range_m':float(np.ptp(reference[:,1])),
        'root_path_rmse_m':float(np.sqrt(np.mean(np.sum(delta**2,axis=-1)))),
        'root_height_mae_m':float(np.abs(delta[:,1]).mean()),
        'root_endpoint_error_m':float(np.linalg.norm(predicted[-1]-reference[-1])),
        'rollout_completion_fraction':len(predicted)/len(reference),
        'requested_frames':len(reference),'generated_frames':len(predicted),
        'error_scope':'generated prefix only; missing frames are an explicit failed rollout',
        'predicted_path_length_m':float(np.linalg.norm(np.diff(predicted,axis=0),axis=-1).sum()),
        'furthest_checked_milestone_error_m':errors[-1]}


def traversal_checks(metrics):
    # Declared trial criteria, not claims of human perceptual/physical validity.
    return {'completes_ordered_route':metrics['ordered_route_fraction']==1.,
        'complete_rollout':metrics['rollout_completion_fraction']==1.,
        'arrives_near_source_destination':metrics['root_endpoint_error_m']<=.5,
        'matches_source_elevation':metrics['root_height_mae_m']<=.2,
        'full_known_surface':metrics.get('audited_vertices_per_frame')==18056 and metrics.get('unknown_body_fraction')==0,
        'body_penetration_p99':metrics.get('body_penetration_p99_m',float('inf')) is not None
            and metrics.get('body_penetration_p99_m',float('inf'))<=.03}


def traversal_pilot_checks(initial,report):
    """Budget decision is separate from successful final traversal acceptance."""
    before=initial.get('traversal',[]);after=report.get('traversal',[])
    key=lambda r:(r['split'],r['source_id'],r['start_frame'],r.get('requested_frames',r['emitted_frames']))
    complete=bool(before) and {key(r) for r in before}=={key(r) for r in after}
    complete=complete and {r['split'] for r in after}=={'train','validation'}
    coverage=report.get('traversal_coverage',{})
    complete=complete and bool(coverage) and all(not row['missing_families'] for row in coverage.values())
    a={key(r):r for r in before}
    mean=lambda rows,field:float(np.mean([r['metrics'][field] for r in rows])) if rows else float('inf')
    checks={'same_real_source_cases':complete,
        'finite_traversal':bool(after) and all(np.isfinite(r['metrics']['root_path_rmse_m']) for r in after),
        'native_motion_retained':report.get('short_mean_mpjpe_m',float('inf'))<=1.1*initial['short_mean_mpjpe_m']+.01}
    for split in ('train','validation'):
        rows=[r for r in after if r['split']==split];prior=[a[key(r)] for r in rows if key(r) in a]
        full=[r for r in rows if key(r) in a and a[key(r)]['metrics'].get('rollout_completion_fraction',1.)==1.]
        old_full=[a[key(r)] for r in full]
        checks[split+'_complete_cases_retained']=bool(full) and all(r['metrics'].get('rollout_completion_fraction',1.)==1. for r in full)
        checks[split+'_path_retained']=bool(old_full) and mean(full,'root_path_rmse_m')<=1.1*mean(old_full,'root_path_rmse_m')+.05
        checks[split+'_route_progress_retained']=bool(prior) and mean(rows,'ordered_route_fraction')>=mean(prior,'ordered_route_fraction')-.05
    common=[r for r in after if key(r) in a and r['metrics'].get('rollout_completion_fraction',1.)==1.
            and a[key(r)]['metrics'].get('rollout_completion_fraction',1.)==1.]
    checks['real_terrain_learning_signal']=complete and (
        bool(common) and mean(common,'root_path_rmse_m')<.95*mean([a[key(r)] for r in common],'root_path_rmse_m') or
        mean(after,'ordered_route_fraction')>mean(before,'ordered_route_fraction')+.05)
    old_flat=initial.get('flat',[]);new_flat=report.get('flat',[])
    flat_complete=len(old_flat)==len(new_flat)==4
    checks['flat_control_review_retained']=flat_complete
    if flat_complete:
        drift=lambda rows:float(np.mean([r['metrics']['drift_max_m'] for r in rows]))
        progress=lambda rows:min(r['metrics']['start_progress_ratio'] for r in rows if 'start_progress_ratio' in r['metrics'])
        checks['rest_not_sacrificed']=drift(new_flat)<=1.1*drift(old_flat)+.05
        checks['start_not_sacrificed']=progress(new_flat)>=.8*progress(old_flat)
    if 'directional' in initial or 'directional' in report:
        old=initial.get('directional',[]);new=report.get('directional',[])
        checks['directional_coverage']=len(old)==len(new)==4
        if len(old)==len(new)==4:
            checks['directional_error_retained']=max(r['metrics']['cross_track_max_m'] for r in new)<=1.1*max(r['metrics']['cross_track_max_m'] for r in old)+.05
            checks['directional_progress_retained']=min(r['metrics']['progress_ratio'] for r in new)>=.8*min(r['metrics']['progress_ratio'] for r in old)
    # Wall stopping scores deliberately do not participate in this decision.
    return checks
