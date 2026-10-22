"""Fixed, task-level pilot selection; loss alone cannot approve a controller."""

import math


def summarize(report):
    if report.get('evaluation_goal')=='real-paired-terrain-traversal-v1':
        if len(report.get('flat',[]))!=4 or not report.get('traversal') or len(report.get('short',[]))<8:
            raise ValueError('Incomplete real-terrain task coverage')
        flat=[r['metrics'] for r in report['flat']];terrain=[r['metrics'] for r in report['traversal']]
        average=lambda rows,key:sum(r[key] for r in rows)/len(rows)
        return {'evaluation_goal':report['evaluation_goal'],
            'fk_mpjpe_m':average(report['short'],'fk_mpjpe_m'),
            'idle_drift_m':average(flat,'drift_max_m'),
            'idle_speed_mps':average(flat,'settled_speed_p95_mps'),
            'foot_penetration_m':average(flat,'foot_penetration_p99_m'),
            'foot_gap_m':sum(abs(m['median_higher_foot_gap_m']) for m in flat)/len(flat),
            'start_progress_ratio':min(m['start_progress_ratio'] for m in flat if 'start_progress_ratio' in m),
            'terrain_route_fraction':average(terrain,'ordered_route_fraction'),
            'terrain_path_rmse_m':average(terrain,'root_path_rmse_m'),
            'terrain_height_mae_m':average(terrain,'root_height_mae_m'),
            'terrain_penetration_m':average(terrain,'body_penetration_p99_m'),
            'unknown_body_fraction':max(m['unknown_body_fraction'] for m in flat+terrain),
            'full_surface_audit':all(m.get('audited_vertices_per_frame')==18056 for m in flat+terrain),
            'absolute_gates_passed':bool(report.get('absolute_gates_passed',False))}
    if len(report.get('flat',[]))!=4 or not report.get('walls') or len(report.get('short',[]))<8:
        raise ValueError('Incomplete held-out task coverage')
    def average(values):
        values=list(values)
        if not values or any(v is None or not math.isfinite(v) for v in values):
            raise ValueError('Unassessed or nonfinite task metric')
        return sum(values)/len(values)
    flat=[r['metrics'] for r in report['flat']];walls=[r['metrics'] for r in report['walls']]
    unknown=max(m['unknown_body_fraction'] for m in flat+walls)
    return {'fk_mpjpe_m':average(r['fk_mpjpe_m'] for r in report['short']),
        'idle_drift_m':average(m['drift_max_m'] for m in flat),
        'idle_speed_mps':average(m['settled_speed_p95_mps'] for m in flat),
        'foot_penetration_m':average(m['foot_penetration_p99_m'] for m in flat),
        'foot_gap_m':average(m['median_higher_foot_gap_m'] for m in flat),
        'start_progress_ratio':min(m['start_progress_ratio'] for m in flat if 'start_progress_ratio' in m),
        'wall_penetration_m':average(m['body_penetration_max_m'] for m in walls),
        'wall_final_speed_mps':average(m['final_speed_mps'] for m in walls),
        'unknown_body_fraction':unknown,
        'full_surface_audit':all(m.get('audited_vertices_per_frame',0)==18056 for m in flat+walls),
        'absolute_gates_passed':bool(report.get('absolute_gates_passed',False))}


def quality_score(values):
    if values.get('evaluation_goal')=='real-paired-terrain-traversal-v1':
        return (values['terrain_path_rmse_m']/.35+values['terrain_height_mae_m']/.2+
            (1-values['terrain_route_fraction'])+values['terrain_penetration_m']/.03+
            values['idle_drift_m']/.1+values['idle_speed_mps']/.1)/6
    return (values['idle_drift_m']/.1+values['idle_speed_mps']/.1+
        values['foot_penetration_m']/.02+values['foot_gap_m']/.05+
        values['wall_penetration_m']/.1+values['wall_final_speed_mps']/.15)/6


def pilot_decision(initial,baseline,candidate):
    a,b,c=map(summarize,(initial,baseline,candidate))
    score=quality_score(c)
    criteria={
        'complete_known_surface_audit':c['full_surface_audit'] and c['unknown_body_fraction']==0,
        'learned_start_instead_of_freezing':c['start_progress_ratio']>=.5,
        'start_retained_against_baseline':c['start_progress_ratio']>=.8*b['start_progress_ratio'],
        'reconstruction_retained':c['fk_mpjpe_m']<=1.1*b['fk_mpjpe_m']+.01,
        'reconstruction_retained_from_start':c['fk_mpjpe_m']<=1.1*a['fk_mpjpe_m']+.01,
        'physical_score_improves_over_baseline':score<=.9*quality_score(b),
        'physical_score_improves_from_start':score<=.9*quality_score(a),
        'wall_collision_not_hidden_by_other_wins':c['wall_penetration_m']<=1.1*b['wall_penetration_m']+.005,
        'idle_not_hidden_by_wall_wins':c['idle_drift_m']<=1.1*b['idle_drift_m']+.01,
    }
    return {'eligible_for_long_training':all(criteria.values()),'quality_approved':False,
        'criteria':criteria,'initial':a,'native_control':b,'candidate':c,
        'score':{'initial':quality_score(a),'native_control':quality_score(b),'candidate':score},
        'meaning':'pilot evidence to spend the remaining training budget; not a deployment or PFNN comparison'}
