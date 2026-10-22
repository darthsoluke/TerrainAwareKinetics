"""Compare the four small decoder pilots without declaring a production winner."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--render', action='store_true')
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    names = ('baseline', 'feedback', 'terrain', 'both')
    contracts, results, reviews = {}, {}, {}
    for name in names:
        contracts[name] = read(args.runs/name/'contract.json')
        results[name] = read(args.runs/name/'training_result.json')
        reviews[name] = read(args.runs/name/'review/metrics.json')
    for key in ('source_bundle_sha256', 'data_manifest_sha256', 'steps', 'seed', 'batch_windows', 'primitives_per_update'):
        if len({json.dumps(row[key], sort_keys=True) for row in contracts.values()}) != 1:
            raise ValueError('Unmatched comparison contract: '+key)
    if len({r['sampling_sha256'] for r in results.values()}) != 1:
        raise ValueError('Training windows/draws differ across arms')
    if any(r['updates'] != 500 for r in results.values()):
        raise ValueError('Incomplete500-update comparison')
    identities = [[(r['source_id'], r['start'], r['history']) for r in reviews[name]['rows']] for name in names]
    if any(value != identities[0] for value in identities[1:]):
        raise ValueError('Held-out comparison windows differ')
    for name in ('feedback', 'both'):
        if results[name]['counts']['generated_second'] < 1:
            raise ValueError('Feedback arm never trained on generated history')
    source = read(args.runs/'baseline/initial_review/metrics.json')
    table = []
    for name, review in [('source', source), *reviews.items()]:
        metrics = review['summary']['played_history']
        table.append({'arm': name, **metrics,
            'rest_drift_max_m': max((r['drift_max_m'] for r in review['rest']), default=None),
            'rest_penetration_p99_max_m': max((r['penetration_p99_m'] for r in review['rest']), default=None),
            'higher_foot_gap_max_median_m': max((r['higher_foot_gap_median_m'] for r in review['rest']), default=None),
            'invalid_predicted_points': sum(r['invalid_predicted_points'] for r in review['rows'] if r['history']=='played_history')})
    oracle = {}
    for token in ('model', 'source'):
        for root in ('model', 'source'):
            rows = [r for r in source['oracle'] if r['tokens']==token and r['root']==root]
            oracle[token+'_tokens_'+root+'_root'] = {key: float(np.mean([r[key] for r in rows]))
                for key in ('root_y_mae_m', 'fk_mpjpe_m')}
    report = {'schema': 'motionbricks.terrain_decoder_pilot_comparison.v1', 'quality_approved': False,
        'table': table, 'oracle': oracle, 'training': results,
        'matched_source_sha256': contracts['baseline']['source_bundle_sha256'],
        'matched_data_sha256': contracts['baseline']['data_manifest_sha256'],
        'sampling_sha256': results['baseline']['sampling_sha256'],
        'notes': ['24 train clips/96 fixed windows;12 validation clips/36 fixed windows; single seed',
                  'Each arm adapts only the decoder for500 updates; Root/Pose and dictionary frozen',
                  'Source row is the unadapted source bundle; baseline is equally budgeted decoder finetuning',
                  'Short terrain feedback test is32 frames; rest/start-stop is30 seconds after stop',
                  'No IK, smoothing, root projection, force solver or live deployment',
                  'Oracle source tokens/Root are diagnostic only, not deployed observations',
                  'Overlapping windows and one seed do not establish population-level significance']}
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# Small Decoder Terrain/Feedback Pilot', '',
        'All four arms completed500 updates on the same samples. Lower values are better.', '',
        '| Arm | Root Y MAE (mm) | FK MPJPE (mm) | Penetration mean (mm) | Slow-point speed (mm/s) | Rest drift max (m) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in table:
        values = [row[k] for k in ('root_y_mae_m','fk_mpjpe_m','penetration_mean_m','slow_vertex_speed_mps')]
        values = ['unavailable' if v is None else f'{v*1000:.2f}' for v in values]
        drift = 'unavailable' if row['rest_drift_max_m'] is None else f"{row['rest_drift_max_m']:.3f}"
        lines.append('| '+row['arm']+' | '+' | '.join(values)+' | '+drift+' |')
    lines += ['', *['- '+note for note in report['notes']], '',
        'Physical skill quality remains unapproved. Inspect contact coverage and long-rollout drift, not only reconstruction error.']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (key, label, scale) in zip(axes.flat, [
        ('root_y_mae_m','Root height MAE (mm)',1000), ('fk_mpjpe_m','FK MPJPE (mm)',1000),
        ('slow_vertex_speed_mps','Source-slow vertex speed (mm/s)',1000), ('rest_drift_max_m','Rest/start-stop drift (m)',1)]):
        values = [np.nan if row[key] is None else row[key]*scale for row in table]
        ax.bar([r['arm'] for r in table], values, color=['#6b7280','#3874a8','#c08035','#32846b','#9f526f'])
        ax.set_title(label);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(args.output/'comparison.png', dpi=150);plt.close(fig)
    subprocess.run([sys.executable, '-u', '-m', 'scripts.probe_decoder_terrain_sensitivity',
        '--base', str(args.runs.parent), '--output', str(args.output/'terrain_sensitivity.json')], check=True)
    if args.render:
        try:
            import torch
            from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
            from motionbricks.data.soma_skin_reference import load_official_skinner
            import scripts.render_parc_frame_ablation as renderer
            torch.set_num_threads(4)
            renderer.LABELS = ('Source motion', 'Baseline decoder', 'Terrain + feedback')
            skin = load_official_skinner(SOMASkeleton30())
            videos = []
            for index, label in ((0, 'Held-out flat motion'), (27, 'Held-out generated terrain')):
                videos.append(renderer.render(args.runs/'baseline/review', args.runs/'both/review',
                    args.output, skin, index, label))
            report['videos'] = videos
        except Exception as error:
            report['render_error'] = type(error).__name__+': '+str(error)
        (args.output/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'table': table, 'oracle': oracle, 'render_error': report.get('render_error')}, indent=2))


if __name__ == '__main__':
    main()
