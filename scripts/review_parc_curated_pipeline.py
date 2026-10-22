"""Render corrected references / physics, with explicit non-approval diagnostics."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from render_parc_long_course import DEFAULT_REPO, _official_fk
from render_parc_physics_comparison import load_motion, render_pair


def metrics(path, repo):
    motion = load_motion(path)
    _, names, pos, _, _ = _official_fk(motion, repo, repo / 'data/assets/humanoid.xml')
    rot = Rotation.from_quat(motion['joint_rot'].reshape(-1, 4)).as_matrix().reshape(-1, 14, 3, 3)
    change = rot[1:] @ rot[:-1].swapaxes(-1, -2)
    angle = Rotation.from_matrix(change.reshape(-1, 3, 3)).magnitude() * (180 / np.pi)
    jerk = np.linalg.norm(np.diff(pos, n=3, axis=0), axis=-1) * motion['fps'] ** 3
    hands = [names.index(name) for name in ('left_hand', 'right_hand')]
    feet = [names.index(name) for name in ('left_foot', 'right_foot')]
    contacts = motion['contacts'] > .9
    return {'frames': len(pos), 'joint_step_max_deg': float(angle.max()),
            'joint_step_p99_deg': float(np.quantile(angle, .99)),
            'body_jerk_p99_mps3': float(np.quantile(jerk, .99)),
            'hand_contact_frames': int(contacts[:, hands].any(-1).sum()),
            'no_foot_contact_frames': int((~contacts[:, feet].any(-1)).sum()),
            'contact_interpretation': 'source labels for reference, force-derived for simulation; no-foot contact alone is not certified flight'}


def run(root, output, repo, render, corrections=None):
    corrections = corrections or root / 'corrected'
    report = json.loads((corrections / 'correction_report.json').read_text())
    exported = json.loads((root / 'export/manifest.json').read_text())
    cases = {case['case']: case for case in report['cases']}
    output.mkdir(parents=True, exist_ok=False)
    rows, sections = [], []
    for record in exported['recordings']:
        name = record['candidate_key']
        case = cases[name]
        directory = corrections / name / 'variant_0'
        source = directory / 'final.pkl'
        physical = root / 'export' / record['converted_path']
        row = {'name': name, 'whole_motion': record['whole_motion'],
               'raw': metrics(directory / 'raw.pkl', repo),
               'optimized': metrics(directory / 'optimized.pkl', repo),
               'final_reference': metrics(source, repo),
               'physics': metrics(physical, repo), 'training_approved': False}
        if render and record['whole_motion']:
            render_pair(record, source, physical, output / name, repo,
                        source_stage='OFFICIAL OPTIMIZED REFERENCE')
            sections.append('<h2>' + html.escape(name) + '</h2><video controls preload="metadata" src="' +
                            name + '/raw_vs_physics_1x.mp4"></video>')
        rows.append(row)
    (output / 'metrics.json').write_text(json.dumps({'cases': rows, 'training_approved': False}, indent=2))
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PARC corrected / physical</title>'
        '<style>body{font:16px system-ui;max-width:1280px;margin:24px auto;padding:0 16px;background:#f4f6f6}video{width:100%}</style>'
        '<h1>PARC official optimized references / physics</h1><p>30 FPS, 1x. No runtime smoothing or IK. Quality review pending.</p>' + ''.join(sections))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--corrections', type=Path)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.repo.resolve(), args.render,
        args.corrections.resolve() if args.corrections else None)
