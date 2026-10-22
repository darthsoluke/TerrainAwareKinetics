"""Compare browser SOMA rig decoding with official LBS and inspect two viewports."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from playwright.sync_api import sync_playwright

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.soma_skin_reference import load_official_skinner
from scripts.check_soma_lab import CANVAS_CHECK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:28880/?mode=terrain')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    skinner = load_official_skinner(build_native_representation().skeleton)
    strongest_extra = skinner.lbs_weights[:, 4:].sum(-1).argsort(descending=True)[:200].tolist()
    indices = sorted(set(list(range(0, len(skinner.bind_vertices), 31)) + strongest_extra))
    report = {'reference': 'official SOMASkin.lbs', 'asset_influences': skinner.lbs_weights.shape[-1],
              'weights_dropped': False, 'screens': [], 'motion_quality_approved': False}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True,
            args=['--enable-webgl', '--ignore-gpu-blocklist'])
        for name, size in [('desktop', {'width': 1440, 'height': 900}),
                           ('mobile', {'width': 390, 'height': 844})]:
            page = browser.new_page(viewport=size)
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('console', lambda message: errors.append(message.text)
                    if message.type == 'error' and 'THREE.' in message.text else None)
            page.goto(args.url)
            page.wait_for_function('somaLab.skinRig && somaLab.stats.frame >= 16', timeout=60000)
            page.locator('#pause-button').click()
            sample = page.evaluate('''async indices => {
              const THREE = await import('/vendor/three.module.js');
              const mesh = somaLab.mesh;
              return {rig: Array.from(somaLab.skinRig),
                points: indices.map(i => mesh.localToWorld(mesh.getVertexPosition(i, new THREE.Vector3())).toArray()),
                gpu_skin: mesh.isSkinnedMesh,
                extra_influences: mesh.geometry.attributes.skinWeight2.itemSize};
            }''', indices)
            with torch.inference_mode():
                rig = torch.tensor(sample['rig'], dtype=torch.float32).reshape(1, 77, 4, 4)
                reference = skinner.lbs(rig)[0, indices].numpy()
            error = float(np.linalg.norm(np.asarray(sample['points'])-reference, axis=-1).max())
            canvas = page.evaluate(CANVAS_CHECK)
            assert sample['gpu_skin'] and sample['extra_influences'] == 4
            assert error < 2e-5, error
            assert not errors, errors
            assert canvas['colors'] > 100 and not canvas['overflow']
            assert all(abs(value) < 1 for value in canvas['mesh_center_ndc'][:2])
            page.screenshot(path=str(args.output / f'{name}.png'))
            report['screens'].append({'viewport': name, 'vertices_checked': len(indices),
                'reference_max_error_m': error, 'canvas': canvas, 'page_errors': errors})
            page.close()
        browser.close()
    report['status'] = 'passed'
    (args.output / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
