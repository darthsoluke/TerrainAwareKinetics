"""Check real online diagnostics, transport sources, controls and rendered layers."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from scripts.check_soma_lab import CANVAS_CHECK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:28880/?mode=terrain')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors, plans, report = [], [], {'quality_approved': False}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True,
            args=['--enable-webgl', '--ignore-gpu-blocklist'])
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        page.on('pageerror', lambda error: errors.append(str(error)))

        def receive(payload):
            if isinstance(payload, str):
                packet = json.loads(payload)
                if packet.get('type') == 'plan':
                    plans.append(packet)

        page.on('websocket', lambda ws: ws.on('framereceived', receive))
        page.goto(args.url)
        page.wait_for_function('somaLab.ready && somaLab.stats.control.mode === "terrain" && somaLab.stats.frame >= 16', timeout=90000)
        page.locator('#pause-button').click()
        initial = page.evaluate('somaLab.stats.diagnostics')
        assert initial['observation_source'] == 'current_played_history'
        assert initial['input_height_source'] == 'not_conditioned'
        assert initial['future_height_source'] == 'model'
        assert initial['played_frames'] == page.evaluate('somaLab.stats.frame')
        assert not page.evaluate('somaLab.stats.camera.collision')
        assert len(plans) >= 2
        for packet in plans:
            debug = packet['debug']
            assert debug['version'] == 'soma_motion_debug_v1'
            assert len(debug['root_stage_world_xyz_m']) >= 8
            assert len(debug['decoded_world_xyz_m']) >= 8
            assert len(debug['joints_world_xyz_m']) == 8
            assert all(len(frame) == 30 for frame in debug['joints_world_xyz_m'])
            assert debug['input_intent']['height_source'] == 'not_conditioned'
            assert len(debug['terrain_patches']) == 2
            for patch in debug['terrain_patches']:
                assert patch['shape'] == [9, 9]
                assert len(patch['world_xyz_m']) == len(patch['valid']) == 81
        report['transport'] = {'plans': len(plans), 'future_height_source': 'model',
                               'full_soma_joints': 30, 'terrain_patch_shape': [2, 9, 9]}
        page.wait_for_timeout(400)
        assert page.evaluate('somaLab.stats.diagnostics.played_frames') == initial['played_frames']
        # Camera-relative input changes only the planar guide, not the frozen model geometry.
        page.keyboard.down('w')
        assert page.evaluate('somaLab.stats.diagnostics.layers.intent.segments') == 2
        page.keyboard.up('w')
        for name in initial['layers']:
            checkbox = page.locator(f'[data-debug-layer="{name}"]')
            checkbox.uncheck()
            assert not page.evaluate(f'somaLab.scene.getObjectByName("diagnostic-{name}").visible')
            checkbox.check()
            assert page.evaluate(f'somaLab.scene.getObjectByName("diagnostic-{name}").visible')
        assert page.evaluate('somaLab.stats.diagnostics.layers.skeleton.segments') == 29
        assert initial['layers']['near']['points'] > 0 and initial['layers']['terrainIntent']['points'] > 0
        page.locator('#camera-button').click()
        report['desktop'] = page.evaluate(CANVAS_CHECK)
        assert report['desktop']['colors'] > 100 and not report['desktop']['overflow']
        page.screenshot(path=str(args.output / 'diagnostics-desktop.png'))
        page.set_viewport_size({'width': 390, 'height': 844})
        page.locator('#camera-button').click()
        page.wait_for_timeout(100)
        bounds = page.locator('#diagnostics-panel').bounding_box()
        toolbar = page.locator('.toolbar').bounding_box()
        assert bounds['y'] >= toolbar['y'] + toolbar['height']
        assert bounds['x'] >= 0 and bounds['x'] + bounds['width'] <= 390
        report['mobile'] = page.evaluate(CANVAS_CHECK)
        assert report['mobile']['colors'] > 100 and not report['mobile']['overflow']
        page.screenshot(path=str(args.output / 'diagnostics-mobile.png'))
        page.locator('#diagnostics-toggle').click()
        assert page.locator('#diagnostics-panel').is_hidden()
        page.locator('#diagnostics-toggle').click()
        assert page.locator('#diagnostics-panel').is_visible()
        page.locator('#reset-button').click()
        page.wait_for_function('somaLab.ready && somaLab.stats.frame === 0')
        assert page.evaluate('somaLab.stats.diagnostics.played_frames') == 0
        assert page.evaluate('somaLab.stats.diagnostics.layers.root.points') == 0
        assert page.evaluate('somaLab.stats.diagnostics.layers.near.points') == 0
        page.locator('#pause-button').click()
        page.keyboard.down('w')
        page.wait_for_function('somaLab.stats.frame >= 16', timeout=60000)
        page.keyboard.up('w')
        page.locator('#pause-button').click()
        assert page.evaluate('somaLab.stats.diagnostics.layers.history.segments') >= 15
        report['moving'] = page.evaluate('somaLab.stats.diagnostics')
        page.set_viewport_size({'width': 1440, 'height': 900})
        page.locator('#camera-button').click()
        page.mouse.move(700, 480)
        page.mouse.down(button='right')
        page.mouse.move(940, 490, steps=8)
        page.mouse.up(button='right')
        page.wait_for_timeout(100)
        page.screenshot(path=str(args.output / 'diagnostics-moving.png'))
        page.locator('#replay-mode').click()
        page.wait_for_function('somaLab.stats.control.mode === "replay" && somaLab.stats.frame >= 8', timeout=60000)
        page.locator('#pause-button').click()
        replay = page.evaluate('somaLab.stats.diagnostics')
        assert replay['layers']['root']['points'] == 0
        assert replay['layers']['decoded']['points'] == 0
        assert replay['layers']['near']['points'] == 0
        assert replay['layers']['history']['segments'] >= 7
        assert replay['future_height_source'] is None
        report['replay'] = {'no_fake_predictions': True, 'played_history_present': True}
        assert not errors, errors
        report.update(status='passed', page_errors=errors,
            scope='Read-only trajectories, source labels, material geometry and layer rendering; not motion approval')
        (args.output / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        browser.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
