"""Real WASD requests and raw online playback with the relative-height artifact."""

import argparse
import json
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from scripts.check_soma_lab import CANVAS_CHECK
from scripts.check_soma_tps import BASIS, dot, direction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:28880/?mode=terrain')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-model', default='relative-height.pt')
    p.add_argument('--vertical-frame', default='observed-pelvis-y-v1')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plans, errors, commands, rows = [], [], [], []
    with sync_playwright() as engine:
        browser = engine.chromium.launch(channel='msedge', headless=True,
            args=['--enable-webgl', '--ignore-gpu-blocklist'])
        page = browser.new_page(viewport={'width':1440, 'height':900})
        page.on('pageerror', lambda e: errors.append(str(e)))
        def receive(payload):
            if isinstance(payload, str):
                message = json.loads(payload)
                if message.get('type') == 'plan':
                    plans.append(message)
                elif message.get('type') == 'error':
                    errors.append(message['message'])
        def sent(payload):
            if isinstance(payload, str):
                message = json.loads(payload)
                if message.get('type') in ('plan', 'prefetch'):
                    commands.append(message)
        def socket(ws):
            ws.on('framereceived', receive)
            ws.on('framesent', sent)
        page.on('websocket', socket)
        page.goto(args.url)
        page.wait_for_function('somaLab.ready && somaLab.stats.control.mode === "terrain" && somaLab.stats.frame >= 8', timeout=60000)
        config = page.request.get(args.url.split('?')[0]+'api/config').json()
        assert config['terrain_mode']['vertical_frame'] == args.vertical_frame
        assert config['terrain_mode']['model_file'] == args.expected_model
        page.locator('#pause-button').click()
        for key, axis, sign in [('w','forward',1), ('a','right',-1), ('s','forward',-1), ('d','right',1)]:
            page.locator('#reset-button').click()
            page.wait_for_function('somaLab.ready && somaLab.stats.frame === 0', timeout=30000)
            start = page.evaluate('({root:[...somaLab.stats.root], frame:somaLab.stats.frame})')
            request_start = len(commands)
            page.keyboard.down(key)
            basis = page.evaluate(BASIS)
            assert dot(direction(basis['control']['velocity_xz']), basis[axis])*sign > .99999
            page.locator('#pause-button').click()
            page.wait_for_function('somaLab.stats.frame >= 32', timeout=30000)
            page.locator('#pause-button').click()
            page.keyboard.up(key)
            end = page.evaluate('({root:[...somaLab.stats.root], frame:somaLab.stats.frame})')
            requested = basis['control']['velocity_xz']
            assert any(np.allclose(c['velocity_xz'], requested, atol=1e-6) for c in commands[request_start:])
            displacement = np.asarray(end['root'])-np.asarray(start['root'])
            rows.append({'key':key, 'input_velocity_xz':requested, 'start':start, 'end':end,
                         'displacement_xyz_m':displacement.tolist(),
                         'displacement_along_intent_m':float(displacement[[0,2]]@direction(requested))})
            if key == 'w':
                page.screenshot(path=str(args.output/'wasd-forward.png'))
        assert plans and any(m['generation']['neural_generation'] for m in plans)
        assert all(m['generation']['future_height_source'] == 'model' and
                   m['generation']['future_pose_source'] == 'model' for m in plans)
        assert not errors, errors
        canvas = page.evaluate(CANVAS_CHECK)
        assert canvas['colors'] > 100 and not canvas['overflow']
        page.locator('#reset-button').click()
        page.wait_for_function('somaLab.ready && somaLab.stats.frame === 0', timeout=30000)
        page.screenshot(path=str(args.output/'ready.png'))
        browser.close()
    result = {'status':'passed', 'model':config['terrain_mode'], 'keys':rows,
              'neural_generations':sum(m['generation']['neural_generation'] for m in plans),
              'input_and_output_contract':'real keyboard intent, model-generated Root/Pose, official SOMA skin',
              'camera_collision':False, 'page_errors':errors, 'motion_quality_approved':False}
    (args.output/'checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
