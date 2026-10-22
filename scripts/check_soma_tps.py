"""Real browser TPS controls and rendering checks; not motion approval."""

import argparse
import json
import math
from pathlib import Path

from playwright.sync_api import sync_playwright
from scripts.check_soma_lab import CANVAS_CHECK


BASIS = """() => {
  const camera = somaLab.orbitCamera;
  camera.updateMatrixWorld(true);
  const m = camera.matrixWorld.elements;
  const f = [-m[8], -m[10]], r = [m[0], m[2]];
  const normalize = v => v.map(x => x / Math.hypot(...v));
  return {forward: normalize(f), right: normalize(r), control: somaLab.stats.control,
    camera: somaLab.stats.camera};
}"""


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def direction(values):
    length = sum(value * value for value in values) ** .5
    return [value / length for value in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:28880/?mode=terrain')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report, errors = {'quality_approved': False}, []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True,
            args=['--enable-webgl', '--ignore-gpu-blocklist'])
        context = browser.new_context(viewport={'width': 1440, 'height': 900})
        page = context.new_page()
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function('somaLab.ready && somaLab.stats.control.mode === "terrain" && somaLab.stats.frame >= 8', timeout=60000)
        page.locator('#pause-button').click()
        report['desktop'] = page.evaluate(CANVAS_CHECK)
        assert report['desktop']['colors'] > 100 and not report['desktop']['overflow']
        assert all(abs(v) < 1 for v in report['desktop']['mesh_center_ndc'][:2])
        page.screenshot(path=str(args.output / 'tps-desktop.png'))
        assert not page.evaluate('somaLab.stats.camera.collision')
        assert page.evaluate('somaLab.camera.position.distanceTo(somaLab.orbitCamera.position)') < 1e-6
        for key, axis, sign in [('w', 'forward', 1), ('s', 'forward', -1),
                                ('d', 'right', 1), ('a', 'right', -1)]:
            page.keyboard.down(key)
            basis = page.evaluate(BASIS)
            assert dot(direction(basis['control']['velocity_xz']), basis[axis]) * sign > .99999, basis
            page.keyboard.up(key)
        page.keyboard.down('w')
        before = page.evaluate(BASIS)
        page.mouse.move(700, 450)
        page.mouse.down(button='right')
        page.mouse.move(970, 450, steps=10)
        page.mouse.up(button='right')
        after = page.evaluate(BASIS)
        assert dot(before['forward'], after['forward']) < .95
        assert dot(direction(after['control']['velocity_xz']), after['forward']) > .99999
        report['held_w_rotating_camera'] = {'before': before, 'after': after}
        page.keyboard.down('d')
        diagonal = page.evaluate(BASIS)
        assert abs(diagonal['control']['speed'] - 1.6) < 1e-5
        assert abs(diagonal['control']['heading'] - math.atan2(*diagonal['control']['velocity_xz'])) < 1e-6
        page.keyboard.up('w')
        page.keyboard.up('d')
        assert page.evaluate('somaLab.stats.control.speed') == 0
        def vertical_drag():
            page.mouse.move(700, 440)
            page.mouse.down(button='right')
            page.mouse.move(700, 500, steps=5)
            page.mouse.up(button='right')
        polar = page.evaluate('somaLab.stats.camera.polar')
        vertical_drag()
        normal = page.evaluate('somaLab.stats.camera.polar')
        assert normal < polar
        page.locator('#look-invert-toggle').click()
        vertical_drag()
        inverted = page.evaluate('somaLab.stats.camera.polar')
        assert inverted > normal
        page.locator('#look-invert-toggle').click()
        report['vertical_inversion'] = {'initial': polar, 'normal': normal, 'inverted': inverted}
        distance = page.evaluate('somaLab.stats.camera.desired_distance')
        page.mouse.move(700, 440)
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(100)
        assert page.evaluate('somaLab.stats.camera.desired_distance') < distance
        page.locator('#camera-button').click()
        # Request through a real click, never a synthetic pointer-lock event.
        page.locator('#mouse-look-button').click()
        page.wait_for_timeout(200)
        report['pointer_lock_acquired'] = page.evaluate('document.pointerLockElement === document.querySelector("#scene-canvas")')
        if report['pointer_lock_acquired']:
            page.keyboard.down('w')
            locked_before = page.evaluate(BASIS)
            page.mouse.move(800, 400)
            page.mouse.move(930, 430, steps=8)
            locked_after = page.evaluate(BASIS)
            assert dot(direction(locked_after['control']['velocity_xz']), locked_after['forward']) > .99999
            assert dot(locked_before['forward'], locked_after['forward']) < .999
            page.keyboard.press('Escape')
            page.wait_for_function('!document.pointerLockElement')
            assert page.evaluate('somaLab.stats.control.speed') == 0
            page.keyboard.up('w')
        page.set_viewport_size({'width': 390, 'height': 844})
        page.locator('#camera-button').click()
        page.wait_for_timeout(200)
        report['mobile'] = page.evaluate(CANVAS_CHECK)
        assert report['mobile']['colors'] > 100 and not report['mobile']['overflow']
        assert all(abs(v) < 1 for v in report['mobile']['mesh_center_ndc'][:2])
        page.screenshot(path=str(args.output / 'tps-mobile.png'))
        assert not errors, errors
        report.update(status='passed', page_errors=errors,
            camera_collision=False,
            scope='TPS intent, mouse/zoom/inversion and rendering; not terrain skill')
        (args.output / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        browser.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
