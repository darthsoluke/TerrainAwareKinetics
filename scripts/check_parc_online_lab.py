"""Actual browser/renderer/input checks, explicitly not motion-quality approval."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from scripts.check_soma_lab import CANVAS_CHECK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:28880/?mode=terrain')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors, report = [], {'quality_approved': False}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True,
                                              args=['--enable-webgl', '--ignore-gpu-blocklist'])
        page = browser.new_page(viewport={'width': 1280, 'height': 800})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function('somaLab.ready && !document.querySelector("#terrain-mode").disabled', timeout=60000)
        page.locator('#terrain-mode').click()
        page.wait_for_function('somaLab.ready && somaLab.stats.control.mode === "terrain" && somaLab.stats.frame >= 8', timeout=60000)
        page.locator('#pause-button').click()
        report['desktop'] = page.evaluate(CANVAS_CHECK)
        assert report['desktop']['colors'] > 100 and report['desktop']['pixel_range'] > 30
        assert report['desktop']['vertices'] == 18056 and not report['desktop']['overflow']
        assert page.locator('#terrain-warning').is_visible()
        assert page.locator('#error-toast').is_hidden()
        page.screenshot(path=str(args.output / 'terrain-desktop.png'))
        report['generation'] = page.evaluate('somaLab.stats.generation')
        assert report['generation']['future_height_source'] == 'model'
        assert report['generation']['future_pose_source'] == 'model'
        assert report['generation']['inference_config']['num_inference_step'] == 4
        page.locator('#pause-button').click()
        applied = page.evaluate('somaLab.stats.applied')
        page.keyboard.down('w')
        page.wait_for_function('(old) => somaLab.stats.applied >= old + 16', arg=applied, timeout=60000)
        report['wasd_control'] = page.evaluate('somaLab.stats.control')
        assert sum(value * value for value in report['wasd_control']['velocity_xz']) > 0
        page.keyboard.up('w')
        page.locator('#pause-button').click()
        report['wasd'] = page.evaluate(CANVAS_CHECK)
        page.screenshot(path=str(args.output / 'terrain-wasd.png'))
        page.locator('#reset-button').click()
        page.wait_for_function('somaLab.ready && somaLab.stats.control.mode === "terrain"', timeout=60000)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(500)
        report['mobile'] = page.evaluate(CANVAS_CHECK)
        assert report['mobile']['colors'] > 100 and not report['mobile']['overflow']
        page.screenshot(path=str(args.output / 'terrain-mobile.png'))
        assert page.locator('#error-toast').is_hidden()
        report['browser_errors'] = errors
        assert not errors, errors
        report['scope'] = 'experimental rendering, native generation and played-history WASD protocol only'
        report['status'] = 'passed'
        (args.output / 'browser_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        browser.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
