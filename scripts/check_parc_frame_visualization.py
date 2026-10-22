"""Verify exported frame counts, media playback and gallery layout."""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    media = []
    for row in manifest['rows']:
        count, duration = imageio_ffmpeg.count_frames_and_secs(str(root/row['video']))
        with Image.open(root/row['gif']) as gif:
            assert gif.n_frames == row['frame_count'] == count
        assert abs(duration-count/row['fps']) < .02
        media.append({'file': row['video'], 'frames': count, 'duration_seconds': duration})
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width':1440, 'height':1000})
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto((root/'index.html').as_uri())
        page.wait_for_function('Array.from(document.querySelectorAll("video")).every(v=>v.readyState>=1)')
        assert page.locator('video').count() == len(media)
        page.select_option('#rate', '0.25')
        assert page.evaluate('Array.from(document.querySelectorAll("video")).every(v=>v.playbackRate===0.25)')
        page.evaluate('document.querySelector("video").play()')
        page.wait_for_function('document.querySelector("video").currentTime > 0.1')
        page.evaluate('document.querySelectorAll("video").forEach(v=>v.pause())')
        page.screenshot(path=str(root/'gallery-desktop.png'))
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(root/'gallery-mobile.png'))
        browser.close()
    assert not errors, errors
    result = {'status':'passed', 'media':media, 'playback_and_rate_control':True,
              'mobile_no_overflow':True, 'page_errors':errors, 'motion_quality_approved':False}
    (root/'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
