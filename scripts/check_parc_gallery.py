"""Check the delivered static gallery in an isolated headless browser."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(output):
    output = output.resolve()
    gallery = output / "gallery/index.html"
    results = []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True,
            executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe")
        try:
            for name, width, height in (("desktop", 1440, 1000), ("mobile", 390, 844)):
                page = browser.new_page(viewport={"width": width, "height": height})
                page.goto(gallery.as_uri(), wait_until="load")
                count = page.locator("article.clip").count()
                assert count > 0
                overflow = page.evaluate("document.documentElement.scrollWidth > innerWidth")
                assert not overflow, f"Horizontal overflow at {width} px"
                page.locator("#coverage").select_option("covered")
                covered = page.locator("article.clip:visible").count()
                assert covered == page.locator('article.clip[data-coverage="covered"]').count()
                if page.locator('#course option[value="extension|0"]').count():
                    page.locator("#course").select_option("extension|0")
                    assert page.locator("article.clip:visible").count() == page.locator(
                        'article.clip[data-course="extension|0"][data-coverage="covered"]').count()
                page.locator("#search").fill("definitely-no-such-candidate")
                assert page.locator("article.clip:visible").count() == 0
                page.locator("#search").fill("")
                page.locator("#course").select_option("")
                page.locator("#coverage").select_option("")
                assert page.locator("article.clip:visible").count() == count
                page.locator("video").first.evaluate("async video => { video.muted = true; await video.play(); }")
                page.wait_for_function("document.querySelector('video').currentTime > 0.1", timeout=15000)
                playing = page.locator("video").first.evaluate(
                    "video => ({width:video.videoWidth,height:video.videoHeight,time:video.currentTime,duration:video.duration})")
                assert playing["width"] > 0 and playing["height"] > 0
                page.locator("#search").fill("definitely-no-such-candidate")
                assert page.locator("video").first.evaluate("video => video.paused")
                page.locator("#search").fill("")
                page.screenshot(path=str(output / f"gallery_{name}.png"))
                results.append({"viewport": name, "width": width, "height": height,
                                "candidate_count": count, "covered_count": covered,
                                "horizontal_overflow": overflow, "playback": playing,
                                "filters_passed": True})
                page.close()
        finally:
            browser.close()
    (output / "gallery_browser_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args().output)
