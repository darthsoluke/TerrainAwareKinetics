"""Browser acceptance checks for skinned offline review, without live inference."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def probe(url, output, executable=None):
    output.mkdir(parents=True, exist_ok=False)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=executable, args=[
            "--use-angle=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
        try:
            for label, viewport in (("desktop", {"width": 1280, "height": 800}),
                                    ("mobile", {"width": 390, "height": 844})):
                page = browser.new_page(viewport=viewport)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url, wait_until="networkidle", timeout=120000)
                page.wait_for_function("window.somaLab?.ready && window.somaLab.stats.frame >= 8",
                                       timeout=120000)
                page.locator("#pause-button").click()
                page.locator("#camera-button").click()
                before = page.evaluate("""() => ({
                    frame: somaLab.stats.frame,
                    positions: Array.from(somaLab.mesh.geometry.attributes.position.array),
                    replay: somaLab.stats.replay,
                    terrain: somaLab.scene.children.some(item => item.name === 'Official PARC voxelized terrain'),
                    bounds: { toolbar: document.querySelector('.toolbar').getBoundingClientRect().toJSON(),
                              telemetry: document.querySelector('.telemetry').getBoundingClientRect().toJSON() }
                })""")
                screenshot = output / f"{label}.png"
                canvas = output / f"{label}_canvas.png"
                page.screenshot(path=str(screenshot))
                page.locator("#scene-canvas").screenshot(path=str(canvas))
                with Image.open(canvas) as image:
                    pixels = np.asarray(image.convert("RGB"))
                color_count = len(np.unique(pixels.reshape(-1, 3), axis=0))
                if color_count < 100 or float(pixels.std()) < 5:
                    raise AssertionError("Rendered canvas is blank or nearly uniform")
                if not before["terrain"] or before["replay"]["role"] not in {"source", "codec"}:
                    raise AssertionError("Real terrain and explicit review role are required")
                if before["bounds"]["toolbar"]["bottom"] >= before["bounds"]["telemetry"]["top"]:
                    raise AssertionError("Toolbar overlaps telemetry")
                # Selecting a new branch must resume an already paused or completed clip.
                options = page.locator("#clip-select option").evaluate_all(
                    "items => items.map(item => item.value)")
                current = page.locator("#clip-select").input_value()
                opposite = current[:-5] + "source" if current.endswith("codec") else current[:-6] + "codec"
                if opposite not in options:
                    raise AssertionError("Review source/codec pair is missing")
                page.locator("#clip-select").select_option(opposite)
                page.wait_for_function("window.somaLab.stats.frame >= 16", timeout=120000)
                after = page.evaluate("""() => ({
                    frame: somaLab.stats.frame,
                    positions: Array.from(somaLab.mesh.geometry.attributes.position.array),
                    replay: somaLab.stats.replay,
                    paused: document.querySelector('#pause-button').getAttribute('aria-pressed'),
                    error: document.querySelector('#error-toast').hidden ? null : document.querySelector('#error-toast').textContent
                })""")
                if after["paused"] != "false" or after["frame"] < 16:
                    raise AssertionError("New review clip does not play")
                positions = np.asarray(before["positions"])
                changed = float(np.max(np.abs(positions - np.asarray(after["positions"]))))
                if changed < 1e-5 or errors or after["error"]:
                    raise AssertionError(f"Review did not update or browser failed: {errors}, {after['error']}")
                results.append({"viewport": label, "canvas_colors": color_count,
                                "max_vertex_change_m": changed, "frame": after["frame"],
                                "review_role": before["replay"]["role"],
                                "opposite_review_role": after["replay"]["role"],
                                "page_errors": errors, "screenshot": str(screenshot.resolve())})
                page.close()
        finally:
            browser.close()
    report = {"url": url, "checks": results,
              "scope": "rendering and offline replay only; not live terrain control or FPS acceptance"}
    (output / "browser_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser-executable", help="optional installed Chromium executable")
    args = parser.parse_args()
    print(json.dumps(probe(args.url, args.output, args.browser_executable), indent=2))


if __name__ == "__main__":
    main()
