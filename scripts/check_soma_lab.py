"""Exercise the running SOMA browser viewer and capture rendered evidence."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


CANVAS_CHECK = """() => {
  const lab = window.somaLab;
  const renderer = lab.renderer;
  renderer.setRenderTarget(null);
  renderer.render(lab.scene, lab.camera);
  const gl = renderer.getContext();
  const width = gl.drawingBufferWidth, height = gl.drawingBufferHeight;
  const pixels = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const colors = new Set();
  let min = 255, max = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    min = Math.min(min, pixels[i]); max = Math.max(max, pixels[i]);
    colors.add((pixels[i] << 16) | (pixels[i + 1] << 8) | pixels[i + 2]);
  }
  const geometry = lab.mesh.geometry;
  const boundsOwner = lab.mesh.isSkinnedMesh ? lab.mesh : geometry;
  boundsOwner.computeBoundingBox();
  const center = boundsOwner.boundingBox.getCenter(lab.camera.position.clone());
  lab.mesh.localToWorld(center);
  center.project(lab.camera);
  return {colors: colors.size, pixel_range: max - min,
    mesh_center_ndc: center.toArray(), root: [...lab.stats.root],
    vertices: geometry.attributes.position.count,
    overflow: document.documentElement.scrollWidth > innerWidth};
}"""

DEPTH_CHECK = """() => new Promise(resolve => requestAnimationFrame(() => {
  const renderer = somaLab.renderer, gl = renderer.getContext();
  const rect = document.querySelector('#depth-frame').getBoundingClientRect();
  const canvas = renderer.domElement.getBoundingClientRect();
  const sx = gl.drawingBufferWidth / canvas.width;
  const sy = gl.drawingBufferHeight / canvas.height;
  const width = Math.floor(rect.width * sx), height = Math.floor(rect.height * sy);
  const pixels = new Uint8Array(width * height * 4);
  gl.readPixels(Math.floor((rect.left - canvas.left) * sx),
    Math.floor((canvas.bottom - rect.bottom) * sy), width, height,
    gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const colors = new Set();
  let min = 255, max = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    colors.add(pixels[i]); min = Math.min(min, pixels[i]); max = Math.max(max, pixels[i]);
  }
  resolve({colors: colors.size, pixel_range: max - min,
    overlay_transparent: ['#depth-inset', '#depth-frame'].every(selector =>
      getComputedStyle(document.querySelector(selector)).backgroundColor === 'rgba(0, 0, 0, 0)'),
    observation: somaLab.stats.observation});
}))"""


def check_page(page, output, name):
    result = page.evaluate(CANVAS_CHECK)
    assert result["colors"] > 100 and result["pixel_range"] > 30, result
    assert all(abs(v) < 1 for v in result["mesh_center_ndc"][:2]), result
    assert result["vertices"] == 18056 and not result["overflow"], result
    page.screenshot(path=str(output / f"{name}.png"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:28780")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    report = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True,
                                             args=["--enable-webgl", "--ignore-gpu-blocklist"])
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function("window.somaLab?.ready && window.somaLab.stats.applied > 8", timeout=30000)
        report["desktop"] = check_page(page, args.output, "desktop")
        page.locator("#depth-toggle").click()
        page.wait_for_timeout(250)
        depth = page.evaluate(DEPTH_CHECK)
        assert depth["colors"] > 20 and depth["pixel_range"] > 50 and depth["overlay_transparent"], depth
        assert depth["observation"]["enabled"] and not depth["observation"]["connected_to_model"], depth
        report["depth_preview"] = depth
        start = page.evaluate("({root:[...somaLab.stats.root], y:somaLab.camera.position.y})")
        page.keyboard.down("w")
        page.wait_for_timeout(2000)
        observation = page.evaluate("somaLab.stats.observation.intent_xz")
        velocity = page.evaluate("somaLab.stats.control.velocity_xz")
        speed = sum(v * v for v in velocity) ** .5
        assert all(abs(observation[i] - velocity[i] / speed) < 1e-5 for i in (0, 1))
        page.screenshot(path=str(args.output / "walking_depth.png"))
        report["walking"] = check_page(page, args.output, "walking")
        page.keyboard.up("w")
        moved = page.evaluate("({root:[...somaLab.stats.root], y:somaLab.camera.position.y, control:somaLab.stats.control})")
        distance = sum((moved["root"][i] - start["root"][i]) ** 2 for i in (0, 2)) ** .5
        assert .1 < distance < 8, (start, moved)
        assert abs(start["y"] - moved["y"]) < 1e-5, (start, moved)
        assert moved["control"]["velocity_xz"] == [0, 0], moved
        report["wasd"] = {"distance_m": distance, "camera_y_change_m": moved["y"] - start["y"]}
        page.keyboard.down("w")
        page.keyboard.down("d")
        command = page.evaluate("somaLab.stats.control.velocity_xz")
        assert sum(v * v for v in command) ** .5 <= 3.000001, command
        page.keyboard.up("w")
        page.keyboard.up("d")
        page.locator("#depth-toggle").click()
        page.locator("#pause-button").click()
        frozen = page.evaluate("somaLab.stats.applied")
        page.wait_for_timeout(500)
        assert page.evaluate("somaLab.stats.applied") == frozen
        page.locator("#pause-button").click()
        page.locator("#reset-button").click()
        page.wait_for_function("somaLab.ready && somaLab.stats.frame > 16", timeout=20000)
        assert page.locator("#error-toast").is_hidden()
        report["reset"] = page.evaluate("({root:[...somaLab.stats.root], frame:somaLab.stats.frame})")
        page.locator("#replay-mode").click()
        page.locator("#clip-select").select_option("three_stair_flights")
        page.wait_for_function("somaLab.stats.frame > 180", timeout=20000)
        assert page.evaluate("somaLab.scene.getObjectByName('Official PARC voxelized terrain')?.geometry.attributes.position.count > 0")
        report["stairs_replay"] = check_page(page, args.output, "stairs_replay")
        page.locator("#depth-toggle").click()
        page.wait_for_timeout(250)
        depth = page.evaluate(DEPTH_CHECK)
        assert depth["colors"] > 20 and depth["pixel_range"] > 50 and depth["overlay_transparent"], depth
        report["stairs_depth"] = depth
        page.screenshot(path=str(args.output / "stairs_depth.png"))
        page.locator("#depth-toggle").click()
        page.locator("#live-mode").click()
        page.wait_for_function("somaLab.ready && somaLab.stats.frame > 16", timeout=20000)
        assert page.locator("#error-toast").is_hidden()
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(700)
        report["mobile"] = check_page(page, args.output, "mobile")
        pad = page.locator('[data-pad="w"]').bounding_box()
        page.mouse.move(pad["x"] + pad["width"] / 2, pad["y"] + pad["height"] / 2)
        page.mouse.down()
        assert sum(v * v for v in page.evaluate("somaLab.stats.control.velocity_xz")) > 0
        page.mouse.up()
        report["final_stats"] = page.evaluate("somaLab.stats")
        report["browser_errors"] = errors
        assert not errors, errors
        report["status"] = "passed"
        report["scope"] = "frontend rendering, input, native playback and source replay; not motion quality approval"
        (args.output / "browser_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        browser.close()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
