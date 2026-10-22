"""Browser acceptance of the authored world and camera, not motion quality."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from scripts.check_soma_lab import CANVAS_CHECK


CAMERA_CHECK = """async () => {
  const {createHeightFollow, stepHeightFollow} = await import('/app.js');
  const bob = createHeightFollow(1);
  for (let i=0; i<600; i++) stepHeightFollow(bob, 1 + .12*Math.sin(i*.21), 1/60);
  const steps = [30,60,144].map(fps => {
    const f=createHeightFollow(1), ys=[];
    for(let i=0;i<fps;i++) ys.push(stepHeightFollow(f,3,1/fps));
    return {fps, final:f.value, first:ys[0],
      monotonic:ys.every((y,i)=>i===0||y>=ys[i-1]), max:Math.max(...ys)};
  });
  const probe=window.somaWorldProbe;
  const root=[...somaLab.stats.root];
  probe.resetCamera();
  const orbitOffset=somaLab.orbitCamera.position.clone().sub(somaLab.cameraControls.target);
  const before=somaLab.cameraControls.target.y;
  const terrain=[];
  for (const [x,z] of [[0,0],[-20,-100],[20,60]]) {
    terrain.push(probe.ground(x,z));
    probe.followRoot([x,root[1],z]);
    probe.updateCameraHeight(1/30);
  }
  const terrainOnlyDelta=somaLab.cameraControls.target.y-before;
  probe.followRoot([root[0],root[1]+2,root[2]]);
  probe.updateCameraHeight(1/60);
  const rootStepDelta=somaLab.cameraControls.target.y-before;
  const offsetError=somaLab.orbitCamera.position.clone().sub(somaLab.cameraControls.target).distanceTo(orbitOffset);
  probe.resetCamera();
  return {bob_delta:bob.value-1, steps, terrain_samples:terrain,
    terrain_only_camera_delta:terrainOnlyDelta, first_2m_root_step_delta:rootStepDelta,
    orbit_offset_error:offsetError, actual_root_unchanged:root.every((v,i)=>v===somaLab.stats.root[i])};
}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:28930/?mode=terrain')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report, errors = {'motion_quality_approved': False}, []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True,
                                   args=['--enable-webgl', '--ignore-gpu-blocklist'])
        page = browser.new_page(viewport={'width':1440, 'height':900})
        page.on('pageerror', lambda error: errors.append(str(error)))

        def instrument(route):
            response = route.fetch()
            # Test-only access exercises the actual follow path; no model/frame changes.
            body = response.text() + '\nwindow.somaWorldProbe={config:()=>state.config,followRoot,updateCameraHeight,resetCamera,ground:terrainGroundAt};'
            route.fulfill(response=response, body=body)

        page.route('**/app.js', instrument)
        page.goto(args.url)
        page.wait_for_function('window.somaLab?.ready && somaLab.stats.frame >= 8', timeout=90000)
        page.locator('#pause-button').click()
        world = page.evaluate('somaWorldProbe.config().world')
        assert len(world['spawns']) == 37
        report['camera'] = page.evaluate(CAMERA_CHECK)
        camera = report['camera']
        assert camera['bob_delta'] == 0
        assert max(camera['terrain_samples'])-min(camera['terrain_samples']) > 1
        assert abs(camera['terrain_only_camera_delta']) < 1e-9
        assert 0 < camera['first_2m_root_step_delta'] < .03
        assert camera['orbit_offset_error'] < 1e-8 and camera['actual_root_unchanged']
        finals = [s['final'] for s in camera['steps']]
        assert max(finals)-min(finals) < 1e-10
        assert all(s['monotonic'] and s['max'] < 2.82 for s in camera['steps'])
        report['entries'] = []

        def select(spawn_id):
            spawn = next(s for s in world['spawns'] if s['id'] == spawn_id)
            page.locator('#world-select').select_option(spawn_id)
            page.wait_for_function('xz => somaLab.ready && somaLab.stats.frame===0 && Math.hypot(somaLab.stats.root[0]-xz[0],somaLab.stats.root[2]-xz[1]) < .002',
                                   arg=spawn['spawn_xz'], timeout=20000)
            return spawn

        initial_y = None
        for spawn in world['spawns']:
            select(spawn['id'])
            root = page.evaluate('somaLab.stats.root')
            if initial_y is None:
                initial_y = root[1]
            assert abs(root[1]-initial_y) < 1e-6
            report['entries'].append({'id':spawn['id'], 'root':root})
        report['live_cases'] = []
        for entry in ['zone-03','zone-09','zone-16','zone-20','zone-28','zone-36']:
            select(entry)
            page.locator('#pause-button').click()
            page.keyboard.down('w')
            page.wait_for_function('somaLab.stats.frame >= 60', timeout=20000)
            page.keyboard.up('w')
            page.locator('#pause-button').click()
            result = page.evaluate(CANVAS_CHECK)
            assert result['colors']>100 and not result['overflow']
            assert all(abs(v)<1 for v in result['mesh_center_ndc'][:2])
            result.update(entry=entry, frame=page.evaluate('somaLab.stats.frame'),
                          camera=page.evaluate('somaLab.stats.camera'),
                          intervals_ms=page.evaluate('somaLab.stats.scheduling.frame_intervals_ms.slice(-50)'))
            report['live_cases'].append(result)
            page.screenshot(path=str(args.output/f'{entry}.png'))
        select('zone-03')
        page.locator('#world-map-toggle').click()
        map_rect=page.locator('#world-map').bounding_box()
        page.mouse.click(map_rect['x']+map_rect['width']/2, map_rect['y']+map_rect['height']/2)
        page.wait_for_function('document.querySelector("#world-select").value==="hub" && Math.hypot(somaLab.stats.root[0],somaLab.stats.root[2])<.002')
        report['map_click_hub'] = True
        select('zone-03')
        for label, width, height in [('desktop',1440,900),('mobile',390,844)]:
            page.set_viewport_size({'width':width,'height':height})
            page.locator('#camera-button').click()
            page.wait_for_timeout(300)
            result=page.evaluate(CANVAS_CHECK)
            assert not result['overflow'] and result['colors']>100
            assert all(abs(v)<1 for v in result['mesh_center_ndc'][:2])
            toolbar=page.locator('.toolbar').bounding_box()
            panel=page.locator('#world-map-panel').bounding_box()
            assert panel['y'] >= toolbar['y']+toolbar['height']
            if label == 'mobile':
                assert page.locator('#diagnostics-panel').is_hidden()
                page.locator('#diagnostics-toggle').click()
                assert page.locator('#world-map-panel').is_hidden()
                page.locator('#world-map-toggle').click()
                assert page.locator('#diagnostics-panel').is_hidden()
            page.screenshot(path=str(args.output/f'{label}-map.png'))
            report[label]=result
        assert not errors, errors
        report.update(status='passed',page_errors=errors,world_size_m=world['size_m'])
        (args.output/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        browser.close()
    print(json.dumps({'status':'passed','entries':len(report['entries']),
                      'live_cases':len(report['live_cases']),'camera':report['camera']}))


if __name__ == '__main__':
    main()
