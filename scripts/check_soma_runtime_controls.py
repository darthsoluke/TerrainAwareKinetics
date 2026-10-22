"""Exercise sliders through real websocket inference and verify applied geometry."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
from scripts.check_soma_lab import CANVAS_CHECK


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:28931/?mode=terrain')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report,errors={'cases':[]},[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,
            args=['--enable-webgl','--ignore-gpu-blocklist'])
        page=browser.new_page(viewport={'width':1440,'height':900})
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_function('window.somaLab?.ready && somaLab.stats.generation?.runtime_controls?.num_tokens===12',timeout=90000)
        page.wait_for_function('somaLab.stats.diagnostics.terrain_sampling==="predicted_decoded_path_corridor"')
        page.locator('#runtime-toggle').click()
        for name,key,value,press in [('horizon','num_tokens',16,'End'),
                                    ('width','lateral_halfwidth_m',6.,'End'),
                                    ('gain','lookahead_gain',4.,'End'),
                                    ('base','lookahead_base_m',10.,'End'),
                                    ('path','path_refinements',2,'End'),
                                    ('steps','inference_steps',2,'Home ArrowRight'),
                                    ('refresh','refresh_frames',16,'Home ArrowRight')]:
            control=page.locator('#runtime-'+name)
            control.focus()
            for part in press.split():control.press(part)
            try:
                page.wait_for_function('args => somaLab.stats.generation?.runtime_controls?.[args[0]]===args[1]',arg=[key,value],timeout=20000)
            except Exception:
                print(json.dumps({'failed_control':name,'errors':errors,'state':page.evaluate('({frame:somaLab.stats.frame,generation:somaLab.stats.generation,toast:document.querySelector("#error-toast").textContent,range:document.querySelector("#runtime-'+name+'").value})')}),flush=True)
                page.screenshot(path=str(args.output/'failure.png'))
                raise
            report['cases'].append(page.evaluate('({frame:somaLab.stats.frame,root:somaLab.stats.root,generation:somaLab.stats.generation})'))
        page.locator('#runtime-auto').uncheck()
        page.locator('#runtime-ahead').focus();page.locator('#runtime-ahead').press('End')
        page.wait_for_function('somaLab.stats.generation?.runtime_controls?.lookahead_m===20',timeout=20000)
        report['manual_range']=page.evaluate('({camera:somaLab.stats.camera,diagnostics:somaLab.stats.diagnostics,generation:somaLab.stats.generation})')
        assert report['manual_range']['diagnostics']['query_proposal_world_xz_m']
        assert len(report['manual_range']['diagnostics']['terrain_corridor_world_xyz_m'])==81
        assert len(report['manual_range']['diagnostics']['survey_world_xyz_m'])==81
        assert report['manual_range']['diagnostics']['motion_observation_extent_m']<=3.400001
        assert report['manual_range']['diagnostics']['motion_observation_halfwidth_m']<=.600001
        page.locator('#runtime-speed').focus();page.locator('#runtime-speed').press('End')
        page.locator('#scene-canvas').click()
        page.keyboard.down('w')
        page.wait_for_function('Math.abs(somaLab.stats.control.speed-12)<.001')
        page.wait_for_function('somaLab.stats.generation?.runtime_controls?.lookahead_m===20')
        report['speed_12']=page.evaluate('somaLab.stats.control')
        page.keyboard.up('w')
        page.keyboard.press('Escape')
        page.locator('#speed-input').fill('1.6');page.locator('#speed-input').press('Tab')
        page.locator('#runtime-preset').select_option('original')
        page.wait_for_function('somaLab.stats.generation?.runtime_controls?.num_tokens===8 && somaLab.stats.generation.runtime_controls.refresh_frames===24',timeout=20000)
        page.wait_for_function('somaLab.stats.diagnostics.terrain_sampling==="input_direction"')
        page.locator('#runtime-reset').click()
        page.wait_for_function('somaLab.stats.generation?.runtime_controls?.num_tokens===12 && somaLab.stats.generation.runtime_controls.refresh_frames===8',timeout=20000)
        for name,width,height in [('desktop',1440,900),('mobile',390,844)]:
            page.set_viewport_size({'width':width,'height':height})
            page.wait_for_timeout(250)
            result=page.evaluate(CANVAS_CHECK)
            assert result['colors']>100 and not result['overflow']
            assert page.locator('#runtime-panel').is_visible()
            assert page.locator('#diagnostics-panel').is_hidden()
            assert page.locator('#world-map-panel').is_hidden()
            page.screenshot(path=str(args.output/f'{name}.png'))
            report[name]=result
        assert not errors,errors
        report.update(status='passed',page_errors=errors,motion_quality_approved=False)
        (args.output/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        browser.close()
    print(json.dumps({'status':'passed','slider_cases':len(report['cases'])+1}))


if __name__=='__main__':main()
