"""Validate SOMA browser skinning, actual motion pixels and responsive comparison views."""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:28920/web/decoder_pilot/')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(channel='msedge',headless=True,args=['--enable-webgl','--ignore-gpu-blocklist'])
        for label,size in [('desktop',{'width':1500,'height':960}),('mobile',{'width':390,'height':844})]:
            page=browser.new_page(viewport=size);errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(args.url)
            page.wait_for_function('window.pilotView && pilotView.panels.every(p=>p.rig)',timeout=120000)
            page.evaluate('pilotView.pause();pilotView.setFrame(0)');page.wait_for_timeout(100)
            qa=page.evaluate('''async()=>{
                const T=await import('/web/soma_lab/vendor/three.module.js');
                return pilotView.panels.map(p=>({reference:p.entry.qa.vertices,
                  actual:p.entry.qa.indices.map(i=>p.mesh.getVertexPosition(i,new T.Vector3()).toArray())}));
            }''')
            skin_error=max(float(np.linalg.norm(np.array(row['actual'])-np.array(row['reference']),axis=-1).max()) for row in qa)
            if not np.isfinite(skin_error) or skin_error>2e-5:
                raise ValueError(f'Browser skin differs from official LBS: {skin_error}')
            first=np.asarray(Image.open(io.BytesIO(page.locator('canvas').first.screenshot())))[:,:,:3]
            page.screenshot(path=str(args.output/(label+'-start.png')),full_page=True)
            page.evaluate('pilotView.setFrame(899)');page.wait_for_timeout(150)
            last=np.asarray(Image.open(io.BytesIO(page.locator('canvas').first.screenshot())))[:,:,:3]
            changed=int((np.abs(first.astype(int)-last.astype(int)).max(-1)>10).sum())
            colors=len(np.unique(last.reshape(-1,3)[::5],axis=0))
            overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth+1')
            assert changed>100 and colors>100 and not overflow and not errors,(changed,colors,overflow,errors)
            page.screenshot(path=str(args.output/(label+'-end.png')),full_page=True)
            page.evaluate("pilotView.selectClip('terrain_short')")
            page.evaluate('pilotView.pause();pilotView.setFrame(24)');page.wait_for_timeout(150)
            page.screenshot(path=str(args.output/(label+'-terrain.png')),full_page=True)
            report.append({'viewport':label,'official_skin_max_error_m':skin_error,'changed_canvas_pixels':changed,
                           'canvas_colors':colors,'overflow':overflow,'page_errors':errors})
            page.close()
        browser.close()
    (args.output/'checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
