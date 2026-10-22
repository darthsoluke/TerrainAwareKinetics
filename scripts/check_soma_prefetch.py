"""Real browser prefetch sequencing and playback timing; not motion approval."""

import argparse
import json
import time
import threading
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright
from scripts.check_soma_lab import CANVAS_CHECK


def timing(intervals):
    values = np.asarray(intervals, dtype=float)
    return {'frames': len(values), 'fps': float(1000 / values.mean()),
            'p95_ms': float(np.percentile(values, 95)), 'max_ms': float(values.max()),
            'over_50_ms': int((values > 50).sum()), 'under_20_ms': int((values < 20).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:28880/?mode=terrain')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report, errors, plans = {'motion_quality_approved': False}, [], []
    sent_at = {}
    gpu_samples = []
    stop_gpu_sampling = threading.Event()
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        def sample_gpu():
            while not stop_gpu_sampling.is_set():
                gpu_samples.append({'time': time.time(),
                    'sm_mhz': pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                    'memory_mhz': pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                    'pstate': pynvml.nvmlDeviceGetPerformanceState(handle)})
                stop_gpu_sampling.wait(.1)
        threading.Thread(target=sample_gpu, daemon=True).start()
    except (ImportError, RuntimeError):
        pass
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True,
            args=['--enable-webgl', '--ignore-gpu-blocklist'])
        page = browser.new_page(viewport={'width': 1440, 'height': 900})
        page.on('pageerror', lambda error: errors.append(str(error)))

        def receive(payload):
            if isinstance(payload, str):
                meta = json.loads(payload)
                if meta.get('type') == 'error':
                    errors.append(meta.get('message', 'Backend error'))
                if meta.get('type') == 'plan':
                    plans.append({'request_id': meta['request_id'], 'prefetched': meta.get('prefetched', False),
                        'neural': meta['generation']['neural_generation'],
                        'inference_ms': meta['inference_ms'], 'skin_ms': meta['skin_ms'],
                        'server_compute_ms': meta.get('server_compute_ms'),
                        'server_queue_ms': meta.get('server_queue_ms'),
                        'metadata_roundtrip_ms': (time.perf_counter() - sent_at.get(meta['request_id'], time.perf_counter())) * 1000})

        def sent(payload):
            if isinstance(payload, str):
                command = json.loads(payload)
                if command.get('type') in ('plan', 'prefetch'):
                    sent_at[command['request_id']] = time.perf_counter()

        def socket_opened(ws):
            ws.on('framereceived', receive)
            ws.on('framesent', sent)

        page.on('websocket', socket_opened)
        page.add_init_script("""(() => {
          const original = WebSocket.prototype.send;
          const listen = WebSocket.prototype.addEventListener;
          WebSocket.prototype.addEventListener = function(type, callback, ...options) {
            if (type !== 'message') return listen.call(this, type, callback, ...options);
            return listen.call(this, type, event => {
              if (typeof event.data === 'string' && JSON.parse(event.data).type === 'prefetch_deferred'
                && window.holdNextSomaDeferral && !window.somaDeferredReplyHeld) {
                window.somaDeferredReplyHeld = true;
                window.releaseSomaDeferral = () => callback.call(this, event);
                return;
              }
              callback.call(this, event);
            }, ...options);
          };
          window.restoreSomaSend = () => { WebSocket.prototype.send = original; };
          WebSocket.prototype.send = function(data) {
            if (typeof data === 'string' && JSON.parse(data).type === 'played' && !window.deferredSomaAck) {
              window.deferredSomaAck = true;
              setTimeout(() => original.call(this, data), 300);
              return;
            }
            return original.call(this, data);
          };
        })();""")
        page.goto(args.url)
        page.wait_for_function('window.deferredSomaAck && somaLab.stats.frame === 8', timeout=60000)
        page.wait_for_timeout(60)
        assert page.evaluate('somaLab.stats.frame') == 8
        assert page.evaluate('somaLab.stats.scheduling.prefetched_blocks') > 0, {
            'stats': page.evaluate('somaLab.stats'), 'errors': errors, 'plans': plans}
        page.evaluate('window.restoreSomaSend()')
        page.wait_for_function('somaLab.stats.frame >= 40', timeout=60000)
        clock = page.evaluate('[somaLab.nextPlaybackDeadline(100, 105), somaLab.nextPlaybackDeadline(100, 125)]')
        assert abs(clock[0] - (100 + 1000 / 30)) < 1e-6
        assert abs(clock[1] - (125 + 1000 / 30)) < 1e-6
        report['clock_phase_and_no_burst_checks'] = True
        report['full_ack_before_next_block'] = True
        page.evaluate('somaLab.stats.scheduling.frame_intervals_ms = []')
        page.wait_for_function('somaLab.stats.scheduling.frame_intervals_ms.length >= 180', timeout=60000)
        report['idle'] = timing(page.evaluate('somaLab.stats.scheduling.frame_intervals_ms'))
        page.evaluate('somaLab.stats.scheduling.frame_intervals_ms = []')
        page.keyboard.down('w')
        page.wait_for_function('somaLab.stats.scheduling.frame_intervals_ms.length >= 90', timeout=60000)
        page.keyboard.up('w')
        report['moving'] = timing(page.evaluate('somaLab.stats.scheduling.frame_intervals_ms'))
        page.locator('#pause-button').click()
        report['scheduling'] = page.evaluate('somaLab.stats.scheduling')
        report['scheduling'].pop('frame_intervals_ms')
        report['root'] = page.evaluate('somaLab.stats.root')
        report['canvas'] = page.evaluate(CANVAS_CHECK)
        report['soma_in_frame'] = all(abs(value) < 1 for value in report['canvas']['mesh_center_ndc'][:2])
        report['framing_status'] = 'pass' if report['soma_in_frame'] else 'fail_model_rollout_outside_view'
        assert not report['canvas']['overflow'] and report['canvas']['colors'] > 100
        page.screenshot(path=str(args.output / 'prefetch-desktop.png'))
        assert not errors, errors
        assert report['scheduling']['prefetch_requests'] >= 20
        assert report['scheduling']['deferred_prefetches'] > 0
        assert all(not row['neural'] for row in plans if row['prefetched'])
        report['prefetch_contains_only_existing_generation'] = True
        # Reset while a deferral reply is pending must release the old request.
        page.evaluate('window.holdNextSomaDeferral = true')
        page.locator('#pause-button').click()
        page.wait_for_function('window.somaDeferredReplyHeld === true', timeout=30000)
        page.locator('#reset-button').click()
        page.evaluate('window.releaseSomaDeferral()')
        page.wait_for_function('somaLab.ready && somaLab.stats.frame >= 8', timeout=30000)
        page.locator('#pause-button').click()
        report['reset_with_deferred_prefetch'] = True
        assert not errors, errors
        report['fps_approved'] = all(29 <= row['fps'] <= 31 and row['p95_ms'] <= 45
            and row['over_50_ms'] == 0 and row['under_20_ms'] == 0
            for row in [report['idle'], report['moving']])
        report.update(status='processed', execution_status='processed', pipeline_checks_status='pass',
            page_errors=errors, plans=plans,
            timing_scope='One headless Edge desktop session, raw 30 Hz frames with WebSocket inspection; not terrain skill')
        stop_gpu_sampling.set()
        report['gpu_clock_samples'] = gpu_samples
        (args.output / 'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        browser.close()
    print(json.dumps({k: v for k, v in report.items() if k not in ('plans', 'gpu_clock_samples', 'scheduling')}, indent=2))


if __name__ == '__main__':
    main()
