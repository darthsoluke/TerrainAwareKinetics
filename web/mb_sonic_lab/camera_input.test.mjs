import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createContext, SourceTextModule } from 'node:vm';
import { ActualRootFollower, IntentStream, behindRootCamera, worldIntent } from './protocol.mjs';

async function cameraFixture(heading) {
  const context = createContext({ console });
  const three = new SourceTextModule(await readFile(new URL('../full_corpus_lab/vendor/three.module.js', import.meta.url), 'utf8'), { context });
  await three.link(() => assert.fail('Unexpected Three dependency'));
  await three.evaluate();
  const orbit = new SourceTextModule(await readFile(new URL('../full_corpus_lab/vendor/OrbitControls.js', import.meta.url), 'utf8'), { context });
  await orbit.link(specifier => { assert.equal(specifier, 'three'); return three; });
  await orbit.evaluate();
  const listeners = new Map();
  const document = { addEventListener() {}, removeEventListener() {} };
  const element = { style: {}, clientWidth: 1242, clientHeight: 952,
    getRootNode: () => document, setPointerCapture() {}, releasePointerCapture() {},
    addEventListener(type, callback) { listeners.set(type, callback); },
    removeEventListener(type) { listeners.delete(type); },
    dispatch(type, point) {
      listeners.get(type)?.({ pointerId: 1, pointerType: 'mouse', button: 0,
        clientX: point[0], clientY: point[1], pageX: point[0], pageY: point[1],
        ctrlKey: false, metaKey: false, shiftKey: false, preventDefault() {} });
    } };
  const THREE = three.namespace;
  THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
  const camera = new THREE.PerspectiveCamera(45, 1, .02, 1000);
  const controls = new orbit.namespace.OrbitControls(camera, element);
  controls.enableDamping = false;
  controls.minPolarAngle = .15; controls.maxPolarAngle = Math.PI * .49;
  const root = [0, 0, .7907370626926422];
  const view = behindRootCamera(root, heading, Math.hypot(2.5, -3), 1.4);
  camera.position.fromArray(view.position); controls.target.fromArray(view.target); controls.update();
  const follower = new ActualRootFollower(); follower.delta(root, true);
  const stream = new IntentStream('isolated-camera-test');
  stream.bindScene({ intent_enabled: true, episode_id: 1,
    scene: { manifest_id: 'fixture', manifest_sha256: 'unit-fixture' } });
  const input = at => {
    const forward = camera.getWorldDirection(new THREE.Vector3()); forward.z = 0; forward.normalize();
    const intent = worldIntent([0, 1], [forward.x, forward.y], 1.2, 0);
    return { forward, command: stream.next(intent.velocity_xy_mps, intent.yaw_rate_radps, at) };
  };
  const follow = root => {
    const delta = new THREE.Vector3().fromArray(follower.delta(root, true));
    camera.position.add(delta); controls.target.add(delta); controls.update();
  };
  return { camera, controls, element, input, follow, stream };
}

test('real OrbitControls plus rigid actual-Root Follow does not churn a held W at100Hz', async t => {
  const app = await readFile(new URL('./app.js', import.meta.url), 'utf8');
  assert.match(app, /camera\.position\.add\(cameraDelta\); controls\.target\.add\(cameraDelta\); controls\.update\(\);/);
  let maximumDrift = 0;
  for (const heading of [-Math.PI / 2, 0, Math.atan2(-.0413, 1.1993)]) {
    const fixture = await cameraFixture(heading);
    const initial = fixture.input(0).forward;
    let nextDraw = 0;
    for (let sample = 1; sample <= 1000; sample++) {
      const at = sample * 10;
      if (at >= nextDraw) {
        fixture.follow([Math.sin(sample / 37) * .2, sample * .002, .79 + Math.sin(sample / 19) * .04]);
        nextDraw += 1000 / 30;
      }
      fixture.controls.update();
      const current = fixture.input(at);
      maximumDrift = Math.max(maximumDrift, current.forward.distanceTo(initial));
      assert.equal(current.command, null);
    }
    assert.equal(fixture.stream.id, 1);
    fixture.controls.dispose();
  }
  assert.ok(maximumDrift < 1e-10);
  t.diagnostic(`1000 samples per heading; maximum planar direction drift=${maximumDrift}`);
});

test('real Orbit pointer rotation changes held-W intent immediately without filtering deliberate changes', async () => {
  const fixture = await cameraFixture(Math.atan2(-.0413, 1.1993));
  const first = fixture.input(0).command;
  fixture.element.dispatch('pointerdown', [100, 100]);
  for (let sample = 1; sample <= 7; sample++) {
    fixture.element.dispatch('pointermove', [100 + sample * .5, 100]);
    const next = fixture.input(sample * 10).command;
    assert.ok(next);
    assert.equal(next.command_seq, sample + 1);
    assert.equal(next.client_input_ms, sample * 10);
    assert.notDeepEqual(next.velocity_xy_mps, first.velocity_xy_mps);
  }
  fixture.element.dispatch('pointerup', [103.5, 100]);
  assert.equal(fixture.input(80).command, null);
  assert.deepEqual(fixture.stream.next([0, 0], 0, 81).velocity_xy_mps, [0, 0]);
  assert.equal(fixture.stream.next([0, 0], .5, 82).yaw_rate_radps, .5);
  fixture.controls.dispose();
});
