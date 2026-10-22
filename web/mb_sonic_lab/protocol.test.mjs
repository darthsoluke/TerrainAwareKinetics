import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createContext, runInNewContext, SourceTextModule, SyntheticModule } from 'node:vm';
import * as protocol from './protocol.mjs';
import { SCHEMA, IntentStream, DisplayLedger, DISPLAY_MS, worldIntent, BrowserDiagnostics, primaryReferencePresentation,
  ActualRootFollower, LatestIntentMailbox, GamepadLatch, runtimeStatus, behindRootCamera } from './protocol.mjs';

const scope = { intent_enabled: true, episode_id: 1,
  scene: { manifest_id: 'fixture', manifest_sha256: 'fixture-hash' } };
function inputStream(session, enabled = true) {
  const stream = new IntentStream(session, enabled); stream.bindScene(scope); return stream;
}

test('error banner displays the current message while full failure history remains in receipts', async () => {
  const source = await readFile(new URL('./app.js', import.meta.url), 'utf8');
  const handler = source.slice(source.indexOf('function error('), source.indexOf('const number ='));
  assert.match(handler, /failures\.push\(\{ message,/);
  assert.match(handler, /el\('error'\)\.textContent = message;/);
  assert.doesNotMatch(handler, /failures\.map|failures\.(?:splice|pop|shift)|failures\s*=/);
  assert.match(source, /gamepad: gamepadState, failures, server: serverReceipt/);
});

test('animation only redraws unchanged physics for camera changes, not episode activity', async () => {
  const source = await readFile(new URL('./app.js', import.meta.url), 'utf8');
  const animation = source.slice(source.indexOf('function animate('), source.indexOf('function boot('));
  assert.match(animation, /ledger\?\.render\(draw,/);
  assert.match(animation, /else if \(cameraDirty\) \{/);
  assert.doesNotMatch(animation, /cameraDirty\s*\|\|\s*episodeActive|active_compositor_paint/);
  assert.match(animation, /reason: 'camera_change'/);
  assert.match(source, /controls\.addEventListener\('change', \(\) => \{ cameraDirty = true; \}\)/);
});

test('actual app allows active manual Reset only for MB, with transition and connection guards', async () => {
  const context = createContext({ URL, self: {} });
  const app = new SourceTextModule(await readFile(new URL('./app.js', import.meta.url), 'utf8'), {
    context, initializeImportMeta: meta => { meta.url = 'http://127.0.0.1:28948/app.js?mesh-worker'; },
    importModuleDynamically: () => assert.fail('This protocol fixture must not start the browser UI'),
  });
  await app.link(async specifier => {
    if (specifier === './protocol.mjs') {
      return new SyntheticModule(Object.keys(protocol), function () {
        for (const [name, value] of Object.entries(protocol)) this.setExport(name, value);
      }, { context });
    }
    assert.equal(specifier, '/vendor/three.module.js');
    return new SyntheticModule([], () => {}, { context });
  });
  await app.evaluate();
  const allowed = app.namespace.canResetEpisode;
  assert.equal(allowed(true, true, false, true), true);
  assert.equal(allowed(false, true, false, true), false);
  assert.equal(allowed(false, false, false, true), true);
  assert.equal(allowed(true, true, true, true), false);
  assert.equal(allowed(true, true, false, false), false);
  const editing = app.namespace.isIntentEditor;
  assert.equal(editing({ id: 'yaw', closest: () => ({ tagName: 'INPUT' }) }), false);
  assert.equal(editing({ id: 'speed', closest: () => ({ tagName: 'INPUT' }) }), true);
  assert.equal(editing({ id: 'joystick', closest: () => null }), false);
});

test('two simultaneous movement/facing controls release independently and right means clockwise', async () => {
  const source = await readFile(new URL('./app.js', import.meta.url), 'utf8');
  const elements = Object.fromEntries(['speed', 'yaw', 'joystick', 'stage'].map(id => [id, {
    value: '0', handlers: {}, pointers: new Set(),
    addEventListener(type, handler) { this.handlers[type] = handler; },
    setPointerCapture(id) { this.pointers.add(id); },
    hasPointerCapture(id) { return this.pointers.has(id); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 112, height: 112 }),
    focus() { assert.fail('A pointer must not move focus away from the other held control'); },
  }]));
  const inputs = [];
  const context = { config: { intent_enabled: true }, episodeBegan: true, episodeActive: true,
    sceneChanging: false, manualYaw: 0, pointerVelocity: [0, 0], el: id => elements[id] };
  context.input = () => inputs.push({ velocity: [...context.pointerVelocity], yaw: context.manualYaw });
  runInNewContext(source.slice(source.indexOf("el('speed').addEventListener"),
    source.indexOf("el('camera').addEventListener")), context);
  const stick = elements.joystick.handlers;
  let prevented = false;
  stick.pointerdown({ pointerId: 1, clientX: 56, clientY: 16,
    preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  elements.yaw.value = '1'; elements.yaw.handlers.input();
  assert.deepEqual(inputs.at(-1), { velocity: [0, 1], yaw: -1 });
  elements.yaw.handlers.pointerup();
  assert.deepEqual(inputs.at(-1), { velocity: [0, 1], yaw: 0 });
  elements.yaw.value = '-.5'; elements.yaw.handlers.input();
  stick.pointermove({ pointerId: 1, clientX: 96, clientY: 56 });
  assert.deepEqual(inputs.at(-1), { velocity: [1, -0], yaw: .5 });
  stick.lostpointercapture();
  assert.deepEqual(inputs.at(-1), { velocity: [0, 0], yaw: .5 });
  elements.yaw.handlers.pointercancel();
  assert.deepEqual(inputs.at(-1), { velocity: [0, 0], yaw: 0 });
});

test('standard Xbox input finds nonzero slots, has a continuous radial deadzone and independent right yaw', () => {
  const read = axes => protocol.standardGamepadState([null,
    { id: 'Xbox XInput', index: 1, connected: true, mapping: 'standard', axes }]);
  assert.equal(protocol.standardGamepadState([null, { connected: false, mapping: 'standard' }]), null);
  assert.equal(protocol.standardGamepadState([{ connected: true, mapping: '', axes: [0, 0, 0, 0] }]), null);
  assert.deepEqual(read([.08, -.08, .1, .9]).velocity, [0, 0]);
  assert.equal(read([0, 0, .1, .9]).yaw, 0);
  assert.ok(read([.120001, 0, 0, 0]).velocity[0] < .00001);
  assert.deepEqual(read([1, 0, 0, 0]).velocity, [1, 0]);
  assert.deepEqual(read([0, -1, 1, 0]).velocity, [0, 1]);
  assert.equal(read([0, -1, 1, 0]).yaw, 1);
  assert.equal(read([0, 0, -1, 0]).yaw, -1);
  const diagonal = read([1, -1, .7, 0]);
  assert.ok(Math.abs(Math.hypot(...diagonal.velocity) - 1) < 1e-12);
  assert.deepEqual(diagonal.raw_axes, [1, -1, .7, 0]);
});

test('small gamepad sensor noise does not interrupt a held gait, while release is immediate', () => {
  const latch = new GamepadLatch();
  latch.sample([0, 0], 0);
  assert.deepEqual(latch.sample([.5, -.3], .4), { velocity: [.5, -.3], yaw: .4 });
  assert.deepEqual(latch.sample([.505, -.296], .406), { velocity: [.5, -.3], yaw: .4 });
  assert.deepEqual(latch.sample([.53, -.3], .42), { velocity: [.53, -.3], yaw: .42 });
  assert.deepEqual(latch.sample([0, 0], 0), { velocity: [0, 0], yaw: 0 });
});

test('scripted Xbox axes use the real mapper/latch and finish neutral without any physics call', () => {
  const replay = new protocol.SimulatedXbox(), latch = new GamepadLatch();
  assert.deepEqual(protocol.standardGamepadState(replay.sample(0)).velocity,[0,0]);
  replay.start(1000);
  const phases = new Set();
  let simultaneous = 0, reverse = 0, varying = 0, previous = null;
  for (let elapsed = 0; elapsed <= 84000; elapsed += 10) {
    const state = protocol.standardGamepadState(replay.sample(1000+elapsed));
    const input = latch.sample(state.velocity,state.yaw);
    phases.add(replay.phase);
    assert.ok(Math.hypot(...input.velocity)<=1+1e-12);
    assert.ok(Math.abs(input.yaw)<=1);
    if (Math.hypot(...input.velocity)>0 && input.yaw!==0) simultaneous++;
    if (input.velocity[1]<0) reverse++;
    if (previous!==null && previous!==JSON.stringify(input)) varying++;
    previous=JSON.stringify(input);
    if (elapsed>=75000 && elapsed<80000) assert.deepEqual(input,{velocity:[0,0],yaw:0});
  }
  assert.ok(simultaneous>500 && reverse>100 && varying>200);
  assert.equal(phases.size,18);
  assert.equal(replay.receipt().stopped.reason,'trace_complete');
  assert.equal(replay.receipt().phase,'Complete');
  assert.ok(replay.receipt().raw_samples.length<=8301);
  assert.deepEqual(protocol.standardGamepadState(replay.sample(100000)).velocity,[0,0]);
});

test('scripted Xbox cannot resume after focus loss without an explicit new episode start', () => {
  const replay = new protocol.SimulatedXbox();
  replay.start(0); replay.sample(6000);
  const count = replay.samples.length;
  replay.stop(6010,'operator_or_focus_stop');
  assert.deepEqual(protocol.standardGamepadState(replay.sample(12000)).velocity,[0,0]);
  assert.equal(replay.samples.length,count);
  assert.equal(replay.phase,'Stopped');
  replay.start(13000);
  assert.equal(replay.samples.length,0);
  assert.deepEqual(protocol.standardGamepadState(replay.sample(13000)).velocity,[0,0]);
});

test('episode starts only after its initial input is accepted by the runtime', async () => {
  const source = await readFile(new URL('./app.js',import.meta.url),'utf8');
  const sent=[];
  const state={ inputMailbox:{flush:()=>({command_id:7})},socket:{readyState:1,bufferedAmount:0},
    WebSocket:{OPEN:1},startCommandPending:true,startIntentId:null,
    config:{start_action:'begin_episode'},intents:{session:'controller',rows:new Map([[7,{}]])},
    send:value=>sent.push(value) };
  runInNewContext(source.slice(source.indexOf('function flushIntent('),source.indexOf("addEventListener('keydown'"))+';flushIntent();',state);
  assert.equal(state.startIntentId,7);
  assert.equal(state.startCommandPending,true);
  assert.equal(sent.length,0);
  const ack=source.slice(source.indexOf("else if (packet.type === 'intent_received')")+5,
    source.indexOf("else if (packet.type === 'failure'"));
  state.packet={type:'intent_received',command_id:8,server_received_ns:1};
  runInNewContext(ack,state);
  assert.equal(sent.length,0);
  state.packet={type:'intent_received',command_id:7,server_received_ns:2};
  runInNewContext(ack,state);
  assert.deepEqual(JSON.parse(JSON.stringify(sent)),[{type:'begin_episode',session_id:'controller'}]);
  assert.equal(state.startCommandPending,false);
  runInNewContext(ack,state);
  assert.equal(sent.length,1);
});

function fixture() {
  const intents = inputStream('test');
  const ledger = new DisplayLedger('test', intents);
  const packet = (id, command = null, onset = null) => ({ type: 'frame', schema: SCHEMA,
    session_id: 'test', packet_id: id, scene_revision: 0,
    state: { frame_id: id * 7, sim_time_s: id / 30, command,
      episode_id: 1, scene_manifest_id: 'fixture', scene_manifest_sha256: 'fixture-hash',
      actual: { body_onset_command_id: onset, body_onset_sim_time_s: onset === null ? null : .04,
        body_onset_session_id: onset === null ? null : 'test', body_onset_server_ns: onset === null ? null : 2e6 },
      reference: null, timing: { chain_ms: null, physics_steps: id * 7 } } });
  return { intents, ledger, packet };
}

test('primary ghost is G1 in both modes and raw SOMA cannot be mislabeled as adapted G1', () => {
  const parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12];
  assert.equal(primaryReferencePresentation({ control_mode: 'official_reference_read_only', reference: { parents } }).label,
    'Official reference');
  const mb = { control_mode: 'frozen_mb_reference', reference: { parents } };
  assert.equal(primaryReferencePresentation(mb).label, 'MB -> G1 reference');
  assert.throws(() => primaryReferencePresentation({ ...mb, reference: { parents: [-1, ...Array(29).fill(0)] } }),
    /adapted G1 14-body/);
  assert.throws(() => primaryReferencePresentation({ ...mb, reference: { parents: parents.map((v, i) => i === 2 ? 3 : v) } }),
    /adapted G1 14-body/);
});

test('Follow translates camera and target by actual Root delta while preserving orbit/zoom and state', () => {
  const follower = new ActualRootFollower();
  const first = [0, 0, .79], next = [0, -2.7, .82];
  const camera = [2.5, -3, 2.19], target = [0, 0, .79];
  const offset = camera.map((value, axis) => value - target[axis]);
  assert.deepEqual(follower.delta(first, true), [0, 0, 0]);
  const delta = follower.delta(next, true);
  delta.forEach((value, axis) => { camera[axis] += value; target[axis] += value; });
  camera.forEach((value, axis) => assert.ok(Math.abs(value - target[axis] - offset[axis]) < 1e-12));
  assert.deepEqual(target, next);
  assert.deepEqual(first, [0, 0, .79]); assert.deepEqual(next, [0, -2.7, .82]);
  assert.deepEqual(follower.delta([0, -3, .82], false), [0, 0, 0]);
  assert.deepEqual(follower.delta([0, -3, .82], true), [0, 0, 0]);
  follower.reset();
  assert.deepEqual(follower.delta([5, 10, 2], true), [0, 0, 0]);
});

test('WASD behind-actual-heading camera makes W forward, while Follow does not chase heading', () => {
  const root = [2, 3, .79];
  for (const heading of [-Math.PI / 2, 0, Math.PI / 2, Math.PI]) {
    const view = behindRootCamera(root, heading, 4, 1.4);
    const forward = view.target.slice(0, 2).map((value, axis) => (value - view.position[axis]) / 4);
    const w = worldIntent([0, 1], forward, 1, 0).velocity_xy_mps;
    assert.ok(w[0] * Math.cos(heading) + w[1] * Math.sin(heading) > .9999);
    assert.deepEqual(view.target, root);
    assert.equal(view.position[2], root[2] + 1.4);
    if (heading === -Math.PI / 2) {
      assert.ok(view.position[1] > root[1]); assert.deepEqual(w, [0, -1]);
    }
    const follower = new ActualRootFollower(); follower.delta(root, true);
    const delta = follower.delta([2, 2.5, .79], true);
    const offset = view.position.map((value, axis) => value - view.target[axis]);
    const movedCamera = view.position.map((value, axis) => value + delta[axis]);
    const movedTarget = view.target.map((value, axis) => value + delta[axis]);
    movedCamera.forEach((value, axis) => assert.ok(Math.abs(value - movedTarget[axis] - offset[axis]) < 1e-12));
  }
  assert.deepEqual(root, [2, 3, .79]);
  assert.throws(() => behindRootCamera(root, null, 4, 1.4), /heading/);
});

test('receiving an intent is not visual association or actual body onset', () => {
  const { intents, ledger, packet } = fixture();
  const command = intents.next([1, 0], 0, 10);
  intents.rows.get(1).server_received_ns = 1e6;
  ledger.offer(packet(1)); ledger.render(() => {}, () => 20);
  assert.equal(ledger.metrics().first_visual_p95_ms, null);
  ledger.offer(packet(2, command)); ledger.render(() => {}, () => 60);
  assert.equal(ledger.metrics().first_visual_p95_ms, 50);
  assert.equal(ledger.metrics().body_onset_display_p95_ms, null);
  ledger.offer(packet(3, command, 1)); ledger.render(() => {}, () => 100);
  assert.equal(ledger.metrics().body_onset_display_p95_ms, 90);
  assert.equal(intents.rows.get(1).body_onset_sim_time_s, .04);
  assert.equal(ledger.metrics().body_onset_server_p95_ms, 1);
});

test('display draw failure neither acknowledges nor associates', () => {
  const { intents, ledger, packet } = fixture();
  const command = intents.next([0, 1], 1, 10);
  ledger.offer(packet(1, command));
  assert.throws(() => ledger.render(() => { throw new Error('render failed'); }, () => 20));
  assert.equal(ledger.rendered, 0);
  assert.equal(intents.rows.get(1).first_visual_ms, null);
  assert.equal(ledger.pending.packet_id, 1);
});

test('late presentation draws latest sample once with no catch-up burst', () => {
  const { ledger, packet } = fixture();
  ledger.offer(packet(1)); ledger.offer(packet(2));
  const drawn = [];
  const ack = ledger.render(state => drawn.push(state.frame_id), () => 1000);
  assert.deepEqual(drawn, [14]); assert.equal(ack.frame_id, 14);
  ledger.offer(packet(3));
  assert.equal(ledger.render(() => assert.fail(), () => 1001), null);
  assert.ok(ledger.render(() => {}, () => 1000 + DISPLAY_MS));
  assert.equal(ledger.dropped, 1);
});

test('a 60Hz browser gives thirty unique 30Hz display samples without simulation calls', () => {
  const { ledger, packet } = fixture();
  let rendered = 0, id = 0;
  for (let tick = 0; tick < 60; tick++) {
    if (tick % 2 === 0) ledger.offer(packet(++id));
    const time = tick * (1000 / 60);
    if (ledger.render(() => {}, () => time)) rendered++;
  }
  assert.equal(rendered, 30);
  assert.ok(Math.abs(ledger.metrics().unique_display_hz - 30) < 1e-9);
});

test('scene revision clears old pending frame and rejects future mismatches', () => {
  const { ledger, packet } = fixture();
  ledger.offer(packet(1)); ledger.changeScene(1);
  assert.equal(ledger.pending, null);
  ledger.offer(packet(2)); assert.equal(ledger.pending, null);
  assert.throws(() => ledger.offer({ ...packet(3), scene_revision: 2 }));
  ledger.offer({ ...packet(3), scene_revision: 1 });
  assert.equal(ledger.pending.packet_id, 3);
});

test('commands deduplicate, stay bounded and foreign-session IDs cannot label frames', () => {
  const { intents, ledger, packet } = fixture();
  const command = intents.next([1, 0], 0, 1);
  assert.equal(intents.next([1, 0], 0, 2), null);
  ledger.offer(packet(1, { ...command, session_id: 'other' }, 1)); ledger.render(() => {}, () => 20);
  assert.equal(ledger.metrics().first_visual_p95_ms, null);
  assert.equal(ledger.metrics().body_onset_display_p95_ms, null);
  for (let i = 0; i < 500; i++) intents.next([i, 0], 0, i);
  assert.equal(intents.rows.size, 256);
  assert.equal(ledger.metrics().chain_p99_ms, null);
});

test('official read-only reference cannot emit an input ID or latency association', () => {
  const intents = inputStream('read-only', false);
  assert.equal(intents.next([1, 0], 0, 10), null);
  assert.equal(intents.id, 0); assert.equal(intents.rows.size, 0);
});

test('translation, strafe, reverse and pure turn retain independent raw yaw intent', () => {
  assert.deepEqual(worldIntent([0, 1], [1, 0], 2, 0), { velocity_xy_mps: [2, 0], yaw_rate_radps: 0 });
  assert.deepEqual(worldIntent([1, 0], [1, 0], 2, 0), { velocity_xy_mps: [0, -2], yaw_rate_radps: 0 });
  assert.deepEqual(worldIntent([0, -1], [1, 0], 2, 0), { velocity_xy_mps: [-2, 0], yaw_rate_radps: 0 });
  assert.deepEqual(worldIntent([0, 0], [1, 0], 2, .75), { velocity_xy_mps: [0, 0], yaw_rate_radps: .75 });
  const intents = inputStream('yaw');
  const command = intents.next([0, -2], .5, 10);
  assert.equal(command.yaw_rate_radps, .5);
  assert.deepEqual(command.velocity_xy_mps, [0, -2]);
  assert.equal(Object.hasOwn(command, 'heading_rad'), false);
});

test('blocked transport keeps one newest raw input with its original time and no fabricated command ID', () => {
  const stream = inputStream('latest'), mailbox = new LatestIntentMailbox(stream), sent = [];
  for (let i = 0; i < 100; i++) {
    mailbox.offer([i / 100, 0], .5, i);
    assert.equal(mailbox.flush(value => sent.push(value), false), null);
  }
  assert.equal(stream.id, 0); assert.equal(stream.rows.size, 0);
  mailbox.offer([.99, 0], .5, 150);
  mailbox.flush(value => sent.push(value), true);
  assert.equal(sent.length, 1); assert.equal(sent[0].command_id, 1);
  assert.equal(sent[0].client_input_ms, 99);
  assert.deepEqual(sent[0].velocity_xy_mps, [.99, 0]);
  mailbox.offer([0, 0], 0, 151); mailbox.flush(value => sent.push(value), true);
  assert.equal(sent[1].yaw_rate_radps, 0); assert.deepEqual(sent[1].velocity_xy_mps, [0, 0]);
});

test('explicit map/reset clears pending input and binds new episode without reusing associations', () => {
  const { intents, ledger, packet } = fixture();
  const mailbox = new LatestIntentMailbox(intents), sent = [];
  mailbox.offer([1, 0], 0, 10); mailbox.clear();
  intents.bindScene({ ...scope, episode_id: 2 }); ledger.changeScene(1);
  assert.equal(mailbox.flush(value => sent.push(value), true), null);
  mailbox.offer([0, 0], 0, 20); mailbox.flush(value => sent.push(value), true);
  assert.equal(sent[0].episode_id, 2); assert.equal(sent[0].scene_id, 'fixture');
  assert.equal(sent[0].scene_manifest_sha256, 'fixture-hash');
  ledger.offer({ ...packet(1, sent[0]), scene_revision: 1 });
  ledger.render(() => {}, () => 30);
  assert.equal(ledger.metrics().first_visual_p95_ms, null);
  const current = packet(2, sent[0]); current.scene_revision = 1; current.state.episode_id = 2;
  ledger.offer(current); ledger.render(() => {}, () => 70);
  assert.equal(ledger.metrics().first_visual_p95_ms, 50);
  assert.throws(() => new IntentStream('unbound').next([0, 0], 0, 1), /bind/);
});

test('gamepad must return to neutral after blur/map change before held axes can move again', () => {
  const latch = new GamepadLatch();
  assert.deepEqual(latch.sample([1, 0], .5), { velocity: [0, 0], yaw: 0 });
  latch.sample([0, 0], 0);
  assert.deepEqual(latch.sample([1, 0], .5), { velocity: [1, 0], yaw: .5 });
  latch.reset();
  assert.deepEqual(latch.sample([1, 0], .5), { velocity: [0, 0], yaw: 0 });
});

test('manual lifecycle labels planning, ready, active and preserved terminal states in either mode', () => {
  assert.equal(runtimeStatus('awaiting_explicit_start_mb_legacy_reference_history_B1', false, false), 'Ready to Start');
  assert.equal(runtimeStatus('awaiting_explicit_start_mb_legacy_reference_history_B1', true, false), 'Planning reference');
  assert.equal(runtimeStatus('actual_history_warmup', true, false), 'Warming Up');
  assert.equal(runtimeStatus('mb_legacy_reference_history_B1', true, true), 'Running');
  assert.equal(runtimeStatus('completed_mb_legacy_reference_history_B1', true, true), 'Completed');
  assert.equal(runtimeStatus('runtime_failure', true, true), 'Failed');
});

test('active physics FPS excludes Ready idle while all rendered samples remain in the receipt', () => {
  const { ledger, packet } = fixture();
  const ready = packet(1); ready.state.timing.physics_steps = 0; ready.state.sim_time_s = 0;
  ledger.offer(ready); ledger.render(() => {}, () => 0);
  for (let i = 0; i < 31; i++) {
    ledger.offer(packet(i + 2));
    assert.ok(ledger.render(() => {}, () => 24000 + i * DISPLAY_MS));
  }
  const m = ledger.metrics();
  assert.ok(Math.abs(m.active_unique_display_hz - 30) < 1e-8);
  assert.equal(m.active_rendered_samples, 31);
  assert.ok(m.unique_display_hz < 2);
  assert.equal(ledger.frames.length, 32);
  assert.equal(ledger.frames[0].physics_steps, 0);
});

test('native 200Hz producer plus ACK prefetch sustains a single 30Hz gate across RAF phases', () => {
  for (const phase of [0, 3, 7, 11]) {
    const { ledger, packet } = fixture();
    let id = 0, inFlight = null, waitingAck = false, ackAt = null, lastNative = -1;
    let nextRaf = phase, count = 0;
    for (let t = 0; t < 2000; t += .25) {
      if (ackAt !== null && t >= ackAt) { ackAt = null; waitingAck = false; }
      const native = Math.floor(t / 5);
      if (!waitingAck && inFlight === null && native !== lastNative) {
        const value = packet(++id); value.state.frame_id = native;
        value.state.timing.physics_steps = native;
        inFlight = { value, arriveAt: t + 1 }; waitingAck = true; lastNative = native;
      }
      if (inFlight && t >= inFlight.arriveAt) { ledger.offer(inFlight.value); inFlight = null; }
      if (t >= nextRaf) {
        if (ledger.render(() => {}, () => t)) { ackAt = t + 1; count++; }
        nextRaf += 1000 / 60;
      }
    }
    assert.ok(count >= 59 && count <= 60, `RAF phase ${phase}: ${count} unique frames`);
    assert.ok(ledger.metrics().active_unique_display_hz > 29.8);
    assert.equal(ledger.dropped, 0);
  }
});

test('browser diagnostics keep occlusion/focus evidence and separate RAF, transport, paint and ACK clocks', () => {
  const d = new BrowserDiagnostics();
  const visible = { visibility_state: 'visible', document_hidden: false, document_has_focus: true };
  const hidden = { visibility_state: 'hidden', document_hidden: true, document_has_focus: false };
  d.record('raf', 10, visible, false, { delta_ms: 1000, previous_was_active: false });
  d.record('start_clicked', 20, visible, true);
  d.record('raf', 33, visible, true, { delta_ms: 23, previous_was_active: false });
  d.record('raf', 66, visible, true, { delta_ms: 33, previous_was_active: true });
  d.record('visibilitychange', 70, hidden, true);
  d.record('raf', 1076, hidden, true, { delta_ms: 1010, previous_was_active: true });
  d.record('frame_received', 40, visible, true, { packet_id: 2, server_ack_to_publish_ms: .8,
    server_snapshot_wall_ms: .5 });
  d.record('frame_received', 1070, hidden, true, { packet_id: 3, server_ack_to_publish_ms: .9 });
  d.record('frame_rendered', 66, visible, true, { render_ms: .6, receive_to_render_ms: 26 });
  d.record('paint_only', 83, visible, true, { render_ms: .5 });
  d.ackSent({ packet_id: 2, frame_id: 7 }, 67, visible, true);
  d.ackReceived({ packet_id: 2, server_publish_to_ack_ms: 28 }, 69, visible, true);
  const r = d.receipt(), m = r.metrics;
  assert.equal(m.active_raf_delta.max_ms, 1010);
  assert.equal(m.active_packet_interval.max_ms, 1030);
  assert.equal(m.active_render_cpu.max_ms, .6);
  assert.equal(m.active_receive_to_render.max_ms, 26);
  assert.equal(m.active_ack_roundtrip.max_ms, 2);
  assert.equal(m.active_server_publish_to_ack.max_ms, 28);
  assert.ok(m.active_hidden_samples > 0 && m.active_unfocused_samples > 0);
  assert.deepEqual(m.active_visibility_states, ['visible', 'hidden']);
  assert.equal(r.lifetime_event_counts.paint_only, 1);
  assert.ok(r.events.some(row => row.kind === 'raf' && !row.active));
});
