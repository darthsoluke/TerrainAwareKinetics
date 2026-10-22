import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../web/full_corpus_lab/app.js', import.meta.url), 'utf8');
const html = await readFile(new URL('../web/full_corpus_lab/index.html', import.meta.url), 'utf8');
const css = await readFile(new URL('../web/full_corpus_lab/styles.css', import.meta.url), 'utf8');
const definitions = source.slice(source.indexOf('function reactiveControl('), source.indexOf('function isGenerativeMode('));
const { ReactivePlayback, reactiveControl, validateReactiveForecast } = vm.runInNewContext(
  `${definitions}; ({ReactivePlayback, reactiveControl, validateReactiveForecast})`);
const control = { schema: 'motionbricks.reactive_kinematic.v4', step_frames: 4,
  choices: [4, 2, 1], future_frames: 24, fps: 30 };
const plain = value => JSON.parse(JSON.stringify(value));

function definition(name) {
  const start = source.search(new RegExp(`^(?:async )?function ${name}\\(`, 'm'));
  assert.ok(start >= 0, name);
  const rest = source.slice(start + 1);
  const next = rest.search(/^(?:async )?function \w+\(/m);
  return source.slice(start, next < 0 ? source.length : start + 1 + next);
}

function session() {
  const value = new ReactivePlayback(control);
  value.ready(7, 1);
  value.intent({ velocity_xz: [0, 1], heading: .3 }, 2);
  value.start();
  return value;
}

function metadata(plan = 1, origin = 0, patch = {}) {
  return { type: 'forecast', schema: control.schema, episode_id: 7, plan_id: plan, origin_frame: origin,
    played_frame: origin, committed_prefix_frames: 0,
    first_frame: origin + 1, frame_count: 24, fps: 30, vertex_count: 18056,
    frame_stride_floats: 1235, encoding: 'float32-le', geometry_format: 'soma77-rig-v1',
    rig_count: 77, input_seq: 1, step_frames: 4, inference_ms: 12, skin_ms: 2,
    timing: { queue_ms: 3, history_frame: origin }, ...patch };
}

function frames(meta) {
  return Array.from({ length: 24 }, (_, i) => ({
    root: new Float32Array([meta.first_frame + i + meta.plan_id * 100, 1, 0]),
    rig: new Float32Array(77 * 16).fill(meta.plan_id),
  }));
}

function packed(meta) {
  const result = new Float32Array(24 * 1235);
  frames(meta).forEach((frame, i) => {
    result.set(frame.root, i * 1235);
    result.set(frame.rig, i * 1235 + 3);
  });
  return result.buffer;
}

function offer(value, meta, now = 5) {
  value.receiveMetadata(meta, now);
  return value.receiveFrames(value.pending, frames(meta), now + 1);
}

function play(value, count = 1) {
  const receipts = [];
  for (let i = 0; i < count; i += 1) {
    const at = 10 + (value.playedFrame + 1) * 1000 / 30;
    assert.ok(value.next(at));
    receipts.push(value.rendered(at + 1));
  }
  return receipts;
}

function frontend(reactive = session()) {
  let now = 10;
  const sent = [], failures = [], rendered = [], appliedOptions = [], history = [];
  const state = { config: { reactive_control: control }, reactive, ready: true, paused: false,
    mode: 'terrain', modeFrame: 0, lastPlayedAt: null, vertexCount: 18056,
    latestCommand: { velocity_xz: [0, 1], heading: .3 }, live: { modelStarted: true },
    diagnostics: {}, socket: { readyState: 1, send: text => sent.push(JSON.parse(text)) } };
  const stats = { root: [0, 0, 0], applied: 0, scheduling: { frame_intervals_ms: [] }, received: 0, underruns: 0 };
  const el = { reactiveStep: {}, reset: {} };
  const context = vm.createContext({ state, stats, el, ArrayBuffer, Blob, Float32Array,
    MAX_BLOCK_FRAMES: 8, SIM_FPS: 30, LITTLE_ENDIAN: true, validateReactiveForecast,
    performance: { now: () => ++now }, WebSocket: { OPEN: 1 },
    isGenerativeMode: () => true, updateReadouts() {}, updatePauseButton() {},
    updateLiveStateTelemetry() {}, updateIntentArrow() {}, clearInput() {}, clearDiagnostics() {},
    setRunStatus() {}, setTelemetryState() {}, showError() {},
    applyFrame: (frame, options = {}) => {
      rendered.push(frame); appliedOptions.push(plain(options));
      if (options.advance !== false) state.modeFrame += 1;
    },
    updatePlayedDiagnostics: root => history.push([...root]), updateDiagnosticPrediction() {},
    acceptDiagnosticPlan: () => { state.diagnostics = {}; },
    failReactive: message => { failures.push(message); state.live.failed = true; state.paused = true; },
  });
  for (const name of ['setCommand', 'sendReactiveIntent', 'startReactivePlayback', 'applyReactiveFrame', 'stepReactive',
    'confirmReactiveRender', 'resetLive', 'togglePause', 'decodeFrameBlock', 'handleReactiveBinary',
    'handleSocketBinary', 'ensureLivePlan', 'handleSocketJson', 'validateBlockMetadata']) {
    vm.runInContext(definition(name), context);
  }
  return { state, stats, el, context, sent, failures, rendered, appliedOptions, history,
    call: expression => vm.runInContext(expression, context) };
}

test('reactive connection is opt-in to the exact config; selector is hidden by default', () => {
  assert.equal(reactiveControl({}), null);
  assert.equal(reactiveControl({ reactive_control: control }), control);
  for (const patch of [{ schema: 'other' }, { schema: 'motionbricks.reactive_kinematic.v1' }, { future_frames: 8 }, { fps: 60 },
    { choices: [1, 2, 4] }, { step_frames: 8 }]) {
    assert.throws(() => reactiveControl({ reactive_control: { ...control, ...patch } }));
  }
  assert.match(html, /id="reactive-step-control"[^>]*hidden/);
  assert.match(css, /\.reactive-control\[hidden\]\s*\{\s*display: none;/);
  assert.match(html, /id="reactive-step"[\s\S]*?value="4"[\s\S]*?value="2"[\s\S]*?value="1"/);
});

test('default websocket remains /ws; only the opt-in controller chooses /ws/reactive', () => {
  for (const reactive of [null, session()]) {
    const urls = [], handlers = {};
    const state = { config: {}, reactive, live: {} };
    class Socket {
      constructor(url) { urls.push(url); }
      addEventListener(name, callback) { handlers[name] = callback; }
    }
    const context = vm.createContext({ state, WebSocket: Socket, location: { protocol: 'http:', host: 'localhost:28950' },
      el: { reset: {} }, setConnection() {}, setRunStatus() {} });
    vm.runInContext(definition('connectSocket'), context);
    vm.runInContext('connectSocket()', context);
    assert.equal(urls[0], `ws://localhost:28950${reactive ? '/ws/reactive' : '/ws'}`);
    assert.equal(state.socket.binaryType, 'arraybuffer');
    assert.ok(handlers.message);
  }
});

test('changed intents are transmitted immediately even with forecast binary outstanding', () => {
  const f = frontend();
  f.state.reactive.receiveMetadata(metadata(), 5);
  f.state.live.inFlight = { id: 99 };
  f.call('setCommand([2,0], .4); setCommand([0,-2], .4); setCommand([0,-2], .4); setCommand([0,-2], .8)');
  assert.deepEqual(f.sent.map(row => row.type), ['intent', 'intent', 'intent']);
  assert.deepEqual(f.sent.map(row => row.seq), [2, 3, 4]);
  assert.deepEqual(f.sent.map(row => row.velocity_xz), [[2, 0], [0, -2], [0, -2]]);
  assert.ok(f.sent[2].client_input_ms > f.sent[0].client_input_ms);
  assert.ok(f.state.reactive.pending);
  assert.equal(f.state.reactive.playedFrame, 0);
});

test('start is sent once per ready and each execution choice leaves the 24-frame horizon intact', () => {
  const value = new ReactivePlayback(control);
  value.ready(1, 1);
  assert.deepEqual(plain(value.start()), { type: 'start' });
  assert.equal(value.start(), null);
  for (const step of [1, 2, 4]) {
    assert.deepEqual(plain(value.configure(step, 2)), { type: 'configure', step_frames: step });
    assert.equal(value.telemetry.future_frames, 24);
    assert.equal(value.telemetry.tokens, 8);
    assert.equal(value.playedFrame, 0);
  }
  assert.throws(() => value.configure(8, 2));
  assert.throws(() => value.ready(1, 3), /duplicate/);
  value.ready(2, 3);
  assert.deepEqual(plain(value.start()), { type: 'start' });
});

test('ready while paused defers input/start; reset ready starts one fresh episode with the selected step', () => {
  const f = frontend(new ReactivePlayback(control));
  f.state.ready = false;
  f.state.paused = true;
  f.state.positionAttribute = {};
  f.state.mapSwitchStarted = null;
  f.state.config.terrain_mode = {};
  f.state.live.epoch = 0;
  Object.assign(f.context, { toVec3: value => value, toFloatArray: value => value,
    initializeRoot() {}, updateCommandFromInput() {}, setConnection() {} });
  vm.runInContext(definition('handleReady'), f.context);
  f.context.readyMessage = { type: 'ready', mode: 'terrain', episode_id: 1,
    vertex_count: 18056, root: [0, 1, 0], vertices: new Float32Array(18056 * 3), fps: 30 };
  f.call('handleReady(readyMessage)');
  assert.equal(f.failures.length, 0);
  assert.equal(f.sent.length, 0);
  assert.equal(f.state.ready, true);
  f.call('togglePause()');
  assert.deepEqual(f.sent.map(row => row.type), ['intent', 'start']);
  f.state.reactive.configure(2, 20);
  f.call('resetLive()');
  f.context.readyMessage.episode_id = 2;
  f.call('handleReady(readyMessage)');
  assert.equal(f.failures.length, 0);
  assert.deepEqual(f.sent.slice(-3).map(row => row.type), ['intent', 'configure', 'start']);
  assert.equal(f.sent.at(-2).step_frames, 2);
  assert.equal(f.sent.at(-3).seq, 2);
  assert.equal(f.state.modeFrame, 0);
});

test('future forecast starts exactly at its boundary and preserves the accepted prefix', () => {
  const value = session();
  offer(value, metadata(10));
  play(value, 5);
  const future = metadata(11, 8, { played_frame: 5, committed_prefix_frames: 3 });
  value.receiveMetadata(future, 200);
  const old = play(value, 2);
  assert.deepEqual(old.map(row => row.plan_id), [10, 10]);
  const history = plain(value.telemetry.played);
  const result = value.receiveFrames(value.pending, frames(future), 260);
  assert.deepEqual(plain(result.message), { type: 'accept', episode_id: 7, plan_id: 11, first_frame: 9 });
  const next = value.next(270);
  assert.equal(next.absoluteFrame, 8);
  assert.equal(next.frame.root[0], 1008);
  assert.equal(value.playedFrame, 7);
  assert.deepEqual(plain(value.telemetry.played), history);
  assert.equal(value.rendered(271).plan_id, 10);
  assert.equal(value.next(300).frame.root[0], 1109);
  const receipt = value.rendered(301);
  assert.equal(receipt.frame, 9);
  assert.equal(receipt.plan_id, 11);
  assert.equal(value.telemetry.handoffs.at(-1).root_diff_m, 100);
  assert.equal(value.telemetry.handoffs.at(-1).achieved_interval_frames, 8);
});

test('all 4/2/1 execution steps retain the accepted prefix and enter frame zero', () => {
  for (const step of [4, 2, 1]) {
    const value = session();
    value.configure(step, 3);
    offer(value, metadata(1, 0, { step_frames: step }));
    play(value, 3);
    const boundary = 3 + step;
    const result = offer(value, metadata(2, boundary - 1,
      { step_frames: step, played_frame: 3, committed_prefix_frames: step - 1 }));
    assert.equal(result.message.first_frame, boundary);
    const prefix = play(value, step - 1);
    assert.ok(prefix.every(receipt => receipt.plan_id === 1));
    const next = value.next(200);
    assert.equal(next.absoluteFrame, boundary);
    assert.equal(next.frame, value.current.frames[0]);
    assert.equal(next.frame.root[0], 200 + boundary);
    assert.equal(value.rendered(201).plan_id, 2);
  }
});

test('multiple accepted future boundaries preserve intermediate plans', () => {
  const value = session();
  offer(value, metadata(1));
  play(value);
  offer(value, metadata(2, 3, { played_frame: 0, committed_prefix_frames: 3 }));
  play(value, 2);
  offer(value, metadata(3, 6, { played_frame: 3, committed_prefix_frames: 3 }));
  assert.deepEqual([...value.queue.keys()], [4, 7]);
  const receipts = play(value, 6);
  assert.deepEqual(receipts.map(row => [row.frame, row.plan_id]),
    [[4, 2], [5, 2], [6, 2], [7, 3], [8, 3], [9, 3]]);
  assert.deepEqual(plain(value.telemetry.handoffs).map(row => [row.frame, row.plan_id]), [[1, 1], [4, 2], [7, 3]]);
});

test('newer plan at the same future boundary supersedes only that unplayed plan', () => {
  const value = session();
  offer(value, metadata(1)); play(value);
  const first = metadata(2, 3, { played_frame: 0, committed_prefix_frames: 3 });
  offer(value, first);
  offer(value, { ...first, plan_id: 3 });
  assert.equal(value.queue.size, 1);
  assert.equal(value.queue.get(4).meta.plan_id, 3);
  assert.equal(value.telemetry.accepted.at(-1).superseded_plan_id, 2);
  assert.deepEqual(play(value, 3).map(row => row.plan_id), [1, 1, 3]);
});

test('switching 4 to 1 preserves the previously committed three-frame prefix', () => {
  const value = session();
  offer(value, metadata(1));
  const future = metadata(2, 3, { played_frame: 0, committed_prefix_frames: 3 });
  offer(value, future);
  value.configure(1, 6);
  offer(value, { ...future, plan_id: 3, step_frames: 1 });
  assert.deepEqual(play(value, 4).map(row => row.plan_id), [1, 1, 1, 3]);
  assert.equal(value.telemetry.handoffs.at(-1).forecast_step_frames, 1);
});

test('binary that misses its first frame is rejected without cropping or stopping current coverage', async () => {
  const f = frontend();
  offer(f.state.reactive, metadata()); play(f.state.reactive, 3);
  const late = metadata(2, 3, { played_frame: 0, committed_prefix_frames: 3 });
  f.state.reactive.receiveMetadata(late, 100);
  play(f.state.reactive, 2);
  const history = plain(f.state.reactive.telemetry.played);
  f.context.payload = packed(late);
  await f.call('handleSocketBinary(payload)');
  assert.deepEqual(f.sent, [{ type: 'reject', episode_id: 7, plan_id: 2, reason: 'missed_handoff' }]);
  assert.equal(f.failures.length, 0);
  assert.equal(f.state.paused, false);
  assert.equal(f.state.reactive.queue.size, 0);
  assert.deepEqual(plain(f.state.reactive.telemetry.played), history);
  assert.equal(f.state.reactive.telemetry.rejected_count, 1);
  assert.equal(f.state.reactive.telemetry.missed_handoff_count, 1);
  assert.equal(play(f.state.reactive)[0].plan_id, 1);
});

test('uncovered future prefix is a protocol error rather than a synthesized initial hold', () => {
  const value = session();
  assert.throws(() => offer(value, metadata(1, 3, { played_frame: 0, committed_prefix_frames: 3 })),
    /prefix does not cover frame 1/);
  assert.equal(value.telemetry.accepted_count, 0);
});

test('an unrendered staged boundary is replaced with frame zero and acknowledged once', async () => {
  const f = frontend();
  offer(f.state.reactive, metadata());
  f.call('stepReactive()');
  assert.equal(f.state.reactive.playedFrame, 0);
  assert.equal(f.sent.length, 0);
  f.state.reactive.receiveMetadata(metadata(2), 15);
  f.context.payload = packed(metadata(2));
  await f.call('handleSocketBinary(payload)');
  assert.deepEqual(f.sent, [{ type: 'accept', episode_id: 7, plan_id: 2, first_frame: 1 }]);
  f.call('stepReactive()');
  assert.equal(f.rendered.length, 2);
  assert.equal(f.rendered.at(-1).root[0], 201);
  assert.deepEqual(f.appliedOptions, [{ advance: false, count: false }, { advance: false, count: false }]);
  assert.equal(f.state.modeFrame, 0);
  assert.equal(f.stats.applied, 0);
  assert.equal(f.history.length, 0);
  f.call('confirmReactiveRender(20)');
  assert.deepEqual(f.sent.at(-1), { type: 'played', episode_id: 7, frame: 1, plan_id: 2, client_render_ms: 20 });
  assert.equal(f.stats.applied, 1);
  assert.deepEqual(f.history, [[201, 1, 0]]);
  f.call('stepReactive(); confirmReactiveRender(54)');
  assert.equal(f.sent.at(-1).frame, 2);
  assert.equal(f.sent.at(-1).plan_id, 2);
  assert.equal(f.stats.applied, 2);
});

test('real applyFrame staging leaves all played counters and root history for render commit', async () => {
  const f = frontend();
  f.state.positionAttribute = {};
  f.state.rawMesh = {};
  f.state.renderer = { shadowMap: {} };
  f.state.skinMesh = { skeleton: { update() {}, bones: Array.from({ length: 77 }, () => ({
    matrixWorld: { fromArray() { return this; }, transpose() { return this; } },
  })) } };
  Object.assign(f.context, { toVec3: value => Array.from(value), followRoot() {} });
  vm.runInContext(definition('applyFrame'), f.context);
  offer(f.state.reactive, metadata());
  f.call('stepReactive()');
  f.state.reactive.receiveMetadata(metadata(2), 15);
  f.context.payload = packed(metadata(2));
  await f.call('handleSocketBinary(payload)');
  assert.equal(f.state.skinRig[0], 2);
  assert.equal(f.state.modeFrame, 0);
  assert.equal(f.stats.applied, 0);
  assert.deepEqual(f.stats.root, [0, 0, 0]);
  assert.equal(f.history.length, 0);
  f.call('confirmReactiveRender(20); confirmReactiveRender(21)');
  assert.equal(f.stats.applied, 1);
  assert.equal(f.state.modeFrame, 1);
  assert.deepEqual(plain(f.stats.root), [201, 1, 0]);
  assert.deepEqual(f.history, [[201, 1, 0]]);
  assert.equal(f.sent.filter(row => row.type === 'played').length, 1);
  assert.equal(f.failures.length, 0);
});

test('production render hook sends played only after drawing and never on an unchanged display', () => {
  const f = frontend(), callbacks = [], events = [];
  offer(f.state.reactive, metadata());
  f.state.renderer = { setRenderTarget() {}, setScissorTest() {}, setViewport() {},
    render() { events.push('draw'); } };
  f.state.scene = {};
  f.state.camera = {};
  f.state.renderDirty = true;
  f.el.canvas = { clientWidth: 100, clientHeight: 100 };
  const send = f.state.socket.send;
  f.state.socket.send = text => { events.push(JSON.parse(text).type); send(text); };
  Object.assign(f.context, { window: { requestAnimationFrame: callback => callbacks.push(callback) },
    pollGamepad() {}, updateCameraHeight() {}, updateThirdPersonCamera() {} });
  vm.runInContext(definition('startRenderLoop'), f.context);
  f.call('stepReactive(); startRenderLoop()');
  callbacks.shift()(20);
  assert.deepEqual(events, ['draw', 'played']);
  assert.equal(f.state.reactive.playedFrame, 1);
  callbacks.shift()(37);
  assert.deepEqual(events, ['draw', 'played']);
  assert.equal(f.failures.length, 0);
});

test('already drawn pending frame cannot be replaced even while its ACK is paused', async () => {
  const f = frontend();
  offer(f.state.reactive, metadata());
  f.call('stepReactive(); togglePause(); confirmReactiveRender(30)');
  f.state.reactive.receiveMetadata(metadata(2), 31);
  f.context.payload = packed(metadata(2));
  await f.call('handleSocketBinary(payload)');
  assert.deepEqual(f.sent, [{ type: 'reject', episode_id: 7, plan_id: 2, reason: 'missed_handoff' }]);
  assert.equal(f.state.reactive.awaitingRender.meta.plan_id, 1);
  assert.equal(f.rendered.length, 1);
  assert.equal(f.state.reactive.playedFrame, 0);
  f.call('togglePause(); confirmReactiveRender(300)');
  assert.deepEqual(f.sent.at(-1), { type: 'played', episode_id: 7, frame: 1, plan_id: 1, client_render_ms: 30 });
  assert.equal(f.failures.length, 0);
});

test('pause suppresses intents and played ACKs; resume forces latest intent without a second start', () => {
  const f = frontend();
  offer(f.state.reactive, metadata());
  f.call('stepReactive(); togglePause(); setCommand([0,-1],.7); confirmReactiveRender(30)');
  assert.equal(f.sent.length, 0);
  assert.equal(f.state.reactive.playedFrame, 0);
  f.call('togglePause(); confirmReactiveRender(300)');
  assert.deepEqual(f.sent.map(row => row.type), ['intent', 'played']);
  assert.deepEqual(f.sent[0].velocity_xz, [0, -1]);
  assert.equal(f.sent[1].frame, 1);
  assert.equal(f.sent[1].client_render_ms, 30);
  f.call('togglePause(); togglePause()');
  assert.equal(f.sent.at(-1).type, 'intent');
  assert.equal(f.sent.at(-1).seq, f.sent[0].seq + 1);
  assert.equal(f.sent.filter(row => row.type === 'start').length, 0);
});

test('duplicate, old-origin, unacknowledged-history and wrong-episode metadata are rejected', () => {
  const value = session();
  offer(value, metadata(5));
  for (const meta of [metadata(5), metadata(4), metadata(6, 1), metadata(6, 0, { episode_id: 9 }),
    metadata(6, 0, { input_seq: 2 }), metadata(6, 0, { first_frame: 2 })]) {
    assert.throws(() => value.receiveMetadata(meta, 10));
  }
  value.receiveMetadata(metadata(6), 11);
  assert.throws(() => value.receiveMetadata(metadata(7), 12), /binary is missing/);
  play(value, 1);
  assert.throws(() => value.rendered(20), /duplicate|sequence/);
  const advanced = session();
  offer(advanced, metadata()); play(advanced, 2);
  offer(advanced, metadata(2, 2));
  assert.throws(() => advanced.receiveMetadata(metadata(3, 1), 30), /conditioning origin/);
});

test('generated old forecast remains playable through all 24 frames, then fails without tail hold', () => {
  const value = session();
  assert.equal(value.next(3), null);
  offer(value, metadata());
  const receipts = play(value, 24);
  assert.deepEqual(receipts.map(row => row.frame), Array.from({ length: 24 }, (_, i) => i + 1));
  assert.throws(() => value.next(900), /coverage exhausted at frame 25/);
  assert.equal(value.playedFrame, 24);
  assert.equal(offer(value, metadata(2), 901).message.type, 'reject');
});

test('reset invalidates old metadata/binary; new episode preserves monotonic input sequence', () => {
  const f = frontend();
  f.state.reactive.receiveMetadata(metadata(), 5);
  const stale = f.state.reactive.pending;
  f.call('resetLive()');
  assert.deepEqual(f.sent, [{ type: 'reset' }]);
  assert.equal(f.state.ready, false);
  assert.equal(f.state.reactive.awaitingRender, null);
  f.state.reactive.ready(8, 20);
  f.state.reactive.receiveFrames(stale, frames(metadata()), 21);
  assert.equal(f.state.reactive.queue.size, 0);
  const input = f.state.reactive.intent({ velocity_xz: [0, 1], heading: .3 }, 22);
  assert.equal(input.seq, 2);
  assert.throws(() => f.state.reactive.receiveMetadata(metadata(2), 23), /episode/);
  offer(f.state.reactive, metadata(1, 0, { episode_id: 8, input_seq: 2 }));
  f.state.reactive.start();
  assert.deepEqual(play(f.state.reactive).map(row => [row.episode_id, row.frame, row.plan_id]), [[8, 1, 1]]);
});

test('shared float32-LE parser accepts 24 rig frames in reactive mode only', () => {
  const f = frontend();
  f.context.meta = metadata();
  const buffer = new ArrayBuffer(24 * 1235 * 4), view = new DataView(buffer);
  for (let i = 0; i < 24; i += 1) {
    view.setFloat32(i * 1235 * 4, i + 1, true);
    view.setFloat32((i * 1235 + 3) * 4, i + .5, true);
  }
  f.context.buffer = buffer;
  const decoded = f.call('decodeFrameBlock(buffer, meta, 18056, true)');
  assert.equal(decoded.length, 24);
  assert.equal(decoded[23].root[0], 24);
  assert.equal(decoded[23].rig[0], 23.5);
  assert.throws(() => f.call('decodeFrameBlock(buffer, meta, 18056)'), /1 to 8/);
  f.context.buffer = buffer.slice(4);
  assert.throws(() => f.call('decodeFrameBlock(buffer, meta, 18056, true)'), /size mismatch/);
  view.setFloat32(4, NaN, true);
  f.context.buffer = buffer;
  assert.throws(() => f.call('decodeFrameBlock(buffer, meta, 18056, true)'), /non-finite/);
});

test('invalid forecast format, count and clock fields are rejected before binary adoption', () => {
  for (const patch of [{ frame_count: 8 }, { fps: 60 }, { vertex_count: 3 },
    { frame_stride_floats: 1234 }, { rig_count: 76 }, { encoding: 'float32-be' },
    { geometry_format: 'vertices' }, { step_frames: 3 }, { origin_frame: 0.5 },
    { plan_id: NaN }, { episode_id: '7' }, { inference_ms: Infinity }, { timing: null },
    { played_frame: -1 }, { committed_prefix_frames: 4 }, { committed_prefix_frames: 1 },
    { schema: 'motionbricks.reactive_kinematic.v1' }]) {
    const value = session();
    assert.throws(() => value.receiveMetadata(metadata(1, 0, patch), 4), /metadata/);
    assert.equal(value.queue.size, 0);
    assert.equal(value.pending, null);
  }
});

test('measured compute lead may exceed the requested interval but remains bounded', () => {
  validateReactiveForecast(metadata(1,6,{played_frame:0,committed_prefix_frames:6,step_frames:1}),18056);
  assert.throws(()=>validateReactiveForecast(metadata(1,7,
    {played_frame:0,committed_prefix_frames:7,step_frames:1}),18056),/metadata/);
});

test('metadata and binary are paired; malformed payload fails explicitly', async () => {
  const f = frontend();
  f.state.skinMesh = {};
  f.context.message = JSON.stringify(metadata());
  f.call('handleSocketJson(message)');
  f.context.payload = new ArrayBuffer(24 * 1235 * 4);
  await f.call('handleSocketBinary(payload)');
  assert.equal(f.state.reactive.queue.get(1).meta.plan_id, 1);
  assert.deepEqual(f.sent, [{ type: 'accept', episode_id: 7, plan_id: 1, first_frame: 1 }]);
  assert.equal(f.stats.received, 1);
  assert.equal(f.failures.length, 0);
  f.context.message = JSON.stringify(metadata(2));
  f.call('handleSocketJson(message)');
  f.context.payload = new ArrayBuffer(8 * 1235 * 4);
  await f.call('handleSocketBinary(payload)');
  assert.match(f.failures.at(-1), /size mismatch/);
  const unexpected = frontend();
  unexpected.context.payload = new ArrayBuffer(4);
  await unexpected.call('handleSocketBinary(payload)');
  assert.match(unexpected.failures.at(-1), /Unexpected reactive binary/);
});

test('delayed Blob from a closed episode cannot publish into a new episode', async () => {
  const f = frontend();
  f.state.reactive.receiveMetadata(metadata(), 5);
  let release;
  const blob = new Blob();
  blob.arrayBuffer = () => new Promise(resolve => { release = resolve; });
  f.context.payload = blob;
  const done = f.call('handleSocketBinary(payload)');
  f.state.reactive.invalidate();
  f.state.reactive.ready(8, 20);
  release(new ArrayBuffer(24 * 1235 * 4));
  await done;
  assert.equal(f.state.reactive.queue.size, 0);
  assert.equal(f.stats.received, 0);
  assert.equal(f.failures.length, 0);
});

test('telemetry is bounded and distinguishes requested, generated and achieved intervals', () => {
  const value = session();
  for (let i = 0; i < 150; i += 1) {
    value.intent({ velocity_xz: [0, i % 2], heading: i / 10 }, i * 34);
    offer(value, metadata(i + 1, i, { input_seq: value.seq, step_frames: 2 }), i * 34);
    play(value);
  }
  assert.equal(value.playedFrame, 150);
  for (const key of ['inputs', 'forecasts', 'handoffs', 'played']) assert.equal(value.telemetry[key].length, 120);
  const handoff = value.telemetry.handoffs.at(-1), forecast = value.telemetry.forecasts.at(-1);
  assert.equal(handoff.requested_step_frames, 4);
  assert.equal(handoff.forecast_step_frames, 2);
  assert.equal(handoff.achieved_interval_frames, 1);
  assert.equal(forecast.origin_interval_frames, 1);
  assert.deepEqual(plain(forecast.timing), { queue_ms: 3, history_frame: 149 });
});

test('default path still sends its legacy plan and decodes eight raw-vertex frames', () => {
  const f = frontend(null);
  f.state.mode = 'live';
  f.state.live = { epoch: 0, retryAt: 0, awaitingCommit: null };
  f.state.requestId = 0;
  f.context.commandTiming = () => ({ version: 1 });
  f.call('ensureLivePlan()');
  assert.equal(f.sent.at(-1).type, 'plan');
  assert.equal(f.sent.at(-1).frame, undefined);
  f.context.meta = { frame_count: 8, vertex_count: 3, frame_stride_floats: 12, encoding: 'float32-le' };
  f.context.buffer = new ArrayBuffer(8 * 12 * 4);
  assert.equal(f.call('decodeFrameBlock(buffer, meta, 3)').length, 8);
});
