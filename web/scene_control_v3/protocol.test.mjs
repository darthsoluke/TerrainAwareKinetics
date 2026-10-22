import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('./app.js', import.meta.url), 'utf8');
const { SCHEMA, STRIDE, FRAME_MS, CommandStream, FrameLedger, decodeRig, skinAttributes, percentile,
  validateRole, writeReceiptSnapshot, formatReferenceParity, formatClose } =
  await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function fixture() {
  const stream = new CommandStream('session'); stream.next([0, 0, 0], 10);
  const ledger = new FrameLedger('session', 'source_replay_reference');
  const meta = { type: 'frame', schema: SCHEMA, session_id: 'session', packet_id: 1, frame_index: 0,
    command_id: 1, client_input_ms: 10, frame_count: 1, fps: 30, vertex_count: 18056, rig_count: 77,
    geometry_format: 'soma77-rig-v1', encoding: 'float32-le', frame_stride_floats: STRIDE,
    benchmark_role: 'source_replay_reference', command_used_for_motion: false, actor_status: 'UNTRAINED',
    optimizer_updates: 0, learned_wasd: false, quality_response: 'unmeasured',
    timings: { native_rig_ms: 2, necessary_inference_ms: null } };
  const values = new Float32Array(STRIDE); values.set([4, 1, -3]);
  for (let i = 0; i < 77; i++) values.set([1, 0, 0, i, 0, 1, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1], 3 + i * 16);
  return { stream, ledger, meta, bytes: values.buffer };
}

test('4940-byte row-major SOMA77 transport preserves world root and matrices', () => {
  const { meta, bytes } = fixture();
  assert.equal(bytes.byteLength, 4940);
  const frame = decodeRig(bytes, meta);
  assert.deepEqual([...frame.root], [4, 1, -3]); assert.equal(frame.rig.length, 77 * 16);
  assert.equal(frame.rig[76 * 16 + 3], 76); assert.equal(frame.rig[7], 2);
  for (const override of [{ fps: 60 }, { vertex_count: 30 }, { frame_count: 4 }, { rig_count: 30 },
    { geometry_format: 'soma-full-vertices-v1' }, { encoding: 'float64-le' }]) {
    assert.throws(() => decodeRig(bytes, { ...meta, ...override }));
  }
  assert.throws(() => decodeRig(bytes.slice(4), meta));
  new DataView(bytes).setFloat32(40, NaN, true); assert.throws(() => decodeRig(bytes, meta), /Non-finite/);
});

test('every native 4..8 skin influence is retained, without top-four truncation or renormalization', () => {
  for (let count = 4; count <= 8; count++) {
    const skin = { format: 'soma77-rig-v1', rig_count: 77, influences_per_vertex: count,
      bind_vertices: Array(18056 * 3).fill(0), inverse_bind: Array(77 * 16).fill(0), faces: [0, 1, 2],
      indices: Array.from({ length: 18056 * count }, (_, i) => i % count),
      weights: Array.from({ length: 18056 * count }, (_, i) => (i % count + 1) / 40) };
    const attrs = skinAttributes(skin);
    for (const vertex of [0, 1234, 18055]) for (let i = 0; i < 8; i++) {
      const offset = vertex * 4 + i % 4;
      assert.equal((i < 4 ? attrs.indices : attrs.extraIndices)[offset], i < count ? i : 0);
      assert.ok(Math.abs((i < 4 ? attrs.weights : attrs.extraWeights)[offset] - (i < count ? (i + 1) / 40 : 0)) < 1e-7);
    }
    assert.throws(() => skinAttributes({ ...skin, influences_per_vertex: 9 }));
  }
});

test('one render submission advances one factual frame, ACK never precedes successful draw', () => {
  const { stream, ledger, meta, bytes } = fixture(); ledger.plan(meta, stream); ledger.binary(bytes);
  assert.throws(() => ledger.render(() => { throw Error('draw failed'); }, () => 20));
  assert.equal(ledger.rendered, 0); assert.equal(ledger.pending.ack, null);
  const times = [30, 32]; const ack = ledger.render(frame => assert.equal(frame.root[0], 4), () => times.shift());
  assert.equal(ack.client_render_completed_ms, 32); assert.equal(ledger.rendered, 1); assert.equal(ledger.committed, 0);
  assert.equal(ledger.render(() => assert.fail(), () => 2000), null);
  assert.throws(() => ledger.plan(meta, stream)); assert.throws(() => ledger.binary(bytes));
  assert.throws(() => ledger.commit({ ...ack, committed_frame_count: 2 }));
  ledger.commit({ ...ack, committed_frame_count: 1 }); assert.equal(ledger.pending, null);
  assert.throws(() => ledger.commit({ ...ack, committed_frame_count: 1 }));
  assert.equal(ledger.associations[0].latency_ms, 22);
});

test('latest commands stay independent from pending frame and first association is sampled once', () => {
  const { stream, ledger, meta, bytes } = fixture(); ledger.plan(meta, stream); ledger.binary(bytes);
  stream.next([1, 0, 0], 15); stream.next([2, 0, 0], 16);
  assert.equal(stream.next([2, 0, 0], 17), null); assert.equal(stream.id, 3);
  const first = ledger.render(() => {}, () => 20); assert.equal(first.command_id, 1);
  ledger.commit({ ...first, committed_frame_count: 1 });
  ledger.plan({ ...meta, packet_id: 2, frame_index: 1 }, stream); ledger.binary(bytes);
  const second = ledger.render(() => {}, () => 55); ledger.commit({ ...second, committed_frame_count: 2 });
  assert.equal(ledger.associations.length, 1);
  assert.equal(ledger.metrics().unique_rendered_hz, 1000 / 35);
  ledger.segment++; assert.equal(ledger.metrics().unique_rendered_hz, null);
  for (let i = 0; i < 1000; i++) stream.next([i, 0, 0], 100 + i);
  assert.equal(stream.rows.size, 256);
});

test('stale, skipped, unknown, falsely trained or mismatched command frames are rejected', () => {
  for (const override of [{ session_id: 'other' }, { frame_index: 1 }, { packet_id: 2 }, { command_id: 9 },
    { client_input_ms: 9 }, { benchmark_role: 'learned_wasd' }, { actor_status: 'TRAINED' },
    { command_used_for_motion: true }, { optimizer_updates: 1 }]) {
    const { stream, ledger, meta } = fixture(); assert.throws(() => ledger.plan({ ...meta, ...override }, stream));
  }
  assert.equal(percentile([], .95), null); assert.equal(percentile([1, 3, 2], .95), 3);
});

test('capacity role has actual necessary-chain timings and never becomes source replay or learned control', () => {
  const { stream, meta, bytes } = fixture();
  assert.equal(validateRole(meta), false);
  const capacity = { ...meta, benchmark_role: 'architecture_capacity_only', command_used_for_motion: true,
    object_pose: [[0, 1.8, 0, 0, 0, 0, 1]], timings: { native_rig_ms: null, necessary_inference_ms: 12.5 },
    query_metadata: { source_counts: { current_body: 256, committed_root_plan: 128, fresh_intent: 128 }, plan_origin_frame_id: 0 } };
  assert.equal(validateRole(capacity), true);
  const ledger = new FrameLedger('session', capacity.benchmark_role);
  assert.throws(() => ledger.plan(meta, stream));
  ledger.plan(capacity, stream); ledger.binary(bytes);
  const ack = ledger.render((_frame, metadata) => assert.deepEqual(metadata.object_pose, capacity.object_pose), () => 30);
  ledger.commit({ ...ack, committed_frame_count: 1 });
  assert.equal(ledger.metrics().necessary_inference_p99_ms, 12.5);
  assert.equal(ledger.metrics().native_rig_p99_ms, null);
  assert.deepEqual(ledger.frames[0].query_metadata, capacity.query_metadata);
  assert.throws(() => validateRole({ ...capacity, learned_wasd: true }));
  assert.throws(() => validateRole({ ...capacity, quality_response: 'passed' }));
  assert.throws(() => validateRole({ ...capacity, command_used_for_motion: false }));
});

test('receipt click serializes one DOM-readable snapshot; parity exposes all18056 maxabs', async () => {
  const snapshot = { actor_status: 'UNTRAINED', frames: [{ frame_index: 2 }], quality_response: 'unmeasured' };
  let writes = 0, value;
  const textarea = { hidden: true, set value(next) { writes++; value = next; } };
  const serialized = writeReceiptSnapshot(textarea, snapshot);
  assert.equal(writes, 1); assert.equal(textarea.hidden, false); assert.equal(value, serialized);
  snapshot.frames.push({ frame_index: 3 }); assert.equal(JSON.parse(value).frames.length, 1);
  assert.equal(formatReferenceParity({ vertex_count: 18056, max_abs_error_m: .000002, max_vertex_error_m: .000003 }),
    'All 18,056 / maxabs 2.000000e-6 m (CPU skin mirror)');
  const html = await readFile(new URL('./index.html', import.meta.url), 'utf8');
  assert.match(html, /textarea id="receipt-snapshot" readonly/);
  assert.match(html, /id="reference-check"[^>]+title="Compare all 18,056 reference vertices"/);
  assert.match(html, /output id="reference-parity" role="status"/);
});

test('initial capacity labels, viewport containment and negative-X camera are explicit', async () => {
  assert.match(source, /el\('inference-time'\)\.textContent = capacity \? 'Awaiting frames'/);
  assert.match(source, /el\('play'\)\.title = `Play \$\{streamName\}`/);
  assert.match(source, /el\('stage'\)\.setAttribute\('aria-label', capacity \?/);
  assert.match(source, /camera\.position\.set\(root\[0\] - 3\.1/);
  const css = await readFile(new URL('./styles.css', import.meta.url), 'utf8');
  assert.match(css, /body \{[^}]*height: 100dvh; overflow: hidden/);
  assert.match(css, /\.workspace \{[^}]*min-height: 0; overflow: hidden/);
  assert.match(css, /aside \{[^}]*min-height: 0; overflow-y: auto/);
  assert.doesNotMatch(css, /min-height: 550px|100dvh - 126px/);
});

test('pause and repeated visibility pause preserve last continuous Hz in the serialized receipt', () => {
  const { stream, ledger, meta, bytes } = fixture();
  assert.equal(ledger.setPaused(false), true);
  let now = 20;
  for (let i = 0; i < 4; i++) {
    ledger.plan({ ...meta, packet_id: i + 1, frame_index: i }, stream); ledger.binary(bytes);
    const ack = ledger.render(() => {}, () => now);
    ledger.commit({ ...ack, committed_frame_count: i + 1 }); now += FRAME_MS;
  }
  const before = ledger.metrics(), segment = ledger.segment, deadline = ledger.nextAt;
  assert.ok(Math.abs(before.unique_rendered_hz - 30) < 1e-9);
  assert.equal(ledger.setPaused(true), false);
  assert.equal(ledger.setPaused(true), false);
  assert.equal(ledger.segment, segment); assert.equal(ledger.nextAt, deadline);
  const element = { value: '', hidden: true };
  const snapshot = JSON.parse(writeReceiptSnapshot(element, { paused: true, metrics: ledger.metrics() }));
  assert.equal(snapshot.metrics.unique_rendered_hz, before.unique_rendered_hz);
  assert.equal(ledger.setPaused(false), true);
  assert.equal(ledger.segment, segment + 1); assert.equal(ledger.nextAt, null);
  assert.equal(ledger.metrics().unique_rendered_hz, null);
  assert.equal(ledger.setPaused(false), false); assert.equal(ledger.segment, segment + 1);
  assert.match(source, /function pause\(value\) \{\s*if \(value === paused\) return;/);
});

test('abnormal close exposes actual server code and reason rather than generic disconnected', () => {
  assert.equal(formatClose({ code: 1011, reason: 'RuntimeError: Unknown scene: no capacity frame published' }),
    'Stream closed (1011): RuntimeError: Unknown scene: no capacity frame published');
  assert.equal(formatClose({ code: 1006, reason: '' }), 'Stream closed (1006)');
});

test('client is the only 30Hz clock: early frames buffer, late frames never burst or skip', () => {
  const { stream, ledger, meta, bytes } = fixture();
  let now = 10;
  for (let i = 0; i < 646; i++) {
    ledger.plan({ ...meta, packet_id: i + 1, frame_index: i }, stream); ledger.binary(bytes);
    if (i) {
      now = ledger.nextAt - .001;
      assert.equal(ledger.render(() => assert.fail('early draw'), () => now), null);
      assert.equal(ledger.rendered, i); now += .001;
    }
    const ack = ledger.render(() => {}, () => now); ledger.commit({ ...ack, committed_frame_count: i + 1 });
  }
  assert.ok(Math.abs(ledger.metrics().unique_rendered_hz - 30) < 1e-8);
  assert.equal(ledger.rendered, 646); assert.equal(ledger.frames.length, 256);
  ledger.plan({ ...meta, packet_id: 647, frame_index: 646 }, stream); ledger.binary(bytes);
  now += 5000;
  const ack = ledger.render(() => {}, () => now);
  assert.equal(ledger.rendered, 647); assert.equal(ledger.nextAt, now + FRAME_MS);
  assert.equal(ledger.render(() => assert.fail('duplicate draw'), () => now), null);
  ledger.commit({ ...ack, committed_frame_count: 647 });
  ledger.resumeClock(); assert.equal(ledger.nextAt, null);
});

test('real WebSocket with fake presentation clock: early buffer, factual ACK, pause and close',
  { skip: !process.env.SCENE_V3_WS_TEST_URL, timeout: 10000 }, async t => {
    const base = process.env.SCENE_V3_WS_TEST_URL;
    const socket = new WebSocket(base.replace('http:', 'ws:') + '/api/scene-control-v3/ws');
    t.after(() => socket.close());
    socket.binaryType = 'arraybuffer';
    const queue = [], waiters = [];
    socket.onmessage = event => {
      const value = typeof event.data === 'string' ? JSON.parse(event.data) : event.data;
      if (waiters.length) waiters.shift()(value); else queue.push(value);
    };
    const receive = async () => {
      if (queue.length) return queue.shift();
      let timer;
      try { return await Promise.race([new Promise(resolve => waiters.push(resolve)),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('WebSocket receive timeout')), 3000); })]); }
      finally { clearTimeout(timer); }
    };
    const until = async type => {
      let value;
      do { value = await receive(); if (value.type === 'error') throw new Error(value.message); } while (value.type !== type);
      return value;
    };
    const send = value => socket.send(JSON.stringify(value));
    const ready = await until('ready'), ledger = new FrameLedger(ready.session_id, ready.benchmark_role), stream = new CommandStream(ready.session_id);
    send(stream.next([1, 0, 0], 0)); send({ type: 'pause', session_id: ready.session_id, paused: false });
    ledger.plan(await until('frame'), stream); ledger.binary(await receive());
    assert.equal(ledger.committed, 0);
    const first = ledger.render(() => {}, () => 0); send(first);
    ledger.commit(await until('committed'));
    ledger.plan(await until('frame'), stream); ledger.binary(await receive());
    assert.equal(ledger.pending.meta.frame_index, 1);
    assert.equal(ledger.render(() => assert.fail('early draw'), () => FRAME_MS - .001), null);
    let server = await (await fetch(base + '/api/scene-control-v3/diagnostics')).json();
    assert.equal(server.sessions[0].committed_frame_count, 1); assert.equal(server.sessions[0].pending_packet_id, 2);
    send({ type: 'pause', session_id: ready.session_id, paused: true }); await until('pause_applied');
    assert.equal(ledger.rendered, 1);
    send({ type: 'pause', session_id: ready.session_id, paused: false }); await until('pause_applied');
    const second = ledger.render(() => {}, () => FRAME_MS); send(second); ledger.commit(await until('committed'));
    ledger.plan(await until('frame'), stream); ledger.binary(await receive());
    assert.equal(ledger.pending.meta.frame_index, 2); assert.equal(ledger.committed, 2);
    const closed = new Promise(resolve => socket.addEventListener('close', resolve, { once: true })); socket.close(); await closed;
    server = await (await fetch(base + '/api/scene-control-v3/diagnostics')).json();
    assert.equal(server.sessions.length, 0);
  });
