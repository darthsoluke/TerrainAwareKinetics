import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('./app.js', import.meta.url), 'utf8');
const { SCHEMA, FRAME_MS, CommandStream, PlaybackLedger, ViewerDiagnostics, decodeFrameBlock, movementIntent } =
  await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const STRIDE = 54171;

function metadata(count = 4, values = {}) {
  return { type: 'plan', schema: SCHEMA, packet_id: 1, scene_revision: 3, settings_revision: 0,
    start_frame: 0, end_frame: count, frame_count: count, fps: 30, encoding: 'float32-le',
    geometry_format: 'soma-full-vertices-v1', vertex_count: 18056, frame_stride_floats: STRIDE,
    command_ids: Array(count / 4).fill(1), ...values };
}
function binary(count = 4) {
  const values = new Float32Array(count * STRIDE);
  for (let i = 0; i < count; i += 1) {
    values.set([i + 10, 2, -i], i * STRIDE);
    values[i * STRIDE + 3] = i + .25;
  }
  return values.buffer;
}
function ready(count = 4) {
  const ledger = new PlaybackLedger(3, 0, count);
  ledger.plan(metadata(count), 1);
  ledger.binary(binary(count), 0);
  return ledger;
}
function play(ledger) {
  const shown = [];
  let ack;
  while (!ack) ack = ledger.tick(ledger.packet.nextAt, false, frame => shown.push(frame.root[0]));
  return { ack, shown };
}
function receipt(ack, committed) { return { ...ack, schema: SCHEMA, committed_frame: committed }; }

test('full18056 bytes decode without translating Root or vertices', () => {
  for (const count of [4, 8]) {
    const bytes = binary(count);
    assert.equal(bytes.byteLength, count === 4 ? 866736 : 1733472);
    const frames = decodeFrameBlock(bytes, metadata(count));
    assert.equal(frames.length, count);
    assert.deepEqual([...frames[2].root], [12, 2, -2]);
    assert.equal(frames[2].vertices.length, 18056 * 3);
    assert.equal(frames[2].vertices[0], 2.25);
  }
});

test('reject truncated, rig/proxy, wrong fps and non-finite geometry', () => {
  assert.throws(() => decodeFrameBlock(binary().slice(0, -4), metadata()));
  for (const override of [{ vertex_count: 30 }, { geometry_format: 'soma77-rig-v1' }, { fps: 60 }, { frame_count: 6 }]) {
    assert.throws(() => decodeFrameBlock(binary(), metadata(4, override)));
  }
  const bytes = binary(); new DataView(bytes).setFloat32(28, NaN, true);
  assert.throws(() => decodeFrameBlock(bytes, metadata()), /Non-finite/);
});

test('4/8 packets show every frame at 30Hz and ACK only after last successful render', () => {
  for (const count of [4, 8]) {
    const ledger = ready(count), times = [];
    let ack;
    for (let i = 1; i <= count; i += 1) {
      const now = ledger.packet.nextAt;
      assert.equal(ledger.tick(now - .001, false, () => assert.fail('early presentation')), null);
      ack = ledger.tick(now, false, () => times.push(now));
      if (i < count) assert.equal(ack, null);
    }
    assert.equal(ledger.displayed, count);
    assert.equal(ack.frame_count, count);
    times.slice(1).forEach((now, i) => assert.ok(Math.abs(now - times[i] - FRAME_MS) < 1e-9));
    assert.equal(ledger.tick(10000, false, () => assert.fail()), null);
    assert.equal(ledger.drained, false);
    ledger.played(receipt(ack, count));
    assert.equal(ledger.drained, true);
  }
});

test('adjacent packets keep a common 30Hz deadline without an extra packet-start delay', () => {
  const ledger = ready();
  const { ack } = play(ledger);
  const nextAt = ledger.nextPresentation;
  ledger.played(receipt(ack, 4));
  ledger.plan(metadata(4, { packet_id: 2, start_frame: 4, end_frame: 8 }), 1);
  ledger.binary(binary(), nextAt - 10);
  assert.equal(ledger.packet.nextAt, nextAt);
  assert.equal(ledger.tick(nextAt - 1, false, () => assert.fail()), null);
  ledger.tick(nextAt, false, () => {});
  assert.equal(ledger.displayed, 5);
});

test('a late clock advances one frame, never bursts, interpolates or skips', () => {
  const ledger = ready(), roots = [];
  ledger.tick(500, false, frame => roots.push(frame.root[0]));
  assert.deepEqual(roots, [10]);
  assert.ok(ledger.packet.nextAt >= 500 + FRAME_MS);
  ledger.tick(500, false, () => assert.fail());
  const { ack, shown } = play(ledger);
  assert.deepEqual([...roots, ...shown], [10, 11, 12, 13]);
  assert.equal(ack.frame_count, 4);
});

test('pause/hidden retain exact cursor; resume requires a new complete display slot', () => {
  const ledger = ready();
  ledger.tick(ledger.packet.nextAt, false, () => {});
  ledger.tick(5000, true, () => assert.fail());
  assert.equal(ledger.packet.index, 1);
  assert.equal(ledger.tick(9000, false, () => assert.fail()), null);
  assert.equal(ledger.packet.nextAt, 9000 + FRAME_MS);
  const { ack, shown } = play(ledger);
  assert.deepEqual(shown, [11, 12, 13]);
  assert.equal(ack.frame_count, 4);
});

test('settings revision changes keep the published packet original ACK identity', () => {
  const ledger = ready();
  ledger.settings(1, 8);
  const { ack } = play(ledger);
  assert.equal(ack.settings_revision, 0);
  assert.throws(() => ledger.played(receipt({ ...ack, settings_revision: 1 }, 4)));
  ledger.played(receipt(ack, 4));
  ledger.plan(metadata(8, { packet_id: 2, settings_revision: 1, start_frame: 4, end_frame: 12, command_ids: [2, 3] }), 3);
  ledger.binary(binary(8), 1000);
  const next = play(ledger);
  assert.equal(next.ack.settings_revision, 1);
  ledger.played(receipt(next.ack, 12));
  assert.equal(ledger.committed, 12);
});

test('reject stale scenes, unknown commands, duplicate plans and out-of-order frontiers', () => {
  for (const override of [{ scene_revision: 2 }, { settings_revision: 1 }, { start_frame: 1 }, { end_frame: 5 },
    { packet_id: 0 }, { command_ids: [2] }, { command_ids: [] }]) {
    assert.throws(() => new PlaybackLedger(3, 0, 4).plan(metadata(4, override), 1));
  }
  const ledger = ready();
  assert.throws(() => ledger.plan(metadata(), 1));
  assert.throws(() => ledger.binary(binary(), 0));
  assert.throws(() => ledger.played(receipt({ packet_id: 1 }, 4)));
  const { ack } = play(ledger);
  assert.throws(() => ledger.played(receipt({ ...ack, frame_count: 3 }, 4)));
  assert.throws(() => ledger.played(receipt(ack, 3)));
  ledger.played(receipt(ack, 4));
  assert.throws(() => ledger.played(receipt(ack, 4)));
  assert.throws(() => ledger.plan(metadata(4, { start_frame: 4, end_frame: 8 }), 1));
});

test('render failure cannot advance cursor or acknowledge an unpresented frame', () => {
  const ledger = ready();
  assert.throws(() => ledger.tick(FRAME_MS, false, () => { throw new Error('GPU failed'); }));
  assert.equal(ledger.packet.index, 0);
  assert.equal(ledger.displayed, 0);
  assert.equal(ledger.packet.ackSent, false);
});

test('commands deduplicate semantic repeats and remain independent of packet inference/ACK', () => {
  const stream = new CommandStream(), ledger = ready();
  assert.equal(stream.next([0, 0], 1, 3, 0).command_id, 1);
  for (let i = 0; i < 1000; i += 1) assert.equal(stream.next([0, 0], 1, 3, 0), null);
  assert.equal(stream.next([2, 0], 1, 3, 0).command_id, 2);
  assert.equal(stream.next([3.6, 0], 1, 3, 0).command_id, 3);
  assert.equal(ledger.packet.index, 0);
  assert.equal(stream.next([3.6, 0], 1, 3, 1).command_id, 4);
  assert.equal(stream.next([3.6, 0], 1, 4, 0).command_id, 5);
  stream.semantic = null;
  assert.equal(stream.next([3.6, 0], 1, 4, 0).command_id, 6);
});

test('WASD camera-relative direction, normalized diagonal, Shift clamp, stop heading', () => {
  const intent = keys => movementIntent(new Set(keys), [0, 5], 2, 3, .75);
  const vector = (actual, expected) => actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < 1e-12));
  vector(intent(['w']).velocity, [0, 2]);
  vector(intent(['s']).velocity, [0, -2]);
  vector(intent(['a']).velocity, [2, 0]);
  vector(intent(['d']).velocity, [-2, 0]);
  assert.ok(Math.abs(Math.hypot(...intent(['w', 'd']).velocity) - 2) < 1e-12);
  assert.equal(Math.hypot(...intent(['w', 'shift']).velocity), 3);
  assert.deepEqual(intent(['w', 's']), { velocity: [0, 0], heading: .75 });
  assert.equal(movementIntent(new Set(['w']), [0, 5], 2, 3, 0, .2).heading, .2);
});

const controls = { packet_frames: 8, lookahead_tokens: 12, dense_debug: false };
function diagnosticSession(log, now = 0) {
  log.received({ type: 'ready', scene_revision: 3, settings_revision: 0, settings: controls }, now);
}

test('diagnostics associate each four-frame segment, first render and ACK with original revisions', () => {
  const log = new ViewerDiagnostics(), stream = new CommandStream();
  diagnosticSession(log);
  log.sent(stream.next([1, 0], 0, 3, 0), 10, controls);
  log.sent(stream.next([2, 0], 0, 3, 0), 20, controls);
  const meta = metadata(8, { command_ids: [1, 2], compute_ms: 37.5 });
  log.received(meta, 30);
  for (let index = 0; index < 8; index += 1) log.presented(meta, index, 100 + index * FRAME_MS);
  log.received({ type: 'settings_applied', scene_revision: 3, settings_revision: 1,
    settings: { ...controls, packet_frames: 4 } }, 350);
  const ack = { type: 'played', packet_id: 1, frame_count: 8, scene_revision: 3, settings_revision: 0 };
  log.sent(ack, 360, { ...controls, packet_frames: 4 });
  log.received(receipt(ack, 8), 370);
  log.presented(meta, 0, 400);
  const snapshot = log.snapshot(), rows = snapshot.events;
  assert.equal(snapshot.latency_metric, 'command-associated first-render');
  const sends = rows.filter(row => row.type === 'command_sent');
  assert.equal(sends[0].first_render_at_ms, 100);
  assert.equal(sends[0].command_associated_first_render_latency_ms, 90);
  assert.equal(sends[1].first_render_at_ms, 100 + 4 * FRAME_MS);
  assert.equal(sends[1].command_associated_first_render_latency_ms, 100 + 4 * FRAME_MS - 20);
  const frames = rows.filter(row => row.type === 'frame_presented');
  assert.deepEqual(frames.slice(0, 8).map(row => row.command_id), [1, 1, 1, 1, 2, 2, 2, 2]);
  assert.deepEqual(frames.slice(0, 8).map(row => row.frame_index), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.equal(frames.filter(row => row.is_command_associated_first_render === true).length, 2);
  assert.equal(frames[0].command_sent_at_ms, 10);
  assert.equal(frames[4].command_sent_at_ms, 20);
  assert.equal(frames[8].is_command_associated_first_render, false);
  assert.equal(frames[8].settings_revision, 0);
  assert.equal(rows.find(row => row.type === 'plan_received').compute_ms, 37.5);
  assert.deepEqual(rows.filter(row => row.type.startsWith('ack_')).map(row =>
    [row.type, row.at_ms, row.packet_id, row.frame_count, row.settings_revision]),
  [['ack_sent', 360, 1, 8, 0], ['ack_received', 370, 1, 8, 0]]);
});

test('diagnostic snapshots are deep copies and the single event buffer never exceeds 500', () => {
  const log = new ViewerDiagnostics(), command = new CommandStream().next([1, 2], .5, 3, 0);
  diagnosticSession(log);
  log.sent(command, 10, controls);
  command.velocity_xz[0] = 999;
  const copy = log.snapshot();
  copy.events[0].settings.packet_frames = 999;
  copy.events[1].velocity_xz[0] = 999;
  copy.events.push({ type: 'injected' });
  assert.equal(log.snapshot().events[0].settings.packet_frames, 8);
  assert.equal(log.snapshot().events[1].velocity_xz[0], 1);
  assert.equal(log.snapshot().events.length, 2);
  for (let index = 0; index < 750; index += 1) {
    log.presented(metadata(), index % 4, 20 + index);
    assert.ok(log.snapshot().events.length <= 500);
  }
  const tail = log.snapshot().events;
  assert.equal(tail.length, 500);
  assert.equal(tail.at(-1).command_sent_at_ms, null);
  assert.equal(tail.at(-1).is_command_associated_first_render, null);
  assert.equal(tail.at(-1).command_associated_first_render_latency_ms, null);
});

test('diagnostic command association cannot cross a reconnected session', () => {
  const log = new ViewerDiagnostics(), command = new CommandStream().next([1, 0], 0, 3, 0);
  diagnosticSession(log);
  log.sent(command, 10, controls);
  log.presented(metadata(), 0, 30);
  diagnosticSession(log, 100);
  log.presented(metadata(), 0, 110);
  assert.equal(log.snapshot().events.at(-1).command_sent_at_ms, null);
  log.sent(command, 120, controls);
  log.presented(metadata(), 1, 160);
  assert.equal(log.snapshot().events.at(-1).session_id, 2);
  assert.equal(log.snapshot().events.at(-1).command_associated_first_render_latency_ms, 40);
});

test('observed empty waits coalesce and exclude normal cadence, paused and hidden checks', () => {
  const log = new ViewerDiagnostics(), ledger = ready();
  const { ack } = play(ledger), deadline = ledger.nextPresentation;
  log.observePlayback(ledger, false, deadline - 1);
  assert.equal(log.snapshot().events.length, 0);
  log.observePlayback(ledger, false, deadline);
  log.observePlayback(ledger, false, deadline + 10);
  const waiting = log.snapshot().events[0];
  assert.equal(waiting.reason, 'ack_receipt');
  assert.equal(waiting.checks, 2);
  assert.equal(waiting.duration_ms, null);
  log.observePlayback(ledger, true, deadline + 20);
  log.observePlayback(ledger, true, deadline + 1000);
  assert.equal(log.snapshot().events.length, 1);
  assert.equal(log.snapshot().events[0].duration_ms, 20);
  ledger.played(receipt(ack, 4));
  log.observePlayback(ledger, false, deadline + 1100);
  ledger.plan(metadata(4, { packet_id: 2, start_frame: 4, end_frame: 8 }), 1);
  ledger.binary(binary(), deadline + 1110);
  log.observePlayback(ledger, false, deadline + 1110);
  assert.equal(log.snapshot().events[1].reason, 'packet');
  assert.equal(log.snapshot().events[1].duration_ms, 10);
  assert.equal(ledger.packet.index, 0);
});

test('diagnostics do not change native frame order, 4/8 playback deadlines or ACK identity', () => {
  for (const count of [4, 8]) {
    const run = instrumented => {
      const ledger = ready(count), log = new ViewerDiagnostics(), frames = [];
      let ack;
      while (!ack) {
        const now = ledger.packet.nextAt;
        if (instrumented) log.observePlayback(ledger, false, now);
        ack = ledger.tick(now, false, (frame, meta, index) => {
          frames.push({ root: [...frame.root], index, at: now });
          if (instrumented) log.presented(meta, index, now);
        });
      }
      return { frames, ack, displayed: ledger.displayed, nextPresentation: ledger.nextPresentation };
    };
    assert.deepEqual(run(true), run(false));
  }
  const ledger = ready(), log = new ViewerDiagnostics();
  const failedRender = () => { throw new Error('render failed'); };
  assert.throws(() => ledger.tick(0, false, (frame, meta, index) => {
    failedRender();
    log.presented(meta, index, 0);
  }));
  assert.equal(log.snapshot().events.length, 0);
  assert.equal(ledger.packet.index, 0);
});

test('empty wait segments keep their exact revision and ACK-versus-packet phase', () => {
  const log = new ViewerDiagnostics(), ledger = ready();
  const { ack } = play(ledger), deadline = ledger.nextPresentation;
  log.observePlayback(ledger, false, deadline);
  ledger.played(receipt(ack, 4));
  log.observePlayback(ledger, false, deadline + 5);
  ledger.settings(1, 8);
  log.observePlayback(ledger, false, deadline + 15);
  log.observePlayback(ledger, true, deadline + 25);
  assert.deepEqual(log.snapshot().events.map(row => [row.reason, row.settings_revision, row.duration_ms]),
    [['ack_receipt', 0, 5], ['packet', 0, 10], ['packet', 1, 10]]);
});

test('observing commands does not add semantic duplicates', () => {
  const log = new ViewerDiagnostics(), stream = new CommandStream();
  for (let index = 0; index < 1000; index += 1) {
    const command = stream.next([1, 0], 0, 3, 0);
    if (command) log.sent(command, index, controls);
  }
  assert.equal(stream.id, 1);
  assert.equal(log.snapshot().events.length, 1);
});

test('DOM diagnostics export serializes only on request and keeps a detached snapshot', () => {
  const log = new ViewerDiagnostics(), output = { value: '', hidden: true };
  diagnosticSession(log);
  log.presented(metadata(), 0, 10);
  assert.deepEqual(output, { value: '', hidden: true });
  log.exportToTextarea(output);
  assert.equal(output.hidden, false);
  const exported = output.value;
  assert.equal(JSON.parse(exported).events.at(-1).at_ms, 10);
  log.presented(metadata(), 1, 20);
  assert.equal(output.value, exported);
  log.exportToTextarea(output);
  assert.equal(JSON.parse(output.value).events.at(-1).at_ms, 20);
  output.value = '{}';
  assert.equal(log.snapshot().events.length, 3);
});
