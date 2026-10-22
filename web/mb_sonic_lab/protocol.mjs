export const SCHEMA = 'motionbricks.mb_sonic.lab.v1';
export const DISPLAY_MS = 1000 / 30;
export const percentile = (values, p) => values.length
  ? [...values].sort((a, b) => a - b)[Math.ceil(values.length * p) - 1] : null;

export function runtimeStatus(status, active, hasReference) {
  if (status.startsWith('completed')) return 'Completed';
  if (['runtime_failure', 'terminal_failure'].includes(status)) return 'Failed';
  if (status === 'actual_history_warmup') return 'Warming Up';
  if (status.startsWith('awaiting_explicit_start')) return active ? 'Planning reference' : 'Ready to Start';
  if (active) return hasReference ? 'Running' : 'Planning reference';
  return status;
}

export class GamepadLatch {
  constructor() { this.reset(); }
  reset() { this.armed = false; this.previous = { velocity: [0, 0], yaw: 0 }; }
  sample(velocity, yaw) {
    if (!velocity.some(value => value !== 0) && yaw === 0) this.armed = true;
    if (!this.armed) return { velocity: [0, 0], yaw: 0 };
    const changed = (value, previous) => value === 0 || Math.abs(value-previous) >= .015 ? value : previous;
    this.previous = { velocity: velocity.map((value, i) => changed(value, this.previous.velocity[i])),
      yaw: changed(yaw, this.previous.yaw) };
    return this.previous;
  }
}

export function standardGamepadState(pads) {
  const pad = Array.from(pads).find(value => value?.connected && value.mapping === 'standard');
  if (!pad) return null;
  const [x, y, yaw, pitch] = pad.axes;
  const deadzone = .12;
  const radius = Math.hypot(x, y);
  const magnitude = Math.max(0, (Math.min(radius, 1) - deadzone) / (1 - deadzone));
  const scale = radius > 0 ? magnitude / radius : 0;
  const turn = Math.sign(yaw) * Math.max(0, (Math.abs(yaw) - deadzone) / (1 - deadzone));
  return { id: pad.id, index: pad.index, raw_axes: [x, y, yaw, pitch],
    velocity: [x * scale + 0, -y * scale + 0], yaw: turn + 0 };
}

export class SimulatedXbox {
  constructor() { this.started = null; this.stopped = null; this.samples = []; this.phase = 'Ready'; }
  start(at) { this.started = at; this.stopped = null; this.samples = []; this.phase = 'Neutral'; }
  stop(at, reason) {
    if (this.started !== null && this.stopped === null) this.stopped = { at, reason };
  }
  sample(at) {
    const t = this.started === null ? 0 : (at - this.started) / 1000;
    let axes = [0, 0, 0, 0], phase = 'Ready';
    if (this.started !== null && this.stopped === null) {
      if (t < 1) phase = 'Neutral';
      else if (t < 4) { phase = 'Forward ramp'; axes[1] = -.7 * (t - 1) / 3; }
      else if (t < 9) { phase = 'Forward'; axes[1] = -.7; }
      else if (t < 14) { phase = 'Forward + turn'; axes = [0, -.7, .35, 0]; }
      else if (t < 19) { phase = 'Right strafe'; axes[0] = .65; }
      else if (t < 23) { phase = 'Reverse strafe'; axes[0] = -.65; }
      else if (t < 28) { phase = 'Backward'; axes[1] = .65; }
      else if (t < 33) { phase = 'Backward + turn'; axes = [0, .55, -.35, 0]; }
      else if (t < 43) {
        phase = 'Analog circle'; const angle = (t - 33) * 2 * Math.PI / 10;
        axes = [.6 * Math.sin(angle), -.6 * Math.cos(angle), .2 * Math.sin(angle), 0];
      } else if (t < 48) { phase = 'Speed ramp'; axes[1] = -.25 - .4 * (t - 43) / 5; }
      else if (t < 54) { phase = 'Rapid strafe reversals'; axes[0] = Math.floor(t - 48) % 2 ? -.55 : .55; }
      else if (t < 58) { phase = 'Turn right'; axes[2] = .35; }
      else if (t < 62) { phase = 'Turn left'; axes[2] = -.35; }
      else if (t < 67) { phase = 'Diagonal + turn'; axes = [.45, -.45, .25, 0]; }
      else if (t < 75) phase = 'Stop';
      else if (t < 80) { phase = 'Neutral sensor noise'; axes = [.06 * Math.sin(t * 17), .06 * Math.cos(t * 19), .07 * Math.sin(t * 23), 0]; }
      else if (t < 83) phase = 'Final neutral';
      else { phase = 'Complete'; this.stopped = { at, reason: 'trace_complete' }; }
      this.samples.push({ client_ms: at, time_s: t, phase, axes: [...axes] });
    } else if (this.stopped !== null) phase = this.stopped.reason === 'trace_complete' ? 'Complete' : 'Stopped';
    this.phase = phase;
    return [{ id: 'Simulated Xbox - scripted axes', index: 0, connected: true, mapping: 'standard',
      timestamp: at, axes, buttons: [] }];
  }
  receipt() {
    return { schema: 'motionbricks.simulated_xbox.v1', source: 'scripted_Gamepad_API_shape_not_physical_device',
      trace_duration_s: 83, started_client_ms: this.started, stopped: this.stopped,
      phase: this.phase, raw_samples: this.samples };
  }
}

export class LatestIntentMailbox {
  constructor(stream) { this.stream = stream; this.pending = null; }
  clear() { this.pending = null; }
  offer(velocity, yaw, at) {
    const semantic = JSON.stringify([velocity, yaw]);
    if (this.pending?.semantic !== semantic) this.pending = { velocity: [...velocity], yaw, at, semantic };
  }
  flush(send, writable) {
    if (!writable || !this.pending) return null;
    const { velocity, yaw, at } = this.pending;
    this.pending = null;
    const command = this.stream.next(velocity, yaw, at);
    if (command) send(command);
    return command;
  }
}

export function worldIntent([x, y], [forwardX, forwardY], speed, yawRate) {
  const norm = Math.hypot(x, y);
  if (norm > 1) { x /= norm; y /= norm; }
  return { velocity_xy_mps: [speed * (forwardX * y + forwardY * x), speed * (forwardY * y - forwardX * x)]
    .map(v => Math.round(v * 10000) / 10000 + 0), yaw_rate_radps: Math.round(yawRate * 10000) / 10000 + 0 };
}

export function behindRootCamera(root, heading, distance, elevation) {
  if (!Number.isFinite(heading) || !Number.isFinite(distance) || distance <= 0
      || !Number.isFinite(elevation) || !Array.isArray(root) || root.length !== 3
      || !root.every(Number.isFinite)) throw new Error('Actual Root heading/camera dimensions unavailable');
  return { position: [root[0] - Math.cos(heading) * distance,
    root[1] - Math.sin(heading) * distance, root[2] + elevation], target: [...root] };
}

export class ActualRootFollower {
  constructor() { this.previous = null; }
  reset() { this.previous = null; }
  delta(actualRoot, enabled) {
    const previous = this.previous;
    this.previous = [...actualRoot];
    return enabled && previous !== null ? actualRoot.map((value, axis) => value - previous[axis]) : [0, 0, 0];
  }
}

export function primaryReferencePresentation(config) {
  const parents = config.reference.parents;
  if (!Array.isArray(parents) || parents.length !== 14
      || parents.some((parent, i) => !Number.isInteger(parent) || parent < -1 || parent >= i)
      || parents.filter(parent => parent === -1).length !== 1) {
    throw new Error('Primary reference must use the ordered adapted G1 14-body hierarchy');
  }
  const official = config.control_mode === 'official_reference_read_only';
  if (config.control_mode === 'fixed_reference_read_only') {
    return { label: 'Continuous MB reference', mode: 'Continuous Reference / Read Only',
      stage: 'Actual MuJoCo G1 and the complete continuous MB reference' };
  }
  return { label: official ? 'Official reference' : 'MB -> G1 reference',
    mode: official ? 'Official Reference / Read Only' : 'MB -> G1 Reference',
    stage: official ? 'Actual MuJoCo G1 and official CSV reference'
      : 'Actual MuJoCo G1 and adapted MotionBricks G1 reference' };
}

function distribution(values) {
  const finite = values.filter(Number.isFinite);
  return { samples: finite.length, p50_ms: percentile(finite, .5), p95_ms: percentile(finite, .95),
    p99_ms: percentile(finite, .99), max_ms: finite.length ? Math.max(...finite) : null };
}

export class BrowserDiagnostics {
  constructor() { this.events = []; this.counts = {}; this.acks = new Map(); this.capacity = 4096; }
  record(kind, at, context, active, detail = {}) {
    this.events.push({ kind, client_ms: at, active, ...context, ...detail });
    this.counts[kind] = (this.counts[kind] ?? 0) + 1;
    if (this.events.length > this.capacity) this.events.shift();
  }
  ackSent(ack, at, context, active) {
    this.acks.set(ack.packet_id, at);
    if (this.acks.size > 256) this.acks.delete(this.acks.keys().next().value);
    this.record('ack_sent', at, context, active, { packet_id: ack.packet_id, frame_id: ack.frame_id });
  }
  ackReceived(packet, at, context, active) {
    const sent = this.acks.get(packet.packet_id);
    this.record('ack_receipt', at, context, active, { packet_id: packet.packet_id,
      ack_roundtrip_ms: sent === undefined ? null : at - sent,
      server_publish_to_ack_ms: packet.server_publish_to_ack_ms });
    this.acks.delete(packet.packet_id);
  }
  metrics() {
    const active = this.events.filter(row => row.active);
    const rows = kind => active.filter(row => row.kind === kind);
    const intervals = kind => {
      const selected = rows(kind);
      return selected.slice(1).map((row, i) => row.client_ms - selected[i].client_ms);
    };
    return {
      active_raf_callbacks: rows('raf').length,
      active_frame_packets: rows('frame_received').length,
      active_ack_sent: rows('ack_sent').length,
      active_paint_only_submissions: rows('paint_only').length,
      active_raf_delta: distribution(rows('raf').filter(row => row.previous_was_active).map(row => row.delta_ms)),
      active_packet_interval: distribution(intervals('frame_received')),
      active_ack_interval: distribution(intervals('ack_sent')),
      active_frame_render_interval: distribution(intervals('frame_rendered')),
      active_render_cpu: distribution(rows('frame_rendered').map(row => row.render_ms)),
      active_receive_to_render: distribution(rows('frame_rendered').map(row => row.receive_to_render_ms)),
      active_ack_roundtrip: distribution(rows('ack_receipt').map(row => row.ack_roundtrip_ms)),
      active_server_publish_to_ack: distribution(rows('ack_receipt').map(row => row.server_publish_to_ack_ms)),
      active_server_ack_to_publish: distribution(rows('frame_received').map(row => row.server_ack_to_publish_ms)),
      active_snapshot_wall: distribution(rows('frame_received').map(row => row.server_snapshot_wall_ms)),
      active_hidden_samples: active.filter(row => row.document_hidden).length,
      active_unfocused_samples: active.filter(row => !row.document_has_focus).length,
      active_visibility_states: [...new Set(active.map(row => row.visibility_state))],
    };
  }
  receipt() {
    return { capacity: this.capacity, lifetime_event_counts: { ...this.counts },
      metrics: this.metrics(), events: this.events,
      context_scope: 'actual document.visibilityState/hidden/hasFocus; native host occlusion not inferred' };
  }
}

export class IntentStream {
  constructor(session, enabled = true) {
    this.session = session; this.enabled = enabled; this.id = 0; this.semantic = null; this.rows = new Map();
  }
  bindScene(config) {
    if (config.intent_enabled && config.episode_id === undefined) throw new Error('Runtime omitted the intent episode ID');
    this.scope = { scene_id: config.scene.manifest_id, scene_manifest_sha256: config.scene.manifest_sha256,
      episode_id: config.episode_id };
    this.semantic = null;
  }
  next(velocity, yawRate, inputAt) {
    if (!this.enabled) return null;
    if (!this.scope) throw new Error('Intent must bind the canonical scene and episode');
    const semantic = JSON.stringify([velocity, yawRate]);
    if (semantic === this.semantic) return null;
    this.semantic = semantic;
    const id = ++this.id;
    const value = { ...this.scope, type: 'intent', session_id: this.session, command_id: id, command_seq: id,
      client_input_ms: inputAt, velocity_xy_mps: [...velocity], yaw_rate_radps: yawRate, input_frame: 'world_z_up' };
    this.rows.set(value.command_id, { ...value, first_visual_ms: null, body_onset_display_ms: null,
      body_onset_sim_time_s: null, body_onset_server_latency_ms: null, server_received_ns: null });
    if (this.rows.size > 256) this.rows.delete(this.rows.keys().next().value);
    return value;
  }
}

export class DisplayLedger {
  constructor(session, intents) {
    this.session = session; this.intents = intents; this.pending = null; this.lastPacket = 0;
    this.nextAt = 0; this.frames = []; this.chain = []; this.rendered = 0; this.dropped = 0;
    this.sceneRevision = 0;
  }
  changeScene(revision) { this.sceneRevision = revision; this.pending = null; this.nextAt = 0; }
  offer(packet) {
    if (packet.scene_revision < this.sceneRevision) return;
    if (packet.schema !== SCHEMA || packet.session_id !== this.session || packet.scene_revision !== this.sceneRevision
        || packet.packet_id <= this.lastPacket) throw new Error('Invalid physics display packet');
    if (this.pending) this.dropped++;
    this.pending = packet; this.lastPacket = packet.packet_id;
  }
  render(draw, clock) {
    const start = clock();
    if (!this.pending || start + .25 < this.nextAt) return null;
    const packet = this.pending, state = packet.state;
    draw(state);
    const end = clock();
    this.nextAt = start - this.nextAt < DISPLAY_MS ? this.nextAt + DISPLAY_MS : start + DISPLAY_MS;
    this.pending = null; this.rendered++;
    const command = state.command;
    if (command?.session_id === this.session) {
      const row = this.intents.rows.get(command.command_id);
      if (row && row.scene_id === state.scene_manifest_id && row.episode_id === state.episode_id
          && row.scene_manifest_sha256 === state.scene_manifest_sha256
          && row.client_input_ms === command.client_input_ms && row.first_visual_ms === null) {
        row.first_visual_ms = end;
      }
      const onset = this.intents.rows.get(state.actual.body_onset_command_id);
      if (onset && onset.scene_id === state.scene_manifest_id && onset.episode_id === state.episode_id
          && onset.scene_manifest_sha256 === state.scene_manifest_sha256
          && state.actual.body_onset_session_id === this.session && onset.body_onset_display_ms === null) {
        onset.body_onset_display_ms = end;
        onset.body_onset_sim_time_s = state.actual.body_onset_sim_time_s;
        if (Number.isFinite(state.actual.body_onset_server_ns) && Number.isFinite(onset.server_received_ns)) {
          onset.body_onset_server_latency_ms = (state.actual.body_onset_server_ns - onset.server_received_ns) / 1e6;
        }
      }
    }
    const ack = { type: 'rendered', session_id: this.session, packet_id: packet.packet_id,
      frame_id: state.frame_id, command_id: command?.command_id ?? null,
      client_render_started_ms: start, client_render_completed_ms: end };
    this.frames.push({ ...ack, sim_time_s: state.sim_time_s, timing: state.timing,
      packet_received_client_ms: packet.client_received_ms ?? null,
      receive_to_render_ms: Number.isFinite(packet.client_received_ms) ? start - packet.client_received_ms : null,
      published_server_ns: packet.published_server_ns,
      consumed_intent: state.command,
      scene_revision: this.sceneRevision, status: state.status,
      physics_steps: state.timing.physics_steps,
      actual_body_onset_command_id: state.actual.body_onset_command_id,
      reference_version: state.reference?.version ?? null, reference_time_s: state.reference?.time_s ?? null });
    if (this.frames.length > 256) this.frames.shift();
    if (Number.isFinite(state.timing.chain_ms)) {
      this.chain.push(state.timing.chain_ms);
      if (this.chain.length > 256) this.chain.shift();
    }
    return ack;
  }
  metrics() {
    const inputs = [...this.intents.rows.values()];
    const span = this.frames.length > 1
      ? this.frames.at(-1).client_render_completed_ms - this.frames[0].client_render_completed_ms : 0;
    const active = this.frames.filter(row => row.scene_revision === this.sceneRevision && row.physics_steps > 0);
    const activeSpan = active.length > 1
      ? active.at(-1).client_render_completed_ms - active[0].client_render_completed_ms : 0;
    return { unique_display_hz: span > 0 ? (this.frames.length - 1) * 1000 / span : null,
      active_unique_display_hz: activeSpan > 0 ? (active.length - 1) * 1000 / activeSpan : null,
      active_window_ms: activeSpan, active_rendered_samples: active.length,
      active_first_sim_time_s: active[0]?.sim_time_s ?? null,
      active_last_sim_time_s: active.at(-1)?.sim_time_s ?? null,
      chain_p99_ms: percentile(this.chain, .99),
      first_visual_p95_ms: percentile(inputs.filter(x => x.first_visual_ms !== null)
        .map(x => x.first_visual_ms - x.client_input_ms), .95),
      body_onset_display_p95_ms: percentile(inputs.filter(x => x.body_onset_display_ms !== null)
        .map(x => x.body_onset_display_ms - x.client_input_ms), .95),
      body_onset_server_p95_ms: percentile(inputs.map(x => x.body_onset_server_latency_ms).filter(Number.isFinite), .95),
      unique_rendered: this.rendered, dropped_display_samples: this.dropped,
      retained_chain_samples: this.chain.length,
      retained_first_visual_samples: inputs.filter(x => x.first_visual_ms !== null).length,
      retained_body_onset_samples: inputs.filter(x => x.body_onset_display_ms !== null).length };
  }
}
