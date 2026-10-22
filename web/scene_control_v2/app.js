export const SCHEMA = 'motionbricks.scene_control_v2.runtime.v1';
export const FRAME_MS = 1000 / 30;
const API = '/api/scene-control-v2';
const VERTICES = 18056;
const STRIDE = 3 + VERTICES * 3;

export function decodeFrameBlock(buffer, meta) {
  if (meta.encoding !== 'float32-le' || meta.geometry_format !== 'soma-full-vertices-v1'
      || meta.vertex_count !== VERTICES || meta.frame_stride_floats !== STRIDE
      || ![4, 8].includes(meta.frame_count) || meta.fps !== 30
      || buffer.byteLength !== meta.frame_count * STRIDE * 4) {
    throw new Error('Invalid full18056 frame block');
  }
  const values = new Float32Array(buffer.byteLength / 4);
  const view = new DataView(buffer);
  for (let i = 0; i < values.length; i += 1) values[i] = view.getFloat32(i * 4, true);
  if (!values.every(Number.isFinite)) throw new Error('Non-finite SOMA geometry');
  return Array.from({ length: meta.frame_count }, (_, index) => {
    const offset = index * STRIDE;
    return { root: values.subarray(offset, offset + 3), vertices: values.subarray(offset + 3, offset + STRIDE) };
  });
}

export class CommandStream {
  constructor() { this.id = 0; this.semantic = null; }
  next(velocity, heading, sceneRevision, settingsRevision) {
    const semantic = JSON.stringify([velocity, heading, sceneRevision, settingsRevision]);
    if (semantic === this.semantic) return null;
    this.semantic = semantic;
    return { type: 'command', command_id: ++this.id, scene_revision: sceneRevision,
      settings_revision: settingsRevision, velocity_xz: [...velocity], heading };
  }
}

export class ViewerDiagnostics {
  #events = [];
  #sequence = 0;
  #session = 0;
  #emptyWait = null;
  #record(type, at, fields) {
    const event = { sequence: ++this.#sequence, session_id: this.#session, type, at_ms: at, ...structuredClone(fields) };
    this.#events.push(event);
    if (this.#events.length > 500 && this.#events.shift() === this.#emptyWait) this.#emptyWait = null;
    return event;
  }
  #packet(meta) {
    return { packet_id: meta.packet_id, scene_revision: meta.scene_revision, settings_revision: meta.settings_revision,
      packet_frames: meta.frame_count, start_frame: meta.start_frame, end_frame: meta.end_frame,
      compute_ms: meta.compute_ms, command_ids: meta.command_ids };
  }
  #finishWait(now) {
    if (!this.#emptyWait) return;
    this.#emptyWait.ended_at_ms = now;
    this.#emptyWait.duration_ms = now - this.#emptyWait.at_ms;
    this.#emptyWait = null;
  }
  sent(message, now, settings) {
    const { type, ...fields } = message;
    if (type === 'command') Object.assign(fields, { packet_frames: settings.packet_frames,
      lookahead_tokens: settings.lookahead_tokens, first_render_at_ms: null,
      command_associated_first_render_latency_ms: null });
    this.#record(type === 'played' ? 'ack_sent' : `${type}_sent`, now, fields);
  }
  received(message, now) {
    if (message.type === 'ready') { this.#finishWait(now); this.#session += 1; }
    if (message.type === 'plan') { this.#record('plan_received', now, this.#packet(message)); return; }
    const keys = ['scene_revision', 'settings_revision', 'command_id', 'packet_id', 'frame_count',
      'committed_frame', 'settings', 'paused', 'duplicate'];
    const fields = Object.fromEntries(keys.filter(key => Object.hasOwn(message, key)).map(key => [key, message[key]]));
    this.#record(message.type === 'ready' ? 'session_ready' : message.type === 'played' ? 'ack_received' : message.type, now, fields);
  }
  presented(meta, frameIndex, now) {
    const commandId = meta.command_ids[Math.floor(frameIndex / 4)];
    const command = this.#events.findLast(event => event.type === 'command_sent' && event.session_id === this.#session
      && event.command_id === commandId && event.scene_revision === meta.scene_revision && event.settings_revision === meta.settings_revision);
    const first = command ? command.first_render_at_ms === null : null;
    if (first) {
      command.first_render_at_ms = now;
      command.command_associated_first_render_latency_ms = now - command.at_ms;
      command.first_render_packet_id = meta.packet_id;
      command.first_render_frame_index = frameIndex;
    }
    this.#record('frame_presented', now, { ...this.#packet(meta), frame_index: frameIndex,
      absolute_frame: meta.start_frame + frameIndex, command_id: commandId, command_sent_at_ms: command ? command.at_ms : null,
      is_command_associated_first_render: first,
      command_associated_first_render_latency_ms: first ? command.command_associated_first_render_latency_ms : null });
  }
  observePlayback(ledger, blocked, now) {
    const empty = !blocked && (!ledger.packet || ledger.packet.ackSent)
      && (ledger.nextPresentation === null || now >= ledger.nextPresentation);
    if (!empty) { this.#finishWait(now); return; }
    const context = {
      scene_revision: ledger.sceneRevision, settings_revision: ledger.settingsRevision,
      waiting_packet_id: ledger.packet ? ledger.packet.meta.packet_id : null,
      reason: ledger.packet?.ackSent ? 'ack_receipt' : 'packet' };
    if (this.#emptyWait && Object.entries(context).some(([key, value]) => this.#emptyWait[key] !== value)) this.#finishWait(now);
    if (!this.#emptyWait) this.#emptyWait = this.#record('empty_wait', now, { ...context, checks: 0,
      last_checked_at_ms: now, ended_at_ms: null, duration_ms: null });
    this.#emptyWait.checks += 1;
    this.#emptyWait.last_checked_at_ms = now;
  }
  snapshot() {
    return structuredClone({ schema: 'motionbricks.scene_control_v2.viewer_diagnostics.v1',
      clock: 'performance.now()', time_origin_ms: performance.timeOrigin, event_limit: 500, session_id: this.#session,
      latency_metric: 'command-associated first-render',
      render_boundary: 'successful renderer.render return; not pose-response or input-to-photon', events: this.#events });
  }
  exportToTextarea(output) {
    output.value = JSON.stringify(this.snapshot());
    output.hidden = false;
  }
}

export function movementIntent(pressed, cameraForward, speed, maxSpeed, heading, headingOffset = 0) {
  const forward = Number(pressed.has('w')) - Number(pressed.has('s'));
  const right = Number(pressed.has('d')) - Number(pressed.has('a'));
  if (!forward && !right) return { velocity: [0, 0], heading };
  const x = cameraForward[0] * forward - cameraForward[1] * right;
  const z = cameraForward[1] * forward + cameraForward[0] * right;
  const magnitude = Math.hypot(x, z);
  const targetSpeed = Math.min(maxSpeed, speed * (pressed.has('shift') ? 1.8 : 1));
  return { velocity: [x / magnitude * targetSpeed, z / magnitude * targetSpeed],
    heading: Math.atan2(x, z) + headingOffset };
}

export class PlaybackLedger {
  constructor(sceneRevision, settingsRevision, packetFrames) {
    this.sceneRevision = sceneRevision;
    this.settingsRevision = settingsRevision;
    this.packetFrames = packetFrames;
    this.metadata = null;
    this.packet = null;
    this.lastPacket = 0;
    this.committed = 0;
    this.displayed = 0;
    this.nextPresentation = null;
  }
  settings(revision, packetFrames) {
    if (revision !== this.settingsRevision + 1 || ![4, 8].includes(packetFrames)) throw new Error('Invalid settings revision');
    this.settingsRevision = revision;
    this.packetFrames = packetFrames;
  }
  plan(meta, lastCommandId) {
    if (this.metadata || this.packet || meta.schema !== SCHEMA
        || !Number.isSafeInteger(meta.packet_id) || meta.packet_id <= this.lastPacket
        || meta.scene_revision !== this.sceneRevision || meta.settings_revision !== this.settingsRevision
        || meta.start_frame !== this.committed || meta.end_frame !== this.committed + meta.frame_count
        || meta.frame_count !== this.packetFrames || !Array.isArray(meta.command_ids)
        || meta.command_ids.length !== meta.frame_count / 4
        || meta.command_ids.some(id => !Number.isSafeInteger(id) || id < 1 || id > lastCommandId)) {
      throw new Error('Packet frontier, command or revision mismatch');
    }
    this.metadata = meta;
  }
  binary(buffer, now) {
    if (!this.metadata) throw new Error('Binary block without plan metadata');
    const frames = decodeFrameBlock(buffer, this.metadata);
    this.packet = { meta: this.metadata, frames, index: 0,
      nextAt: this.nextPresentation === null ? now : Math.max(now, this.nextPresentation), ackSent: false };
    this.lastPacket = this.metadata.packet_id;
    this.metadata = null;
  }
  tick(now, blocked, present) {
    const packet = this.packet;
    if (!packet || packet.ackSent) return null;
    if (blocked) { packet.nextAt = null; this.nextPresentation = null; return null; }
    if (packet.nextAt === null) { packet.nextAt = now + FRAME_MS; return null; }
    if (now < packet.nextAt) return null;
    if (packet.index < packet.frames.length) {
      // Exactly one native frame is rendered per tick; late ticks never catch up.
      present(packet.frames[packet.index], packet.meta, packet.index);
      packet.index += 1;
      this.displayed += 1;
      const next = packet.nextAt + FRAME_MS;
      packet.nextAt = next - now < 20 ? now + FRAME_MS : next;
      this.nextPresentation = packet.nextAt;
      if (packet.index < packet.frames.length) return null;
    }
    packet.ackSent = true;
    return { type: 'played', packet_id: packet.meta.packet_id, frame_count: packet.frames.length,
      scene_revision: packet.meta.scene_revision, settings_revision: packet.meta.settings_revision };
  }
  played(receipt) {
    const packet = this.packet;
    if (!packet?.ackSent || receipt.schema !== SCHEMA || receipt.packet_id !== packet.meta.packet_id
        || receipt.frame_count !== packet.frames.length || receipt.scene_revision !== packet.meta.scene_revision
        || receipt.settings_revision !== packet.meta.settings_revision || receipt.committed_frame !== packet.meta.end_frame) {
      throw new Error('Playback receipt does not match the displayed packet');
    }
    this.committed = receipt.committed_frame;
    this.packet = null;
  }
  get drained() { return !this.metadata && !this.packet; }
}

async function startViewer() {
  const diagnostics = new ViewerDiagnostics();
  Object.defineProperty(window, 'sceneControlV2Diagnostics', { value: () => diagnostics.snapshot() });
  const THREE = await import('three');
  const { OrbitControls } = await import('/vendor/OrbitControls.js');
  const ui = Object.fromEntries(['stage', 'scene', 'status', 'pause', 'camera', 'reconnect', 'settings',
    'settings-toggle', 'settings-close', 'speed', 'speed-value', 'settings-status', 'error', 'frame',
    'command', 'ack', 'revision', 'packet', 'hz', 'diagnostics-export', 'diagnostics-snapshot'].map(id => [id, document.getElementById(id)]));
  const state = { socket: null, ready: false, config: null, settings: null, settingsPending: false,
    sceneRevision: 0, settingsRevision: 0, ledger: null, commands: new CommandStream(), desiredSettings: null,
    paused: false, pauseRequest: null, transition: null, connecting: false, failed: false,
    keyboard: new Map(), touches: new Map(), heading: 0, root: new THREE.Vector3(), following: false, orbiting: false,
    body: null, terrain: null, lastPresented: null, intervals: [], renderDirty: true };
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#dce7ea');
  const camera = new THREE.PerspectiveCamera(45, 1, .05, 1600);
  const renderer = new THREE.WebGLRenderer({ canvas: ui.stage, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  const orbit = new OrbitControls(camera, ui.stage);
  orbit.enableDamping = false;
  orbit.enablePan = true;
  orbit.minDistance = 1.3;
  orbit.maxDistance = 160;
  orbit.maxPolarAngle = Math.PI * .48;
  scene.add(new THREE.HemisphereLight('#f8ffff', '#6d7b80', 2.4));
  const light = new THREE.DirectionalLight('#fff4e1', 3.0);
  scene.add(light, light.target);

  function icons() { window.lucide.createIcons(); }
  function status(text) { ui.status.textContent = text; }
  function send(message) {
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) throw new Error('Session is disconnected');
    const encoded = JSON.stringify(message);
    const sentAt = performance.now();
    state.socket.send(encoded);
    diagnostics.sent(message, sentAt, state.settings);
  }
  function fail(error) {
    state.failed = true;
    state.ready = false;
    state.connecting = false;
    state.transition = null;
    state.settingsPending = false;
    state.pauseRequest = null;
    ui.error.textContent = error.message;
    ui.error.hidden = false;
    status('Stopped');
    state.keyboard.clear();
    state.touches.clear();
    if (state.socket?.readyState === WebSocket.OPEN) state.socket.close(1000, 'Viewer stopped');
    refreshControls();
  }
  function protect(action) {
    return (...args) => { try { action(...args); } catch (error) { fail(error); } };
  }
  function refreshControls() {
    const busy = !state.ready || Boolean(state.transition) || state.settingsPending;
    ui.scene.disabled = busy || state.pauseRequest !== null;
    ui.pause.disabled = busy || state.pauseRequest !== null;
    ui.reconnect.disabled = state.connecting || Boolean(state.transition) || state.settingsPending || state.pauseRequest !== null;
    document.querySelectorAll('input[name="packet"], input[name="preview"]').forEach(input => { input.disabled = busy; });
    document.querySelectorAll('[data-key]').forEach(button => { button.disabled = !state.ready || Boolean(state.transition); });
    ui.pause.title = ui.pause.ariaLabel = state.paused ? 'Resume playback' : 'Pause playback';
    ui.pause.innerHTML = `<i data-lucide="${state.paused ? 'play' : 'pause'}"></i>`;
    ui.revision.textContent = `${state.sceneRevision} / ${state.settingsRevision}`;
    icons();
  }
  function setSettingsControls() {
    document.querySelector(`input[name="packet"][value="${state.settings.packet_frames}"]`).checked = true;
    document.querySelector(`input[name="preview"][value="${state.settings.lookahead_tokens}"]`).checked = true;
    ui['settings-status'].textContent = `Revision ${state.settingsRevision}`;
  }
  function dispose(mesh) {
    if (!mesh) return;
    scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
  }
  function mesh(vertices, faces, material, colors) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setIndex(faces.flat());
    if (colors) geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    return new THREE.Mesh(geometry, material);
  }
  function installConfiguration(config) {
    const contract = config.scene_control_v2;
    if (contract?.schema !== SCHEMA || config.fps !== 30 || config.vertex_count !== VERTICES
        || config.initial_vertices.length !== VERTICES * 3 || config.initial_root.length !== 3
        || !config.terrain_scene?.mesh || !Array.isArray(config.faces) || config.faces.flat().length < 3) {
      throw new Error('Expected current-scene SOMA full18056 and original PFNN mesh');
    }
    state.config = config;
    state.sceneRevision = contract.scene_revision;
    state.settingsRevision = contract.settings_revision;
    state.settings = contract.settings;
    if (!state.desiredSettings) state.desiredSettings = contract.settings;
    state.ledger = new PlaybackLedger(state.sceneRevision, state.settingsRevision, state.settings.packet_frames);
    dispose(state.body);
    dispose(state.terrain);
    state.body = mesh(config.initial_vertices, config.faces,
      new THREE.MeshStandardMaterial({ color: '#bbced6', metalness: .12, roughness: .55, side: THREE.DoubleSide }));
    state.body.geometry.attributes.position.setUsage(THREE.DynamicDrawUsage);
    state.body.frustumCulled = false;
    state.body.name = 'SOMA authoritative full18056';
    const terrain = config.terrain_scene.mesh;
    state.terrain = mesh(terrain.vertices, terrain.faces, new THREE.MeshStandardMaterial({
      color: terrain.colors ? '#ffffff' : '#9caeaa', vertexColors: Boolean(terrain.colors),
      roughness: .95, side: THREE.DoubleSide }), terrain.colors);
    state.terrain.name = 'Original PFNN static terrain';
    scene.add(state.body, state.terrain);
    state.root.fromArray(config.initial_root);
    light.position.copy(state.root).add(new THREE.Vector3(-5, 10, -4));
    light.target.position.copy(state.root);
    const catalog = config.map_catalog;
    ui.scene.replaceChildren(...catalog.entries.map(entry => {
      const option = document.createElement('option');
      option.value = entry.id;
      option.textContent = `${entry.id} / ${entry.label}`;
      return option;
    }));
    ui.scene.value = catalog.current;
    ui.speed.max = String(config.runtime_controls.max_speed_mps);
    state.lastPresented = null;
    state.intervals = [];
    ui.frame.textContent = '0'; ui.ack.textContent = '-'; ui.packet.textContent = '-'; ui.hz.textContent = '0.0';
    setSettingsControls();
    centerCamera();
    state.renderDirty = true;
  }
  function centerCamera() {
    orbit.target.copy(state.root).add(new THREE.Vector3(0, .85, 0));
    camera.position.copy(orbit.target).add(new THREE.Vector3(-Math.sin(state.heading) * 5, 2.4, -Math.cos(state.heading) * 5));
    state.following = true;
    orbit.update();
    state.following = false;
    state.renderDirty = true;
  }
  function present(frame, meta, frameIndex) {
    // Vertices are already world-space. Never apply Root to the body or terrain.
    state.body.geometry.attributes.position.array.set(frame.vertices);
    state.body.geometry.attributes.position.needsUpdate = true;
    state.body.geometry.computeVertexNormals();
    const nextRoot = new THREE.Vector3().fromArray(frame.root);
    const delta = nextRoot.clone().sub(state.root);
    camera.position.add(delta);
    orbit.target.add(delta);
    state.root.copy(nextRoot);
    state.following = true;
    orbit.update();
    state.following = false;
    light.position.copy(state.root).add(new THREE.Vector3(-5, 10, -4));
    light.target.position.copy(state.root);
    renderer.render(scene, camera);
    state.renderDirty = false;
    const now = performance.now();
    diagnostics.presented(meta, frameIndex, now);
    if (state.lastPresented !== null) state.intervals.push(now - state.lastPresented);
    if (state.intervals.length > 30) state.intervals.shift();
    state.lastPresented = now;
    if (state.intervals.length) ui.hz.textContent = (1000 * state.intervals.length / state.intervals.reduce((a, b) => a + b)).toFixed(1);
  }
  function commandFromInput() {
    const pressed = new Set([...state.keyboard.values(), ...state.touches.values()]);
    document.querySelectorAll('[data-key]').forEach(button => {
      button.classList.toggle('is-down', pressed.has(button.dataset.key));
      button.setAttribute('aria-pressed', String(pressed.has(button.dataset.key)));
    });
    if (!state.ready || state.transition || state.settingsPending) return;
    const direction = orbit.target.clone().sub(camera.position);
    const intent = movementIntent(pressed, [direction.x, direction.z], Number(ui.speed.value),
      Number(ui.speed.max), state.heading, state.config.entry_command?.heading_offset_rad ?? 0);
    state.heading = intent.heading;
    const command = state.commands.next(intent.velocity, state.heading, state.sceneRevision, state.settingsRevision);
    if (command) { send(command); ui.command.textContent = String(command.command_id); }
  }
  function clearInput() { state.keyboard.clear(); state.touches.clear(); commandFromInput(); }
  function requestPause(paused) {
    state.pauseRequest = paused;
    send({ type: 'pause', scene_revision: state.sceneRevision, settings_revision: state.settingsRevision, paused });
    refreshControls();
  }
  function togglePause() {
    state.paused = !state.paused;
    if (state.paused) state.ledger.tick(performance.now(), true, present);
    clearInput();
    requestPause(state.paused);
    state.lastPresented = null; state.intervals = []; ui.hz.textContent = '0.0';
    status(state.paused ? 'Paused' : 'Running');
  }
  function beginTransition(kind, mapId = null) {
    state.transition = { kind, mapId, paused: state.paused, pauseReady: false, requesting: false };
    clearInput();
    requestPause(true);
    status('Draining published frames');
    refreshControls();
  }
  async function finishTransition() {
    const change = state.transition;
    if (!change || !change.pauseReady || change.requesting || !state.ledger.drained) return;
    change.requesting = true;
    status(change.kind === 'map' ? 'Loading scene' : 'Reconnecting');
    try {
      let config;
      if (change.kind === 'map') {
        const response = await fetch(`${API}/maps/select`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ map_id: change.mapId }) });
        if (!response.ok) throw new Error(`Scene selection: ${response.status} ${await response.text()}`);
        config = await response.json();
      }
      state.paused = change.paused;
      await connect(config);
    } catch (error) { state.transition = null; fail(error); }
  }
  async function connect(config) {
    if (state.connecting) return;
    state.connecting = true;
    state.ready = false;
    const old = state.socket;
    state.socket = null;
    if (old) old.close(1000, 'New session');
    state.failed = false;
    state.settingsPending = false;
    state.pauseRequest = null;
    state.keyboard.clear(); state.touches.clear();
    ui.error.hidden = true;
    status('Connecting');
    refreshControls();
    try {
      if (!config) {
        const response = await fetch(`${API}/config`, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Configuration: HTTP ${response.status}`);
        config = await response.json();
      }
      installConfiguration(config);
      const url = new URL(config.scene_control_v2.websocket_path, location.href);
      url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(url);
      state.socket = socket;
      socket.binaryType = 'arraybuffer';
      socket.onmessage = protect(event => {
        if (state.socket !== socket || state.failed) return;
        if (typeof event.data === 'string') handleMessage(JSON.parse(event.data));
        else { state.ledger.binary(event.data, performance.now()); scheduleClock(); }
      });
      socket.onerror = () => { if (state.socket === socket) fail(new Error('WebSocket connection failed')); };
      socket.onclose = () => {
        if (state.socket !== socket || state.transition?.requesting || state.failed) return;
        state.connecting = false;
        fail(new Error('Session closed. Reconnect to start a new session.'));
      };
    } catch (error) { state.transition = null; fail(error); }
    finally { state.connecting = false; refreshControls(); }
  }
  function handleMessage(message) {
    const receivedAt = performance.now();
    if (message.schema !== SCHEMA) throw new Error('Unexpected runtime schema');
    switch (message.type) {
      case 'ready':
        if (state.ready || message.scene_revision !== state.sceneRevision || message.settings_revision !== state.settingsRevision
            || message.vertex_count !== VERTICES || !Number.isFinite(message.heading)) throw new Error('Ready/config mismatch');
        diagnostics.received(message, receivedAt);
        state.settings = message.settings;
        state.heading = message.heading;
        state.commands.semantic = null;
        state.transition = null;
        state.ready = true;
        centerCamera();
        setSettingsControls();
        if (state.paused) requestPause(true);
        if (state.desiredSettings.packet_frames !== state.settings.packet_frames
            || state.desiredSettings.lookahead_tokens !== state.settings.lookahead_tokens) {
          requestSettings(state.desiredSettings.packet_frames, state.desiredSettings.lookahead_tokens);
        } else commandFromInput();
        status(state.paused ? 'Paused' : 'Running');
        refreshControls();
        break;
      case 'plan':
        state.ledger.plan(message, state.commands.id);
        diagnostics.received(message, receivedAt);
        ui.packet.textContent = `${message.packet_id} / ${message.frame_count}`;
        break;
      case 'played':
        state.ledger.played(message);
        diagnostics.received(message, receivedAt);
        ui.ack.textContent = `${message.packet_id} / ${message.committed_frame}`;
        void finishTransition();
        break;
      case 'command_received':
        if (message.command_id > state.commands.id) throw new Error('Unknown command receipt');
        diagnostics.received(message, receivedAt);
        break;
      case 'settings_applied':
        if (!state.settingsPending || message.scene_revision !== state.sceneRevision) throw new Error('Unexpected settings receipt');
        state.ledger.settings(message.settings_revision, message.settings.packet_frames);
        diagnostics.received(message, receivedAt);
        state.settingsRevision = message.settings_revision;
        state.settings = message.settings;
        state.settingsPending = false;
        setSettingsControls(); refreshControls(); commandFromInput();
        break;
      case 'pause_applied':
        if (message.scene_revision !== state.sceneRevision || message.settings_revision !== state.settingsRevision
            || message.paused !== state.pauseRequest) throw new Error('Unexpected pause receipt');
        diagnostics.received(message, receivedAt);
        state.pauseRequest = null;
        if (state.transition) { state.transition.pauseReady = true; void finishTransition(); }
        refreshControls();
        break;
      case 'scene_changed':
        if (!state.transition?.requesting) throw new Error('Scene changed in another session. Reconnect.');
        diagnostics.received(message, receivedAt);
        break;
      case 'error': case 'runtime_failure': throw new Error(message.message);
      default: throw new Error(`Unknown runtime message: ${message.type}`);
    }
  }
  function updateSettings() {
    const packet = Number(document.querySelector('input[name="packet"]:checked').value);
    const preview = Number(document.querySelector('input[name="preview"]:checked').value);
    if (packet === state.settings.packet_frames && preview === state.settings.lookahead_tokens) return;
    requestSettings(packet, preview);
  }
  function requestSettings(packet, preview) {
    state.desiredSettings = { packet_frames: packet, lookahead_tokens: preview, dense_debug: state.settings.dense_debug };
    state.settingsPending = true;
    send({ type: 'settings', scene_revision: state.sceneRevision, settings_revision: state.settingsRevision + 1,
      settings: { packet_frames: packet, lookahead_tokens: preview, dense_debug: state.settings.dense_debug } });
    ui['settings-status'].textContent = `Applying revision ${state.settingsRevision + 1}`;
    refreshControls();
  }
  function showSettings(show) {
    ui.settings.hidden = !show;
    document.body.classList.toggle('settings-open', show);
    ui['settings-toggle'].setAttribute('aria-expanded', String(show));
  }
  ui['settings-toggle'].onclick = () => showSettings(ui.settings.hidden);
  ui['settings-close'].onclick = () => showSettings(false);
  ui['diagnostics-export'].onclick = () => diagnostics.exportToTextarea(ui['diagnostics-snapshot']);
  ui.camera.onclick = protect(() => { centerCamera(); commandFromInput(); });
  ui.pause.onclick = protect(togglePause);
  ui.reconnect.onclick = protect(() => {
    if (state.ready) beginTransition('reconnect');
    else { state.transition = null; void connect(); }
  });
  ui.scene.onchange = protect(() => beginTransition('map', ui.scene.value));
  document.querySelectorAll('input[name="packet"], input[name="preview"]').forEach(input => { input.onchange = protect(updateSettings); });
  ui.speed.oninput = protect(() => { ui['speed-value'].value = `${Number(ui.speed.value).toFixed(1)} m/s`; commandFromInput(); });
  const keys = { KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd', ArrowUp: 'w', ArrowLeft: 'a', ArrowDown: 's', ArrowRight: 'd', ShiftLeft: 'shift', ShiftRight: 'shift' };
  document.addEventListener('keydown', protect(event => {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(event.target.tagName) || event.metaKey || event.ctrlKey || event.altKey) return;
    const key = keys[event.code];
    if (!key) return;
    event.preventDefault();
    if (event.repeat) return;
    state.keyboard.set(event.code, key); commandFromInput();
  }));
  document.addEventListener('keyup', protect(event => {
    const key = keys[event.code];
    if (key) { state.keyboard.delete(event.code); commandFromInput(); }
  }));
  document.querySelectorAll('[data-key]').forEach(button => {
    button.onpointerdown = protect(event => {
      event.preventDefault(); button.setPointerCapture(event.pointerId);
      state.touches.set(event.pointerId, button.dataset.key); commandFromInput();
    });
    button.onpointerup = button.onpointercancel = button.onlostpointercapture = protect(event => {
      state.touches.delete(event.pointerId); commandFromInput();
    });
  });
  window.addEventListener('blur', protect(clearInput));
  document.addEventListener('visibilitychange', protect(() => {
    if (document.hidden) { clearInput(); state.ledger?.tick(performance.now(), true, present); }
    state.lastPresented = null; state.intervals = [];
  }));
  orbit.addEventListener('change', protect(() => {
    state.renderDirty = true;
    if (state.orbiting && !state.following) commandFromInput();
  }));
  orbit.addEventListener('start', () => { state.orbiting = true; });
  orbit.addEventListener('end', protect(() => { state.orbiting = false; commandFromInput(); }));
  function resize() {
    const width = ui.stage.clientWidth, height = ui.stage.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / height; camera.updateProjectionMatrix(); state.renderDirty = true;
  }
  window.addEventListener('resize', resize); resize();
  new ResizeObserver(([entry]) => {
    document.documentElement.style.setProperty('--toolbar-height', `${entry.target.getBoundingClientRect().height}px`);
  }).observe(document.querySelector('.toolbar'));
  renderer.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); fail(new Error('WebGL context lost')); });
  let clockTimer;
  function scheduleClock() {
    clearTimeout(clockTimer);
    const packet = state.ledger?.packet;
    const delay = packet && !packet.ackSent && packet.nextAt !== null
      ? Math.max(0, packet.nextAt - performance.now()) : FRAME_MS;
    clockTimer = setTimeout(play, delay);
  }
  function play() {
    try {
      if (state.ready && !state.failed) {
        const blocked = document.hidden || (state.paused && !state.transition?.pauseReady);
        const now = performance.now();
        diagnostics.observePlayback(state.ledger, blocked, now);
        const acknowledgement = state.ledger.tick(now, blocked, present);
        ui.frame.textContent = String(state.ledger.displayed);
        if (acknowledgement) send(acknowledgement);
      }
    } catch (error) { fail(error); }
    scheduleClock();
  }
  function render() {
    try {
      if (state.renderDirty && !document.hidden) { renderer.render(scene, camera); state.renderDirty = false; }
    } catch (error) { fail(error); }
    requestAnimationFrame(render);
  }
  icons(); refreshControls(); scheduleClock(); requestAnimationFrame(render);
  await connect();
}

if (typeof document !== 'undefined') startViewer().catch(error => {
  document.getElementById('error').textContent = error.message;
  document.getElementById('error').hidden = false;
  document.getElementById('status').textContent = 'Stopped';
});
