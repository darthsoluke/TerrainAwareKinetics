import { SCHEMA, IntentStream, DisplayLedger, worldIntent, BrowserDiagnostics, primaryReferencePresentation,
  ActualRootFollower, LatestIntentMailbox, GamepadLatch, standardGamepadState, SimulatedXbox, runtimeStatus, behindRootCamera } from './protocol.mjs';
import * as THREE from '/vendor/three.module.js';

export function canResetEpisode(intentEnabled, active, changing, connected) {
  return connected && !changing && (!active || intentEnabled === true);
}

export function isIntentEditor(target) {
  return target.id !== 'yaw' && Boolean(target.closest('input,select,textarea,button'));
}

if (new URL(import.meta.url).searchParams.has('mesh-worker')) {
  self.onmessage = async ({ data }) => {
    const { asset_id, vertex_count, index_count } = data;
    try {
      const response = await fetch(`/api/mb-sonic/assets/${asset_id}`);
      if (!response.ok) throw new Error(`Mesh asset ${asset_id}: ${response.status}`);
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength !== (vertex_count * 3 + index_count) * 4) throw new Error('Packed mesh length mismatch');
      const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))]
        .map(value => value.toString(16).padStart(2, '0')).join('');
      if (digest !== asset_id) throw new Error('Packed mesh content hash mismatch');
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buffer, 0, vertex_count * 3), 3));
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, vertex_count * 12, index_count), 1));
      geometry.computeVertexNormals(); geometry.computeBoundingSphere();
      const normals = geometry.attributes.normal.array;
      const sphere = { center: geometry.boundingSphere.center.toArray(), radius: geometry.boundingSphere.radius };
      self.postMessage({ asset_id, buffer, normals, sphere }, [buffer, normals.buffer]);
    } catch (error) { self.postMessage({ asset_id, error: error.message }); }
  };
} else {

const { OrbitControls } = await import('/vendor/OrbitControls.js');
const API = '/api/mb-sonic';
const el = id => document.getElementById(id);
const failures = [];
let socket, intents, inputMailbox, ledger, config, lastState, serverReceipt;
let sceneChanging = false;
let episodeBegan = false;
let startRequested = false;
let startCommandPending = false;
let startIntentId = null;
let episodeActive = false;
let previousRafMs = null, previousRafActive = false;
const browserDiagnostics = new BrowserDiagnostics();
const browserContext = () => ({ visibility_state: document.visibilityState,
  document_hidden: document.hidden, document_has_focus: document.hasFocus() });
const keys = new Set();
let pointerVelocity = [0, 0], gamepadVelocity = [0, 0], gamepadYaw = 0, manualYaw = 0;
let gamepadState = null, gamepadIdentity = null;
const YAW_LIMIT = 1.5;
let cameraDirty = true;
let cameraNeedsAlignment = false;
const meshAssets = new Map();
const cameraFollower = new ActualRootFollower();
const gamepadLatch = new GamepadLatch();
const simulatedGamepad = new URL(location.href).searchParams.get('input') === 'simulated-xbox' ? new SimulatedXbox() : null;
let assetWorkerFailure = null;
const assetWorker = new Worker(new URL('./app.js?mesh-worker', import.meta.url), { type: 'module' });
assetWorker.onmessage = ({ data }) => {
  const entry = meshAssets.get(data.asset_id);
  if (data.error) { entry.reject(new Error(data.error)); return; }
  const { vertex_count, index_count } = entry.description;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(data.buffer, 0, vertex_count * 3), 3));
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(data.buffer, vertex_count * 12, index_count), 1));
  geometry.setAttribute('normal', new THREE.BufferAttribute(data.normals, 3));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3().fromArray(data.sphere.center), data.sphere.radius);
  geometry.userData.asset_id = data.asset_id; entry.geometry = geometry; entry.resolve();
};
assetWorker.onerror = event => {
  assetWorkerFailure = new Error(event.message);
  for (const entry of meshAssets.values()) entry.reject(assetWorkerFailure);
  error(assetWorkerFailure);
};

function error(value) {
  simulatedGamepad?.stop(performance.now(), 'runtime_or_display_failure');
  const message = value instanceof Error ? value.message : String(value);
  failures.push({ message, client_ms: performance.now() });
  browserDiagnostics.record('failure', performance.now(), browserContext(), episodeActive, { message });
  episodeActive = false;
  startCommandPending = false;
  startIntentId = null;
  inputMailbox?.clear();
  setDriveEnabled(false);
  el('start').disabled = true;
  el('reset').disabled = sceneChanging || socket?.readyState !== WebSocket.OPEN;
  el('error').hidden = false;
  el('error').textContent = message;
  el('status').textContent = 'Failed'; el('status').dataset.state = 'failed';
}

const number = (value, unit = '', digits = 2) => Number.isFinite(value) ? `${value.toFixed(digits)}${unit}` : 'Unknown';
const send = value => {
  if (socket?.readyState === WebSocket.OPEN) {
    const at = performance.now(); socket.send(JSON.stringify(value)); return at;
  }
  return null;
};

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
const renderer = new THREE.WebGLRenderer({ canvas: el('stage'), antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;
const world = new THREE.Scene(); world.background = new THREE.Color('#dce5e0');
world.add(new THREE.HemisphereLight(0xffffff, 0x718c7b, 2.4));
const sunlight = new THREE.DirectionalLight(0xffffff, 2.8); sunlight.position.set(3, -4, 8); world.add(sunlight);
const camera = new THREE.PerspectiveCamera(45, 1, .02, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.addEventListener('change', () => { cameraDirty = true; });
controls.enableDamping = false; controls.minPolarAngle = .15; controls.maxPolarAngle = Math.PI * .49;
const sceneGroup = new THREE.Group(), robotGroup = new THREE.Group();
world.add(sceneGroup, robotGroup);
const robotGeoms = new Map(), sceneGeoms = new Map();
let referenceLines, referencePoints, contactPoints;
const referenceMaterial = new THREE.LineBasicMaterial({ color: 0xc24e8d, transparent: true, opacity: .8, depthTest: false });

function disposeGroup(group) {
  for (const child of [...group.children]) {
    if (!child.geometry.userData.asset_id) child.geometry.dispose();
    child.material.dispose(); group.remove(child);
  }
}

function geometry(value) {
  switch (value.kind) {
    case 'mesh': {
      return meshAssets.get(value.mesh.asset_id).geometry;
    }
    case 'box': return new THREE.BoxGeometry(...value.size.map(x => x * 2));
    case 'sphere': return new THREE.SphereGeometry(value.size[0], 20, 12);
    case 'capsule': {
      const result = new THREE.CapsuleGeometry(value.size[0], value.size[1] * 2, 6, 14);
      result.rotateX(Math.PI / 2); return result;
    }
    case 'cylinder': {
      const result = new THREE.CylinderGeometry(value.size[0], value.size[0], value.size[1] * 2, 20);
      result.rotateX(Math.PI / 2); return result;
    }
    default: throw new Error(`Unsupported canonical geometry: ${value.kind}`);
  }
}

function pose(mesh, value) {
  mesh.position.fromArray(value.position);
  mesh.quaternion.fromArray(value.quaternion_xyzw);
}

function mesh(value) {
  const rgba = value.rgba;
  return new THREE.Mesh(geometry(value), new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgba[0], rgba[1], rgba[2]), roughness: .72,
    opacity: rgba[3], transparent: rgba[3] < 1, side: THREE.DoubleSide,
  }));
}

async function loadAsset(description) {
  if (assetWorkerFailure) throw assetWorkerFailure;
  const { asset_id } = description;
  if (!meshAssets.has(asset_id)) {
    const entry = { description };
    entry.promise = new Promise((resolve, reject) => { entry.resolve = resolve; entry.reject = reject; });
    meshAssets.set(asset_id, entry);
    assetWorker.postMessage(description);
  }
  await meshAssets.get(asset_id).promise;
}

async function configure(value) {
  if (value.schema !== SCHEMA || value.world_up !== 'z' || value.optimizer_updates !== 0) {
    throw new Error('Expected zero-update G1 runtime in world Z-up coordinates');
  }
  const referencePresentation = primaryReferencePresentation(value);
  el('status').textContent = 'Loading geometry';
  await Promise.all([...value.scene.geometries, ...value.robot.geometries]
    .filter(x => x.kind === 'mesh').map(x => loadAsset(x.mesh)));
  config = value;
  const trackerName = config.tracker_backend === 'protomotions' ? 'ProtoMotions' : 'SONIC';
  el('tracker-title').textContent = `MB + ${trackerName}`;
  el('rates-label').textContent = `Physics / ${trackerName}`;
  el('clock-summary').textContent = `${config.display_hz} Hz display · ${config.controller_hz} Hz ${trackerName} · ${Math.round(1 / config.physics_dt_s)} Hz physics`;
  cameraNeedsAlignment = config.intent_enabled;
  cameraFollower.reset();
  episodeBegan = false;
  episodeActive = false;
  startRequested = false;
  startCommandPending = false;
  inputMailbox?.clear(); gamepadLatch.reset();
  keys.clear(); pointerVelocity = [0, 0]; gamepadVelocity = [0, 0];
  gamepadYaw = 0; manualYaw = 0; el('yaw').value = '0'; el('yaw-value').textContent = '0.00 rad/s';
  if (intents) { intents.enabled = config.intent_enabled; intents.bindScene(config); }
  el('mode').textContent = referencePresentation.mode + (
    config.hardware_execution_profile === 'development_stable_gpu_power' ? ' | Dev GPU' : '');
  el('reference-label').textContent = referencePresentation.label;
  el('stage').setAttribute('aria-label', referencePresentation.stage);
  setDriveEnabled(false);
  el('drive').title = config.intent_enabled ? 'Velocity intent' : config.intent_unavailable_reason;
  el('stick').style.transform = 'translate(0, 0)';
  el('start').disabled = true;
  el('reset').disabled = true;
  disposeGroup(sceneGroup); disposeGroup(robotGroup); sceneGeoms.clear(); robotGeoms.clear();
  const requiredAssets = new Set([...config.scene.geometries, ...config.robot.geometries]
    .filter(value => value.kind === 'mesh').map(value => value.mesh.asset_id));
  for (const [id, entry] of meshAssets) if (!requiredAssets.has(id)) {
    entry.geometry.dispose(); meshAssets.delete(id);
  }
  for (const value of config.scene.geometries) {
    const object = mesh(value); pose(object, value); sceneGroup.add(object); sceneGeoms.set(value.id, object);
  }
  for (const value of config.robot.geometries) {
    const object = mesh(value); object.visible = false; robotGroup.add(object); robotGeoms.set(value.id, object);
  }
  for (const object of [referenceLines, referencePoints, contactPoints]) {
    if (object) {
      world.remove(object); object.geometry.dispose();
      if (object.material !== referenceMaterial) object.material.dispose();
    }
  }
  const count = config.reference.parents.length;
  const lines = new THREE.BufferGeometry();
  lines.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(count * 6), 3));
  referenceLines = new THREE.LineSegments(lines, referenceMaterial); referenceLines.frustumCulled = false;
  const joints = new THREE.BufferGeometry();
  joints.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(count * 3), 3));
  referencePoints = new THREE.Points(joints, new THREE.PointsMaterial({ color: 0xc24e8d, size: .027, depthTest: false }));
  referencePoints.frustumCulled = false;
  const contacts = new THREE.BufferGeometry();
  contacts.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
  contactPoints = new THREE.Points(contacts, new THREE.PointsMaterial({ color: 0xc92732, size: .065, depthTest: false }));
  contactPoints.frustumCulled = false;
  world.add(referenceLines, referencePoints, contactPoints);
  const options = config.scene_catalog.map(scene => {
    const option = document.createElement('option'); option.value = scene.id;
    option.textContent = scene.available ? scene.label : `${scene.label} [unavailable]`;
    option.disabled = !scene.available; option.title = scene.unavailable_reason || scene.label;
    option.selected = scene.id === config.scene.manifest_id; return option;
  });
  el('scene').replaceChildren(...options); el('scene').value = config.scene.manifest_id;
  el('scene').disabled = false;
  el('manifest').textContent = config.scene.manifest_id;
  el('manifest-hash').textContent = config.scene.manifest_sha256;
  el('reference-version').textContent = config.reference.version;
  el('reference-source').textContent = config.reference.source;
  el('coverage').textContent = `${number(config.reference.coverage_start_s, 's')} - ${number(config.reference.coverage_end_s, 's')}`;
  el('rates').textContent = `${number(1 / config.physics_dt_s, '', 0)} / ${number(config.controller_hz, ' Hz', 0)}`;
  el('optimizer').textContent = config.optimizer_updates;
  el('robot-source').textContent = `${config.robot.name} | ${config.robot.source}`;
  camera.position.fromArray(config.initial_camera.position);
  controls.target.fromArray(config.initial_camera.target); controls.update();
  referenceLines.visible = referencePoints.visible = false;
  contactPoints.visible = false; lastState = null;
  sceneChanging = false;
  el('error').hidden = true;
  el('status').dataset.state = '';
  el('status').textContent = config.start_required ? 'Ready to Start' : 'Connected';
  el('start').disabled = true;
  el('reset').disabled = false;
  const title = config.intent_enabled ? 'G1 Xbox' : (
    config.control_mode === 'fixed_reference_read_only' ? 'G1 Continuous Reference' : 'G1 Official Reference');
  document.title = `${title} | MB + ${trackerName}`;
}

function setDriveEnabled(enabled) {
  el('drive').inert = !enabled;
  el('drive').setAttribute('aria-disabled', String(!enabled));
  el('joystick').tabIndex = enabled ? 0 : -1;
  el('joystick').setAttribute('aria-disabled', String(!enabled));
  el('speed').disabled = !enabled; el('yaw').disabled = !enabled;
}

function focusBehindActualRoot(actual) {
  const delta = new THREE.Vector3().subVectors(camera.position, controls.target);
  const view = behindRootCamera(actual.root_position, actual.root_heading_rad, Math.hypot(delta.x, delta.y), delta.z);
  camera.position.fromArray(view.position); controls.target.fromArray(view.target); controls.update();
}

function draw(state) {
  if (state.scene_manifest_id !== config.scene.manifest_id
      || state.scene_manifest_sha256 !== config.scene.manifest_sha256) {
    throw new Error('Physics state and displayed SceneManifest differ');
  }
  if (cameraNeedsAlignment) {
    focusBehindActualRoot(state.actual);
    cameraNeedsAlignment = false;
  }
  const cameraDelta = new THREE.Vector3().fromArray(cameraFollower.delta(state.actual.root_position, el('follow').checked));
  camera.position.add(cameraDelta); controls.target.add(cameraDelta); controls.update();
  for (const value of state.actual.geom_poses) {
    const object = robotGeoms.get(value.id); if (!object) throw new Error(`Unknown official G1 geom ${value.id}`);
    pose(object, value); object.visible = true;
  }
  // Moving manifest objects use the same actual MuJoCo world transforms.
  if (state.scene_geom_poses) for (const value of state.scene_geom_poses) pose(sceneGeoms.get(value.id), value);
  robotGroup.visible = el('actual-visible').checked;
  const reference = state.reference;
  referenceLines.visible = referencePoints.visible = reference !== null && el('reference-visible').checked;
  if (reference !== null) {
    if (reference.version !== config.reference.version) throw new Error('Reference version changed without configuration');
    const points = reference.joint_positions;
    const linePosition = referenceLines.geometry.attributes.position;
    const jointPosition = referencePoints.geometry.attributes.position;
    if (points.length !== config.reference.parents.length) throw new Error('Reference topology does not match runtime');
    points.forEach((point, index) => {
      jointPosition.setXYZ(index, ...point);
      const parent = config.reference.parents[index];
      linePosition.setXYZ(index * 2, ...point);
      linePosition.setXYZ(index * 2 + 1, ...(parent < 0 ? point : points[parent]));
    });
    linePosition.needsUpdate = true; jointPosition.needsUpdate = true;
  }
  contactPoints.geometry.setAttribute('position', new THREE.Float32BufferAttribute(state.contacts.flatMap(x => x.position), 3));
  contactPoints.visible = el('contacts-visible').checked;
  renderer.render(world, camera);
  lastState = state;
  el('status').textContent = runtimeStatus(state.status, episodeActive, reference !== null);
  el('start').disabled = !(config.start_required && !episodeBegan && !startRequested
    && state.status.startsWith('awaiting_explicit_start'));
  el('reset').disabled = !canResetEpisode(config.intent_enabled, episodeActive, sceneChanging,
    socket.readyState === WebSocket.OPEN);
  el('sim-time').textContent = number(state.sim_time_s, ' s');
  el('ticks').textContent = `${state.timing.physics_steps ?? 'Unknown'} / ${state.timing.controller_steps ?? 'Unknown'}`;
  el('rtf').textContent = number(state.timing.realtime_factor, 'x');
  el('tracking').textContent = number(state.timing.tracking_error_m, ' m', 3);
  el('heading-error').textContent = number(state.timing.heading_error_rad, ' rad', 3);
  el('consumed').textContent = state.command ? `#${state.command.command_id}` : 'None';
  el('reference-time').textContent = number(reference?.time_s, ' s');
  el('onset-sim').textContent = number(state.actual.body_onset_sim_time_s, ' s');
  el('contact-count').textContent = state.contacts.length;
  const rows = state.contacts.slice(0, 8).map(row => {
    const item = document.createElement('li');
    item.textContent = `${row.geom1} / ${row.geom2}: ${number(row.normal_force_n, ' N', 1)}`; return item;
  });
  el('contacts').replaceChildren(...rows);
  for (const failure of state.errors) {
    const message = typeof failure === 'string' ? failure : JSON.stringify(failure);
    if (!failures.some(row => row.message === message)) error(message);
  }
}

function measure() {
  const m = ledger.metrics();
  el('display-rate').textContent = number(m.active_unique_display_hz, ' Hz', 1);
  el('chain').textContent = number(m.chain_p99_ms, ' ms');
  el('visual').textContent = number(m.first_visual_p95_ms, ' ms');
  el('onset').textContent = number(m.body_onset_display_p95_ms, ' ms');
  el('onset-server').textContent = number(m.body_onset_server_p95_ms, ' ms');
}

function input(inputAt = performance.now()) {
  if (!intents || !config.intent_enabled || !episodeBegan || !episodeActive || sceneChanging
      || document.hidden || !document.hasFocus()) return;
  let x = Number(keys.has('KeyD')) - Number(keys.has('KeyA'));
  let y = Number(keys.has('KeyW')) - Number(keys.has('KeyS'));
  if (x === 0 && y === 0) [x, y] = pointerVelocity.some(v => v !== 0) ? pointerVelocity : gamepadVelocity;
  const norm = Math.hypot(x, y);
  if (norm > 1) { x /= norm; y /= norm; }
  const forward = camera.getWorldDirection(new THREE.Vector3()); forward.z = 0; forward.normalize();
  const speed = Number(el('speed').value);
  if (!Number.isFinite(speed) || speed < 0 || speed > 3) return;
  const keyYaw = Number(keys.has('KeyQ')) - Number(keys.has('KeyE'));
  const yawRate = keyYaw !== 0 ? keyYaw * YAW_LIMIT : manualYaw !== 0 ? manualYaw : -gamepadYaw * YAW_LIMIT;
  const intent = worldIntent([x, y], [forward.x, forward.y], speed, yawRate);
  const velocity = intent.velocity_xy_mps;
  inputMailbox.offer(velocity, intent.yaw_rate_radps, inputAt); flushIntent();
  el('intent').textContent = `${velocity.map(v => v.toFixed(2)).join(', ')} m/s`;
  el('yaw-value').textContent = `${intent.yaw_rate_radps.toFixed(2)} rad/s`;
  el('stick').style.transform = `translate(${x * 35}px, ${-y * 35}px)`;
  el('joystick').setAttribute('aria-valuenow', Math.round(Math.hypot(x, y) * 100));
}

function stopInput() {
  simulatedGamepad?.stop(performance.now(), 'operator_or_focus_stop');
  keys.clear(); pointerVelocity = [0, 0]; gamepadVelocity = [0, 0]; gamepadYaw = 0; manualYaw = 0;
  gamepadLatch.reset();
  el('yaw').value = '0'; el('yaw-value').textContent = '0.00 rad/s';
  el('intent').textContent = '0.00, 0.00 m/s'; el('stick').style.transform = 'translate(0, 0)';
  el('joystick').setAttribute('aria-valuenow', '0');
  if (config?.intent_enabled && episodeActive && !sceneChanging) {
    inputMailbox.offer([0, 0], 0, performance.now()); flushIntent(true);
  } else inputMailbox?.clear();
}

function flushIntent(urgentStop = false) {
  const command = inputMailbox?.flush(send, socket?.readyState === WebSocket.OPEN && (urgentStop || socket.bufferedAmount === 0));
  if (startCommandPending && startIntentId === null && command) startIntentId = command.command_id;
}

addEventListener('keydown', event => {
  if (!config?.intent_enabled || !episodeActive || !episodeBegan || sceneChanging
      || !['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE'].includes(event.code)
      || isIntentEditor(event.target)) return;
  event.preventDefault(); keys.add(event.code); input();
});
addEventListener('keyup', event => {
  if (!keys.delete(event.code)) return; event.preventDefault(); input();
});
addEventListener('blur', stopInput);
addEventListener('pagehide', stopInput);
document.addEventListener('focusin', event => {
  if (isIntentEditor(event.target)) stopInput();
});
for (const kind of ['focus', 'blur', 'pageshow', 'pagehide']) {
  addEventListener(kind, () => browserDiagnostics.record(kind, performance.now(), browserContext(), episodeActive));
}
document.addEventListener('visibilitychange', () => {
  browserDiagnostics.record('visibilitychange', performance.now(), browserContext(), episodeActive);
});
document.addEventListener('visibilitychange', () => { if (document.hidden) stopInput(); });
el('speed').addEventListener('input', () => input());
el('yaw').addEventListener('input', () => { manualYaw = -Number(el('yaw').value); input(); });
for (const event of ['pointerup', 'pointercancel', 'keyup', 'blur']) {
  el('yaw').addEventListener(event, () => { manualYaw = 0; el('yaw').value = '0'; input(); });
}
el('joystick').addEventListener('pointerdown', event => {
  if (!config?.intent_enabled || !episodeBegan || !episodeActive || sceneChanging) return;
  event.preventDefault();
  el('joystick').setPointerCapture(event.pointerId); updateStick(event);
});
function updateStick(event) {
  if (!config?.intent_enabled) return;
  const rect = el('joystick').getBoundingClientRect();
  let x = (event.clientX - rect.left - rect.width / 2) / 40;
  let y = -(event.clientY - rect.top - rect.height / 2) / 40;
  const norm = Math.hypot(x, y); if (norm > 1) { x /= norm; y /= norm; }
  pointerVelocity = [x, y]; input();
}
el('joystick').addEventListener('pointermove', event => {
  if (el('joystick').hasPointerCapture(event.pointerId)) updateStick(event);
});
for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  el('joystick').addEventListener(type, () => { pointerVelocity = [0, 0]; input(); });
}
el('camera').addEventListener('click', () => {
  if (!lastState) return;
  if (config.intent_enabled) { focusBehindActualRoot(lastState.actual); return; }
  const delta = new THREE.Vector3().subVectors(camera.position, controls.target);
  controls.target.fromArray(lastState.actual.root_position);
  camera.position.copy(controls.target).add(delta); controls.update();
});
el('scene').addEventListener('change', () => {
  startCommandPending = false;
  stopInput(); inputMailbox?.clear(); episodeActive = false;
  sceneChanging = true; setDriveEnabled(false); el('scene').disabled = true;
  el('start').disabled = true; el('reset').disabled = true;
  el('status').textContent = 'Loading scene';
  browserDiagnostics.record('scene_requested', performance.now(), browserContext(), false,
    { scene_id: el('scene').value, previous_scene_revision: ledger.sceneRevision });
  send({ type: 'scene', session_id: intents.session, scene_id: el('scene').value });
});
el('reset').addEventListener('click', () => {
  if (!canResetEpisode(config.intent_enabled, episodeActive, sceneChanging,
    socket.readyState === WebSocket.OPEN)) return;
  startCommandPending = false; startRequested = false; episodeActive = false; sceneChanging = true;
  stopInput();
  inputMailbox?.clear(); setDriveEnabled(false); el('status').textContent = 'Resetting';
  el('reset').disabled = true; el('scene').disabled = true; el('start').disabled = true;
  browserDiagnostics.record('reset_clicked', performance.now(), browserContext(), false,
    { scene_id: config.scene.manifest_id, previous_scene_revision: ledger.sceneRevision });
  send({ type: 'reset_episode', session_id: intents.session });
});
el('start').addEventListener('click', () => {
  if (!config.start_required || episodeBegan || startRequested) return;
  startRequested = true;
  episodeActive = true;
  browserDiagnostics.record('start_clicked', performance.now(), browserContext(), true);
  el('start').disabled = true;
  el('reset').disabled = !canResetEpisode(config.intent_enabled, episodeActive, sceneChanging,
    socket.readyState === WebSocket.OPEN);
  el('status').textContent = config.intent_enabled ? 'Planning reference' : 'Starting';
  if (config.intent_enabled) {
    startIntentId = null;
    startCommandPending = true;
    inputMailbox.offer([0, 0], 0, performance.now()); flushIntent();
  } else send({ type: config.start_action, session_id: intents.session });
  el('stage').focus({ preventScroll: true });
});

function downloadReceipt() {
  if (!ledger) return;
  const rect = el('stage').getBoundingClientRect();
  const bodyStyle = getComputedStyle(document.body), canvasStyle = getComputedStyle(el('stage'));
  const rootNdc = lastState === null || lastState === undefined ? null
    : new THREE.Vector3().fromArray(lastState.actual.root_position).project(camera);
  const receipt = { schema: SCHEMA, config, last_physics_frame: lastState?.frame_id ?? null,
    captured_client_ms: performance.now(), display_measurement: 'renderer_submission_not_scanout',
    body_onset_measurement: 'runtime_measured_onset_then_client_render_submission',
    metrics: ledger.metrics(), frames: ledger.frames, commands: [...intents.rows.values()],
    browser_diagnostics: browserDiagnostics.receipt(),
    browser_surface: { ...browserContext(), viewport_width: innerWidth, viewport_height: innerHeight,
      canvas_rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      canvas_buffer: { width: renderer.domElement.width, height: renderer.domElement.height },
      body_style: { display: bodyStyle.display, visibility: bodyStyle.visibility, opacity: bodyStyle.opacity },
      canvas_style: { display: canvasStyle.display, visibility: canvasStyle.visibility, opacity: canvasStyle.opacity },
      actual_layer_visible: robotGroup.visible,
      follow_actual_root: el('follow').checked,
      camera_position_world: camera.position.toArray(), camera_target_world: controls.target.toArray(),
      actual_meshes_visible: [...robotGeoms.values()].filter(mesh => mesh.visible && robotGroup.visible).length,
      actual_meshes_expected: robotGeoms.size,
      actual_root_camera_ndc: rootNdc === null ? null : rootNdc.toArray(),
      scope: 'actual DOM/render state; not screenshot pixel proof or native host occlusion detection' },
    gamepad: gamepadState, failures, server: serverReceipt ?? null,
    input_simulation: simulatedGamepad?.receipt() ?? null };
  const serialized = JSON.stringify(receipt, null, 2);
  el('receipt-snapshot').value = serialized;
  const url = URL.createObjectURL(new Blob([serialized], { type: 'application/json' }));
  const anchor = document.createElement('a'); anchor.href = url;
  anchor.download = 'mb-sonic-g1-measurements.json'; anchor.click(); URL.revokeObjectURL(url);
}
el('receipt').addEventListener('click', () => send({ type: 'receipt', session_id: intents.session }));

const observer = new ResizeObserver(() => {
  const rect = el('stage').getBoundingClientRect();
  camera.aspect = rect.width / rect.height; camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height, false);
  cameraDirty = true;
});
observer.observe(el('stage'));

function animate(rafTimestamp) {
  try {
    const callbackAt = performance.now();
    browserDiagnostics.record('raf', callbackAt, browserContext(), episodeActive,
      { raf_timestamp_ms: rafTimestamp, delta_ms: previousRafMs === null ? null : callbackAt - previousRafMs,
        previous_was_active: previousRafActive });
    previousRafMs = callbackAt; previousRafActive = episodeActive;
    if (!document.hidden) {
      controls.update();
      const receipt = sceneChanging ? null : ledger?.render(draw, () => performance.now());
      if (receipt) {
        const frame = ledger.frames.at(-1);
        browserDiagnostics.record('frame_rendered', receipt.client_render_completed_ms, browserContext(), episodeActive,
          { packet_id: receipt.packet_id, frame_id: receipt.frame_id,
            render_ms: receipt.client_render_completed_ms - receipt.client_render_started_ms,
            receive_to_render_ms: frame.receive_to_render_ms });
        cameraDirty = false;
        const sentAt = send(receipt);
        if (sentAt !== null) browserDiagnostics.ackSent(receipt, sentAt, browserContext(), episodeActive);
        measure();
        if (lastState.status.startsWith('completed') || ['terminal_failure', 'runtime_failure'].includes(lastState.status)) {
          browserDiagnostics.record('episode_ended', performance.now(), browserContext(), episodeActive,
            { status: lastState.status });
          episodeActive = false;
          inputMailbox?.clear(); setDriveEnabled(false);
          el('reset').disabled = false;
        }
      } else if (cameraDirty) {
        const began = performance.now(); renderer.render(world, camera); cameraDirty = false;
        browserDiagnostics.record('paint_only', performance.now(), browserContext(), episodeActive,
          { render_ms: performance.now() - began, reason: 'camera_change',
            actual_physics_frame_id: lastState?.frame_id ?? null, actual_sim_time_s: lastState?.sim_time_s ?? null });
      }
    }
    requestAnimationFrame(animate);
  } catch (failure) { stopInput(); error(failure); socket?.close(1011, 'Display failed'); }
}

function boot() {
  socket = new WebSocket(`ws://${location.host}${API}/ws`);
  let messages = Promise.resolve();
  const inputClock = setInterval(() => {
    flushIntent();
    if (!config?.intent_enabled || document.hidden || !document.hasFocus()) return;
    const sampleAt = performance.now();
    gamepadState = standardGamepadState(simulatedGamepad === null ? navigator.getGamepads() : simulatedGamepad.sample(sampleAt));
    if (simulatedGamepad !== null) {
      el('gamepad-status').dataset.phase = simulatedGamepad.phase;
      el('gamepad-status').title = `Simulated Xbox | ${simulatedGamepad.phase}`;
    }
    const identity = gamepadState === null ? null : `${gamepadState.index}:${gamepadState.id}`;
    if (identity !== gamepadIdentity) {
      gamepadLatch.reset(); gamepadIdentity = identity;
      el('gamepad-name').textContent = simulatedGamepad !== null ? 'Simulated Xbox' : gamepadState === null ? 'No gamepad'
        : /xbox|xinput/i.test(gamepadState.id) ? 'Xbox' : 'Gamepad';
      el('gamepad-status').dataset.connected = String(gamepadState !== null);
      el('gamepad-status').title = gamepadState === null ? 'No standard gamepad detected'
        : `${gamepadState.id} | Left stick: movement | Right stick: facing`;
      browserDiagnostics.record('gamepad_changed', performance.now(), browserContext(), episodeActive,
        { id: gamepadState?.id ?? null, index: gamepadState?.index ?? null });
    }
    if (gamepadState === null) { gamepadVelocity = [0, 0]; gamepadYaw = 0; }
    else {
      const latched = gamepadLatch.sample(gamepadState.velocity, gamepadState.yaw);
      gamepadVelocity = latched.velocity; gamepadYaw = latched.yaw;
    }
    input();
  }, 10);
  socket.onmessage = event => {
    const receivedAt = performance.now(), receivedContext = browserContext(), receivedWhileActive = episodeActive;
    messages = messages.then(async () => {
      const packet = JSON.parse(event.data);
      if (packet.type === 'frame') {
        packet.client_received_ms = receivedAt;
        browserDiagnostics.record('frame_received', receivedAt, receivedContext, receivedWhileActive,
          { packet_id: packet.packet_id, frame_id: packet.state.frame_id,
            scene_revision: packet.scene_revision,
            processing_started_client_ms: performance.now(),
            published_server_ns: packet.published_server_ns,
            server_ack_to_publish_ms: packet.server_ack_to_publish_ms,
            server_snapshot_wall_ms: packet.server_snapshot_wall_ms });
      }
      if (packet.type === 'ready') {
        await configure(packet.config); intents = new IntentStream(packet.session_id, config.intent_enabled);
        intents.bindScene(config); inputMailbox = new LatestIntentMailbox(intents);
        ledger = new DisplayLedger(packet.session_id, intents); el('receipt').disabled = false;
        ledger.changeScene(packet.scene_revision);
        el('status').textContent = config.start_required ? 'Ready to Start' : 'Connected'; input();
      } else if (packet.type === 'frame') ledger.offer(packet);
      else if (packet.type === 'config') {
        ledger.changeScene(packet.scene_revision);
        await configure(packet.config);
      }
      else if (packet.type === 'diagnostics') { serverReceipt = packet.diagnostics; downloadReceipt(); }
      else if (packet.type === 'episode_started') {
        if (sceneChanging || !startRequested) return;
        simulatedGamepad?.start(receivedAt);
        episodeActive = true;
        browserDiagnostics.record('episode_started', receivedAt, receivedContext, true);
        episodeBegan = true; el('start').disabled = true; el('status').textContent = 'Starting';
        setDriveEnabled(config.intent_enabled); input();
      }
      else if (packet.type === 'render_receipt') {
        browserDiagnostics.ackReceived(packet, receivedAt, receivedContext, receivedWhileActive);
      }
      else if (packet.type === 'intent_received') {
        const row = intents.rows.get(packet.command_id);
        if (row) row.server_received_ns = packet.server_received_ns;
        if (startCommandPending && packet.command_id === startIntentId) {
          startCommandPending = false;
          send({ type: config.start_action, session_id: intents.session });
        }
      }
      else if (packet.type === 'failure' || packet.type === 'error') {
        if (packet.type === 'error') stopInput();
        sceneChanging = false; el('scene').disabled = false;
        el('scene').value = config.scene.manifest_id; error(packet.message); stopInput();
      }
    }).catch(failure => { stopInput(); error(failure); socket.close(1011, 'Protocol failed'); });
  };
  socket.onerror = () => error('G1 lab connection failed');
  socket.onclose = event => {
    browserDiagnostics.record('socket_closed', performance.now(), browserContext(), episodeActive, { code: event.code });
    episodeActive = false;
    clearInterval(inputClock);
    el('scene').disabled = true;
    el('start').disabled = true;
    el('reset').disabled = true;
    if (!failures.length) error(`Connection closed (${event.code}): ${event.reason}`);
    el('receipt').onclick = downloadReceipt;
  };
  requestAnimationFrame(animate);
}

function initialize() {
  window.lucide.createIcons();
  boot();
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize, { once: true });
else initialize();

}
