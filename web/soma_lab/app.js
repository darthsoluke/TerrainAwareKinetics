import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const SIM_FPS = 30;
const TICK_MS = 1000 / SIM_FPS;
const MAX_BLOCK_FRAMES = 8;
const CAMERA_TARGET_HEIGHT = 0.98;
const LOOK_RADIANS_PER_PIXEL = 0.0025;
const RUN_MULTIPLIER = 1.8;
const DEPTH_LAYER = 1;
const DEPTH_RENDER_INTERVAL = 1000 / 12;
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

function isGenerativeMode(mode = state.mode) {
  return mode === 'live' || mode === 'terrain';
}

function terrainModeAvailable() {
  if (state.config?.terrain_mode?.available !== true) return false;
  const terrain = state.config?.terrain_scene;
  const rows = heightfieldRows(terrain?.hf);
  return rows.length > 0 && rows.every(row => row.length > 0)
    && rows.some(row => Array.from(row).some(value => finiteHeight(value) !== null))
    && Array.isArray(terrain?.min_point) && terrain.min_point.length === 3
    && terrain.min_point.every(value => Number.isFinite(Number(value)))
    && Number.isFinite(Number(terrain.dx)) && Number(terrain.dx) > 0;
}

const el = {
  canvas: document.querySelector('#scene-canvas'),
  connection: document.querySelector('#connection-state'),
  connectionLabel: document.querySelector('#connection-label'),
  liveMode: document.querySelector('#live-mode'),
  terrainMode: document.querySelector('#terrain-mode'),
  terrainWarning: document.querySelector('#terrain-warning'),
  replayMode: document.querySelector('#replay-mode'),
  clipSelect: document.querySelector('#clip-select'),
  worldSelect: document.querySelector('#world-select'),
  worldSelectControl: document.querySelector('#world-select-control'),
  worldMapToggle: document.querySelector('#world-map-toggle'),
  worldMapPanel: document.querySelector('#world-map-panel'),
  worldMap: document.querySelector('#world-map'),
  speed: document.querySelector('#speed-input'),
  intent: document.querySelector('#intent-toggle'),
  depthToggle: document.querySelector('#depth-toggle'),
  depthInset: document.querySelector('#depth-inset'),
  depthFrame: document.querySelector('#depth-frame'),
  reset: document.querySelector('#reset-button'),
  pause: document.querySelector('#pause-button'),
  camera: document.querySelector('#camera-button'),
  mouseLook: document.querySelector('#mouse-look-button'),
  invertLook: document.querySelector('#look-invert-toggle'),
  diagnosticsToggle: document.querySelector('#diagnostics-toggle'),
  diagnosticsPanel: document.querySelector('#diagnostics-panel'),
  runtimeToggle: document.querySelector('#runtime-toggle'),
  runtimePanel: document.querySelector('#runtime-panel'),
  predictionHorizon: document.querySelector('#prediction-horizon'),
  observationAge: document.querySelector('#observation-age'),
  telemetry: document.querySelector('.telemetry'),
  runStatus: document.querySelector('#run-status'),
  frame: document.querySelector('#frame-readout'),
  fps: document.querySelector('#fps-readout'),
  root: document.querySelector('#root-readout'),
  style: document.querySelector('#style-readout'),
  inference: document.querySelector('#infer-readout'),
  skin: document.querySelector('#skin-readout'),
  underruns: document.querySelector('#underrun-readout'),
  error: document.querySelector('#error-toast'),
  touchPad: document.querySelector('#touch-pad'),
};

const stats = {
  received: 0,
  applied: 0,
  underruns: 0,
  root: [0, 0, 0],
  control: { velocity_xz: [0, 0], heading: 0, speed: 1.6, mode: 'live' },
  frame: 0,
  inference_ms: null,
  skin_ms: null,
  style: null,
  generation: null,
  camera: { pointer_locked: false, invert_y: false },
  diagnostics: { version: 'soma_motion_debug_v1', layers: {}, played_frames: 0 },
  scheduling: { prefetch_requests: 0, prefetched_blocks: 0, deferred_prefetches: 0, frame_intervals_ms: [], clock_rebases: 0, boundaries: [] },
  observation: { enabled: false, connected_to_model: false, kind: 'depth-preview', intent_xz: [0, 1], sensor_root: [0, 0, 0] },
};

const state = {
  renderer: null,
  renderDirty: true,
  scene: null,
  camera: null,
  orbitCamera: null,
  controls: null,
  mesh: null,
  rawMesh: null,
  skinMesh: null,
  skinRig: null,
  positionAttribute: null,
  vertexCount: 0,
  config: null,
  runtimeControls: null,
  clips: [],
  mode: 'live',
  paused: false,
  ready: false,
  hasVisualFrame: false,
  modeFrame: 0,
  lastPlayedAt: null,
  cameraHeight: createHeightFollow(CAMERA_TARGET_HEIGHT),
  cameraObservedY: CAMERA_TARGET_HEIGHT,
  cameraRoot: null,
  cameraDrag: null,
  invertLookY: false,
  latestCommand: { velocity_xz: [0, 0], heading: 0 },
  lastIntentDirection: [0, 1],
  serverStyle: null,
  intentVisible: false,
  intentArrow: null,
  depthEnabled: false,
  depthCamera: null,
  depthMaterial: null,
  depthTarget: null,
  depthBlitScene: null,
  depthBlitCamera: null,
  depthBlitMaterial: null,
  lastDepthRender: 0,
  depthHasFrame: false,
  floor: null,
  terrain: null,
  terrainInfo: null,
  diagnostics: { layers: {}, history: [], plan: null, consumed: 0, joints: null,
    observationStartFrame: null, requestId: null },
  keyState: new Set(),
  touchState: new Set(),
  socket: null,
  reconnectTimer: null,
  reconnectDelay: 500,
  requestId: 0,
  errorTimer: null,
  tickDeadline: 0,
  live: {
    epoch: 0,
    requestId: null,
    inFlight: null,
    pendingPacket: null,
    current: null,
    currentIndex: 0,
    currentStarted: false,
    next: null,
    awaitingCommit: null,
    modelStarted: false,
    resetPending: false,
    resetQueued: false,
    serverGeneration: null,
    retryAt: 0,
  },
  replay: {
    epoch: 0,
    meta: null,
    current: null,
    currentIndex: 0,
    currentStarted: false,
    next: null,
    nextStart: 0,
    inFlight: null,
    retryAt: 0,
  },
};

window.somaLab = {
  get renderer() { return state.renderer; },
  get scene() { return state.scene; },
  get camera() { return state.camera; },
  get orbitCamera() { return state.orbitCamera; },
  get cameraControls() { return state.controls; },
  get mesh() { return state.mesh; },
  get skinRig() { return state.skinRig; },
  stats,
  nextPlaybackDeadline,
  get ready() { return state.ready; },
  setCommand(velocityXZ, heading) {
    const x = Number(velocityXZ?.[0]) || 0;
    const z = Number(velocityXZ?.[1]) || 0;
    const magnitude = Math.hypot(x, z);
    const maximum = maxMovementSpeed();
    const scale = magnitude > maximum ? maximum / magnitude : 1;
    const velocity = [x * scale, z * scale];
    const nextHeading = Number.isFinite(Number(heading))
      ? Number(heading)
      : magnitude > 0.001 ? Math.atan2(velocity[0], velocity[1]) : state.latestCommand.heading;
    setCommand(velocity, nextHeading);
  },
};

initializeScene();
bindControls();
refreshIcons();
window.addEventListener('load', refreshIcons, { once: true });
window.addEventListener('resize', resizeRenderer);
document.addEventListener('input', () => { state.renderDirty = true; });
document.addEventListener('click', () => { state.renderDirty = true; });
window.addEventListener('blur', clearInput);

startRenderLoop();
startSimulationClock();
loadConfiguration();

function initializeScene() {
  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color('#d5e0e9');
  state.scene.fog = new THREE.Fog('#d5e0e9', 42, 145);
  state.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 240);
  state.camera.position.set(3.7, 2.65, -4.8);
  state.orbitCamera = state.camera.clone();

  try {
    state.renderer = new THREE.WebGLRenderer({ canvas: el.canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
    state.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    state.renderer.outputColorSpace = THREE.SRGBColorSpace;
    state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    state.renderer.toneMappingExposure = 1;
    state.renderer.shadowMap.enabled = true;
    state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    state.renderer.shadowMap.autoUpdate = false;
    state.renderer.shadowMap.needsUpdate = true;
    resizeRenderer();
  } catch (error) {
    showError(`3D renderer unavailable: ${error.message}`);
    setRunStatus('Renderer unavailable', 'stalled');
  }

  state.controls = new OrbitControls(state.orbitCamera, el.canvas);
  state.controls.enablePan = false;
  state.controls.enableDamping = false;
  state.controls.mouseButtons.LEFT = null;
  state.controls.mouseButtons.RIGHT = null;
  state.controls.screenSpacePanning = false;
  state.controls.minDistance = 1.6;
  state.controls.maxDistance = 14;
  state.controls.minPolarAngle = Math.PI * 0.12;
  state.controls.maxPolarAngle = Math.PI * 0.85;
  state.controls.target.set(0, CAMERA_TARGET_HEIGHT, 0);
  state.controls.update();
  state.controls.addEventListener('change', () => {
    state.renderDirty = true;
    updateThirdPersonCamera();
    // A held movement key must track look changes without another keydown.
    if (state.keyState.size || state.touchState.size) updateCommandFromInput();
  });

  state.scene.add(new THREE.HemisphereLight(0xe7f2ff, 0x72776a, 0.8));

  const keyLight = new THREE.DirectionalLight(0xfff2da, 2.8);
  keyLight.position.set(-6, 10, 3);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.left = -12;
  keyLight.shadow.camera.right = 12;
  keyLight.shadow.camera.top = 12;
  keyLight.shadow.camera.bottom = -12;
  keyLight.shadow.camera.near = 0.5;
  keyLight.shadow.camera.far = 32;
  keyLight.shadow.bias = -0.00006;
  keyLight.shadow.normalBias = 0.008;
  keyLight.target.position.set(0, 0, 0);
  state.scene.add(keyLight, keyLight.target);
  state.keyLight = keyLight;
  state.fillLight = new THREE.DirectionalLight(0xcfe5ff, 1.1);
  state.rimLight = new THREE.DirectionalLight(0xffffff, 1.2);
  state.scene.add(state.fillLight, state.fillLight.target, state.rimLight, state.rimLight.target);
  updateLightRig();

  const floorCanvas = document.createElement('canvas');
  floorCanvas.width = 128;
  floorCanvas.height = 128;
  const context = floorCanvas.getContext('2d');
  context.fillStyle = '#cdd3d6';
  context.fillRect(0, 0, 128, 128);
  context.fillStyle = '#bfc6ca';
  context.fillRect(0, 0, 64, 64);
  context.fillRect(64, 64, 64, 64);
  const checker = new THREE.CanvasTexture(floorCanvas);
  checker.colorSpace = THREE.SRGBColorSpace;
  checker.wrapS = THREE.RepeatWrapping;
  checker.wrapT = THREE.RepeatWrapping;
  checker.repeat.set(90, 90);
  checker.anisotropy = 8;
  state.floor = new THREE.Mesh(
    new THREE.PlaneGeometry(180, 180),
    new THREE.MeshStandardMaterial({ map: checker, roughness: 1, metalness: 0 }),
  );
  state.floor.rotation.x = -Math.PI / 2;
  state.floor.position.y = -0.012;
  state.floor.receiveShadow = true;
  state.scene.add(state.floor);
  state.floor.layers.enable(DEPTH_LAYER);

  state.intentArrow = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0.07, 0), 1.2, 0x397d72, 0.22, 0.13,
  );
  state.intentArrow.visible = false;
  state.scene.add(state.intentArrow);
  initializeDiagnostics();

  state.depthCamera = new THREE.PerspectiveCamera(68, 16 / 9, 0.08, 14);
  state.depthCamera.layers.set(DEPTH_LAYER);
  state.depthMaterial = new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking });
  state.depthTarget = new THREE.WebGLRenderTarget(256, 144, {
    format: THREE.RGBAFormat,
    type: THREE.UnsignedByteType,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: true,
    stencilBuffer: false,
  });
  state.depthTarget.texture.generateMipmaps = false;
  state.depthBlitScene = new THREE.Scene();
  state.depthBlitCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
  state.depthBlitCamera.position.z = 1;
  state.depthBlitMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uDepth: { value: state.depthTarget.texture },
      uNear: { value: state.depthCamera.near },
      uFar: { value: state.depthCamera.far },
    },
    vertexShader: 'varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position,1.0); }',
    fragmentShader: `
      uniform sampler2D uDepth;
      uniform float uNear;
      uniform float uFar;
      varying vec2 vUv;
      #include <packing>
      void main() {
        float depth = unpackRGBAToDepth(texture2D(uDepth, vUv));
        float distance = -perspectiveDepthToViewZ(depth, uNear, uFar);
        float gray = clamp(1.0 - distance / uFar, 0.0, 1.0);
        gl_FragColor = vec4(vec3(gray), 1.0);
      }
    `,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  state.depthBlitScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), state.depthBlitMaterial));
}

function initializeDiagnostics() {
  const colors = { intent: 0xbd6914, submitted: 0x737b82, root: 0x0069db,
    decoded: 0xc12983, history: 0x0b7b51, near: 0x0b788b,
    terrainIntent: 0xa0630c,survey:0x667a1c,skeleton: 0x54477d,
    sdfBody:0x148da0,sdfIntent:0xd18416,sdfVelocity:0x4074d5,boneSdf:0x54477d };
  for (const [name, color] of Object.entries(colors)) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(1024 * 3), 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    const options = { color, depthTest: false, depthWrite: false, toneMapped: false,
      fog: false, transparent: true, opacity: 0.9 };
    const dashed = name === 'intent' || name === 'submitted';
    const lines = new THREE.LineSegments(geometry, dashed
      ? new THREE.LineDashedMaterial({ ...options, dashSize: 0.09, gapSize: 0.055 })
      : new THREE.LineBasicMaterial(options));
    lines.frustumCulled = false;
    lines.renderOrder = 10;
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(512 * 3), 3).setUsage(THREE.DynamicDrawUsage));
    const sdfLayer = name.startsWith('sdf') || name === 'boneSdf';
    if (sdfLayer) pointGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(512 * 3), 3).setUsage(THREE.DynamicDrawUsage));
    pointGeometry.setDrawRange(0, 0);
    const points = new THREE.Points(pointGeometry,
      new THREE.PointsMaterial({ ...options, color:sdfLayer ? 0xffffff : color,
        vertexColors:sdfLayer,size:name === 'boneSdf' ? 7 : 4,sizeAttenuation:false }));
    points.frustumCulled = false;
    points.renderOrder = 11;
    const group = new THREE.Group();
    group.name = `diagnostic-${name}`;
    group.add(lines, points);
    const enabled = el.diagnosticsPanel.querySelector(`[data-debug-layer="${name}"]`).checked;
    group.visible = enabled;
    state.scene.add(group);
    state.diagnostics.layers[name] = { group, lines, points, enabled, dashed, segmentCount: 0, pointCount: 0 };
  }
  new ResizeObserver(() => {
    document.querySelector('.lab').style.setProperty('--toolbar-bottom', `${document.querySelector('.toolbar').getBoundingClientRect().bottom + 10}px`);
  }).observe(document.querySelector('.toolbar'));
}

function validDebugPoint(point) {
  return Array.isArray(point) && point.length === 3 && point.every(Number.isFinite);
}

function drawDiagnostic(name, segments = [], points = []) {
  const layer = state.diagnostics.layers[name];
  if (!layer) return;
  const linePositions = layer.lines.geometry.attributes.position;
  let count = 0;
  for (const [a, b] of segments) {
    if (!validDebugPoint(a) || !validDebugPoint(b) || count >= 512) continue;
    linePositions.array.set(a, count * 6);
    linePositions.array.set(b, count * 6 + 3);
    count += 1;
  }
  layer.lines.geometry.setDrawRange(0, count * 2);
  linePositions.needsUpdate = true;
  if (layer.dashed) layer.lines.computeLineDistances();
  const pointPositions = layer.points.geometry.attributes.position;
  let pointCount = 0;
  for (const point of points) {
    if (!validDebugPoint(point) || pointCount >= 512) continue;
    pointPositions.array.set(point, pointCount * 3);
    pointCount += 1;
  }
  layer.points.geometry.setDrawRange(0, pointCount);
  pointPositions.needsUpdate = true;
  layer.segmentCount = count;
  layer.pointCount = pointCount;
}

function drawDiagnosticPath(name, path) {
  const points = Array.isArray(path) ? path : [];
  const segments = points.slice(1).map((point, index) => [points[index], point]);
  drawDiagnostic(name, segments, points.filter((_, index) => index % 4 === 0 || index === points.length - 1));
}

function colorSdfPoints(name, distances, valid) {
  const attribute = state.diagnostics.layers[name]?.points.geometry.attributes.color;
  if (!attribute) return;
  const colors = [new THREE.Color(0x148da0),new THREE.Color(0xe63946),new THREE.Color(0xe3ab24),new THREE.Color(0x909090)];
  for (let i=0;i<Math.min(distances?.length ?? 0,attribute.count);i+=1) {
    const d=distances[i];
    const color=!valid?.[i] || !Number.isFinite(d) ? colors[3] : d < -0.005 ? colors[1] : Math.abs(d)<0.08 ? colors[2] : colors[0];
    attribute.setXYZ(i,color.r,color.g,color.b);
  }
  attribute.needsUpdate=true;
}

function acceptSdfObservation(observation) {
  const names=['sdfBody','sdfIntent','sdfVelocity'];
  for (const name of names) drawDiagnostic(name);
  if (observation?.role !== 'model_input') return;
  const n=observation.points_per_region, points=observation.world_xyz_m;
  if (!Number.isInteger(n) || n<1 || n>512 || points?.length!==3*n
      || observation.signed_distance_m?.length!==points.length || observation.valid?.length!==points.length) return;
  const grid=Math.round(Math.cbrt(n));
  for (let r=0;r<3;r+=1) {
    const region=points.slice(r*n,(r+1)*n), segments=[];
    if (grid**3===n) {
      const corner=(x,y,z)=>region[(x*(grid-1)*grid+y*(grid-1))*grid+z*(grid-1)];
      for (let x=0;x<2;x+=1) for(let y=0;y<2;y+=1) for(let z=0;z<2;z+=1) {
        if (!x) segments.push([corner(x,y,z),corner(1,y,z)]);
        if (!y) segments.push([corner(x,y,z),corner(x,1,z)]);
        if (!z) segments.push([corner(x,y,z),corner(x,y,1)]);
      }
    }
    drawDiagnostic(names[r],segments,region);
    colorSdfPoints(names[r],observation.signed_distance_m.slice(r*n,(r+1)*n),observation.valid.slice(r*n,(r+1)*n));
  }
  document.querySelector('#sdf-summary').textContent=`SDF INPUT: ${points.length} samples / ${observation.valid.filter(Boolean).length} known`;
}

function updateInputGuide() {
  if (!state.diagnostics.layers.intent) return;
  if (!isGenerativeMode()) {
    drawDiagnostic('intent');
    return;
  }
  // This is a planar intent guide at the observed Root Y, never a height target.
  const seconds = state.diagnostics.plan?.input_intent?.horizon_seconds ?? 1;
  const [x, z] = state.latestCommand.velocity_xz;
  const end = [stats.root[0] + x * seconds, stats.root[1], stats.root[2] + z * seconds];
  const yaw = state.latestCommand.heading;
  const tip = [end[0] + Math.sin(yaw) * 0.18, end[1], end[2] + Math.cos(yaw) * 0.18];
  drawDiagnostic('intent', [[stats.root, end], [end, tip]], [end]);
}

function acceptDiagnosticPlan(meta) {
  const debug = meta.debug ?? {};
  state.diagnostics.plan = debug;
  state.diagnostics.consumed = 0;
  state.diagnostics.requestId = meta.request_id;
  if (meta.generation?.neural_generation !== false) {
    state.diagnostics.observationStartFrame = state.modeFrame - (debug.observation_lag_frames ?? 0);
  }
  const intent = debug.input_intent;
  if (validDebugPoint(intent?.start_world_xyz_m) && Array.isArray(intent?.endpoint_world_xz_m)) {
    const end = [intent.endpoint_world_xz_m[0], intent.start_world_xyz_m[1], intent.endpoint_world_xz_m[1]];
    drawDiagnostic('submitted', [[intent.start_world_xyz_m, end]], [end]);
  } else drawDiagnostic('submitted');
  for (const name of ['near','terrainIntent','survey']) drawDiagnostic(name);
  for (const patch of debug.terrain_patches ?? []) {
    const name = patch.name==='near' ? 'near' : patch.name==='intent' ? 'terrainIntent' : patch.name==='survey' ? 'survey' : null;
    if (!name || !Array.isArray(patch.shape) || !Array.isArray(patch.world_xyz_m) || !Array.isArray(patch.valid)) continue;
    const [rows, columns] = patch.shape;
    if (!Number.isInteger(rows) || !Number.isInteger(columns) || rows < 1 || columns < 1
      || rows * columns !== patch.world_xyz_m.length || rows * columns > 512) continue;
    const segments = [], points = [];
    for (let i = 0; i < rows * columns; i += 1) {
      if (!patch.valid[i] || !validDebugPoint(patch.world_xyz_m[i])) continue;
      points.push(patch.world_xyz_m[i]);
      for (const j of [i % columns < columns - 1 ? i + 1 : -1, i + columns < rows * columns ? i + columns : -1]) {
        if (j >= 0 && patch.valid[j]) segments.push([patch.world_xyz_m[i], patch.world_xyz_m[j]]);
      }
    }
    drawDiagnostic(name, segments, points);
  }
  acceptSdfObservation(debug.unified_sdf);
  updateDiagnosticPrediction();
}

function updateDiagnosticPrediction() {
  const diagnostic = state.diagnostics;
  for (const [name, key] of [['root', 'root_stage_world_xyz_m'], ['decoded', 'decoded_world_xyz_m']]) {
    const path = diagnostic.plan?.[key];
    drawDiagnosticPath(name, Array.isArray(path) ? path.slice(diagnostic.consumed) : []);
  }
  const remaining = diagnostic.plan?.decoded_world_xyz_m?.length ?? 0;
  el.predictionHorizon.textContent = remaining ? `${(Math.max(0, remaining - diagnostic.consumed) / SIM_FPS).toFixed(2)} s` : '-- s';
  const age = diagnostic.observationStartFrame;
  el.observationAge.textContent = diagnostic.plan?.terrain_patches?.length
    ? `Observation snapshot: ${((state.modeFrame - age) / SIM_FPS).toFixed(2)} s ago`
    : 'No model observation';
  refreshDiagnosticStats();
}

function updatePlayedDiagnostics(root, joints, advance, boneSdf = null) {
  const diagnostic = state.diagnostics;
  if (advance) {
    diagnostic.history.push([...root]);
    if (diagnostic.history.length > SIM_FPS * 5) diagnostic.history.shift();
    drawDiagnosticPath('history', diagnostic.history);
  }
  const parents = state.config?.skeleton?.parents;
  diagnostic.joints = joints ?? null;
  const segments = [];
  if (Array.isArray(joints) && Array.isArray(parents) && joints.length === parents.length) {
    parents.forEach((parent, child) => {
      if (Number.isInteger(parent) && parent >= 0 && parent < joints.length) segments.push([joints[parent], joints[child]]);
    });
  }
  drawDiagnostic('skeleton', segments, joints ?? []);
  if (boneSdf?.role === 'diagnostic_only_not_model_input' && boneSdf.distances?.length === joints?.length) {
    drawDiagnostic('boneSdf',segments,joints);
    colorSdfPoints('boneSdf',boneSdf.distances,boneSdf.valid);
  } else drawDiagnostic('boneSdf');
  refreshDiagnosticStats();
}

function refreshDiagnosticStats() {
  const diagnostic = state.diagnostics;
  stats.diagnostics = { version: 'soma_motion_debug_v1', played_frames: diagnostic.history.length,
    request_id: diagnostic.requestId, consumed_prediction_frames: diagnostic.consumed,
    future_height_source: isGenerativeMode() ? 'model' : null,
    input_height_source: 'not_conditioned',
    observation_source: diagnostic.plan?.observation_source ?? null,
    terrain_sampling: diagnostic.plan?.terrain_sampling ?? 'input_direction',
    query_proposal_world_xz_m: diagnostic.plan?.query_proposal_world_xz_m ?? [],
    terrain_corridor_world_xyz_m: diagnostic.plan?.terrain_patches?.find(patch=>patch.name==='intent')?.world_xyz_m ?? [],
    survey_world_xyz_m:diagnostic.plan?.terrain_patches?.find(patch=>patch.name==='survey')?.world_xyz_m ?? [],
    motion_observation_extent_m:diagnostic.plan?.motion_observation_extent_m ?? null,
    motion_observation_halfwidth_m:diagnostic.plan?.motion_observation_halfwidth_m ?? null,
    layers: Object.fromEntries(Object.entries(diagnostic.layers).map(([name, layer]) =>
      [name, { enabled: layer.enabled, segments: layer.segmentCount, points: layer.pointCount }])) };
}

function clearDiagnostics() {
  const diagnostic = state.diagnostics;
  diagnostic.history = [];
  diagnostic.plan = null;
  diagnostic.consumed = 0;
  diagnostic.observationStartFrame = null;
  diagnostic.requestId = null;
  diagnostic.joints = null;
  const sdfSummary = document.querySelector('#sdf-summary');
  if (sdfSummary) sdfSummary.textContent = 'SDF INPUT: --';
  for (const name of Object.keys(diagnostic.layers)) drawDiagnostic(name);
  updateDiagnosticPrediction();
  updateInputGuide();
}

function bindControls() {
  el.diagnosticsToggle.addEventListener('click', () => {
    el.diagnosticsPanel.hidden = !el.diagnosticsPanel.hidden;
    el.diagnosticsToggle.setAttribute('aria-expanded', String(!el.diagnosticsPanel.hidden));
    if (!el.diagnosticsPanel.hidden) {
      el.runtimePanel.hidden = true;
      el.runtimeToggle.setAttribute('aria-expanded','false');
    }
    fitInspectionPanels('diagnostics');
  });
  for (const checkbox of el.diagnosticsPanel.querySelectorAll('[data-debug-layer]')) {
    checkbox.addEventListener('change', () => {
      const layer = state.diagnostics.layers[checkbox.dataset.debugLayer];
      layer.enabled = checkbox.checked;
      layer.group.visible = layer.enabled;
      refreshDiagnosticStats();
    });
  }
  el.liveMode.addEventListener('click', () => switchMode('live'));
  el.terrainMode.addEventListener('click', () => switchMode('terrain'));
  el.replayMode.addEventListener('click', () => switchMode('replay'));
  el.clipSelect.addEventListener('change', () => {
    if (state.mode === 'replay' && el.clipSelect.value) loadReplayClip(el.clipSelect.value);
  });
  el.speed.addEventListener('input', updateCommandFromInput);
  el.speed.addEventListener('change', normalizeSpeedInput);
  el.intent.addEventListener('click', () => {
    state.intentVisible = !state.intentVisible;
    el.intent.setAttribute('aria-pressed', String(state.intentVisible));
    el.intent.title = state.intentVisible ? 'Hide movement intent' : 'Show movement intent';
    el.intent.setAttribute('aria-label', el.intent.title);
    updateIntentArrow();
  });
  el.depthToggle.addEventListener('click', toggleDepthPreview);
  el.reset.addEventListener('click', resetLive);
  el.pause.addEventListener('click', togglePause);
  el.camera.addEventListener('click', resetCamera);
  el.mouseLook.addEventListener('click', toggleMouseLook);
  el.invertLook.addEventListener('click', () => {
    state.invertLookY = !state.invertLookY;
    el.invertLook.setAttribute('aria-pressed', String(state.invertLookY));
    updateCameraStats();
  });
  bindThirdPersonLook();

  window.addEventListener('keydown', (event) => {
    if (event.code === 'Escape' && document.pointerLockElement === el.canvas) {
      document.exitPointerLock?.();
      clearInput();
      return;
    }
    if (!isGenerativeMode() || isEditableTarget(event.target)) return;
    const key = keyName(event.code);
    if (!key) return;
    if (['w', 'a', 's', 'd', 'shift'].includes(key)) event.preventDefault();
    if (state.keyState.has(key)) return;
    state.keyState.add(key);
    updateCommandFromInput();
  });
  window.addEventListener('keyup', (event) => {
    const key = keyName(event.code);
    if (!key || !state.keyState.delete(key)) return;
    updateCommandFromInput();
  });

  for (const button of el.touchPad.querySelectorAll('[data-pad]')) {
    const key = button.dataset.pad;
    button.addEventListener('pointerdown', (event) => {
      if (!isGenerativeMode()) return;
      event.preventDefault();
      button.setPointerCapture?.(event.pointerId);
      state.touchState.add(key);
      button.classList.add('is-down');
      updateCommandFromInput();
    });
    const release = () => {
      if (!state.touchState.delete(key)) return;
      button.classList.remove('is-down');
      updateCommandFromInput();
    };
    button.addEventListener('pointerup', release);
    button.addEventListener('pointercancel', release);
    button.addEventListener('lostpointercapture', release);
  }
}

function isEditableTarget(target) {
  return Boolean(target?.closest?.('input, select, textarea, [contenteditable="true"]'));
}

function updateCameraStats() {
  if (!state.controls) return;
  stats.camera = {
    pointer_locked: document.pointerLockElement === el.canvas,
    invert_y: state.invertLookY,
    azimuth: state.controls.getAzimuthalAngle(),
    polar: state.controls.getPolarAngle(),
    desired_distance: state.controls.getDistance(),
    distance: state.camera.position.distanceTo(state.controls.target),
    target: state.controls.target.toArray(),
    vertical_source: 'played-root',
    vertical_goal: state.cameraHeight.goal,
    observed_root_y: state.cameraObservedY,
  };
}

function updateThirdPersonCamera() {
  if (!state.controls || !state.orbitCamera) return;
  state.camera.position.copy(state.orbitCamera.position);
  state.camera.lookAt(state.controls.target);
  state.camera.updateMatrixWorld(true);
  updateCameraStats();
  stats.camera.collision = false;
}

function updateLightRig() {
  const [x, , z] = stats.root;
  const y = stats.root[1] - CAMERA_TARGET_HEIGHT;
  for (const [light, offset] of [[state.keyLight, [-6, 10, 3]],
    [state.fillLight, [5, 5, -7]], [state.rimLight, [3, 7, 6]]]) {
    if (!light) continue;
    light.position.set(x + offset[0], y + offset[1], z + offset[2]);
    light.target.position.set(x, y + 0.6, z);
  }
}

function rotateThirdPersonLook(dx, dy) {
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || !state.controls) return;
  const offset = state.orbitCamera.position.clone().sub(state.controls.target);
  const orbit = new THREE.Spherical().setFromVector3(offset);
  orbit.theta -= dx * LOOK_RADIANS_PER_PIXEL;
  const vertical = state.invertLookY ? -dy : dy;
  orbit.phi = THREE.MathUtils.clamp(orbit.phi - vertical * LOOK_RADIANS_PER_PIXEL,
    state.controls.minPolarAngle, state.controls.maxPolarAngle);
  state.orbitCamera.position.copy(state.controls.target).add(offset.setFromSpherical(orbit));
  state.controls.update();
}

function toggleMouseLook() {
  if (document.pointerLockElement === el.canvas) {
    document.exitPointerLock?.();
    return;
  }
  if (!el.canvas.requestPointerLock) return;
  el.canvas.focus({ preventScroll: true });
  try {
    // Embedded browsers may deny capture; right-drag remains available.
    el.canvas.requestPointerLock()?.catch?.(() => updateCameraStats());
  } catch (_) {
    updateCameraStats();
  }
}

function bindThirdPersonLook() {
  el.canvas.addEventListener('click', (event) => {
    if (event.button === 0 && document.pointerLockElement !== el.canvas) toggleMouseLook();
  });
  el.canvas.addEventListener('contextmenu', event => event.preventDefault());
  document.addEventListener('pointerlockchange', () => {
    const locked = document.pointerLockElement === el.canvas;
    state.cameraDrag = null;
    state.controls.enableRotate = !locked;
    el.mouseLook.setAttribute('aria-pressed', String(locked));
    el.canvas.classList.toggle('mouse-captured', locked);
    updateCameraStats();
    if (!locked) clearInput();
  });
  document.addEventListener('pointerlockerror', updateCameraStats);
  document.addEventListener('mousemove', (event) => {
    if (document.pointerLockElement === el.canvas) rotateThirdPersonLook(event.movementX, event.movementY);
  });
  el.canvas.addEventListener('pointerdown', (event) => {
    if (event.pointerType !== 'mouse' || event.button !== 2 || document.pointerLockElement) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    el.canvas.focus({ preventScroll: true });
    state.cameraDrag = { id: event.pointerId, x: event.clientX, y: event.clientY };
    el.canvas.setPointerCapture(event.pointerId);
  }, { capture: true });
  el.canvas.addEventListener('pointermove', (event) => {
    const drag = state.cameraDrag;
    if (!drag || drag.id !== event.pointerId) return;
    event.stopImmediatePropagation();
    rotateThirdPersonLook(event.clientX - drag.x, event.clientY - drag.y);
    drag.x = event.clientX;
    drag.y = event.clientY;
  }, { capture: true });
  const releaseDrag = (event) => {
    if (state.cameraDrag?.id !== event.pointerId) return;
    state.cameraDrag = null;
    if (el.canvas.hasPointerCapture(event.pointerId)) el.canvas.releasePointerCapture(event.pointerId);
  };
  el.canvas.addEventListener('pointerup', releaseDrag, { capture: true });
  el.canvas.addEventListener('pointercancel', releaseDrag, { capture: true });
  el.canvas.addEventListener('lostpointercapture', releaseDrag);
  window.addEventListener('blur', () => { state.cameraDrag = null; });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) clearInput();
  });
}

function keyName(code) {
  if (code === 'KeyW' || code === 'ArrowUp') return 'w';
  if (code === 'KeyA' || code === 'ArrowLeft') return 'a';
  if (code === 'KeyS' || code === 'ArrowDown') return 's';
  if (code === 'KeyD' || code === 'ArrowRight') return 'd';
  if (code === 'ShiftLeft' || code === 'ShiftRight') return 'shift';
  return null;
}

function clearInput() {
  const hadInput = state.keyState.size || state.touchState.size;
  state.keyState.clear();
  state.touchState.clear();
  el.touchPad.querySelectorAll('.is-down').forEach(button => button.classList.remove('is-down'));
  if (hadInput) updateCommandFromInput();
}

function updateCommandFromInput() {
  if (!isGenerativeMode()) return;
  const pressed = new Set([...state.keyState, ...state.touchState]);
  let forward = Number(pressed.has('w')) - Number(pressed.has('s'));
  let right = Number(pressed.has('d')) - Number(pressed.has('a'));
  const directionLength = Math.hypot(forward, right);
  if (directionLength > 0) {
    forward /= directionLength;
    right /= directionLength;
  }

  let velocity = [0, 0];
  let heading = state.latestCommand.heading;
  if (directionLength > 0) {
    const cameraForward = new THREE.Vector3();
    state.orbitCamera.getWorldDirection(cameraForward);
    cameraForward.y = 0;
    if (cameraForward.lengthSq() < 1e-6) cameraForward.set(0, 0, 1);
    cameraForward.normalize();
    const cameraRight = new THREE.Vector3().crossVectors(cameraForward, state.camera.up).normalize();
    const worldDirection = cameraForward.multiplyScalar(forward).add(cameraRight.multiplyScalar(right)).normalize();
    state.lastIntentDirection = [worldDirection.x, worldDirection.z];
    const baseSpeed = readSpeed();
    const running = pressed.has('shift');
    const speed = Math.min(maxMovementSpeed(),baseSpeed*(running ? RUN_MULTIPLIER : 1));
    velocity = [worldDirection.x * speed, worldDirection.z * speed];
    heading = Math.atan2(worldDirection.x, worldDirection.z)+(state.config?.entry_command?.heading_offset_rad||0);
  }
  setCommand(velocity, heading);
  if(directionLength>0 && state.config?.start_paused && state.paused && !state.live.modelStarted && !state.live.failed)togglePause();
}

function normalizeSpeedInput() {
  const value = readSpeed();
  el.speed.value = value.toFixed(1);
  updateCommandFromInput();
}

function readSpeed() {
  const parsed = Number(el.speed.value);
  const value = Number.isFinite(parsed) ? parsed : 1.6;
  return Math.max(0.3,Math.min(maxMovementSpeed(),value));
}

function maxMovementSpeed() { return state.config?.runtime_controls?.max_speed_mps ?? 3; }

function setCommand(velocity, heading) {
  state.latestCommand = {
    velocity_xz: [Number(velocity[0]) || 0, Number(velocity[1]) || 0],
    heading: Number.isFinite(heading) ? heading : state.latestCommand.heading,
  };
  const magnitude = Math.hypot(...state.latestCommand.velocity_xz);
  if (magnitude > 0.001) {
    state.lastIntentDirection = [
      state.latestCommand.velocity_xz[0] / magnitude,
      state.latestCommand.velocity_xz[1] / magnitude,
    ];
  }
  stats.control = {
    velocity_xz: [...state.latestCommand.velocity_xz],
    heading: state.latestCommand.heading,
    speed: magnitude,
    mode: state.mode,
  };
  updateLiveStateTelemetry();
  updateIntentArrow();
}

function updateIntentArrow() {
  state.renderDirty = true;
  updateInputGuide();
  if (!state.intentArrow) return;
  const [x, z] = state.latestCommand.velocity_xz;
  const magnitude = Math.hypot(x, z);
  state.intentArrow.visible = state.intentVisible && isGenerativeMode() && magnitude > 0.01;
  if (!state.intentArrow.visible) return;
  state.intentArrow.position.set(...stats.root);
  state.intentArrow.setDirection(new THREE.Vector3(x, 0, z).normalize());
  state.intentArrow.setLength(Math.min(1.8, 0.8 + magnitude * 0.28), 0.22, 0.13);
}

async function loadConfiguration() {
  try {
    const response = await fetch('/api/config', { cache: 'no-store' });
    if (!response.ok) throw new Error(`Config request failed (${response.status})`);
    const config = await response.json();
    validateConfig(config);
    state.config = config;
    const unified=Boolean(config.terrain_mode?.unified_schema);
    for (const item of document.querySelectorAll('[data-unified-sdf]')) item.hidden=!unified;
    if (unified) {
      for (const name of ['near','terrainIntent','survey','sdfBody','sdfIntent','sdfVelocity','boneSdf']) {
        const enabled=name.startsWith('sdf') || name==='boneSdf';
        el.diagnosticsPanel.querySelector(`[data-debug-layer="${name}"]`).checked=enabled;
        state.diagnostics.layers[name].enabled=enabled;
        state.diagnostics.layers[name].group.visible=enabled;
      }
    }
    if(config.start_paused){state.paused=true;updatePauseButton();}
    if(config.entry_command)el.speed.value=String(config.entry_command.speed_mps);
    state.vertexCount = Number(config.vertex_count);
    state.clips = Array.isArray(config.clips) ? config.clips : [];
    createSomaMesh(config.initial_vertices, config.faces);
    if (config.soma_skin) createSomaGpuSkin(config.soma_skin, config.faces);
    initializeRoot(config.initial_root, true);
    populateSelectors();
    initializeWorldControls();
    initializeRuntimeControls();
    setRunStatus('Connecting');
    connectSocket();
    const query = new URLSearchParams(window.location.search);
    const requestedClip = query.get('clip');
    if (requestedClip && state.clips.some(clip => clip.id === requestedClip)) {
      el.clipSelect.value = requestedClip;
      switchMode('replay');
    } else if ((query.get('mode') === 'terrain' || config.default_mode === 'terrain') && terrainModeAvailable()) {
      switchMode('terrain');
    }
    refreshIcons();
    if (unified) {
      const skinVisible = state.skinMesh?.visible;
      if (state.skinMesh) state.skinMesh.visible = true;
      state.renderer.compile(state.scene, state.camera);
      if (state.skinMesh) state.skinMesh.visible = skinVisible;
    }
  } catch (error) {
    setConnection('error', 'OFFLINE');
    setRunStatus('Config unavailable', 'stalled');
    showError(error.message);
  }
}

function validateConfig(config) {
  if (!config || !Number.isFinite(Number(config.vertex_count)) || Number(config.vertex_count) < 3) {
    throw new Error('Invalid SOMA configuration');
  }
  const vertices = toFloatArray(config.initial_vertices);
  if (vertices.length !== Number(config.vertex_count) * 3) throw new Error('Initial vertex count does not match config');
  if (!Array.isArray(config.faces) || flattenNumbers(config.faces).length < 3) throw new Error('SOMA faces are missing');
  if (!Array.isArray(config.initial_root) || config.initial_root.length !== 3) throw new Error('Initial root is missing');
}

function populateSelectors() {
  el.clipSelect.replaceChildren();
  for (const clip of state.clips) {
    const option = document.createElement('option');
    option.value = String(clip.id);
    option.textContent = String(clip.label ?? clip.id);
    el.clipSelect.append(option);
  }
  const hasClips = state.clips.length > 0;
  el.clipSelect.hidden = !hasClips || state.mode !== 'replay';
  el.clipSelect.disabled = !hasClips || state.mode !== 'replay';
  el.speed.disabled = !isGenerativeMode();
  el.speed.closest('.speed-control').hidden = !isGenerativeMode();
  el.touchPad.hidden = !isGenerativeMode();
  el.replayMode.disabled = !hasClips;
  el.terrainMode.disabled = !terrainModeAvailable();
  el.terrainWarning.hidden = state.mode !== 'terrain';
  if (state.config?.default_mode === 'terrain') {
    el.liveMode.hidden = true;
    el.replayMode.hidden = !hasClips;
    el.clipSelect.closest('.clip-control').hidden = !hasClips;
  }
  if (state.config?.terrain_mode?.model_label) {
    el.terrainWarning.textContent = `${state.config.terrain_mode.model_label} · ${state.config.terrain_mode.visual_baseline_approved ? 'LIVE TRIAL' : 'EXPERIMENTAL'}`;
    el.terrainWarning.title = state.config.terrain_mode.quality_note ?? '';
    el.terrainWarning.title = state.config.terrain_mode.model_sha256 || state.config.terrain_mode.model_file;
  }
  if (!hasClips) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'NO CLIPS';
    el.clipSelect.append(option);
  }
  el.reset.disabled = !state.ready || !state.socket || state.socket.readyState !== WebSocket.OPEN || !isGenerativeMode();
  refreshIcons();
}

function createSomaMesh(verticesInput, facesInput) {
  const vertices = toFloatArray(verticesInput);
  const faces = flattenNumbers(facesInput);
  const geometry = new THREE.BufferGeometry();
  state.positionAttribute = new THREE.BufferAttribute(vertices, 3);
  state.positionAttribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute('position', state.positionAttribute);
  geometry.setIndex(faces);
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  const material = new THREE.MeshStandardMaterial({
    color: '#bcc6cf',
    roughness: 0.6,
    metalness: 0.12,
    side: THREE.DoubleSide,
  });
  state.mesh = new THREE.Mesh(geometry, material);
  state.mesh.name = 'SOMA authoritative vertex mesh';
  state.mesh.castShadow = true;
  state.mesh.receiveShadow = true;
  state.scene.add(state.mesh);
  state.rawMesh = state.mesh;
}

function somaSkinMaterial(material) {
  material.onBeforeCompile = shader => {
    shader.vertexShader = 'attribute vec4 skinIndex2;\nattribute vec4 skinWeight2;\n' + shader.vertexShader;
    const vertexTerms = ['x', 'y', 'z', 'w'].map(c =>
      `skinned += getBoneMatrix(skinIndex2.${c}) * skinVertex * skinWeight2.${c};`).join('\n');
    const normalTerms = ['x', 'y', 'z', 'w'].map(c =>
      `skinMatrix += skinWeight2.${c} * getBoneMatrix(skinIndex2.${c});`).join('\n');
    shader.vertexShader = shader.vertexShader.replace('#include <skinning_vertex>',
      THREE.ShaderChunk.skinning_vertex.replace('transformed =',
        `${vertexTerms}\ntransformed =`));
    shader.vertexShader = shader.vertexShader.replace('#include <skinnormal_vertex>',
      THREE.ShaderChunk.skinnormal_vertex.replace('skinMatrix = bindMatrixInverse',
        `${normalTerms}\nskinMatrix = bindMatrixInverse`));
  };
  material.customProgramCacheKey = () => 'official-soma-eight-weight-v1';
  return material;
}

class SomaGpuMesh extends THREE.SkinnedMesh {
  // Preserve every asset influence in both GPU and CPU picking/bounds paths.
  applyBoneTransform(index, target) {
    const base = target.clone().applyMatrix4(this.bindMatrix);
    const point = new THREE.Vector3();
    const matrix = new THREE.Matrix4();
    const attributes = this.geometry.attributes;
    target.set(0, 0, 0);
    for (let influence = 0; influence < 8; influence += 1) {
      const offset = index * 4 + influence % 4;
      const joint = influence < 4 ? attributes.skinIndex.array[offset] : attributes.skinIndex2.array[offset];
      const weight = influence < 4 ? attributes.skinWeight.array[offset] : attributes.skinWeight2.array[offset];
      matrix.multiplyMatrices(this.skeleton.bones[joint].matrixWorld, this.skeleton.boneInverses[joint]);
      target.addScaledVector(point.copy(base).applyMatrix4(matrix), weight);
    }
    return target.applyMatrix4(this.bindMatrixInverse);
  }
}

function createSomaGpuSkin(skin, faces) {
  const influences = skin.influences_per_vertex;
  if (skin.format !== 'soma77-rig-v1' || skin.rig_count !== 77
    || skin.bind_vertices.length !== state.vertexCount * 3
    || !Number.isInteger(influences) || influences < 4 || influences > 8
    || skin.indices.length !== state.vertexCount * influences || skin.weights.length !== state.vertexCount * influences
    || skin.inverse_bind.length !== 77 * 16) throw new Error('Invalid official SOMA skin asset');
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(skin.bind_vertices, 3));
  geometry.setIndex(flattenNumbers(faces));
  const indices = new Uint16Array(state.vertexCount * 4);
  const weights = new Float32Array(state.vertexCount * 4);
  const extraIndices = new Uint16Array(state.vertexCount * 4);
  const extraWeights = new Float32Array(state.vertexCount * 4);
  for (let vertex = 0; vertex < state.vertexCount; vertex += 1) {
    for (let j = 0; j < 4; j += 1) {
      indices[vertex * 4 + j] = skin.indices[vertex * influences + j];
      weights[vertex * 4 + j] = skin.weights[vertex * influences + j];
    }
    for (let j = 4; j < influences; j += 1) {
      extraIndices[vertex * 4 + j - 4] = skin.indices[vertex * influences + j];
      extraWeights[vertex * 4 + j - 4] = skin.weights[vertex * influences + j];
    }
  }
  geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(indices, 4));
  geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(weights, 4));
  geometry.setAttribute('skinIndex2', new THREE.Uint16BufferAttribute(extraIndices, 4));
  geometry.setAttribute('skinWeight2', new THREE.Float32BufferAttribute(extraWeights, 4));
  geometry.computeVertexNormals();
  const bones = [], inverses = [];
  for (let i = 0; i < 77; i += 1) {
    const bone = new THREE.Bone();
    bone.matrixAutoUpdate = false;
    bone.matrixWorldAutoUpdate = false;
    inverses.push(new THREE.Matrix4().fromArray(skin.inverse_bind, i * 16).transpose());
    bone.matrixWorld.copy(inverses[i]).invert();
    bones.push(bone);
  }
  const mesh = new SomaGpuMesh(geometry, somaSkinMaterial(state.rawMesh.material.clone()));
  mesh.bind(new THREE.Skeleton(bones, inverses), new THREE.Matrix4());
  mesh.bindMode = THREE.DetachedBindMode;
  mesh.customDepthMaterial = somaSkinMaterial(new THREE.MeshDepthMaterial({ depthPacking: THREE.RGBADepthPacking }));
  mesh.customDistanceMaterial = somaSkinMaterial(new THREE.MeshDistanceMaterial());
  mesh.name = 'SOMA official GPU skin';
  mesh.frustumCulled = false;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.visible = false;
  state.scene.add(mesh);
  state.skinMesh = mesh;
}

function initializeRoot(rootInput, resetView = false) {
  const root = toVec3(rootInput);
  stats.root = [...root];
  if (resetView || !state.cameraRoot) {
    resetCamera();
  } else {
    followRoot(root);
  }
  updateReadouts();
  updateIntentArrow();
}

function applyFrame(frame, { count = true, advance = true, joints = null, boneSdf = null } = {}) {
  state.renderDirty = true;
  if (!state.positionAttribute || (!frame.rig && frame.vertices.length !== state.vertexCount * 3)) {
    throw new Error('Frame vertex count differs from the active SOMA mesh');
  }
  // Server vertices are authoritative world-space positions; root only drives camera and telemetry.
  if (frame.rig) {
    if (!state.skinMesh || frame.rig.length !== 77 * 16) throw new Error('Invalid SOMA rig frame');
    state.rawMesh.visible = false;
    state.skinMesh.visible = true;
    state.mesh = state.skinMesh;
    state.skinRig = frame.rig;
    for (let i = 0; i < 77; i += 1) {
      state.mesh.skeleton.bones[i].matrixWorld.fromArray(frame.rig, i * 16).transpose();
    }
    state.mesh.skeleton.update();
  } else {
    if (state.skinMesh) state.skinMesh.visible = false;
    state.rawMesh.visible = true;
    state.mesh = state.rawMesh;
    state.skinRig = null;
    state.positionAttribute.array.set(frame.vertices);
    state.positionAttribute.needsUpdate = true;
    state.mesh.geometry.computeVertexNormals();
    state.mesh.geometry.computeBoundingSphere();
  }
  state.renderer.shadowMap.needsUpdate = true;
  const root = toVec3(frame.root);
  stats.root = [...root];
  followRoot(root);
  state.hasVisualFrame = true;
  if (count) stats.applied += 1;
  if (advance) state.modeFrame += 1;
  if (advance) {
    const now = performance.now();
    if (state.lastPlayedAt !== null) {
      stats.scheduling.frame_intervals_ms.push(now - state.lastPlayedAt);
      if (stats.scheduling.frame_intervals_ms.length > 900) stats.scheduling.frame_intervals_ms.shift();
    }
    state.lastPlayedAt = now;
  }
  stats.frame = state.modeFrame;
  updatePlayedDiagnostics(root, joints, advance, boneSdf);
  if (isGenerativeMode()) updateDiagnosticPrediction();
  updateReadouts();
  updateIntentArrow();
}

function followRoot(root) {
  state.cameraObservedY = root[1];
  if (!state.cameraRoot) {
    state.cameraRoot = [root[0], root[2]];
    state.cameraHeight = createHeightFollow(root[1]);
    state.controls.target.set(root[0], root[1], root[2]);
    return;
  }
  const dx = root[0] - state.cameraRoot[0];
  const dz = root[2] - state.cameraRoot[1];
  state.orbitCamera.position.x += dx;
  state.orbitCamera.position.z += dz;
  state.controls.target.x += dx;
  state.controls.target.z += dz;
  state.cameraRoot = [root[0], root[2]];
  updateLightRig();
}

export function createHeightFollow(height) {
  return { value: height, goal: height, velocity: 0 };
}

export function stepHeightFollow(follow, rootHeight, dt) {
  if (!Number.isFinite(rootHeight) || !Number.isFinite(dt) || dt <= 0) return follow.value;
  // Camera-only dead zone rejects gait bob; sustained elevation moves its boundary.
  const deadZone = 0.18;
  const delta = rootHeight - follow.goal;
  if (Math.abs(delta) > deadZone) follow.goal = rootHeight - Math.sign(delta) * deadZone;
  // Exact critically damped spring integration for a stationary goal.
  const omega = 8;
  const offset = follow.value - follow.goal;
  const impulse = follow.velocity + omega * offset;
  const decay = Math.exp(-omega * dt);
  follow.value = follow.goal + (offset + impulse * dt) * decay;
  follow.velocity = (follow.velocity - omega * impulse * dt) * decay;
  return follow.value;
}

function updateCameraHeight(dt) {
  if (!state.cameraRoot || !state.controls) return;
  const y = stepHeightFollow(state.cameraHeight, state.cameraObservedY, dt);
  const dy = y - state.controls.target.y;
  if (Math.abs(dy) < 1e-8) return;
  state.orbitCamera.position.y += dy;
  state.controls.target.y = y;
  state.renderDirty = true;
}

function resetCamera() {
  if (!state.camera || !state.controls) return;
  const x = stats.root[0];
  const z = stats.root[2];
  state.cameraObservedY = stats.root[1];
  state.cameraHeight = createHeightFollow(state.cameraObservedY);
  state.controls.target.set(x, state.cameraObservedY, z);
  const framing = Math.max(1, Math.min(2.5, 0.85 / state.camera.aspect));
  const heading = state.latestCommand.heading-(state.config?.entry_command?.heading_offset_rad||0);
  const forward = new THREE.Vector3(Math.sin(heading), 0, Math.cos(heading));
  const right = new THREE.Vector3().crossVectors(forward, state.camera.up);
  state.orbitCamera.position.copy(state.controls.target)
    .addScaledVector(forward, -4.7 * framing).addScaledVector(right, (state.config?.entry_command?0:.45) * framing);
  state.orbitCamera.position.y += 1.4 * framing;
  state.cameraRoot = [x, z];
  updateLightRig();
  state.controls.update();
  updateCameraStats();
}

function connectSocket() {
  if (!state.config) return;
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const socketUrl = `${protocol}//${location.host}/ws`;
  let ws;
  try {
    ws = new WebSocket(socketUrl);
  } catch (error) {
    scheduleReconnect(error.message);
    return;
  }
  state.socket = ws;
  ws.binaryType = 'arraybuffer';
  setConnection('connecting', 'CONNECTING');
  el.reset.disabled = true;

  ws.addEventListener('open', () => {
    if (state.socket !== ws) return;
    state.reconnectDelay = 500;
    setConnection('connected', 'ONLINE');
    setRunStatus('Awaiting model');
  });
  ws.addEventListener('message', event => {
    if (state.socket !== ws) return;
    if (typeof event.data === 'string') {
      handleSocketJson(event.data);
    } else {
      handleSocketBinary(event.data);
    }
  });
  ws.addEventListener('error', () => {
    if (state.socket === ws) setRunStatus('Connection error', 'stalled');
  });
  ws.addEventListener('close', () => {
    if (state.socket !== ws) return;
    state.ready = false;
    state.live.epoch += 1;
    state.live.inFlight = null;
    state.live.pendingPacket = null;
    state.live.awaitingCommit = null;
    state.live.current = null;
    state.live.currentIndex = 0;
    state.live.currentStarted = false;
    state.live.next = null;
    state.live.modelStarted = false;
    state.live.resetQueued = false;
    state.live.resetPending = false;
    state.hasVisualFrame = false;
    state.requestId = 0;
    setConnection('error', 'RECONNECTING');
    el.reset.disabled = true;
    scheduleReconnect('WebSocket closed');
  });
}

function scheduleReconnect(message) {
  setConnection('error', 'RECONNECTING');
  setRunStatus('Reconnecting', 'stalled');
  if (message) showError(message);
  if (state.reconnectTimer) return;
  const delay = state.reconnectDelay;
  state.reconnectDelay = Math.min(5000, state.reconnectDelay * 1.7);
  state.reconnectTimer = window.setTimeout(() => {
    state.reconnectTimer = null;
    connectSocket();
  }, delay);
}

function handleSocketJson(text) {
  let message;
  try {
    message = JSON.parse(text);
  } catch {
    showError('Invalid server message');
    return;
  }
  if (message.type === 'ready') {
    handleReady(message);
    return;
  }
  if (message.type === 'plan') {
    handlePlanMetadata(message);
    return;
  }
  if (message.type === 'played') {
    handlePlayedAcknowledgement(message);
    return;
  }
  if (message.type === 'prefetch_deferred') {
    handlePrefetchDeferred(message);
    return;
  }
  if (message.type === 'error') {
    if(state.mode==='terrain'){haltTerrainPlayback(String(message.message??'Inference failed'));return;}
    if (state.live.inFlight && Number(message.request_id) === state.live.inFlight.id) {
      state.live.inFlight = null;
      state.live.retryAt = performance.now() + 800;
      sendQueuedReset();
    } else if (state.live.awaitingCommit !== null
      && Number(message.request_id) === state.live.awaitingCommit) {
      state.live.awaitingCommit = null;
      state.live.retryAt = performance.now() + 800;
      if (isGenerativeMode()) resetLive();
    }
    showError(String(message.message ?? 'Backend error'));
    return;
  }
  showError(`Unknown server message: ${String(message.type ?? 'missing type')}`);
}

function handleReady(message) {
  try {
    if (message.mode && message.mode !== state.mode) {
      if (isGenerativeMode() && !state.live.resetPending) resetLive();
      return;
    }
    const root = toVec3(message.root);
    const vertices = toFloatArray(message.vertices);
    const vertexCount = Number(message.vertex_count ?? state.vertexCount);
    if (vertexCount !== state.vertexCount || vertices.length !== state.vertexCount * 3) {
      throw new Error('Ready mesh differs from configured SOMA topology');
    }
    if (!state.positionAttribute) throw new Error('SOMA geometry is not initialized');
    if (!state.hasVisualFrame || state.live.resetPending) {
      if (state.mode === 'terrain' && Number.isFinite(message.heading)) {
        setCommand([0, 0], state.config?.entry_command?.heading_rad ?? message.heading);
      }
      applyFrame({ root, vertices }, { count: false, advance: false, joints: message.joints_world_xyz_m });
      state.modeFrame = 0;
      stats.frame = 0;
      if (state.live.resetPending) initializeRoot(root, true);
      if(state.config?.entry_command)resetCamera();
    }
    if (state.live.resetPending) {
      state.requestId = 0;
      state.live.epoch += 1;
      state.live.serverGeneration = null;
      state.live.awaitingCommit = null;
    }
    state.live.resetPending = false;
    state.live.resetQueued = false;
    state.live.failed=false;
    state.ready = true;
    setConnection('connected', 'ONLINE');
    el.reset.disabled = !isGenerativeMode();
    if (message.fps) stats.server_fps = Number(message.fps);
    setRunStatus(state.mode === 'terrain' ? (state.config.terrain_mode.visual_baseline_approved ? 'Accepted motion · live trial' : 'Terrain preview · not approved')
      : state.mode === 'live' ? 'Flat live ready' : 'Replay ready');
    updateReadouts();
    if (isGenerativeMode() && !state.paused) ensureLivePlan();
  } catch (error) {
    showError(error.message);
    setRunStatus('Invalid model state', 'stalled');
  }
}

function handlePlanMetadata(meta) {
  stats.scheduling.plan_timings ??= [];
  stats.scheduling.plan_timings.push({request_id:meta.request_id,neural:meta.generation?.neural_generation,
    prefetched:Boolean(meta.prefetched),inference_ms:meta.inference_ms,skin_ms:meta.skin_ms,
    queue_ms:meta.server_queue_ms,compute_ms:meta.server_compute_ms});
  if (stats.scheduling.plan_timings.length>60) stats.scheduling.plan_timings.shift();
  const boundary = stats.scheduling.boundaries.findLast(row => row.next_request_id === Number(meta.request_id));
  if (boundary) boundary.metadata_at = performance.now();
  if (state.live.pendingPacket) {
    showError('Plan metadata arrived before the previous binary block');
    return;
  }
  const request = state.live.inFlight;
  const matching = Boolean(request)
    && Number(meta.request_id) === request.id
    && meta.mode === request.mode
    && request.epoch === state.live.epoch;
  const packet = { meta, request, accepted: matching, error: null };
  if (!matching) {
    packet.error = 'Stale plan response ignored';
  } else {
    try {
      validateBlockMetadata(meta, state.vertexCount, false);
    } catch (error) {
      packet.accepted = false;
      packet.error = error.message;
    }
  }
  state.live.pendingPacket = packet;
}

async function handleSocketBinary(data) {
  const packet = state.live.pendingPacket;
  if (!packet) {
    showError('Unexpected binary frame block');
    return;
  }
  state.live.pendingPacket = null;
  let buffer = data;
  if (data instanceof Blob) buffer = await data.arrayBuffer();
  if (!(buffer instanceof ArrayBuffer)) {
    finishLiveResponse(packet, null, 'Unsupported binary frame block');
    return;
  }
  if (!packet.accepted) {
    finishLiveResponse(packet, null, packet.error);
    return;
  }
  try {
    const frames = decodeFrameBlock(buffer, packet.meta, state.vertexCount);
    const boundary = stats.scheduling.boundaries.findLast(row => row.next_request_id === Number(packet.meta.request_id));
    if (boundary) boundary.binary_at = performance.now();
    const live = state.live;
    if (!live.inFlight || live.inFlight.id !== packet.request.id || live.inFlight.epoch !== packet.request.epoch || packet.request.epoch !== live.epoch) {
      finishLiveResponse(packet, null, 'Stale plan response ignored');
      return;
    }
    if (live.resetQueued) {
      live.inFlight = null;
      sendQueuedReset();
      return;
    }
    if (live.next) throw new Error('The live frame queue is full');
    live.inFlight = null;
    live.next = Object.freeze({ frames, meta: packet.meta });
    live.modelStarted = true;
    live.retryAt = 0;
    stats.received += 1;
    stats.inference_ms = finiteOrNull(packet.meta.inference_ms);
    stats.skin_ms = finiteOrNull(packet.meta.skin_ms);
    stats.style = packet.meta.style ?? null;
    stats.generation = packet.meta.generation ?? null;
    if (packet.meta.prefetched) stats.scheduling.prefetched_blocks += 1;
    state.serverStyle = packet.meta.style ?? null;
    updateReadouts();
    if (isGenerativeMode() && !state.paused) setRunStatus(state.mode === 'terrain' ? 'Terrain preview' : 'Flat Live');
    wakeLivePlayback();
  } catch (error) {
    finishLiveResponse(packet, null, error.message);
  }
}

function finishLiveResponse(packet, _frames, message) {
  const live = state.live;
  if (live.inFlight && packet.request && live.inFlight.id === packet.request.id && live.inFlight.epoch === packet.request.epoch) {
    live.inFlight = null;
    live.retryAt = performance.now() + 800;
  }
  if (message && message !== 'Stale plan response ignored') showError(message);
  if (message && state.mode === 'terrain' && !live.resetQueued) {
    haltTerrainPlayback(message);
    return;
  }
  if (live.resetQueued) sendQueuedReset();
  if (isGenerativeMode() && state.ready && !state.paused && !live.current && !live.next) ensureLivePlan();
}

function haltTerrainPlayback(message){
  state.paused=true;state.live.failed=true;state.live.inFlight=null;
  state.live.pendingPacket=null;state.live.awaitingCommit=null;state.live.next=null;
  updatePauseButton();updateLiveStateTelemetry();showError(message);setRunStatus('Reset required','stalled');
}

function handlePlayedAcknowledgement(message) {
  const live = state.live;
  if (live.awaitingCommit === null || Number(message.request_id) !== live.awaitingCommit) {
    showError('Unexpected playback acknowledgement');
    return;
  }
  live.awaitingCommit = null;
  const boundary = stats.scheduling.boundaries.findLast(row => row.request_id === Number(message.request_id));
  if (boundary) boundary.ack_at = performance.now();
  if (live.resetQueued) {
    sendQueuedReset();
    return;
  }
  if (isGenerativeMode() && state.ready && !state.paused) ensureLivePlan();
  wakeLivePlayback();
}

function wakeLivePlayback() {
  const live = state.live;
  const now = performance.now();
  if (!isGenerativeMode() || state.paused || !state.ready || live.current || !live.next
    || live.awaitingCommit !== null || state.lastPlayedAt === null
    || now - state.lastPlayedAt < TICK_MS) return;
  // A just-late packet must not wait an additional timer period. Publish one
  // due frame, then retain full spacing; never catch up by bursting/skipping.
  stepLive();
  state.tickDeadline = performance.now() + TICK_MS;
}

function handlePrefetchDeferred(message) {
  const live = state.live;
  const request = live.inFlight;
  if (!request || request.kind !== 'prefetch' || request.id !== Number(message.request_id)
    || request.sourceRequestId !== Number(message.source_request_id)) return;
  live.inFlight = null;
  if (live.resetQueued || request.epoch !== live.epoch) {
    sendQueuedReset();
    return;
  }
  stats.scheduling.deferred_prefetches += 1;
  ensureLivePlan();
}

function validateBlockMetadata(meta, vertexCount, replay) {
  const frameCount = Number(meta.frame_count);
  const actualVertexCount = Number(meta.vertex_count);
  const stride = Number(meta.frame_stride_floats);
  const encoding = meta.encoding ?? (replay ? 'float32-le' : null);
  if (!Number.isInteger(frameCount) || frameCount < 1 || frameCount > MAX_BLOCK_FRAMES
    || (!replay && frameCount !== MAX_BLOCK_FRAMES)) {
    throw new Error(replay ? `Frame block must contain 1 to ${MAX_BLOCK_FRAMES} frames`
      : `Live plan must contain exactly ${MAX_BLOCK_FRAMES} frames`);
  }
  if (actualVertexCount !== vertexCount) throw new Error('Frame block vertex count differs from the SOMA mesh');
  const rig = meta.geometry_format === 'soma77-rig-v1';
  if (rig && (meta.rig_count !== 77 || meta.influences_per_vertex !== state.config.soma_skin?.influences_per_vertex || !state.skinMesh)) {
    throw new Error('Official SOMA rig assets are unavailable');
  }
  if (stride !== 3 + (rig ? 77 * 16 : vertexCount * 3)) throw new Error('Frame block stride is invalid');
  if (encoding !== 'float32-le') throw new Error('Unsupported frame encoding');
}

function validateClipMetadata(meta, vertexCount) {
  const frameCount = Number(meta.frame_count);
  const actualVertexCount = Number(meta.vertex_count);
  const stride = Number(meta.frame_stride_floats);
  if (!Number.isInteger(frameCount) || frameCount < 1) throw new Error('Replay clip contains no frames');
  if (actualVertexCount !== vertexCount) throw new Error('Replay vertex count differs from the SOMA mesh');
  if (stride !== 3 + vertexCount * 3) throw new Error('Replay frame stride is invalid');
  if (meta.encoding !== undefined && meta.encoding !== 'float32-le') throw new Error('Unsupported replay encoding');
}

function decodeFrameBlock(buffer, meta, vertexCount) {
  validateBlockMetadata(meta, vertexCount, true);
  const frameCount = Number(meta.frame_count);
  const stride = Number(meta.frame_stride_floats);
  const expectedBytes = frameCount * stride * Float32Array.BYTES_PER_ELEMENT;
  if (buffer.byteLength !== expectedBytes) {
    throw new Error(`Frame block size mismatch (${buffer.byteLength} bytes)`);
  }
  let floats;
  if (LITTLE_ENDIAN) {
    floats = new Float32Array(buffer);
  } else {
    floats = new Float32Array(buffer.byteLength / 4);
    const view = new DataView(buffer);
    for (let i = 0; i < floats.length; i += 1) floats[i] = view.getFloat32(i * 4, true);
  }
  const frames = new Array(frameCount);
  for (let index = 0; index < frameCount; index += 1) {
    const offset = index * stride;
    const root = floats.subarray(offset, offset + 3);
    const values = floats.subarray(offset + 3, offset + stride);
    frames[index] = Object.freeze(meta.geometry_format === 'soma77-rig-v1' ? { root, rig: values } : { root, vertices: values });
  }
  return Object.freeze(frames);
}

function ensureLivePlan() {
  const live = state.live;
  const socketOpen = state.socket?.readyState === WebSocket.OPEN;
  if (!state.ready || !socketOpen || !isGenerativeMode() || state.paused) return;
  if (live.inFlight || live.pendingPacket || live.next || live.awaitingCommit !== null) return;
  if (live.current && !live.currentStarted) return;
  if (state.mode === 'terrain' && live.current) return;
  if (performance.now() < live.retryAt) return;

  const id = ++state.requestId;
  const request = { id, epoch: live.epoch, mode: state.mode };
  const command = state.latestCommand;
  const message = {
    type: 'plan',
    mode: state.mode,
    velocity_xz: [...command.velocity_xz],
    heading: command.heading,
    request_id: id,
    ...(state.runtimeControls ? {runtime_controls:{...state.runtimeControls}} : {}),
  };
  try {
    live.inFlight = request;
    state.socket.send(JSON.stringify(message));
  } catch (error) {
    live.inFlight = null;
    live.retryAt = performance.now() + 800;
    showError(`Plan request failed: ${error.message}`);
  }
}

function prefetchTerrainPlan() {
  const live = state.live;
  const prefix = state.config?.terrain_mode?.prefetch?.played_frame_count;
  if (state.mode !== 'terrain' || !['played-buffer-v2','committed-buffer-v3'].includes(state.config?.terrain_mode?.prefetch?.version)
    || state.paused || live.resetQueued || live.resetPending || !state.ready
    || state.socket?.readyState !== WebSocket.OPEN || live.inFlight || live.pendingPacket
    || live.next || live.awaitingCommit !== null || !live.current || live.currentIndex !== prefix) return;
  const id = ++state.requestId;
  const command = state.latestCommand;
  const request = { id, epoch: live.epoch, mode: 'terrain', kind: 'prefetch',
    sourceRequestId: live.current.meta.request_id };
  try {
    live.inFlight = request;
    state.socket.send(JSON.stringify({ type: 'prefetch', mode: 'terrain', request_id: id,
      source_request_id: live.current.meta.request_id, played_frame_count: live.currentIndex,
      velocity_xz: [...command.velocity_xz], heading: command.heading,
      ...(state.runtimeControls ? {runtime_controls:{...state.runtimeControls}} : {}) }));
    stats.scheduling.prefetch_requests += 1;
  } catch (error) {
    live.inFlight = null;
    showError(`Prefetch request failed: ${error.message}`);
    resetLive();
  }
}

function resetLive() {
  if (state.socket?.readyState !== WebSocket.OPEN || !isGenerativeMode()) return;
  const live = state.live;
  live.epoch += 1;
  if (live.pendingPacket) live.pendingPacket.accepted = false;
  live.current = null;
  live.currentIndex = 0;
  live.currentStarted = false;
  live.next = null;
  live.modelStarted = false;
  live.resetQueued = true;
  live.retryAt = 0;
  state.ready = false;
  state.modeFrame = 0;
  stats.frame = 0;
  clearDiagnostics();
  state.lastPlayedAt = null;
  stats.scheduling.frame_intervals_ms = [];
  stats.scheduling.prefetch_requests = 0;
  stats.scheduling.prefetched_blocks = 0;
  stats.scheduling.deferred_prefetches = 0;
  stats.scheduling.clock_rebases = 0;
  stats.scheduling.commit_wait_ticks = 0;
  stats.scheduling.boundaries = [];
  stats.scheduling.plan_timings = [];
  stats.underruns = 0;
  el.reset.disabled = true;
  setRunStatus('Resetting');
  sendQueuedReset();
}

function sendQueuedReset() {
  const live = state.live;
  if (!live.resetQueued || live.inFlight || live.pendingPacket || live.awaitingCommit !== null
      || state.socket?.readyState !== WebSocket.OPEN) return;
  live.resetQueued = false;
  live.resetPending = true;
  try {
    state.socket.send(JSON.stringify({ type: 'reset', mode: state.mode,
      ...(state.mode === 'terrain' && state.config?.world ? {spawn_id: el.worldSelect.value} : {}) }));
  } catch (error) {
    showError(`Reset request failed: ${error.message}`);
    live.resetPending = false;
  }
}

function togglePause() {
  if(state.live.failed){setRunStatus('Reset required','stalled');return;}
  const restartReplay = state.paused && state.mode === 'replay' && replayIsComplete();
  state.paused = !state.paused;
  updatePauseButton();
  updateLiveStateTelemetry();
  if (restartReplay) {
    loadReplayClip(el.clipSelect.value);
    return;
  }
  if (state.paused) {
    setRunStatus('Paused');
  } else if (isGenerativeMode()) {
    state.lastPlayedAt=null;
    ensureLivePlan();
    setRunStatus(state.mode === 'terrain' ? 'Terrain preview' : 'Flat Live');
  } else {
    setRunStatus(state.replay.meta ? 'Replay' : 'Loading replay');
  }
}

function switchMode(mode) {
  if (mode === state.mode) return;
  if (mode === 'replay' && !state.clips.length) return;
  if (mode === 'terrain' && !terrainModeAvailable()) return;
  state.mode = mode;
  clearDiagnostics();
  state.lastPlayedAt = null;
  stats.scheduling.frame_intervals_ms = [];
  state.modeFrame = 0;
  stats.frame = 0;
  state.keyState.clear();
  state.touchState.clear();
  el.touchPad.querySelectorAll('.is-down').forEach(button => button.classList.remove('is-down'));
  if (isGenerativeMode()) updateCommandFromInput();
  el.liveMode.classList.toggle('is-active', mode === 'live');
  el.terrainMode.classList.toggle('is-active', mode === 'terrain');
  el.replayMode.classList.toggle('is-active', mode === 'replay');
  el.liveMode.setAttribute('aria-pressed', String(mode === 'live'));
  el.terrainMode.setAttribute('aria-pressed', String(mode === 'terrain'));
  el.replayMode.setAttribute('aria-pressed', String(mode === 'replay'));
  el.clipSelect.hidden = mode !== 'replay' || !state.clips.length;
  el.clipSelect.disabled = mode !== 'replay' || !state.clips.length;
  el.speed.closest('.speed-control').hidden = !isGenerativeMode();
  el.speed.disabled = !isGenerativeMode();
  el.touchPad.hidden = !isGenerativeMode();
  el.reset.disabled = !isGenerativeMode() || !state.ready || state.socket?.readyState !== WebSocket.OPEN;
  el.terrainWarning.hidden = mode !== 'terrain';
  state.floor.visible = mode === 'live';
  stats.control.mode = mode;
  state.intentArrow.visible = false;
  updateReadouts();

  if (mode === 'replay') {
    const clipId = el.clipSelect.value || String(state.clips[0].id);
    if (el.clipSelect.value !== clipId) el.clipSelect.value = clipId;
    loadReplayClip(clipId);
  } else {
    cancelReplayLoad();
    if (mode === 'terrain') createTerrain(state.config.terrain_scene);
    else clearTerrain();
    followRoot(stats.root);
    resetLive();
    setRunStatus('Resetting');
    updateIntentArrow();
  }
}

function cancelReplayLoad() {
  const replay = state.replay;
  replay.epoch += 1;
  replay.inFlight?.controller.abort();
  replay.inFlight = null;
  replay.meta = null;
  replay.current = null;
  replay.next = null;
  replay.currentIndex = 0;
  replay.currentStarted = false;
  replay.nextStart = 0;
}

async function loadReplayClip(clipId) {
  clearDiagnostics();
  const replay = state.replay;
  state.paused = false;
  updatePauseButton();
  replay.epoch += 1;
  const epoch = replay.epoch;
  replay.inFlight?.controller.abort();
  replay.inFlight = null;
  replay.meta = null;
  replay.current = null;
  replay.next = null;
  replay.currentIndex = 0;
  replay.currentStarted = false;
  replay.nextStart = 0;
  replay.retryAt = 0;
  clearTerrain();
  state.modeFrame = 0;
  stats.frame = 0;
  updateReadouts();
  setRunStatus('Loading replay');

  try {
    const url = `/api/clips/${encodeURIComponent(clipId)}`;
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Clip request failed (${response.status})`);
    const meta = await response.json();
    if (epoch !== replay.epoch || state.mode !== 'replay') return;
    validateClipMetadata(meta, state.vertexCount);
    replay.meta = meta;
    stats.replay = { id: meta.id, role: meta.role ?? 'source', task: meta.task ?? 'offline physics replay' };
    createTerrain(meta.terrain);
    followRoot(stats.root);
    const firstCount = Math.min(MAX_BLOCK_FRAMES, Number(meta.frame_count));
    const first = await fetchClipChunk(clipId, 0, firstCount, epoch);
    if (epoch !== replay.epoch || state.mode !== 'replay') return;
    replay.current = first;
    replay.currentIndex = 0;
    replay.currentStarted = false;
    replay.nextStart = first.end;
    setRunStatus('Replay ready');
  } catch (error) {
    if (epoch !== replay.epoch || error.name === 'AbortError') return;
    showError(error.message);
    setRunStatus('Replay unavailable', 'stalled');
  }
}

async function fetchClipChunk(clipId, start, count, epoch) {
  const replay = state.replay;
  const controller = new AbortController();
  const request = { start, epoch, controller };
  replay.inFlight = request;
  try {
    const query = new URLSearchParams({ start: String(start), count: String(count) });
    const response = await fetch(`/api/clips/${encodeURIComponent(clipId)}/frames?${query}`, {
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Replay frames request failed (${response.status})`);
    const buffer = await response.arrayBuffer();
    if (epoch !== replay.epoch) return null;
    const meta = {
      frame_count: count,
      vertex_count: Number(replay.meta.vertex_count),
      frame_stride_floats: Number(replay.meta.frame_stride_floats),
      encoding: replay.meta.encoding ?? 'float32-le',
    };
    const frames = decodeFrameBlock(buffer, meta, state.vertexCount);
    return Object.freeze({ start, end: start + count, frames });
  } finally {
    if (replay.inFlight === request) replay.inFlight = null;
  }
}

function prefetchReplayChunk() {
  const replay = state.replay;
  if (state.mode !== 'replay' || !replay.meta || !replay.current || replay.inFlight || replay.next) return;
  if (replay.nextStart >= Number(replay.meta.frame_count) || performance.now() < replay.retryAt) return;
  const start = replay.nextStart;
  const count = Math.min(MAX_BLOCK_FRAMES, Number(replay.meta.frame_count) - start);
  const clipId = replay.meta.id ?? el.clipSelect.value;
  const epoch = replay.epoch;
  fetchClipChunk(clipId, start, count, epoch).then(block => {
    if (!block || epoch !== replay.epoch || state.mode !== 'replay') return;
    replay.next = block;
    replay.nextStart = block.end;
    replay.retryAt = 0;
  }).catch(error => {
    if (epoch !== replay.epoch || error.name === 'AbortError') return;
    replay.retryAt = performance.now() + 900;
    showError(error.message);
  });
}

function startSimulationClock() {
  state.tickDeadline = performance.now() + TICK_MS;
  const tick = () => {
    const now = performance.now();
    if (now >= state.tickDeadline) {
      const deadline = nextPlaybackDeadline(state.tickDeadline, now);
      if (deadline - state.tickDeadline > TICK_MS + 1e-4) stats.scheduling.clock_rebases += 1;
      state.tickDeadline = deadline;
      simulationStep();
    }
    window.setTimeout(tick, Math.max(0, state.tickDeadline - performance.now()));
  };
  window.setTimeout(tick, TICK_MS);
}

function nextPlaybackDeadline(previous, now) {
  const next = previous + TICK_MS;
  // Preserve phase for small timer delays; severe overruns never burst or skip motion.
  return next - now < 20 ? now + TICK_MS : next;
}

function simulationStep() {
  if (state.paused || !state.mesh) {
    if (state.paused) setTelemetryState('paused');
    return;
  }
  if (isGenerativeMode()) {
    stepLive();
  } else {
    stepReplay();
  }
}

function stepLive() {
  const live = state.live;
  if (!live.modelStarted) {
    ensureLivePlan();
    return;
  }
  if (!live.current && live.next && live.awaitingCommit === null) {
    live.current = live.next;
    live.next = null;
    live.currentIndex = 0;
    live.currentStarted = false;
  }
  if (!live.current) {
    if (live.awaitingCommit !== null) {
      stats.scheduling.commit_wait_ticks = (stats.scheduling.commit_wait_ticks ?? 0) + 1;
      setRunStatus('Confirming played frames');
      return;
    }
    stats.underruns += 1;
    updateReadouts();
    setRunStatus('Buffer wait', 'stalled');
    setTelemetryState('stalled');
    ensureLivePlan();
    return;
  }
  if (!live.currentStarted) {
    live.currentStarted = true;
    acceptDiagnosticPlan(live.current.meta);
    if (state.mode === 'live') ensureLivePlan();
    if (state.mode === 'terrain' && state.config?.terrain_mode?.prefetch?.played_frame_count===0) prefetchTerrainPlan();
  }
  const frame = live.current.frames[live.currentIndex];
  if (!frame) {
    live.current = null;
    live.currentIndex = 0;
    live.currentStarted = false;
    if (state.mode === 'terrain') haltTerrainPlayback('Invalid motion frame queue');
    return;
  }
  state.diagnostics.consumed = live.currentIndex + 1;
  const boneCheck=live.current.meta.debug?.bone_sdf;
  applyFrame(frame, { joints: live.current.meta.debug?.joints_world_xyz_m?.[live.currentIndex],
    boneSdf:boneCheck ? {role:boneCheck.role,distances:boneCheck.signed_distance_m?.[live.currentIndex],
      valid:boneCheck.valid?.[live.currentIndex]} : null });
  live.currentIndex += 1;
  if (live.currentIndex === state.config?.terrain_mode?.prefetch?.played_frame_count) prefetchTerrainPlan();
  setRunStatus(state.mode === 'terrain' ? 'Terrain preview' : 'Flat Live');
  setTelemetryState('running');
  if (live.currentIndex >= live.current.frames.length) {
    const completed = live.current;
    live.current = null;
    live.currentIndex = 0;
    live.currentStarted = false;
    if (state.mode === 'terrain') acknowledgePlayedBlock(completed);
  }
}

function acknowledgePlayedBlock(block) {
  const live = state.live;
  const requestId = Number(block?.meta?.request_id);
  if (!Number.isInteger(requestId) || state.socket?.readyState !== WebSocket.OPEN) {
    showError('Cannot confirm the played motion block');
    if (isGenerativeMode()) resetLive();
    return;
  }
  live.awaitingCommit = requestId;
  try {
    const message = { type: 'played', request_id: requestId, frame_count: block.frames.length };
    const boundary = {request_id:requestId,sent_at:performance.now(),last_played_at:state.lastPlayedAt};
    if (!live.next && !live.inFlight && !live.pendingPacket && !live.resetQueued && !state.paused) {
      const id = ++state.requestId;
      live.inFlight = { id, epoch: live.epoch, mode: state.mode };
      message.next_plan = { type: 'plan', mode: state.mode, request_id: id,
        velocity_xz: [...state.latestCommand.velocity_xz], heading: state.latestCommand.heading,
        ...(state.runtimeControls ? {runtime_controls:{...state.runtimeControls}} : {}) };
      boundary.next_request_id = id;
    }
    stats.scheduling.boundaries.push(boundary);
    if (stats.scheduling.boundaries.length > 100) stats.scheduling.boundaries.shift();
    state.socket.send(JSON.stringify(message));
  } catch (error) {
    live.awaitingCommit = null;
    showError(`Playback acknowledgement failed: ${error.message}`);
    if (isGenerativeMode()) resetLive();
  }
}

function stepReplay() {
  const replay = state.replay;
  if (!replay.meta || !replay.current) return;
  if (!replay.currentStarted) {
    replay.currentStarted = true;
    prefetchReplayChunk();
  }
  if (!replay.current || replay.currentIndex >= replay.current.frames.length) {
    if (replay.next) {
      replay.current = replay.next;
      replay.next = null;
      replay.currentIndex = 0;
      replay.currentStarted = false;
    } else if (replay.nextStart < Number(replay.meta.frame_count)) {
      stats.underruns += 1;
      updateReadouts();
      setRunStatus('Replay buffer wait', 'stalled');
      setTelemetryState('stalled');
      prefetchReplayChunk();
      return;
    } else {
      state.paused = true;
      updatePauseButton();
      setRunStatus('Replay complete', 'paused');
      setTelemetryState('paused');
      return;
    }
  }
  if (!replay.currentStarted) {
    replay.currentStarted = true;
    prefetchReplayChunk();
  }
  const frame = replay.current.frames[replay.currentIndex];
  if (!frame) return;
  applyFrame(frame);
  replay.currentIndex += 1;
  setRunStatus('Replay');
  setTelemetryState('running');
  prefetchReplayChunk();
}

function updatePauseButton() {
  el.pause.setAttribute('aria-pressed', String(state.paused));
  el.pause.title = state.paused ? 'Resume motion' : 'Pause motion';
  el.pause.setAttribute('aria-label', el.pause.title);
  el.pause.innerHTML = `<i data-lucide="${state.paused ? 'play' : 'pause'}" aria-hidden="true"></i>`;
  el.telemetry.dataset.state = state.paused ? 'paused' : '';
  refreshIcons();
}

function toggleDepthPreview() {
  state.depthEnabled = !state.depthEnabled;
  el.depthInset.hidden = !state.depthEnabled;
  el.depthToggle.setAttribute('aria-pressed', String(state.depthEnabled));
  el.depthToggle.title = state.depthEnabled ? 'Hide depth observation preview' : 'Show depth observation preview';
  el.depthToggle.setAttribute('aria-label', el.depthToggle.title);
  el.depthToggle.innerHTML = `<i data-lucide="${state.depthEnabled ? 'eye' : 'eye-off'}" aria-hidden="true"></i>`;
  stats.observation.enabled = state.depthEnabled;
  stats.observation.connected_to_model = false;
  state.lastDepthRender = 0;
  state.depthHasFrame = false;
  refreshIcons();
}

function replayIsComplete() {
  const replay = state.replay;
  return Boolean(replay.meta && replay.current && !replay.next
    && replay.currentIndex >= replay.current.frames.length
    && replay.nextStart >= Number(replay.meta.frame_count));
}

function startRenderLoop() {
  let previousTime = null;
  const render = time => {
    window.requestAnimationFrame(render);
    // Cap time after a hidden tab resumes so the observation camera never snaps.
    const dt = previousTime === null ? 0 : Math.min(0.05, (time - previousTime) / 1000);
    previousTime = time;
    updateCameraHeight(dt);
    state.controls?.update();
    updateThirdPersonCamera();
    if (state.renderer && state.scene && state.camera && (state.renderDirty || state.depthEnabled)) {
      state.renderDirty = false;
      state.renderer.setRenderTarget(null);
      state.renderer.setScissorTest(false);
      state.renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
      state.renderer.render(state.scene, state.camera);
      const now = performance.now();
      if (state.depthEnabled) {
        if (!state.depthHasFrame || now - state.lastDepthRender >= DEPTH_RENDER_INTERVAL) {
          state.lastDepthRender = now;
          captureDepthObservation();
        }
        blitDepthObservation();
      }
    }
  };
  window.requestAnimationFrame(render);
}

function captureDepthObservation() {
  const renderer = state.renderer;
  const scene = state.scene;
  const camera = state.depthCamera;
  const target = state.depthTarget;
  if (!renderer || !scene || !camera || !target || !state.depthEnabled) return;

  const root = stats.root;
  const direction = state.lastIntentDirection;
  const sensorPosition = new THREE.Vector3(root[0], root[1] + 0.08, root[2]);
  camera.position.copy(sensorPosition);
  camera.lookAt(sensorPosition.x + direction[0] * 10, sensorPosition.y, sensorPosition.z + direction[1] * 10);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  stats.observation = {
    enabled: true,
    connected_to_model: false,
    kind: 'depth-preview',
    intent_xz: [...direction],
    sensor_root: [...root],
  };

  const previousBackground = scene.background;
  const previousOverride = scene.overrideMaterial;
  const previousClear = renderer.getClearColor(new THREE.Color()).clone();
  const previousAlpha = renderer.getClearAlpha();
  try {
    scene.background = null;
    scene.overrideMaterial = state.depthMaterial;
    renderer.setRenderTarget(target);
    renderer.setClearColor(0xffffff, 1);
    renderer.clear(true, true, true);
    renderer.render(scene, camera);
  } finally {
    scene.background = previousBackground;
    scene.overrideMaterial = previousOverride;
    renderer.setRenderTarget(null);
    renderer.setClearColor(previousClear, previousAlpha);
  }

  state.depthHasFrame = true;
}

function blitDepthObservation() {
  const renderer = state.renderer;
  if (!renderer || !state.depthHasFrame) return;
  const frameRect = el.depthFrame.getBoundingClientRect();
  const canvasRect = el.canvas.getBoundingClientRect();
  if (!frameRect.width || !frameRect.height || !canvasRect.width || !canvasRect.height) return;
  const x = frameRect.left - canvasRect.left;
  const y = canvasRect.bottom - frameRect.bottom;
  renderer.setViewport(x, y, frameRect.width, frameRect.height);
  renderer.setScissor(x, y, frameRect.width, frameRect.height);
  renderer.setScissorTest(true);
  renderer.render(state.depthBlitScene, state.depthBlitCamera);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, window.innerWidth, window.innerHeight);
}

function resizeRenderer() {
  state.renderDirty = true;
  fitInspectionPanels();
  if (!state.renderer) return;
  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);
  state.renderer.setSize(width, height, false);
  state.camera.aspect = width / height;
  state.camera.updateProjectionMatrix();
  if (state.orbitCamera) {
    state.orbitCamera.aspect = state.camera.aspect;
    state.orbitCamera.updateProjectionMatrix();
  }
}

function createTerrain(terrainData) {
  state.renderDirty = true;
  clearTerrain();
  const rows = heightfieldRows(terrainData?.hf);
  if (!state.scene) return;
  const origin = terrainData?.min_point ? toVec3(terrainData.min_point) : [0, 0, 0];
  const spacing = terrainSpacing(terrainData?.dx);
  if (rows.length) state.terrainInfo = { rows, origin, spacing };

  const officialMesh = terrainData?.mesh;
  if (officialMesh?.vertices && officialMesh?.faces) {
    const vertices = toFloatArray(officialMesh.vertices);
    const faces = flattenNumbers(officialMesh.faces);
    if (vertices.length % 3 !== 0 || faces.length < 3) throw new Error('Official replay terrain mesh is invalid');
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geometry.setIndex(faces);
    if (officialMesh.colors) {
      const colors = toFloatArray(officialMesh.colors);
      if (colors.length !== vertices.length) throw new Error('Terrain colors differ from the mesh vertices');
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    const material = new THREE.MeshStandardMaterial({
      color: officialMesh.colors ? '#ffffff' : '#99a39f', vertexColors: Boolean(officialMesh.colors),
      roughness: 0.94, metalness: 0.01, side: THREE.DoubleSide,
    });
    const terrainMesh = new THREE.Mesh(geometry, material);
    terrainMesh.name = officialMesh.semantics || 'Official PARC voxelized terrain';
    terrainMesh.castShadow = true;
    terrainMesh.receiveShadow = true;
    terrainMesh.layers.enable(DEPTH_LAYER);
    state.terrain = terrainMesh;
    state.scene.add(terrainMesh);
    return;
  }

  if (!rows.length) return;
  let minHeight = Infinity;
  for (const row of rows) {
    if (!row || typeof row[Symbol.iterator] !== 'function') continue;
    for (const value of row) {
      const height = finiteHeight(value);
      if (height !== null) minHeight = Math.min(minHeight, height);
    }
  }
  if (!Number.isFinite(minHeight)) return;
  const baseY = origin[1] + minHeight - Math.max(0.5, Math.max(...spacing) * 0.5);
  state.terrainInfo = { rows, origin, spacing, baseY };
  const cells = [];
  for (let row = 0; row < rows.length; row += 1) {
    const values = rows[row];
    if (!values || typeof values[Symbol.iterator] !== 'function') continue;
    let column = 0;
    for (const rawHeight of values) {
      const height = finiteHeight(rawHeight);
      if (height !== null) {
        const topY = origin[1] + height;
        cells.push({
          x: origin[0] + column * spacing[0],
          y: (topY + baseY) * 0.5,
          z: origin[2] + row * spacing[1],
          sx: spacing[0],
          sy: topY - baseY,
          sz: spacing[1],
          height,
        });
      }
      column += 1;
    }
  }
  if (!cells.length) return;
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshStandardMaterial({ color: '#99a39f', roughness: 0.94, metalness: 0.01 });
  const instances = new THREE.InstancedMesh(geometry, material, cells.length);
  instances.name = 'Replay voxel heightfield';
  instances.castShadow = true;
  instances.receiveShadow = true;
  instances.layers.enable(DEPTH_LAYER);
  const transform = new THREE.Object3D();
  for (let i = 0; i < cells.length; i += 1) {
    const cell = cells[i];
    transform.position.set(cell.x, cell.y, cell.z);
    transform.scale.set(cell.sx, cell.sy, cell.sz);
    transform.updateMatrix();
    instances.setMatrixAt(i, transform.matrix);
    const shade = THREE.MathUtils.clamp(0.92 + cell.height * 0.025, 0.92, 1.08);
    instances.setColorAt(i, new THREE.Color().setRGB(shade, shade, shade));
  }
  instances.instanceMatrix.needsUpdate = true;
  if (instances.instanceColor) instances.instanceColor.needsUpdate = true;
  state.terrain = instances;
  state.scene.add(instances);
}

function heightfieldRows(hf) {
  let source = hf;
  if (!Array.isArray(hf) && !ArrayBuffer.isView(hf) && hf && typeof hf === 'object') {
    if (Array.isArray(hf.data) || ArrayBuffer.isView(hf.data)) source = hf.data;
    else if (Array.isArray(hf.values) || ArrayBuffer.isView(hf.values)) source = hf.values;
    else return [];
  }
  if (!source) return [];
  if (ArrayBuffer.isView(source)) {
    const shape = hf?.shape;
    if (Array.isArray(shape) && shape.length >= 2) {
      const rowCount = Number(shape[0]);
      const columnCount = Number(shape[1]);
      const rows = [];
      for (let row = 0; row < rowCount; row += 1) {
        rows.push(source.subarray(row * columnCount, (row + 1) * columnCount));
      }
      return rows;
    }
    return [source];
  }
  if (!Array.isArray(source)) return [];
  if (source.length && (Array.isArray(source[0]) || ArrayBuffer.isView(source[0]))) return source;
  if (!Array.isArray(source) || !source.length) return [];
  const shape = !Array.isArray(hf) && Array.isArray(hf?.shape) ? hf.shape : null;
  if (shape && shape.length >= 2) {
    const columns = Number(shape[1]);
    if (!Number.isInteger(columns) || columns < 1) return [];
    const rows = [];
    for (let index = 0; index < source.length; index += columns) rows.push(source.slice(index, index + columns));
    return rows;
  }
  if (Array.isArray(source[0]) || ArrayBuffer.isView(source[0])) return source;
  return [source];
}

function terrainSpacing(dx) {
  if (Array.isArray(dx) || ArrayBuffer.isView(dx)) return [positive(dx[0], 1), positive(dx[1] ?? dx[0], 1)];
  if (dx && typeof dx === 'object') return [positive(dx.x, 1), positive(dx.z ?? dx.y, 1)];
  return [positive(dx, 1), positive(dx, 1)];
}

function clearTerrain() {
  if (state.terrain) {
    state.scene.remove(state.terrain);
    state.terrain.geometry.dispose();
    state.terrain.material.dispose();
    state.terrain = null;
  }
  state.terrainInfo = null;
}

function terrainGroundAt(x, z) {
  const info = state.terrainInfo;
  if (!info) return 0;
  const column = Math.floor((x - info.origin[0]) / info.spacing[0] + 0.5);
  const row = Math.floor((z - info.origin[2]) / info.spacing[1] + 0.5);
  if (row < 0 || row >= info.rows.length) return 0;
  const values = info.rows[row];
  if (!values || column < 0 || column >= values.length) return 0;
  const height = finiteHeight(values[column]);
  return height !== null ? info.origin[1] + height : 0;
}

function finiteHeight(value) {
  if (value === null || value === undefined || value === '') return null;
  const height = Number(value);
  return Number.isFinite(height) ? height : null;
}

function updateLiveStateTelemetry() {
  el.telemetry.dataset.liveState=JSON.stringify({ready:state.ready,paused:state.paused,
    model_started:state.live.modelStarted,in_flight:state.live.inFlight?.id??null,
    received:stats.received,applied:stats.applied,reset_pending:state.live.resetPending,
    reset_queued:state.live.resetQueued,command:state.latestCommand,
    submitted_command:state.diagnostics.plan?.input_intent??null});
}

function updateReadouts() {
  updateLiveStateTelemetry();
  drawWorldMap();
  updateRuntimeReadout();
  el.frame.textContent = String(state.modeFrame);
  const intervals=stats.scheduling.frame_intervals_ms.slice(-120);
  if(intervals.length){
    const mean=intervals.reduce((a,b)=>a+b,0)/intervals.length;
    const sorted=[...intervals].sort((a,b)=>a-b);
    el.fps.textContent=(1000/mean).toFixed(1);
    el.telemetry.dataset.playbackTiming=JSON.stringify({frames:stats.frame,count:intervals.length,fps:1000/mean,
      mean_ms:mean,p95_ms:sorted[Math.floor((sorted.length-1)*.95)],max_ms:sorted.at(-1),
      prefetched_blocks:stats.scheduling.prefetched_blocks,deferred_prefetches:stats.scheduling.deferred_prefetches,
      underruns:stats.underruns,clock_rebases:stats.scheduling.clock_rebases,
      commit_wait_ticks:stats.scheduling.commit_wait_ticks??0,
      recent_ack_ms:stats.scheduling.boundaries.filter(row=>Number.isFinite(row.ack_at)).slice(-9).map(row=>row.ack_at-row.sent_at),
      recent_plans:(stats.scheduling.plan_timings??[]).slice(-9)});
  }else{el.fps.textContent='--';delete el.telemetry.dataset.playbackTiming;}
  el.root.textContent = stats.root.map(value => Number(value).toFixed(2)).join(', ');
  el.style.textContent = state.mode === 'replay' ? String(state.replay.meta?.role ?? 'replay').toUpperCase()
    : state.mode === 'terrain' ? 'EXPERIMENTAL' : displayStyle(stats.style);
  el.inference.textContent = !isGenerativeMode() || stats.inference_ms === null ? '-- ms' : `${stats.inference_ms.toFixed(1)} ms`;
  el.skin.textContent = !isGenerativeMode() || stats.skin_ms === null ? '-- ms' : `${stats.skin_ms.toFixed(1)} ms`;
  el.underruns.textContent = String(stats.underruns);
}

let worldMapBackground = null;
let worldMapLastDraw = 0;
function fitInspectionPanels(preferred = 'map') {
  if (window.innerWidth > 700) return;
  if (preferred === 'diagnostics' && !el.diagnosticsPanel.hidden) {
    el.worldMapPanel.hidden = true;
    el.worldMapToggle.setAttribute('aria-pressed', 'false');
  } else if (preferred === 'runtime' && !el.runtimePanel.hidden) {
    el.worldMapPanel.hidden = true;
    el.worldMapToggle.setAttribute('aria-pressed','false');
  } else if (!el.worldMapPanel.hidden) {
    el.diagnosticsPanel.hidden = true;
    el.diagnosticsToggle.setAttribute('aria-expanded', 'false');
    el.runtimePanel.hidden = true;
    el.runtimeToggle.setAttribute('aria-expanded','false');
  }
}

function initializeRuntimeControls() {
  const config = state.config.runtime_controls;
  if (!config) return;
  state.runtimeControls = {...config.defaults};
  el.speed.max=String(config.max_speed_mps);
  el.runtimeToggle.hidden = false;
  el.runtimeToggle.onclick = () => {
    el.runtimePanel.hidden = !el.runtimePanel.hidden;
    el.runtimeToggle.setAttribute('aria-expanded',String(!el.runtimePanel.hidden));
    if (!el.runtimePanel.hidden) {
      el.diagnosticsPanel.hidden = true;
      el.diagnosticsToggle.setAttribute('aria-expanded','false');
      fitInspectionPanels('runtime');
    }
  };
  const input = id => document.getElementById(`runtime-${id}`);
  const setValues = values => {
    state.runtimeControls = {...values};
    input('horizon').value=values.num_tokens;
    input('steps').value=values.inference_steps;
    input('auto').checked=values.lookahead_m<0;
    input('ahead').disabled=values.lookahead_m<0;
    input('gain').disabled=values.lookahead_m>=0;
    input('base').disabled=values.lookahead_m>=0;
    if(values.lookahead_m>=0) input('ahead').value=values.lookahead_m;
    input('width').value=values.lateral_halfwidth_m;
    input('gain').value=values.lookahead_gain ?? 1;
    input('base').value=values.lookahead_base_m ?? .4;
    input('refresh').value=values.refresh_frames;
    input('path').value=values.path_refinements ?? 0;
    updateRuntimeReadout();
  };
  for (const name of ['horizon','steps','auto','ahead','width','gain','base','refresh','path']) {
    input(name).addEventListener('input',()=> {
      state.runtimeControls = {num_tokens:Number(input('horizon').value),
        inference_steps:Number(input('steps').value),
        lookahead_m:input('auto').checked ? -1 : Number(input('ahead').value),
        lateral_halfwidth_m:Number(input('width').value),refresh_frames:Number(input('refresh').value),
        lookahead_gain:Number(input('gain').value),lookahead_base_m:Number(input('base').value),
        path_refinements:Number(input('path').value)};
      input('ahead').disabled=input('auto').checked;
      input('gain').disabled=!input('auto').checked;
      input('base').disabled=!input('auto').checked;
      input('preset').value='custom';
      updateRuntimeReadout();
    });
  }
  input('preset').onchange=()=> {
    const key=input('preset').value;
    setValues(key==='original' ? config.trained_defaults : key==='long'
      ? {...config.defaults,num_tokens:16} : config.defaults);
  };
  input('reset').onclick=()=>{input('preset').value='responsive';setValues(config.defaults);};
  input('speed').oninput=()=>{el.speed.value=input('speed').value;updateCommandFromInput();updateRuntimeReadout();};
  setValues(config.defaults);
}

function updateRuntimeReadout() {
  const values=state.runtimeControls;
  if (!values) return;
  const text=(name,value)=>{document.getElementById(`runtime-${name}-value`).textContent=value;};
  const seconds=(4*values.num_tokens-4)/SIM_FPS;
  text('horizon',`${seconds.toFixed(2)} s`);
  text('steps',String(values.inference_steps));
  const extent=values.lookahead_m<0 ? Math.min(20,Math.max(.8,stats.control.speed*seconds*(values.lookahead_gain??1)+(values.lookahead_base_m??.4))) : values.lookahead_m;
  text('speed',`${readSpeed().toFixed(1)} m/s`);
  document.getElementById('runtime-speed').value=readSpeed();
  if (values.lookahead_m<0) document.getElementById('runtime-ahead').value=extent;
  text('gain',`${(values.lookahead_gain??1).toFixed(2)} x`);
  text('base',`${(values.lookahead_base_m??.4).toFixed(1)} m`);
  text('path',values.path_refinements ? `${values.path_refinements} x route` : 'Input only');
  text('ahead',`${extent.toFixed(2)} m`);
  text('width',`${values.lateral_halfwidth_m.toFixed(2)} m`);
  const refreshFrames=8*Math.floor(Math.min(values.refresh_frames,4*values.num_tokens-8)/8);
  text('refresh',`${(refreshFrames/SIM_FPS).toFixed(2)} s`);
  const applied=stats.generation?.runtime_controls;
  const pending=!applied || Object.keys(values).some(key=>values[key]!==applied[key]);
  document.getElementById('runtime-applied').textContent=pending ? 'Pending next replan'
    : `Motion view ${stats.diagnostics.motion_observation_extent_m?.toFixed(2) ?? '--'} m · ${stats.inference_ms?.toFixed(1) ?? '--'} ms`;
}

function initializeWorldControls() {
  const world = state.config?.world;
  if (!world || !el.worldSelect) return;
  el.worldSelectControl.hidden = false;
  el.worldMapToggle.hidden = false;
  el.worldSelect.replaceChildren();
  for (const spawn of world.spawns) {
    const option = document.createElement('option');
    option.value = spawn.id;
    option.textContent = spawn.id === 'hub' ? spawn.label : `${spawn.id.slice(-2)} ${spawn.label}`;
    el.worldSelect.append(option);
  }
  el.worldSelect.value = world.default_spawn;
  document.getElementById('world-map-size').textContent = `${world.size_m[0]} x ${world.size_m[1]} m`;
  el.worldSelect.onchange = () => { state.keyState.clear(); state.touchState.clear(); updateCommandFromInput(); resetLive(); worldMapLastDraw = 0; };
  el.worldMapToggle.onclick = () => {
    el.worldMapPanel.hidden = !el.worldMapPanel.hidden;
    el.worldMapToggle.setAttribute('aria-pressed', String(!el.worldMapPanel.hidden));
    fitInspectionPanels();
    worldMapLastDraw = 0; drawWorldMap();
  };
  const terrain = state.config.terrain_scene;
  const rows = heightfieldRows(terrain.hf);
  worldMapBackground = document.createElement('canvas');
  worldMapBackground.width = worldMapBackground.height = 256;
  const context = worldMapBackground.getContext('2d'), image = context.createImageData(256,256);
  for (let y=0;y<256;y++) for (let x=0;x<256;x++) {
    const height = Number(rows[Math.min(rows.length-1,Math.floor((255-y)*rows.length/256))][Math.min(rows[0].length-1,Math.floor(x*rows[0].length/256))]);
    const base = height < 0 ? [145,167,190] : [225-height*17,231-height*10,222-height*21];
    const index = (y*256+x)*4;
    for(let k=0;k<3;k++) image.data[index+k] = Math.max(50,base[k]);
    image.data[index+3]=255;
  }
  context.putImageData(image,0,0);
  el.worldMap.onclick = event => {
    const rect = el.worldMap.getBoundingClientRect();
    const point = [terrain.min_point[0]+(event.clientX-rect.left)/rect.width*world.size_m[0],
      terrain.min_point[2]+(1-(event.clientY-rect.top)/rect.height)*world.size_m[1]];
    const nearest = [...world.spawns].sort((a,b) => Math.hypot(point[0]-(a.center_xz||a.spawn_xz)[0],point[1]-(a.center_xz||a.spawn_xz)[1])-Math.hypot(point[0]-(b.center_xz||b.spawn_xz)[0],point[1]-(b.center_xz||b.spawn_xz)[1]))[0];
    el.worldSelect.value = nearest.id; el.worldSelect.onchange();
  };
}
function drawWorldMap() {
  if (!worldMapBackground || el.worldMapPanel.hidden || performance.now()-worldMapLastDraw<150) return;
  worldMapLastDraw = performance.now();
  const ctx=el.worldMap.getContext('2d'), world=state.config.world, origin=state.config.terrain_scene.min_point;
  ctx.drawImage(worldMapBackground,0,0);
  const point=(x,z)=>[(x-origin[0])/world.size_m[0]*256,256-(z-origin[2])/world.size_m[1]*256];
  ctx.font='10px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';
  for (const spawn of world.spawns) {
    const center=spawn.center_xz||spawn.spawn_xz, p=point(...center);
    ctx.fillStyle=spawn.id===el.worldSelect.value?'#ab452c':'#3d5550';
    ctx.fillText(spawn.id==='hub'?'H':spawn.id.slice(-2),p[0],p[1]);
    if(spawn.id===el.worldSelect.value){ctx.strokeStyle='#ab452c';ctx.strokeRect(p[0]-15,p[1]-15,30,30);}
  }
  const root=point(stats.root[0],stats.root[2]);ctx.fillStyle='#0069db';ctx.beginPath();ctx.arc(root[0],root[1],3.5,0,Math.PI*2);ctx.fill();
  document.getElementById('world-map-label').textContent=world.spawns.find(s=>s.id===el.worldSelect.value)?.label||'';
}

function displayStyle(style) {
  if (style === null || style === undefined || style === '') return '--';
  return String(style);
}

function setRunStatus(text, stateName = '') {
  el.runStatus.textContent = text;
  setTelemetryState(stateName);
}

function setTelemetryState(stateName) {
  el.telemetry.dataset.state = stateName;
}

function setConnection(stateName, label) {
  el.connection.dataset.state = stateName;
  el.connectionLabel.textContent = label;
  if (state.config) el.reset.disabled = stateName !== 'connected' || !state.ready || !isGenerativeMode();
}

function showError(message) {
  el.error.textContent = String(message);
  el.error.hidden = false;
  if (state.errorTimer) window.clearTimeout(state.errorTimer);
  state.errorTimer = window.setTimeout(() => { el.error.hidden = true; }, 6500);
}

function refreshIcons() {
  window.lucide?.createIcons?.();
}

function toFloatArray(value) {
  if (ArrayBuffer.isView(value)) return new Float32Array(value.buffer, value.byteOffset, value.length);
  return new Float32Array(flattenNumbers(value));
}

function flattenNumbers(value) {
  if (!Array.isArray(value)) return [];
  if (!value.length || !Array.isArray(value[0]) && !ArrayBuffer.isView(value[0])) return value.map(Number);
  const output = [];
  for (const row of value) for (const item of row) output.push(Number(item));
  return output;
}

function toVec3(value) {
  if (ArrayBuffer.isView(value)) value = Array.from(value);
  if (!Array.isArray(value)) throw new Error('Expected a three-dimensional vector');
  if (value.length === 3 && !Array.isArray(value[0])) return [Number(value[0]), Number(value[1]), Number(value[2])];
  if (value.length >= 3 && !Array.isArray(value[0])) return [Number(value[0]), Number(value[1]), Number(value[2])];
  throw new Error('Expected a three-dimensional vector');
}

function finiteOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function positive(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}
