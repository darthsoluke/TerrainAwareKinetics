export const SCHEMA = 'motionbricks.scene_control_v3.viewer.v1';
export const FRAME_MS = 1000 / 30;
export const STRIDE = 3 + 77 * 16;
const API = '/api/scene-control-v3';
export const ROLE_COMMAND = { source_replay_reference: false, architecture_capacity_only: true };

export function validateRole(value) {
  if (!Object.hasOwn(ROLE_COMMAND, value.benchmark_role)
      || value.command_used_for_motion !== ROLE_COMMAND[value.benchmark_role]
      || value.actor_status !== 'UNTRAINED' || value.optimizer_updates !== 0
      || value.quality_response !== 'unmeasured' || value.learned_wasd !== false) {
    throw new Error('Expected an explicit source-reference or UNTRAINED capacity role');
  }
  return value.benchmark_role === 'architecture_capacity_only';
}

export function decodeRig(buffer, meta) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength !== STRIDE * 4
      || meta.schema !== SCHEMA || meta.geometry_format !== 'soma77-rig-v1'
      || meta.frame_stride_floats !== STRIDE || meta.frame_count !== 1 || meta.fps !== 30
      || meta.vertex_count !== 18056 || meta.rig_count !== 77 || meta.encoding !== 'float32-le') {
    throw new Error('Expected one complete official SOMA77 frame');
  }
  const values = new Float32Array(STRIDE), view = new DataView(buffer);
  for (let i = 0; i < STRIDE; i++) values[i] = view.getFloat32(i * 4, true);
  if (!values.every(Number.isFinite)) throw new Error('Non-finite SOMA rig');
  return { root: values.subarray(0, 3), rig: values.subarray(3) };
}

export function skinAttributes(skin) {
  const n = skin.influences_per_vertex;
  if (skin.format !== 'soma77-rig-v1' || skin.rig_count !== 77 || !Number.isInteger(n) || n < 4 || n > 8
      || skin.bind_vertices.length !== 18056 * 3 || skin.inverse_bind.length !== 77 * 16
      || skin.indices.length !== 18056 * n || skin.weights.length !== 18056 * n || !skin.faces?.length) {
    throw new Error('Complete official18056 skin with all native influences is required');
  }
  const indices = new Uint16Array(18056 * 4), weights = new Float32Array(18056 * 4);
  const extraIndices = new Uint16Array(18056 * 4), extraWeights = new Float32Array(18056 * 4);
  for (let v = 0; v < 18056; v++) {
    for (let j = 0; j < n; j++) {
      (j < 4 ? indices : extraIndices)[v * 4 + j % 4] = skin.indices[v * n + j];
      (j < 4 ? weights : extraWeights)[v * 4 + j % 4] = skin.weights[v * n + j];
    }
  }
  return { indices, weights, extraIndices, extraWeights };
}

export class CommandStream {
  constructor(session) { this.session = session; this.id = 0; this.semantic = null; this.rows = new Map(); }
  next(vector, inputAt, force = false) {
    const semantic = JSON.stringify(vector);
    if (!force && semantic === this.semantic) return null;
    this.semantic = semantic;
    const value = { type: 'command', session_id: this.session, command_id: ++this.id,
      command: [...vector], client_input_ms: inputAt };
    this.rows.set(value.command_id, { ...value, first_render_ms: null });
    if (this.rows.size > 256) this.rows.delete(this.rows.keys().next().value);
    return value;
  }
}

export const percentile = (values, p) => values.length
  ? [...values].sort((a, b) => a - b)[Math.ceil(values.length * p) - 1] : null;

export function writeReceiptSnapshot(element, snapshot) {
  const serialized = JSON.stringify(snapshot, null, 2);
  element.value = serialized; element.hidden = false;
  return serialized;
}

export function formatReferenceParity(value) {
  return `All ${value.vertex_count.toLocaleString('en-US')} / maxabs ${value.max_abs_error_m.toExponential(6)} m (CPU skin mirror)`;
}

export function formatClose(value) {
  return `Stream closed (${value.code})${value.reason ? `: ${value.reason}` : ''}`;
}

export class FrameLedger {
  constructor(session, role) {
    if (!Object.hasOwn(ROLE_COMMAND, role)) throw new Error('Unknown stream role');
    this.session = session; this.pending = null; this.committed = 0; this.rendered = 0;
    this.role = role;
    this.paused = true;
    this.lastPacket = 0; this.frames = []; this.associations = []; this.segment = 0; this.nextAt = null;
  }
  plan(meta, commands) {
    validateRole(meta);
    const command = commands.rows.get(meta.command_id);
    if (this.pending || meta.session_id !== this.session || meta.type !== 'frame'
        || meta.frame_index !== this.committed || meta.packet_id !== this.lastPacket + 1
        || !command || meta.client_input_ms !== command.client_input_ms
        || meta.benchmark_role !== this.role || meta.command_used_for_motion !== ROLE_COMMAND[this.role]
        || meta.actor_status !== 'UNTRAINED' || meta.optimizer_updates !== 0) {
      throw new Error('Unexpected frame frontier, role or command association');
    }
    this.pending = { meta, command, frame: null, ack: null };
  }
  binary(buffer) {
    if (!this.pending || this.pending.frame) throw new Error('Unexpected rig bytes');
    this.pending.frame = decodeRig(buffer, this.pending.meta);
  }
  render(draw, clock) {
    const p = this.pending;
    if (!p?.frame || p.ack) return null;
    const start = clock();
    if (this.nextAt !== null && start < this.nextAt) return null;
    draw(p.frame, p.meta);
    const end = clock();
    this.nextAt = this.nextAt === null || start - this.nextAt > FRAME_MS
      ? start + FRAME_MS : this.nextAt + FRAME_MS;
    p.ack = { type: 'rendered', session_id: this.session, packet_id: p.meta.packet_id,
      frame_index: p.meta.frame_index, command_id: p.meta.command_id,
      client_input_ms: p.meta.client_input_ms, client_render_started_ms: start, client_render_completed_ms: end };
    if (p.command.first_render_ms === null) {
      p.command.first_render_ms = end;
      this.associations.push({ ...p.ack, latency_ms: end - p.command.client_input_ms });
      if (this.associations.length > 256) this.associations.shift();
    }
    this.rendered++;
    this.frames.push({ ...p.ack, segment: this.segment, render_submit_ms: end - start,
      timings: p.meta.timings, object_pose: p.meta.object_pose ?? null,
      query_metadata: p.meta.query_metadata ?? null });
    if (this.frames.length > 256) this.frames.shift();
    return p.ack;
  }
  commit(value) {
    const p = this.pending;
    if (!p?.ack || ['session_id', 'packet_id', 'frame_index', 'command_id', 'client_input_ms']
      .some(key => value[key] !== p.ack[key]) || value.committed_frame_count !== this.committed + 1) {
      throw new Error('Unexpected ACK receipt');
    }
    this.committed++; this.lastPacket = p.meta.packet_id; this.pending = null;
  }
  resumeClock() { this.segment++; this.nextAt = null; }
  setPaused(value) {
    const resumed = this.paused && !value;
    this.paused = value;
    if (resumed) this.resumeClock();
    return resumed;
  }
  metrics() {
    const frames = this.frames.filter(row => row.segment === this.segment);
    const span = frames.length > 1 ? frames.at(-1).client_render_completed_ms - frames[0].client_render_completed_ms : 0;
    return { unique_rendered_hz: span > 0 ? (frames.length - 1) * 1000 / span : null,
      command_associated_render_p95_ms: percentile(this.associations.map(row => row.latency_ms), .95),
      render_submission_p95_ms: percentile(this.frames.map(row => row.render_submit_ms), .95),
      native_rig_p99_ms: percentile(this.frames.map(row => row.timings.native_rig_ms).filter(Number.isFinite), .99),
      necessary_inference_p99_ms: percentile(this.frames.map(row => row.timings.necessary_inference_ms).filter(Number.isFinite), .99),
      retained_frame_samples: this.frames.length, retained_association_samples: this.associations.length };
  }
}

if (typeof document !== 'undefined') boot().catch(showError);

function showError(error) {
  document.querySelector('#error').hidden = false;
  document.querySelector('#error').textContent = error.message;
  document.querySelector('#status').textContent = 'Stopped';
}

async function boot() {
  const THREE = await import('three');
  const { OrbitControls } = await import('/vendor/OrbitControls.js');
  const el = id => document.getElementById(id);
  const request = async path => {
    const result = await fetch(API + path);
    if (!result.ok) throw new Error(await result.text());
    return result.json();
  };
  const config = await request('/config');
  const capacity = validateRole(config), streamName = capacity ? 'capacity probe' : 'reference';
  if (config.schema !== SCHEMA || config.scene?.kind !== (capacity ? 'triangle_meshes' : 'bound_source_heightfield')) {
    throw new Error('Scene geometry does not match the declared benchmark role');
  }
  el('role').textContent = capacity ? 'ARCHITECTURE CAPACITY / UNTRAINED' : 'SOURCE REPLAY / REFERENCE';
  el('stream-title').textContent = capacity ? 'Untrained Capacity Stream' : 'Reference Stream';
  el('scope').textContent = capacity ? 'Commands enter the random actor. Synthetic contacts. No learned WASD.'
    : 'Reference motion ignores commands. No learned WASD.';
  el('budget-title').textContent = capacity ? 'Proposed Capacity Budget' : 'Transport Budget';
  el('inference-budget').textContent = capacity ? '\u2264 20 ms' : 'N/A (source replay)';
  el('inference-time').textContent = capacity ? 'Awaiting frames' : 'N/A (source replay)';
  el('play').title = `Play ${streamName}`;
  el('play').setAttribute('aria-label', el('play').title);
  el('stage').setAttribute('aria-label', capacity ? 'Official SOMA UNTRAINED capacity body and actual query scene'
    : 'Official SOMA full-body source reference and record-bound scene');
  el('identity-label').textContent = capacity ? 'Actor' : 'Record';
  el('source-name').textContent = capacity ? `H${config.capacity.hidden_dim} / ${config.capacity.device}` : config.source.source_id;
  el('scene-name').textContent = capacity ? config.capacity.scene_id : 'Record-bound world';
  const renderer = new THREE.WebGLRenderer({ canvas: el('stage'), antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene(); scene.background = new THREE.Color('#d5dfda');
  scene.add(new THREE.HemisphereLight(0xffffff, 0x637467, 2.3));
  const light = new THREE.DirectionalLight(0xffffff, 2.6); light.position.set(4, 8, 3); scene.add(light);
  const camera = new THREE.PerspectiveCamera(45, 1, .03, 300);
  const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = false;

  // Same eight-influence vertex and normal chunks as full_corpus_lab.
  function skinMaterial(material) {
    material.onBeforeCompile = shader => {
      shader.vertexShader = 'attribute vec4 skinIndex2;\nattribute vec4 skinWeight2;\n' + shader.vertexShader;
      const terms = ['x', 'y', 'z', 'w'];
      const vertices = terms.map(c => `skinned += getBoneMatrix(skinIndex2.${c}) * skinVertex * skinWeight2.${c};`).join('\n');
      const normals = terms.map(c => `skinMatrix += skinWeight2.${c} * getBoneMatrix(skinIndex2.${c});`).join('\n');
      shader.vertexShader = shader.vertexShader.replace('#include <skinning_vertex>',
        THREE.ShaderChunk.skinning_vertex.replace('transformed =', `${vertices}\ntransformed =`));
      shader.vertexShader = shader.vertexShader.replace('#include <skinnormal_vertex>',
        THREE.ShaderChunk.skinnormal_vertex.replace('skinMatrix = bindMatrixInverse', `${normals}\nskinMatrix = bindMatrixInverse`));
    };
    material.customProgramCacheKey = () => 'official-soma-eight-weight-v1';
    return material;
  }
  const skin = config.soma_skin, attrs = skinAttributes(skin);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(skin.bind_vertices, 3));
  geometry.setIndex(skin.faces);
  geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(attrs.indices, 4));
  geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(attrs.weights, 4));
  geometry.setAttribute('skinIndex2', new THREE.Uint16BufferAttribute(attrs.extraIndices, 4));
  geometry.setAttribute('skinWeight2', new THREE.Float32BufferAttribute(attrs.extraWeights, 4));
  geometry.computeVertexNormals();
  const bones = [], inverses = [];
  for (let i = 0; i < 77; i++) {
    const bone = new THREE.Bone(); bone.matrixAutoUpdate = false; bone.matrixWorldAutoUpdate = false;
    const inverse = new THREE.Matrix4().fromArray(skin.inverse_bind, i * 16).transpose();
    bone.matrixWorld.copy(inverse).invert(); bones.push(bone); inverses.push(inverse);
  }
  const mesh = new THREE.SkinnedMesh(geometry, skinMaterial(new THREE.MeshStandardMaterial({ color: 0x20796e, roughness: .7 })));
  mesh.bind(new THREE.Skeleton(bones, inverses), new THREE.Matrix4()); mesh.bindMode = THREE.DetachedBindMode;
  mesh.frustumCulled = false; scene.add(mesh);

  const sceneObjects = [];
  function objectPoses(poses) {
    if (!Array.isArray(poses) || poses.length !== sceneObjects.length) throw new Error('Missing queried object transforms');
    poses.forEach((pose, index) => {
      if (!Array.isArray(pose) || pose.length !== 7 || !pose.every(Number.isFinite)) throw new Error('Invalid queried object transform');
      sceneObjects[index].position.fromArray(pose);
      sceneObjects[index].quaternion.fromArray(pose, 3);
    });
  }
  if (capacity) {
    if (!Array.isArray(config.scene.objects) || !config.scene.objects.length) throw new Error('Missing actual query meshes');
    config.scene.objects.forEach((object, index) => {
      if (object.id !== index || !object.known || !object.vertices.length || !object.faces.length) throw new Error('Unknown query geometry');
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(object.vertices, 3)); geometry.setIndex(object.faces);
      geometry.computeVertexNormals();
      const surface = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: [0xaab7ad, 0xc0ae81, 0x8cabb5][index % 3], roughness: .95, side: THREE.DoubleSide }));
      surface.visible = object.enabled; scene.add(surface); sceneObjects.push(surface);
    });
    objectPoses(config.scene.objects.map(object => object.pose));
  } else {
    const ground = config.scene, rows = ground.heightfield_y_m, origin = ground.min_point_world_xyz_m, dx = ground.dx_m;
    if (!Array.isArray(rows) || rows.length < 2 || !Array.isArray(origin) || origin.length !== 3
        || !origin.every(Number.isFinite) || !Number.isFinite(dx) || dx <= 0) throw new Error('Missing bound world geometry');
    const positions = [], faces = [], width = rows[0].length;
    for (let z = 0; z < rows.length; z++) {
      if (rows[z].length !== width) throw new Error('Malformed bound world geometry');
      for (let x = 0; x < width; x++) positions.push(origin[0] + x * dx, origin[1] + (Number.isFinite(rows[z][x]) ? rows[z][x] : 0), origin[2] + z * dx);
    }
    for (let z = 0; z + 1 < rows.length; z++) for (let x = 0; x + 1 < width; x++) {
      if (![rows[z][x], rows[z][x + 1], rows[z + 1][x], rows[z + 1][x + 1]].every(Number.isFinite)) continue;
      const a = z * width + x; faces.push(a, a + width, a + 1, a + 1, a + width, a + width + 1);
    }
    if (!faces.length) throw new Error('Bound scene has no known surface');
    const terrainGeometry = new THREE.BufferGeometry(); terrainGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    terrainGeometry.setIndex(faces); terrainGeometry.computeVertexNormals();
    const terrain = new THREE.Mesh(terrainGeometry, new THREE.MeshStandardMaterial({ color: 0xaab7ad, roughness: .95, side: THREE.DoubleSide })); scene.add(terrain);
  }
  let root = [...config.initial_root], paused = true, userPaused = true, socket, commands, ledger, dirty = true, failed = false, referenceResult = null, closeInfo = null;
  function apply(frame, meta) {
    if (capacity && meta) objectPoses(meta.object_pose);
    for (let i = 0; i < 77; i++) bones[i].matrixWorld.fromArray(frame.rig, i * 16).transpose();
    mesh.skeleton.update();
    const delta = new THREE.Vector3(frame.root[0] - root[0], frame.root[1] - root[1], frame.root[2] - root[2]);
    camera.position.add(delta); controls.target.add(delta); root = [...frame.root]; controls.update();
  }
  function center() {
    controls.target.set(root[0], root[1] + .2, root[2]);
    camera.position.set(root[0] - 3.1, root[1] + 1.8, root[2] + 4.2); controls.update(); dirty = true;
  }
  apply({ root: config.initial_root, rig: config.initial_rig }); center();
  new ResizeObserver(() => {
    const { width, height } = el('stage').getBoundingClientRect();
    renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix(); dirty = true;
  }).observe(el('stage'));
  controls.addEventListener('change', () => { dirty = true; });
  const send = value => { if (value) socket.send(JSON.stringify(value)); };
  function command(force = false) {
    const vector = ['vx', 'vz', 'heading'].map(key => Number(el(key).value));
    ['vx', 'vz', 'heading'].forEach((key, i) => { el(`${key}-value`).textContent = `${vector[i].toFixed(1)} ${i === 2 ? 'rad' : 'm/s'}`; });
    if (commands) send(commands.next(vector, performance.now(), force));
  }
  function pause(value) {
    if (value === paused) return;
    paused = value;
    if (ledger?.setPaused(value)) command(true);
    send({ type: 'pause', session_id: ledger.session, paused: value });
    el('play').innerHTML = `<i data-lucide="${value ? 'play' : 'pause'}"></i>`;
    el('play').title = `${value ? 'Play' : 'Pause'} ${streamName}`; el('play').setAttribute('aria-label', el('play').title);
    window.lucide.createIcons(); el('status').textContent = value ? 'Paused' : `Streaming ${streamName}`;
  }
  ['vx', 'vz', 'heading'].forEach(key => el(key).addEventListener('input', () => command()));
  el('play').onclick = () => { userPaused = !userPaused; pause(userPaused || document.hidden); };
  document.addEventListener('visibilitychange', () => { if (ledger) pause(userPaused || document.hidden); });
  el('camera').onclick = center; el('reconnect').onclick = () => location.reload();
  const fmt = value => value === null ? '-' : `${value.toFixed(2)} ms`;
  function update() {
    if (!ledger) return;
    const metrics = ledger.metrics();
    el('hz').textContent = metrics.unique_rendered_hz === null ? '-' : metrics.unique_rendered_hz.toFixed(2);
    el('frame').textContent = ledger.frames.at(-1)?.frame_index ?? '-';
    el('command').textContent = `${commands.id} / ${ledger.frames.at(-1)?.command_id ?? '-'}`;
    el('committed').textContent = ledger.committed;
    el('latency').textContent = fmt(metrics.command_associated_render_p95_ms);
    el('render-time').textContent = fmt(metrics.render_submission_p95_ms); el('rig-time').textContent = fmt(metrics.native_rig_p99_ms);
    el('inference-time').textContent = capacity ? fmt(metrics.necessary_inference_p99_ms) : 'N/A (source replay)';
  }
  async function receipt() {
    const server = await request('/diagnostics');
    return structuredClone({ schema: SCHEMA, benchmark_role: config.benchmark_role, actor_status: 'UNTRAINED', optimizer_updates: 0,
      captured_client_ms: performance.now(),
      learned_wasd: false, quality_response: 'unmeasured', command_used_for_motion: capacity,
      display_measurement: 'client_render_submission_not_scanout', time_origin_ms: performance.timeOrigin,
      presentation_clock: 'client_monotonic_30hz_server_ack_immediate_prepare',
      source: config.source, capacity: config.capacity ?? null,
      reference_parity: referenceResult,
      connection_close: closeInfo,
      scene_role: capacity ? 'actual_query_meshes' : 'record_bound_world', metrics: ledger?.metrics() ?? null,
      frames: ledger?.frames ?? [], associations: ledger?.associations ?? [], commands: [...(commands?.rows.values() ?? [])],
      server, renderer: { webgl: renderer.capabilities.isWebGL2 ? 2 : 1,
        width: renderer.domElement.width, height: renderer.domElement.height, influences: skin.influences_per_vertex,
        vertices: 18056, triangles: skin.faces.length / 3, program_count: renderer.info.programs.length },
      proposed_budgets: config.proposed_budgets, gpu_pixel_proof: 'parent_pending', quality_pass: null });
  }
  el('receipt').onclick = async () => {
    try {
      const serialized = writeReceiptSnapshot(el('receipt-snapshot'), await receipt());
      const url = URL.createObjectURL(new Blob([serialized], { type: 'application/json' }));
      const link = document.createElement('a'); link.href = url; link.download = `scene_control_v3_${capacity ? 'untrained_capacity' : 'source_reference'}.json`; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { showError(error); }
  };
  socket = new WebSocket(`ws://${location.host}${config.websocket_path}`); socket.binaryType = 'arraybuffer';
  socket.onmessage = event => {
    try {
      if (event.data instanceof ArrayBuffer) { ledger.binary(event.data); return; }
      const value = JSON.parse(event.data);
      if (value.type === 'ready') {
        validateRole(value);
        if (value.benchmark_role !== config.benchmark_role) throw new Error('Stream role changed');
        ledger = new FrameLedger(value.session_id, value.benchmark_role); commands = new CommandStream(value.session_id);
        el('play').disabled = false; el('command-controls').disabled = false; el('status').textContent = 'Paused'; command(true);
      } else if (value.type === 'frame') ledger.plan(value, commands);
      else if (value.type === 'committed') {
        ledger.commit(value); update();
        if (value.completed) { userPaused = true; pause(true); el('status').textContent = capacity ? 'Capacity limit reached' : 'Reference complete'; el('play').disabled = true; }
      } else if (value.type === 'error') throw new Error(value.message);
    } catch (error) { failed = true; socket.close(); showError(error); }
  };
  socket.onclose = event => {
    closeInfo = { code: event.code, reason: event.reason, was_clean: event.wasClean, at_ms: performance.now() };
    paused = true; ledger?.setPaused(true); el('play').disabled = true; el('command-controls').disabled = true;
    if (event.code !== 1000 || event.reason) showError(new Error(formatClose(closeInfo)));
    el('status').textContent = `Disconnected (${event.code})`;
  };
  socket.onerror = () => { failed = true; showError(new Error('Workbench connection failed')); };
  function animate() {
    if (failed) return;
    try {
      if (!paused && !document.hidden && ledger?.pending?.frame && !ledger.pending.ack) {
        const ack = ledger.render((frame, meta) => { apply(frame, meta); renderer.render(scene, camera); }, () => performance.now());
        if (ack) { send(ack); dirty = false; update(); }
        else if (dirty) { renderer.render(scene, camera); dirty = false; }
      } else if (dirty) { renderer.render(scene, camera); dirty = false; }
      requestAnimationFrame(animate);
    } catch (error) { failed = true; socket.close(); showError(error); }
  }
  async function referenceParity() {
      const reference = await request('/reference');
      const matrices = inverses.map((inverse, i) => new THREE.Matrix4().fromArray(config.initial_rig, i * 16).transpose().multiply(inverse));
      const point = new THREE.Vector3(), actual = new THREE.Vector3(), bind = new THREE.Vector3();
      let maxError = 0, maxAbs = 0;
      for (let v = 0; v < 18056; v++) {
        bind.fromArray(skin.bind_vertices, v * 3); actual.set(0, 0, 0);
        for (let j = 0; j < skin.influences_per_vertex; j++) {
          const offset = v * skin.influences_per_vertex + j;
          actual.addScaledVector(point.copy(bind).applyMatrix4(matrices[skin.indices[offset]]), skin.weights[offset]);
        }
        point.fromArray(reference.vertices, v * 3).sub(actual);
        maxError = Math.max(maxError, point.length()); maxAbs = Math.max(maxAbs, Math.abs(point.x), Math.abs(point.y), Math.abs(point.z));
      }
      return { frame_index: capacity ? null : 0, reference_kind: capacity ? 'native_initial_seed' : 'source_frame0',
        vertex_count: 18056, influences: skin.influences_per_vertex,
        max_abs_error_m: maxAbs, max_vertex_error_m: maxError, method: 'browser_cpu_mirror_of_all_influence_gpu_shader',
        reference_provenance: reference.provenance, outside_timed_loop: true };
  }
  el('reference-check').disabled = false;
  el('reference-check').onclick = async () => {
    el('reference-check').disabled = true;
    try {
      if (ledger && !paused) { userPaused = true; pause(true); }
      el('reference-parity').textContent = 'Comparing all 18,056 reference vertices';
      referenceResult = await referenceParity();
      el('reference-parity').textContent = formatReferenceParity(referenceResult);
    } catch (error) { el('reference-parity').textContent = `Reference comparison failed: ${error.message}`; }
    finally { el('reference-check').disabled = false; }
  };
  window.sceneControlV3 = { renderer, scene, camera, mesh, receipt, referenceParity, get ledger() { return ledger; } };
  window.lucide.createIcons(); animate();
}
