// Review fitted BONES terrain: skeleton playback with minimal (solid) and maximal (wire) boxes.
// Serve the repository root (python -m http.server) and open
// /web/terrain_fit_review/?index=/path/to/viz/index.json  or use "Open JSON".
import * as THREE from 'three';
import {OrbitControls} from '/web/soma_lab/vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const view = $('view');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(matchMedia('(prefers-color-scheme: dark)').matches ? 0x202124 : 0xeef0f2);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.0); sun.position.set(3, 6, 2); scene.add(sun);
const dynamic = new THREE.Group(); scene.add(dynamic);

let clips = [], clip = null, frame = 0, playing = true, last = performance.now(), accumulator = 0;
let skeleton = null, joints = null, trail = null;

function resize() {
  const {clientWidth: w, clientHeight: h} = view;
  renderer.setSize(w, h); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

const ROLE_COLORS = {ground: 0x9aa0a6, support: 0xe8710a, ceiling: 0x00897b, wall: 0x8e24aa, clutter: 0x78909c};

function boxMesh(box, wire) {
  const geometry = new THREE.BoxGeometry(...box.half_extent_xyz_m.map((v) => 2 * v));
  const role = box.role;
  const object = wire
    ? new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({color: 0x5f6368}))
    : new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
        color: ROLE_COLORS[role] ?? 0xe8710a, transparent: true,
        opacity: role === 'ground' ? 0.35 : role === 'ceiling' ? 0.3 : 0.55, depthWrite: role === 'ground'}));
  const r = box.rotation_local_to_world;
  const m = new THREE.Matrix4().set(r[0][0], r[0][1], r[0][2], box.center_xyz_m[0],
                                    r[1][0], r[1][1], r[1][2], box.center_xyz_m[1],
                                    r[2][0], r[2][1], r[2][2], box.center_xyz_m[2], 0, 0, 0, 1);
  object.applyMatrix4(m);
  return object;
}

function build() {
  dynamic.clear();
  const variant = $('variant').value;
  if (variant.startsWith('scene')) {
    for (const box of (clip.variants || [])[+variant.slice(5)] || []) dynamic.add(boxMesh(box, false));
  } else {
    for (const name of ['minimal', 'maximal']) {
      if (variant !== 'both' && variant !== name) continue;
      for (const box of clip.boxes[name]) {
        if (name === 'maximal' && box.role === 'ground' && variant === 'both') continue;
        dynamic.add(boxMesh(box, name === 'maximal'));
      }
    }
  }
  const positions = new Float32Array(clip.segments.length * 6);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  skeleton = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({color: 0x1a73e8}));
  dynamic.add(skeleton);
  joints = new THREE.InstancedMesh(new THREE.SphereGeometry(0.025, 8, 6),
    new THREE.MeshStandardMaterial({color: 0x1a73e8}), clip.joint_names.length);
  dynamic.add(joints);
  const hips = clip.joint_names.indexOf('Hips');
  if (hips >= 0) {
    const path = clip.frames_mm.map((f) => new THREE.Vector3(f[hips][0] / 1000, clip.ground_m + 0.005, f[hips][2] / 1000));
    trail = new THREE.Line(new THREE.BufferGeometry().setFromPoints(path), new THREE.LineBasicMaterial({color: 0x188038}));
    trail.visible = $('trail').checked; dynamic.add(trail);
  }
  const all = clip.frames_mm.flat().map((p) => p.map((v) => v / 1000));
  const center = all.reduce((a, p) => a.add(new THREE.Vector3(...p)), new THREE.Vector3()).divideScalar(all.length);
  controls.target.copy(center); camera.position.copy(center).add(new THREE.Vector3(2.5, 1.8, 3.0)); controls.update();
  $('timeline').max = clip.frames_mm.length - 1;
  showInfo(); setFrame(0);
}

function setFrame(index) {
  frame = index; $('timeline').value = index;
  $('time').textContent = `${(index / clip.fps).toFixed(2)} s`;
  const points = clip.frames_mm[index];
  const attribute = skeleton.geometry.attributes.position;
  clip.segments.forEach(([a, b], i) => {
    attribute.setXYZ(2 * i, points[a][0] / 1000, points[a][1] / 1000, points[a][2] / 1000);
    attribute.setXYZ(2 * i + 1, points[b][0] / 1000, points[b][1] / 1000, points[b][2] / 1000);
  });
  attribute.needsUpdate = true; skeleton.geometry.computeBoundingSphere();
  const matrix = new THREE.Matrix4();
  points.forEach((p, i) => { matrix.makeTranslation(p[0] / 1000, p[1] / 1000, p[2] / 1000); joints.setMatrixAt(i, matrix); });
  joints.instanceMatrix.needsUpdate = true;
}

function showInfo() {
  const m = clip.meta || {};
  const rows = [['clip', m.filename], ['stem', m.stem], ['group', m.group],
    ['levels (m)', JSON.stringify(clip.levels_m.map((v) => +v.toFixed(3)))], ['flags', (clip.flags || []).join(', ') || 'none']];
  for (const name of ['minimal', 'maximal']) {
    const k = clip.metrics[name];
    const skin = k.skin;
    rows.push([`${name} ${skin ? 'skin' : 'landmark'} penetration max`,
               `${((skin ? skin.max_m : k.penetration_max_m) * 100).toFixed(1)} cm`],
              [`${name} frames > 2 cm`, skin ? skin.frames_over_report : k.penetration_frames_over_report],
              [`${name} unsupported evidence`, `${k.support_evidence_unsupported} / ${k.support_evidence_points}`]);
  }
  const scene = clip.scene_report;
  if (scene) {
    const c = scene.ceiling;
    rows.push(['ceiling', c.status === 'fitted'
      ? `${c.tight_m.toFixed(2)}-${c.loose_m.toFixed(2)} m above floor; explains ${(100 * (c.explanation?.tight ?? 0)).toFixed(0)}% of low frames`
      : c.status], ['walls', scene.walls.status]);
  }
  for (const box of clip.boxes.maximal.filter((b) => b.role === 'support')) {
    const minimal = clip.boxes.minimal.find((b) => b.name === box.name);
    const size = (b) => `${(2 * b.half_extent_xyz_m[0]).toFixed(2)} x ${(2 * b.half_extent_xyz_m[2]).toFixed(2)}`;
    rows.push([box.name, `top ${(box.top_m - clip.ground_m).toFixed(3)} m; min ${size(minimal)}; max ${size(box)}`]);
  }
  $('info').innerHTML = '<table>' + rows.map(([k, v]) => `<tr><td>${k}</td><td>${v ?? ''}</td></tr>`).join('') + '</table>';
}

function select(index) {
  clip = clips[index];
  const options = [['both', 'fit: both'], ['minimal', 'fit: minimal / tight'], ['maximal', 'fit: maximal / loose']]
    .concat((clip.variants || []).map((_, i) => [`scene${i}`, `sampled scene ${i + 1}`]));
  const current = $('variant').value;
  $('variant').innerHTML = options.map(([v, label]) => `<option value="${v}">${label}</option>`).join('');
  $('variant').value = options.some(([v]) => v === current) ? current : 'both';
  build(); $('status').textContent = `${clips.length} clip(s)`;
}

function setClips(list) {
  clips = list; $('clip').innerHTML = clips.map((c, i) => `<option value="${i}">${c.meta?.filename ?? `clip ${i}`}</option>`).join('');
  if (clips.length) select(0);
}

$('clip').onchange = (e) => select(+e.target.value);
$('variant').onchange = () => clip && build();
$('trail').onchange = (e) => { if (trail) trail.visible = e.target.checked; };
$('timeline').oninput = (e) => { playing = false; $('play').textContent = 'Play'; setFrame(+e.target.value); };
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
$('file').onchange = async (e) => setClips(await Promise.all([...e.target.files].map(async (f) => JSON.parse(await f.text()))));

async function loadIndex() {
  const url = new URLSearchParams(location.search).get('index');
  if (!url) { $('status').textContent = 'Open payload JSON files, or pass ?index=.../index.json'; return; }
  const index = await (await fetch(url)).json();
  const base = new URL(url, location.href);
  setClips(await Promise.all(index.files.map(async (name) => (await fetch(new URL(name, base))).json())));
}

function tick(now) {
  const dt = (now - last) / 1000; last = now;
  if (clip && playing) {
    accumulator += dt * +$('speed').value;
    const step = 1 / clip.fps;
    while (accumulator >= step) { accumulator -= step; setFrame((frame + 1) % clip.frames_mm.length); }
  }
  controls.update(); renderer.render(scene, camera); requestAnimationFrame(tick);
}
resize(); loadIndex().catch((error) => { $('status').textContent = `Load failed: ${error.message}`; });
requestAnimationFrame(tick);
