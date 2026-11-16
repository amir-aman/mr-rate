import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import S from '../aggregates/stats_public.json';
import V from '../aggregates/volmeta.json';

const $ = (q, el = document) => el.querySelector(q);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const fmt = n => n.toLocaleString('en-US');
const SLICE = ['#F0A63A', '#3CC2B4', '#E0607E'];
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------------- slice rule, same as imaging.axial_slices in the repository ---------------- */
const fgMax = Math.max(...V.fg);
const minimum = Math.max(4, Math.floor(V.nx * V.ny * 0.005), Math.floor(fgMax * 0.1));
const eligible = V.fg.map((c, k) => (c >= minimum ? k : -1)).filter(k => k >= 0);
const e0 = eligible[0], e1 = eligible[eligible.length - 1];
const sliceIndex = f => Math.round(e0 + f * (e1 - e0));
const sliceZ = f => V.z0 + sliceIndex(f) * V.step;

const state = { fr: [0.2, 0.5, 0.8], cut: -1 };
const view = { ready: false };

/* ---------------- slice atlas (MNI152 template) ---------------- */
const atlas = new Image();
const atlasC = document.createElement('canvas');
let atlasCtx = null;
atlas.onload = () => {
  atlasC.width = atlas.width; atlasC.height = atlas.height;
  atlasCtx = atlasC.getContext('2d', { willReadFrequently: true });
  atlasCtx.drawImage(atlas, 0, 0);
  for (let s = 0; s < 3; s++) drawTile(s);
  updateScene();
};
atlas.src = 'assets/mni_slices.png';
const rect = k => [(k % V.cols) * V.nx, Math.floor(k / V.cols) * V.ny];

/* ---------------- slice cards ---------------- */
const tiles = [], ranges = [], outs = [], cuts = [];
const list = $('#slices');
for (let s = 0; s < 3; s++) {
  const li = el('li', 'slice'); li.style.setProperty('--c', SLICE[s]);
  const cv = el('canvas'); cv.width = V.nx * 2; cv.height = V.ny * 2;
  cv.setAttribute('role', 'img'); cv.setAttribute('aria-label', 'Axial slice ' + (s + 1) + ' of the MNI152 template');
  const body = el('div', 'slice-body');
  const head = el('div', 'slice-head');
  const lab = el('label', null, 'Slice ' + (s + 1)); lab.htmlFor = 'r' + s;
  const out = el('output'); out.htmlFor = 'r' + s;
  head.append(lab, out);
  const r = el('input'); r.type = 'range'; r.id = 'r' + s; r.min = 0; r.max = 100; r.step = 1;
  const b = el('button', 'cut', 'Cut the brain here'); b.type = 'button'; b.setAttribute('aria-pressed', 'false');
  body.append(head, r, b); li.append(cv, body); list.append(li);
  tiles.push(cv); ranges.push(r); outs.push(out); cuts.push(b);
  r.addEventListener('input', () => { state.fr[s] = r.value / 100; stopAuto(); refresh(s); });
  b.addEventListener('click', () => { state.cut = state.cut === s ? -1 : s; stopAuto(); syncCuts(); updateScene(); });
}
function syncCuts() {
  cuts.forEach((b, s) => { const on = state.cut === s; b.setAttribute('aria-pressed', on); b.textContent = on ? 'Show the whole brain' : 'Cut the brain here'; });
}
function drawTile(s) {
  if (!atlasCtx) return;
  const c = tiles[s].getContext('2d'); const [x, y] = rect(sliceIndex(state.fr[s]));
  c.imageSmoothingQuality = 'high';
  c.fillStyle = '#000'; c.fillRect(0, 0, tiles[s].width, tiles[s].height);
  c.drawImage(atlasC, x, y, V.nx, V.ny, 0, 0, tiles[s].width, tiles[s].height);
}
function refresh(s) {
  ranges[s].value = Math.round(state.fr[s] * 100);
  outs[s].textContent = Math.round(state.fr[s] * 100) + '%';
  drawTile(s); updateScene();
}
$('#reset').addEventListener('click', () => { state.fr = [0.2, 0.5, 0.8]; state.cut = -1; syncCuts(); for (let s = 0; s < 3; s++) refresh(s); });

/* ---------------- 3D scene ---------------- */
const stage = $('#stage'), canvas = $('#brain');
let renderer, scene, camera, controls, brainMat, planes = [], cap, capTex, capC, capCtx, clip, center;
function stopAuto() { if (controls) controls.autoRotate = false; }

function initScene(buf) {
  const dv = new DataView(buf);
  const nv = dv.getUint32(4, true), nf = dv.getUint32(8, true);
  const lo = [0, 1, 2].map(i => dv.getFloat32(12 + 4 * i, true)), hi = [0, 1, 2].map(i => dv.getFloat32(24 + 4 * i, true));
  const q = new Uint16Array(buf, 36, nv * 3);
  const shade = new Uint8Array(buf, 36 + nv * 6, nv);
  let off = 36 + nv * 7; off += (4 - (off % 4)) % 4;
  const idx = new Uint32Array(buf, off, nf * 3);
  const pos = new Float32Array(nv * 3), col = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    for (let a = 0; a < 3; a++) pos[3 * i + a] = lo[a] + (q[3 * i + a] / 65535) * (hi[a] - lo[a]);
    const t = 0.34 + 0.66 * (shade[i] / 255);
    col[3 * i] = t * 0.93; col[3 * i + 1] = t * 0.95; col[3 * i + 2] = t;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  center = new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.localClippingEnabled = true;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(30, 1, 0.1, 50);
  camera.position.set(-2.75, 1.25, -3.4);

  scene.add(new THREE.HemisphereLight(0xd6f1f2, 0x0a2630, 1.15));
  const key = new THREE.DirectionalLight(0xfff1df, 2.1); key.position.set(-2, 3, -2.5); scene.add(key);
  const rim = new THREE.DirectionalLight(0x3cc2b4, 1.5); rim.position.set(3, 0.5, 3); scene.add(rim);

  // outer: RAS millimetres to a y-up scene. inner: centred millimetre space.
  const outer = new THREE.Group(); outer.rotation.x = -Math.PI / 2; outer.scale.setScalar(0.01); scene.add(outer);
  const mm = new THREE.Group(); mm.position.copy(center).multiplyScalar(-1); outer.add(mm);

  clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);
  brainMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.62, metalness: 0.02, side: THREE.DoubleSide, clippingPlanes: [] });
  mm.add(new THREE.Mesh(geo, brainMat));

  const w = hi[0] - lo[0] + 10, h = hi[1] - lo[1] + 10;
  for (let s = 0; s < 3; s++) {
    const g = new THREE.Group();
    g.add(new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: SLICE[s], transparent: true, opacity: 0.2, side: THREE.DoubleSide, depthWrite: false })));
    const edge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w, h)), new THREE.LineBasicMaterial({ color: SLICE[s] }));
    g.add(edge); g.position.set(center.x, center.y, 0); mm.add(g); planes.push(g);
  }
  capC = document.createElement('canvas'); capC.width = V.nx; capC.height = V.ny; capCtx = capC.getContext('2d');
  capTex = new THREE.CanvasTexture(capC); capTex.colorSpace = THREE.SRGBColorSpace;
  cap = new THREE.Mesh(new THREE.PlaneGeometry(V.nx * V.step, V.ny * V.step), new THREE.MeshBasicMaterial({ map: capTex, transparent: true, alphaTest: 0.5, side: THREE.DoubleSide }));
  cap.position.set(V.x0 + (V.nx - 1) * V.step / 2, V.y0 + (V.ny - 1) * V.step / 2, 0); cap.visible = false; mm.add(cap);

  controls = new OrbitControls(camera, canvas);
  controls.enableZoom = false; controls.enablePan = false; controls.enableDamping = true;
  controls.minPolarAngle = 0.25; controls.maxPolarAngle = Math.PI - 0.25;
  controls.autoRotate = !reduced; controls.autoRotateSpeed = 0.9;
  controls.addEventListener('start', stopAuto);
  canvas.style.touchAction = 'pan-y';
  canvas.addEventListener('keydown', e => {
    const up = new THREE.Vector3(0, 1, 0); let used = true;
    if (e.key === 'ArrowLeft') camera.position.applyAxisAngle(up, -0.15);
    else if (e.key === 'ArrowRight') camera.position.applyAxisAngle(up, 0.15);
    else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      const right = new THREE.Vector3().crossVectors(camera.position, up).normalize();
      const p = camera.position.clone().applyAxisAngle(right, e.key === 'ArrowUp' ? 0.12 : -0.12);
      const polar = p.angleTo(up); if (polar > 0.25 && polar < Math.PI - 0.25) camera.position.copy(p);
    } else used = false;
    if (used) { e.preventDefault(); stopAuto(); }
  });

  const fit = () => {
    const w2 = stage.clientWidth, h2 = stage.clientHeight; if (!w2 || !h2) return;
    renderer.setSize(w2, h2, false); camera.aspect = w2 / h2;
    camera.fov = camera.aspect < 1.1 ? 42 : 30; camera.updateProjectionMatrix();
  };
  new ResizeObserver(fit).observe(stage); fit();
  view.ready = true; stage.classList.add('ready');
  updateScene();
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
}

function updateScene() {
  if (!view.ready) return;
  const zs = state.fr.map(sliceZ), cutZ = state.cut < 0 ? null : zs[state.cut];
  planes.forEach((g, s) => { g.position.z = zs[s]; g.visible = cutZ == null || (s !== state.cut && zs[s] < cutZ); });
  if (cutZ == null || !atlasCtx) { brainMat.clippingPlanes = []; cap.visible = false; }
  else {
    clip.constant = (cutZ - center.z) * 0.01 + 0.002; brainMat.clippingPlanes = [clip];
    const [x, y] = rect(sliceIndex(state.fr[state.cut]));
    const im = atlasCtx.getImageData(x, y, V.nx, V.ny), d = im.data;
    for (let i = 0; i < d.length; i += 4) d[i + 3] = d[i] > 4 ? 255 : 0;
    capCtx.putImageData(im, 0, 0); capTex.needsUpdate = true;
    cap.position.z = cutZ; cap.visible = true;
  }
  brainMat.needsUpdate = true;
}

fetch('assets/brain.bin').then(r => { if (!r.ok) throw new Error('mesh'); return r.arrayBuffer(); }).then(initScene).catch(() => {
  $('#stage-msg').textContent = 'The 3D view needs WebGL, which is not available here. The slice previews below still work.';
  stage.classList.add('failed');
});
syncCuts(); for (let s = 0; s < 3; s++) refresh(s);

/* ---------------- tooltip ---------------- */
const tip = $('#tip');
function showTip(target, value, label) {
  tip.replaceChildren(el('strong', null, value), el('span', null, label));
  tip.hidden = false;
  const r = target.getBoundingClientRect(), t = tip.getBoundingClientRect();
  let x = r.left + r.width / 2 - t.width / 2; x = Math.max(8, Math.min(innerWidth - t.width - 8, x));
  let y = r.top - t.height - 8; if (y < 8) y = r.bottom + 8;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
function bindTip(node, value, label) {
  node.tabIndex = 0; node.setAttribute('role', 'img'); node.setAttribute('aria-label', label + ', ' + value);
  node.addEventListener('pointerenter', () => showTip(node, value, label));
  node.addEventListener('focus', () => showTip(node, value, label));
  node.addEventListener('pointerleave', () => { tip.hidden = true; });
  node.addEventListener('blur', () => { tip.hidden = true; });
}

/* ---------------- numbers ---------------- */
$('#n-volumes').textContent = fmt(S.dataset.volumes);
$('#n-studies').textContent = fmt(S.dataset.studies);
$('#n-patients').textContent = fmt(S.dataset.patients);
$('#sample-line').textContent = 'Charts below use the first batch of the release, ' + fmt(S.n_studies) + ' studies from ' + fmt(S.n_patients) + ' patients with ' + fmt(S.n_series) + ' MRI series, and the pathology label table for ' + fmt(S.labels_n) + ' studies.';

/* ---------------- age pyramid ---------------- */
(() => {
  const root = $('#age'); const A = S.age;
  const rows = A.bins.map((b, i) => ({ b, f: A.F[i], m: A.M[i] })).filter(r => r.f + r.m > 0).reverse();
  const mx = Math.max(...rows.map(r => Math.max(r.f, r.m)));
  rows.forEach(r => {
    const row = el('div', 'pyr-row');
    const name = r.b >= 90 ? '90 and over' : r.b + ' to ' + (r.b + 4);
    const f = el('div', 'pyr-f'), fb = el('i'); fb.style.width = (100 * r.f / mx) + '%'; f.append(fb);
    const m = el('div', 'pyr-m'), mb = el('i'); mb.style.width = (100 * r.m / mx) + '%'; m.append(mb);
    bindTip(fb, fmt(r.f) + ' studies', 'Women aged ' + name);
    bindTip(mb, fmt(r.m) + ' studies', 'Men aged ' + name);
    row.append(f, el('span', 'pyr-age', r.b >= 90 ? '90+' : String(r.b)), m); root.append(row);
  });
  $('#age-note').textContent = 'Median age ' + A.median + ' years, range ' + A.min + ' to ' + A.max + '. ' + fmt(S.sex.F) + ' studies of women and ' + fmt(S.sex.M) + ' of men.';
})();

/* ---------------- simple ranked bars ---------------- */
function bars(root, items, total, unit) {
  const mx = Math.max(...items.map(d => d.n));
  items.forEach(d => {
    const row = el('div', 'hb');
    const bar = el('div', 'hb-bar'), fill = el('i'); fill.style.width = Math.max(0.6, 100 * d.n / mx) + '%'; bar.append(fill);
    row.append(el('span', 'hb-name', d.name), bar, el('span', 'hb-val', fmt(d.n)));
    if (d.sub) row.append(el('span', 'hb-sub', d.sub));
    row.title = d.name + ', ' + fmt(d.n) + ' ' + unit + (total ? ', ' + (100 * d.n / total).toFixed(1) + '%' : '');
    root.append(row);
  });
}
bars($('#scanners'), S.scanner.map(d => ({ name: d.name, n: d.n, sub: d.field })), S.n_studies, 'studies');
bars($('#modality'), Object.entries(S.modality).map(([name, n]) => ({ name, n })), S.n_series, 'series');

/* ---------------- pathology labels and age profile ---------------- */
(() => {
  const root = $('#labels'), prof = $('#profile'), title = $('#profile-title'), note = $('#profile-note');
  const B = S.by_age; let selected = 'Cerebral atrophy'; let showAll = false;
  const mx = S.labels[0].pct; const buttons = new Map();
  S.labels.forEach((d, i) => {
    const has = !!B.labels[d.name];
    const b = el('button', 'lab'); b.type = 'button'; b.disabled = !has;
    if (i >= 12) b.classList.add('extra');
    const bar = el('span', 'lab-bar'), fill = el('i'); fill.style.width = Math.max(0.8, 100 * d.pct / mx) + '%'; bar.append(fill);
    b.append(el('span', 'lab-name', d.name), bar, el('span', 'lab-val', d.pct.toFixed(1) + '%'));
    b.title = fmt(d.n) + ' of ' + fmt(S.labels_n) + ' studies' + (has ? '' : '. Too few cases in the first batch for an age profile');
    b.addEventListener('click', () => { selected = d.name; draw(); });
    buttons.set(d.name, b); root.append(b);
  });
  const more = $('#labels-more');
  more.addEventListener('click', () => { showAll = !showAll; root.classList.toggle('all', showAll); more.textContent = showAll ? 'Show the 12 most common' : 'Show all ' + S.labels.length + ' labels'; });
  more.textContent = 'Show all ' + S.labels.length + ' labels';
  function draw() {
    buttons.forEach((b, k) => b.setAttribute('aria-pressed', k === selected));
    const v = B.labels[selected]; title.textContent = selected + ' by age';
    const top = Math.max(10, ...v.map(x => x || 0)); const scale = top > 50 ? 100 : top > 25 ? 50 : top > 10 ? 25 : 10;
    prof.replaceChildren();
    v.forEach((p, i) => {
      const colm = el('div', 'col'); const plot = el('div', 'col-plot');
      if (p == null) { const s = el('i', 'sup'); plot.append(el('span', 'col-val', '<5'), s); colm.append(plot); bindTip(s, 'fewer than 5 studies', B.bands[i] + ', not shown'); }
      else { const f = el('i'); f.style.height = 'calc((100% - 24px) * ' + Math.max(0.01, p / scale).toFixed(4) + ')'; plot.append(el('span', 'col-val', p.toFixed(p < 10 ? 1 : 0) + '%'), f); colm.append(plot); bindTip(f, p.toFixed(1) + '% of ' + fmt(B.n[i]) + ' studies', selected + ', age ' + B.bands[i]); }
      colm.append(el('span', 'col-name', ['<20', '20s', '30s', '40s', '50s', '60s', '70s', '80+'][i]));
      prof.append(colm);
    });
    note.textContent = 'Share of studies with this label in each age band, first batch only (' + fmt(B.n_studies) + ' studies). Axis runs from 0 to ' + scale + '%. Cells with fewer than 5 studies are not shown.';
  }
  draw();
})();

$('#report-stat').textContent = 'A typical impression is ' + S.reports.impression_words_median + ' words long. One in ten runs past ' + S.reports.impression_words_p90 + ' words, which matters because the text encoder reads at most 64 tokens.';
window.__explorer = { state, view, refresh: () => { syncCuts(); for (let s = 0; s < 3; s++) refresh(s); }, stopAuto, camera: () => camera, eligible: [e0, e1] };
