import * as THREE from 'three';
import S from '../aggregates/stats_public.json';
import V from '../aggregates/volmeta.json';
import MESH from '../aggregates/brain.bin.gz';
import ATLAS from '../aggregates/mni_slices.png';

const $ = (q, r = document) => r.querySelector(q);
const $$ = (q, r = document) => [...r.querySelectorAll(q)];
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const fmt = n => n.toLocaleString('en-US');
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const SLICE = ['#F0A63A', '#3CC2B4', '#E0607E'];
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- slice rule, same as imaging.axial_slices in the repository ---------- */
const fgMax = Math.max(...V.fg);
const minimum = Math.max(4, Math.floor(V.nx * V.ny * 0.005), Math.floor(fgMax * 0.1));
const eligible = V.fg.map((c, k) => (c >= minimum ? k : -1)).filter(k => k >= 0);
const e0 = eligible[0], e1 = eligible[eligible.length - 1];
const sliceIndex = f => Math.round(e0 + f * (e1 - e0));
const sliceZ = f => V.z0 + sliceIndex(f) * V.step;

const state = { fr: [0.2, 0.5, 0.8], cut: -1 };
let slide = 0;
const cutFor = () => (slide === 2 ? 1 : slide === 3 ? state.cut : -1);

/* ---------- slice atlas (MNI152 template, embedded) ---------- */
let atlasCtx = null;
const atlasImg = new Image();
atlasImg.onload = () => {
  const c = document.createElement('canvas'); c.width = atlasImg.width; c.height = atlasImg.height;
  atlasCtx = c.getContext('2d', { willReadFrequently: true }); atlasCtx.drawImage(atlasImg, 0, 0);
  for (let s = 0; s < 3; s++) drawSlice(s);
  updateScene();
};
atlasImg.src = ATLAS;
const rect = k => [(k % V.cols) * V.nx, Math.floor(k / V.cols) * V.ny];

const tiles = [[], [], []];
function drawSlice(s) {
  if (!atlasCtx) return;
  const [x, y] = rect(sliceIndex(state.fr[s]));
  tiles[s].forEach(cv => {
    const c = cv.getContext('2d'); c.imageSmoothingQuality = 'high';
    c.fillStyle = '#000'; c.fillRect(0, 0, cv.width, cv.height);
    c.drawImage(atlasCtx.canvas, x, y, V.nx, V.ny, 0, 0, cv.width, cv.height);
  });
}
function makeTile(s, label) {
  const cv = el('canvas'); cv.width = V.nx * 2; cv.height = V.ny * 2;
  cv.setAttribute('role', 'img'); cv.setAttribute('aria-label', label);
  tiles[s].push(cv); return cv;
}

/* thumbnails on the model-input slide */
const thumbs = $('#thumbs');
for (let s = 0; s < 3; s++) {
  const f = el('figure'); f.style.setProperty('--c', SLICE[s]);
  f.append(makeTile(s, 'Axial slice ' + (s + 1)), el('figcaption', null, 'Slice ' + (s + 1)));
  thumbs.append(f);
}

/* interactive slice cards */
const ranges = [], outs = [], cuts = [];
const list = $('#slices');
for (let s = 0; s < 3; s++) {
  const li = el('li', 'slice'); li.style.setProperty('--c', SLICE[s]);
  const body = el('div', 'slice-body'), head = el('div', 'slice-head');
  const lab = el('label', null, 'Slice ' + (s + 1)); lab.htmlFor = 'r' + s;
  const out = el('output'); out.htmlFor = 'r' + s; head.append(lab, out);
  const r = el('input'); r.type = 'range'; r.id = 'r' + s; r.min = 0; r.max = 100; r.step = 1;
  const b = el('button', 'cut', 'Cut here'); b.type = 'button'; b.setAttribute('aria-pressed', 'false');
  body.append(head, r, b); li.append(makeTile(s, 'Axial slice ' + (s + 1) + ' of the MNI152 template'), body); list.append(li);
  ranges.push(r); outs.push(out); cuts.push(b);
  r.addEventListener('input', () => { state.fr[s] = r.value / 100; userActed(); refresh(s); });
  b.addEventListener('click', () => { state.cut = state.cut === s ? -1 : s; userActed(); syncCuts(); updateScene(); });
}
function syncCuts() { cuts.forEach((b, s) => { const on = state.cut === s; b.setAttribute('aria-pressed', on); b.textContent = on ? 'Close the cut' : 'Cut here'; }); }
function refresh(s) { ranges[s].value = Math.round(state.fr[s] * 100); outs[s].textContent = Math.round(state.fr[s] * 100) + '%'; drawSlice(s); updateScene(); }
$('#reset').addEventListener('click', () => { state.fr = [0.2, 0.5, 0.8]; state.cut = -1; userActed(); syncCuts(); for (let s = 0; s < 3; s++) refresh(s); });

/* ---------- 3D scene ---------- */
const hero = $('#hero'), canvas = $('#brain');
let renderer, scene, camera, brainMat, planes = [], cap, capTex, capCtx, clip, center, ready = false, W = 1, H = 1;
const P = [
  { az: -2.46, pol: 1.25, zoom: 1.0 },
  { az: -1.92, pol: 1.36, zoom: 1.0 },
  { az: -2.62, pol: 0.74, zoom: 0.95 },
  { az: -2.2, pol: 1.05, zoom: 1.05 },
];
const narrow = () => W / H < 0.9;
function preset(i) {
  const p = { ...P[i] }, n = narrow();
  p.ox = n ? 0 : (i === 3 ? 0.16 : -0.14);
  p.oy = n ? (i === 3 ? 0.3 : 0.16) : 0.03;
  if (n) p.zoom *= i === 3 ? 1.35 : 1.0;
  return p;
}
const cur = { ...preset(0) }, goal = { ...cur };
let dragged = false;

async function loadMesh() {
  const s = atob(MESH), bin = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bin[i] = s.charCodeAt(i);
  if (typeof DecompressionStream === 'undefined') throw new Error('no DecompressionStream');
  return new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

function initScene(buf) {
  const dv = new DataView(buf);
  const nv = dv.getUint32(4, true), nf = dv.getUint32(8, true);
  const lo = [0, 1, 2].map(i => dv.getFloat32(12 + 4 * i, true)), hi = [0, 1, 2].map(i => dv.getFloat32(24 + 4 * i, true));
  const q = new Uint16Array(buf, 36, nv * 3), shade = new Uint8Array(buf, 36 + nv * 6, nv);
  let off = 36 + nv * 7; off += (4 - (off % 4)) % 4;
  const idx = new Uint32Array(buf, off, nf * 3);
  const pos = new Float32Array(nv * 3), col = new Float32Array(nv * 3);
  for (let i = 0; i < nv; i++) {
    for (let a = 0; a < 3; a++) pos[3 * i + a] = lo[a] + (q[3 * i + a] / 65535) * (hi[a] - lo[a]);
    const t = 0.3 + 0.7 * (shade[i] / 255);
    col[3 * i] = t * 0.92; col[3 * i + 1] = t * 0.93; col[3 * i + 2] = t * 0.95;
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
  camera = new THREE.PerspectiveCamera(30, 1, 0.1, 60);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x101414, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 2.3); key.position.set(-2.5, 3, -2.5); scene.add(key);
  const rim = new THREE.DirectionalLight(0x5fd0c4, 1.4); rim.position.set(3, 1, 3); scene.add(rim);

  // RAS millimetres -> y-up scene, centred on the brain
  const outer = new THREE.Group(); outer.rotation.x = -Math.PI / 2; outer.scale.setScalar(0.01); scene.add(outer);
  const mm = new THREE.Group(); mm.position.copy(center).multiplyScalar(-1); outer.add(mm);

  clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);
  brainMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.58, metalness: 0.03, side: THREE.DoubleSide, clippingPlanes: [] });
  mm.add(new THREE.Mesh(geo, brainMat));

  const w = hi[0] - lo[0] + 10, h = hi[1] - lo[1] + 10;
  for (let s = 0; s < 3; s++) {
    const g = new THREE.Group();
    g.add(new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: SLICE[s], transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false })));
    g.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(w, h)), new THREE.LineBasicMaterial({ color: SLICE[s] })));
    g.position.set(center.x, center.y, 0); g.visible = false; mm.add(g); planes.push(g);
  }
  const cc = document.createElement('canvas'); cc.width = V.nx; cc.height = V.ny; capCtx = cc.getContext('2d');
  capTex = new THREE.CanvasTexture(cc); capTex.colorSpace = THREE.SRGBColorSpace;
  cap = new THREE.Mesh(new THREE.PlaneGeometry(V.nx * V.step, V.ny * V.step), new THREE.MeshBasicMaterial({ map: capTex, transparent: true, alphaTest: 0.5, side: THREE.DoubleSide }));
  cap.position.set(V.x0 + (V.nx - 1) * V.step / 2, V.y0 + (V.ny - 1) * V.step / 2, 0); cap.visible = false; mm.add(cap);

  resize(); new ResizeObserver(resize).observe(hero);
  ready = true; hero.classList.add('ready');
  updateScene();
}

function resize() {
  W = hero.clientWidth; H = hero.clientHeight; if (!renderer || !W || !H) return;
  renderer.setSize(W, H, false);
  camera.aspect = W / H; camera.fov = narrow() ? 40 : 30;
  const p = preset(slide); goal.ox = p.ox; goal.oy = p.oy; goal.zoom = p.zoom;
}
function placeCamera() {
  const vf = camera.fov * Math.PI / 180, hf = 2 * Math.atan(Math.tan(vf / 2) * camera.aspect), R = narrow() ? 1.3 : 1.55;
  const d = Math.max(R / Math.tan(vf / 2), R / Math.tan(hf / 2)) * cur.zoom;
  camera.position.set(d * Math.sin(cur.pol) * Math.sin(cur.az), d * Math.cos(cur.pol), d * Math.sin(cur.pol) * Math.cos(cur.az));
  camera.lookAt(0, 0, 0);
  camera.setViewOffset(W, H, cur.ox * W, cur.oy * H, W, H);
}

function updateScene() {
  if (!ready) return;
  const zs = state.fr.map(sliceZ), c = cutFor(), cutZ = c < 0 ? null : zs[c];
  planes.forEach((g, s) => { g.position.z = zs[s]; g.visible = slide >= 1 && (cutZ == null || (s !== c && zs[s] < cutZ)); });
  if (cutZ == null || !atlasCtx) { brainMat.clippingPlanes = []; cap.visible = false; }
  else {
    clip.constant = (cutZ - center.z) * 0.01 + 0.002; brainMat.clippingPlanes = [clip];
    const [x, y] = rect(sliceIndex(state.fr[c]));
    const im = atlasCtx.getImageData(x, y, V.nx, V.ny), d = im.data;
    for (let i = 0; i < d.length; i += 4) d[i + 3] = d[i] > 4 ? 255 : 0;
    capCtx.putImageData(im, 0, 0); capTex.needsUpdate = true;
    cap.position.z = cutZ; cap.visible = true;
  }
  brainMat.needsUpdate = true;
}

let heroVisible = true, last = performance.now();
new IntersectionObserver(([e]) => { heroVisible = e.isIntersecting; schedule(); }).observe(hero);
function frame(t) {
  const dt = Math.min(0.05, (t - last) / 1000); last = t;
  if (ready && heroVisible) {
    if (slide === 0 && !dragged && !reduced && !drag) goal.az += dt * 0.16;
    const k = reduced ? 1 : 1 - Math.exp(-dt * 3.2);
    for (const key of ['az', 'pol', 'zoom', 'ox', 'oy']) cur[key] += (goal[key] - cur[key]) * k;
    placeCamera(); renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
}

/* rotate by drag and keys */
let drag = null;
canvas.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY }; dragged = true; canvas.setPointerCapture(e.pointerId); userActed(); });
canvas.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  cur.az -= dx * 0.008; goal.az = cur.az;
  cur.pol = goal.pol = clamp(cur.pol - dy * 0.008, 0.25, Math.PI - 0.25);
});
const endDrag = () => { drag = null; };
canvas.addEventListener('pointerup', endDrag); canvas.addEventListener('pointercancel', endDrag);
canvas.addEventListener('keydown', e => {
  const m = { ArrowLeft: [0.15, 0], ArrowRight: [-0.15, 0], ArrowUp: [0, -0.12], ArrowDown: [0, 0.12] }[e.key];
  if (!m) return; e.preventDefault(); dragged = true; userActed();
  goal.az += m[0]; goal.pol = clamp(goal.pol + m[1], 0.25, Math.PI - 0.25);
});

/* ---------- slides ---------- */
const slides = $$('.slide'), dots = $$('.dot'), pauseBtn = $('#pause');
const DUR = 7500; let auto = !reduced, timer = null;
hero.style.setProperty('--dur', DUR + 'ms');
function setSlide(i, user) {
  slide = (i + slides.length) % slides.length;
  if (user) stopAuto();
  slides.forEach((s, k) => { const on = k === slide; s.classList.toggle('active', on); s.inert = !on; s.setAttribute('aria-hidden', String(!on)); });
  dots.forEach((d, k) => d.setAttribute('aria-current', String(k === slide)));
  hero.dataset.slide = slide;
  const p = preset(slide); goal.az = cur.az + wrap(p.az - cur.az); goal.pol = p.pol; goal.zoom = p.zoom; goal.ox = p.ox; goal.oy = p.oy;
  updateScene(); schedule();
}
function schedule() {
  clearTimeout(timer);
  const playing = auto && heroVisible;
  hero.classList.toggle('playing', playing);
  if (playing) timer = setTimeout(() => setSlide(slide + 1, false), DUR);
  pauseBtn.textContent = auto ? 'Pause' : 'Play';
  pauseBtn.setAttribute('aria-label', auto ? 'Pause the slideshow' : 'Play the slideshow');
}
function stopAuto() { auto = false; schedule(); }
function userActed() { stopAuto(); }
dots.forEach((d, k) => d.addEventListener('click', () => setSlide(k, true)));
$('#prev').addEventListener('click', () => setSlide(slide - 1, true));
$('#next').addEventListener('click', () => setSlide(slide + 1, true));
pauseBtn.addEventListener('click', () => { auto = !auto; if (auto) setSlide(slide, false); else schedule(); });
hero.addEventListener('focusin', e => { if (e.target !== pauseBtn && e.target.matches(':focus-visible')) stopAuto(); });
$$('[data-goto]').forEach(a => a.addEventListener('click', e => { e.preventDefault(); setSlide(Number(a.dataset.goto), true); }));

syncCuts(); for (let s = 0; s < 3; s++) refresh(s);
setSlide(0, false);
requestAnimationFrame(frame);
loadMesh().then(initScene).catch(() => {
  $('#stage-msg').textContent = 'This browser cannot show the 3D brain. The slice previews and charts still work.';
  hero.classList.add('failed');
});

/* ---------- tooltip ---------- */
const tip = $('#tip');
function showTip(target, value, label) {
  tip.replaceChildren(el('strong', null, value), el('span', null, label)); tip.hidden = false;
  const r = target.getBoundingClientRect(), t = tip.getBoundingClientRect();
  const x = clamp(r.left + r.width / 2 - t.width / 2, 8, innerWidth - t.width - 8);
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

/* ---------- numbers ---------- */
$('#n-volumes').textContent = fmt(S.dataset.volumes);
$('#n-studies').textContent = fmt(S.dataset.studies);
$('#n-patients').textContent = fmt(S.dataset.patients);
$('#sample-line').textContent = 'The charts use the first batch of the release, ' + fmt(S.n_studies) + ' studies from ' + fmt(S.n_patients) + ' patients with ' + fmt(S.n_series) + ' MRI series, and the pathology label table for ' + fmt(S.labels_n) + ' studies.';

/* age pyramid */
(() => {
  const root = $('#age'), A = S.age;
  const rows = A.bins.map((b, i) => ({ b, f: A.F[i], m: A.M[i] })).filter(r => r.f + r.m > 0).reverse();
  const mx = Math.max(...rows.map(r => Math.max(r.f, r.m)));
  rows.forEach(r => {
    const row = el('div', 'pyr-row'), name = r.b >= 90 ? '90 and over' : r.b + ' to ' + (r.b + 4);
    const f = el('div', 'pyr-f'), fb = el('i'); fb.style.width = (100 * r.f / mx) + '%'; f.append(fb);
    const m = el('div', 'pyr-m'), mb = el('i'); mb.style.width = (100 * r.m / mx) + '%'; m.append(mb);
    bindTip(fb, fmt(r.f) + ' studies', 'Women aged ' + name); bindTip(mb, fmt(r.m) + ' studies', 'Men aged ' + name);
    row.append(f, el('span', 'pyr-age', r.b >= 90 ? '90+' : String(r.b)), m); root.append(row);
  });
  $('#age-note').textContent = 'Median age ' + A.median + ' years, range ' + A.min + ' to ' + A.max + '. ' + fmt(S.sex.F) + ' studies of women and ' + fmt(S.sex.M) + ' of men.';
})();

/* ranked bars */
function bars(root, items) {
  const mx = Math.max(...items.map(d => d.n));
  items.forEach(d => {
    const row = el('div', 'hb'), bar = el('div', 'hb-bar'), fill = el('i');
    fill.style.width = Math.max(0.6, 100 * d.n / mx) + '%'; bar.append(fill);
    row.append(el('span', 'hb-name', d.name), el('span', 'hb-val', fmt(d.n)), bar);
    if (d.sub) row.append(el('span', 'hb-sub', d.sub));
    root.append(row);
  });
}
bars($('#scanners'), S.scanner.map(d => ({ name: d.name, n: d.n, sub: d.field })));
bars($('#modality'), Object.entries(S.modality).map(([name, n]) => ({ name, n })));

/* pathology labels and age profile */
(() => {
  const root = $('#labels'), prof = $('#profile'), title = $('#profile-title'), note = $('#profile-note');
  const B = S.by_age, mx = S.labels[0].pct, buttons = new Map(); let selected = 'Cerebral atrophy', showAll = false;
  S.labels.forEach((d, i) => {
    const has = !!B.labels[d.name], b = el('button', 'lab'); b.type = 'button'; b.disabled = !has;
    if (i >= 12) b.classList.add('extra');
    const bar = el('span', 'lab-bar'), fill = el('i'); fill.style.width = Math.max(0.8, 100 * d.pct / mx) + '%'; bar.append(fill);
    b.append(el('span', 'lab-name', d.name), el('span', 'lab-val', d.pct.toFixed(1) + '%'), bar);
    b.title = fmt(d.n) + ' of ' + fmt(S.labels_n) + ' studies' + (has ? '' : '. Too few cases in the first batch for an age profile');
    b.addEventListener('click', () => { selected = d.name; draw(); });
    buttons.set(d.name, b); root.append(b);
  });
  const more = $('#labels-more');
  more.textContent = 'Show all ' + S.labels.length + ' labels';
  more.addEventListener('click', () => { showAll = !showAll; root.classList.toggle('all', showAll); more.textContent = showAll ? 'Show the 12 most common' : 'Show all ' + S.labels.length + ' labels'; });
  const names = ['<20', '20s', '30s', '40s', '50s', '60s', '70s', '80+'];
  function draw() {
    buttons.forEach((b, k) => b.setAttribute('aria-pressed', String(k === selected)));
    const v = B.labels[selected]; title.textContent = selected + ' by age';
    const top = Math.max(10, ...v.map(x => x || 0)), scale = top > 50 ? 100 : top > 25 ? 50 : top > 10 ? 25 : 10;
    prof.replaceChildren();
    v.forEach((p, i) => {
      const colm = el('div', 'col'), plot = el('div', 'col-plot');
      if (p == null) { const s = el('i', 'sup'); plot.append(el('span', 'col-val', '<5'), s); bindTip(s, 'fewer than 5 studies', B.bands[i] + ', not shown'); }
      else {
        const f = el('i'); f.style.height = 'calc((100% - 24px) * ' + Math.max(0.01, p / scale).toFixed(4) + ')';
        plot.append(el('span', 'col-val', p.toFixed(p < 10 ? 1 : 0) + '%'), f);
        bindTip(f, p.toFixed(1) + '% of ' + fmt(B.n[i]) + ' studies', selected + ', age ' + B.bands[i]);
      }
      colm.append(plot, el('span', 'col-name', names[i])); prof.append(colm);
    });
    note.textContent = 'Share of studies with this label in each age band, first batch only (' + fmt(B.n_studies) + ' studies). Axis runs from 0 to ' + scale + '%. Cells with fewer than 5 studies are not shown.';
  }
  draw();
})();

$('#report-stat').textContent = 'A typical impression is ' + S.reports.impression_words_median + ' words long. One in ten runs past ' + S.reports.impression_words_p90 + ' words, which matters because the text encoder reads at most 64 tokens.';
window.__explorer = { state, get slide() { return slide; }, get ready() { return ready; }, setSlide, stopAuto, refresh: () => { syncCuts(); for (let s = 0; s < 3; s++) refresh(s); } };
