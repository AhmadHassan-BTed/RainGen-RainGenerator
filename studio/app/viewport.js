/**
 * The 3-D viewport of the 4D Environment Composer (Three.js r160).
 *
 * A world, not knobs (MASTER_PLAN 7.6): objects, a listener with its
 * "hearing footprint" (for ground impacts heard at height h, 50 % of the
 * intensity comes from within radius h and 90 % from within 3h, section 3.5),
 * storm cells as actors on editable track splines, wind streamlines, lightning
 * from the engine's channel geometry and a thunder wavefront expanding at c(T).
 *
 * World coordinates: metres, x east, y north, z up. Display (Three.js) space:
 * X = x, Y = z, Z = -y. Far-field items (storms, strikes) are drawn with a log
 * distance compression beyond 150 m so a 20 km storm still sits in view; labels
 * always show true distances.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { WeatherFX } from './visuals.js';
import { MATERIALS, DEMO_WALK, estimateImpactRate, detectShelter, stormPosition, rangeBearing } from './presets.js';
import { flashBang, fmtNum, compassWord, clamp, OKABE_ITO } from './a11y.js';
import { instantiate, objectLabel } from './library.js';

const DEG = Math.PI / 180;
const NEAR_R = 150;
const FAR_K = 60;

export function compressDist(d) {
  return d <= NEAR_R ? d : NEAR_R + FAR_K * Math.log(d / NEAR_R);
}

export function expandDist(r) {
  return r <= NEAR_R ? r : NEAR_R * Math.exp((r - NEAR_R) / FAR_K);
}

export function webglAvailable() {
  try {
    const c = document.createElement('canvas');
    return !!(c.getContext('webgl2') || c.getContext('webgl'));
  } catch (_) {
    return false;
  }
}

function hashStr(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(a) {
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function prismGeometry(w, d, h, overhang = 0) {
  const x = w / 2 + overhang;
  const z = d / 2 + overhang;
  const A = [-x, 0, z];
  const B = [x, 0, z];
  const C = [x, 0, -z];
  const D = [-x, 0, -z];
  const E = [-x, h, 0];
  const F = [x, h, 0];
  const tris = [A, B, F, A, F, E, C, D, E, C, E, F, A, E, D, B, C, F, A, D, C, A, C, B];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(tris.flat(), 3));
  geo.computeVertexNormals();
  return geo;
}

let cloudTexture = null;
function getCloudTexture() {
  if (cloudTexture) return cloudTexture;
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 4, 64, 64, 62);
  grad.addColorStop(0, 'rgba(255,255,255,0.95)');
  grad.addColorStop(0.45, 'rgba(255,255,255,0.55)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 128, 128);
  const rnd = mulberry32(42);
  g.globalCompositeOperation = 'destination-out';
  for (let i = 0; i < 70; i++) {
    g.fillStyle = `rgba(0,0,0,${0.05 + rnd() * 0.12})`;
    g.beginPath();
    g.arc(rnd() * 128, rnd() * 128, 4 + rnd() * 14, 0, Math.PI * 2);
    g.fill();
  }
  cloudTexture = new THREE.CanvasTexture(c);
  cloudTexture.colorSpace = THREE.SRGBColorSpace;
  return cloudTexture;
}

const SEASON_LEAF = { spring: '#7fa64a', summer: '#4f7c35', autumn: '#a1772c', winter: '#6d6658' };
const HEAT_RAMP = [new THREE.Color(OKABE_ITO.blue), new THREE.Color(OKABE_ITO.sky), new THREE.Color(OKABE_ITO.yellow), new THREE.Color(OKABE_ITO.vermillion)];

function heatColor(x) {
  const t = clamp(x, 0, 1) * (HEAT_RAMP.length - 1);
  const i = Math.min(HEAT_RAMP.length - 2, Math.floor(t));
  return HEAT_RAMP[i].clone().lerp(HEAT_RAMP[i + 1], t - i);
}

export class Viewport {
  /**
   * @param {HTMLElement} container
   * @param {{store: any, onListenerLive?: Function, onCaption?: Function, onDemoEnd?: Function, onModeChange?: Function}} opts
   */
  constructor(container, opts) {
    if (!webglAvailable()) throw new Error('WebGL is not available in this browser.');
    this.container = container;
    this.store = opts.store;
    this.onListenerLive = opts.onListenerLive || (() => {});
    this.onCaption = opts.onCaption || (() => {});
    this.onDemoEnd = opts.onDemoEnd || (() => {});
    this.onModeChange = opts.onModeChange || (() => {});
    this.mode = 'director';
    this.visualsMode = 'full';
    this.timelineTime = 0;
    this.snap = false;
    this.layers = { footprint: true, wind: true, wavefront: true, rays: false, heatmap: false, labels: false, grid: true };
    this.liveRates = new Map();
    this.nodes = new Map();
    this.stormNodes = new Map();
    this.matCache = new Map();
    this.keys = new Set();
    this.autoShelter = true;
    this.pulse = 0;
    this._lastListenerPush = 0;
    this._lastShelterCheck = 0;
    this._heatTimer = 0;
    this._raysTimer = 0;
    this.clock = new THREE.Clock();

    this._initRenderer();
    this._initScene();
    this._initControls();
    this._initListener();
    this.fx = new WeatherFX({
      scene: this.scene,
      renderer: this.renderer,
      camera: this.camera,
      ground: this.ground,
      getFocus: () => this.getFocus(),
      getListener: () => this.listenerEarDisplay(),
      channelToDisplay: (pts) => pts.map((p) => this.toDisplay(p, true)),
      strikeToDisplay: (km, bearing) => this.strikeDisplay(km, bearing),
      footprints: () => this._footprints(),
      estimateRate: (id) => {
        const o = this.store.getObject(id);
        return o ? estimateImpactRate(o, this.store.scene) : 0;
      },
    });
    this.fx.onWaveArrive = () => { this.pulse = 1; };
    this._bindStore();
    this._bindPointer();
    this._bindDrop();
    this._bindKeys();
    this.syncAll();
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this._frame());
  }

  /* ------------------------------------------------------------------ setup */
  _initRenderer() {
    const r = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.domElement.className = 'vp-canvas';
    r.domElement.tabIndex = 0;
    r.domElement.setAttribute('role', 'application');
    r.domElement.setAttribute('aria-label', '3-D environment viewport. Arrow keys move the selected object or the listener; 1 Director, 2 Top, 3 Walk camera.');
    this.container.appendChild(r.domElement);
    this.renderer = r;
    const lr = new CSS2DRenderer();
    lr.domElement.className = 'vp-labels';
    this.container.appendChild(lr.domElement);
    this.labelRenderer = lr;
  }

  _initScene() {
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.05, 8000);
    this.camera.rotation.order = 'YXZ';
    this.camera.position.set(-16, 13, 20);

    const groundMat = new THREE.MeshStandardMaterial({ color: '#2f3a2c', roughness: 0.95, metalness: 0 });
    const ground = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000), groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = false;
    ground.name = 'ground';
    this.ground = ground;
    this.scene.add(ground);

    const minor = new THREE.GridHelper(80, 80, 0x9fb3c8, 0x9fb3c8);
    minor.material.transparent = true;
    minor.material.opacity = 0.07;
    minor.material.depthWrite = false;
    minor.position.y = 0.015;
    const major = new THREE.GridHelper(1000, 100, 0x9fb3c8, 0x9fb3c8);
    major.material.transparent = true;
    major.material.opacity = 0.05;
    major.material.depthWrite = false;
    major.position.y = 0.012;
    this.grid = new THREE.Group();
    this.grid.add(minor, major);
    for (const [txt, x, z] of [['N', 0, -46], ['E', 46, 0], ['S', 0, 46], ['W', -46, 0]]) {
      const l = this._label(txt, 'compass-label');
      l.position.set(x, 0.2, z);
      this.grid.add(l);
    }
    this.scene.add(this.grid);

    this.objectsGroup = new THREE.Group();
    this.objectsGroup.name = 'objects';
    this.scene.add(this.objectsGroup);
    this.stormGroup = new THREE.Group();
    this.scene.add(this.stormGroup);
    this.raysGroup = new THREE.Group();
    this.raysGroup.visible = false;
    this.scene.add(this.raysGroup);
    this.raycaster = new THREE.Raycaster();
    this.groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  }

  _initControls() {
    const orbit = new OrbitControls(this.camera, this.renderer.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.08;
    orbit.maxPolarAngle = 89 * DEG;
    orbit.minDistance = 1.5;
    orbit.maxDistance = 2500;
    orbit.screenSpacePanning = false;
    orbit.target.set(0, 1, 0);
    this.orbit = orbit;
    this.directorPose = { pos: this.camera.position.clone(), target: orbit.target.clone() };

    const tc = new TransformControls(this.camera, this.renderer.domElement);
    tc.setSpace('world');
    tc.size = 0.9;
    tc.addEventListener('dragging-changed', (e) => {
      orbit.enabled = !e.value && this.mode !== 'walk' && this.mode !== 'demo';
    });
    tc.addEventListener('mouseDown', () => {
      this.store.beginGesture('transform');
      this._tcDragging = true;
    });
    tc.addEventListener('objectChange', () => this._onGizmoChange(false));
    tc.addEventListener('mouseUp', () => {
      this._tcDragging = false;
      this._onGizmoChange(true);
      this.store.endGesture();
    });
    this.scene.add(tc);
    this.transform = tc;

    const walk = new PointerLockControls(this.camera, this.renderer.domElement);
    walk.addEventListener('lock', () => this.container.classList.add('walk-locked'));
    walk.addEventListener('unlock', () => this.container.classList.remove('walk-locked'));
    this.walk = walk;
  }

  _initListener() {
    const g = new THREE.Group();
    g.name = 'listener';
    const skin = new THREE.MeshStandardMaterial({ color: '#f1e4d2', roughness: 0.55 });
    const accent = new THREE.MeshStandardMaterial({ color: OKABE_ITO.orange, roughness: 0.4, emissive: new THREE.Color(OKABE_ITO.orange), emissiveIntensity: 0.25 });
    const headPivot = new THREE.Group();
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 16), skin);
    const nose = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.16, 12), accent);
    nose.geometry.rotateX(-Math.PI / 2);
    nose.position.set(0, 0, -0.17);
    const earGeo = new THREE.SphereGeometry(0.04, 12, 8);
    const earL = new THREE.Mesh(earGeo, accent);
    earL.position.set(-0.135, 0, 0);
    const earR = new THREE.Mesh(earGeo, accent);
    earR.position.set(0.135, 0, 0);
    headPivot.add(head, nose, earL, earR);
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.15, 0.19, 1, 14),
      new THREE.MeshStandardMaterial({ color: '#c9d4e0', roughness: 0.7, transparent: true, opacity: 0.55 }),
    );
    const wedge = new THREE.Mesh(
      new THREE.CircleGeometry(1.6, 32, Math.PI / 2 - Math.PI / 6, Math.PI / 3),
      new THREE.MeshBasicMaterial({ color: OKABE_ITO.orange, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide }),
    );
    wedge.geometry.rotateX(-Math.PI / 2);
    wedge.position.y = 0.04;
    const ringMat50 = new THREE.MeshBasicMaterial({ color: OKABE_ITO.yellow, transparent: true, opacity: 0.75, depthWrite: false, side: THREE.DoubleSide });
    const ringMat90 = new THREE.MeshBasicMaterial({ color: OKABE_ITO.sky, transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide });
    const ring50 = new THREE.Mesh(new THREE.RingGeometry(0.965, 1, 96), ringMat50);
    ring50.geometry.rotateX(-Math.PI / 2);
    ring50.position.y = 0.035;
    const ring90 = new THREE.Mesh(new THREE.RingGeometry(0.985, 1, 128), ringMat90);
    ring90.geometry.rotateX(-Math.PI / 2);
    ring90.position.y = 0.03;
    const l50 = this._label('50 % of rain sound', 'ring-label ring-50');
    const l90 = this._label('90 %', 'ring-label ring-90');
    const pick = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 1, 8), new THREE.MeshBasicMaterial({ visible: false }));
    pick.userData.kind = 'listener';
    const nameLabel = this._label('Listener', 'listener-label');
    g.add(headPivot, body, wedge, ring50, ring90, l50, l90, pick, nameLabel);
    this.listener = { group: g, headPivot, body, wedge, ring50, ring90, l50, l90, pick, ringMat50, ringMat90, nameLabel };
    this.scene.add(g);
  }

  _label(text, cls = '') {
    const el = document.createElement('div');
    el.className = `vp-label ${cls}`;
    el.textContent = text;
    el.setAttribute('aria-hidden', 'true');
    return new CSS2DObject(el);
  }

  /* ------------------------------------------------------------------ coordinates */
  get listenerWorld() {
    return this.store.scene.listener.position;
  }

  /** World [x, y, z] -> display Vector3; compress=true applies far-field log compression about the listener. */
  toDisplay(p, compress = false) {
    const x = p[0] ?? 0;
    const y = p[1] ?? 0;
    const z = p[2] ?? 0;
    if (!compress) return new THREE.Vector3(x, z, -y);
    const L = this.listenerWorld;
    const dx = x - L[0];
    const dy = y - L[1];
    const d = Math.hypot(dx, dy);
    const k = d > NEAR_R ? compressDist(d) / d : 1;
    return new THREE.Vector3(L[0] + dx * k, z * k, -(L[1] + dy * k));
  }

  /** Display Vector3 (on the ground) -> world [x, y, z]; expand=true inverts the far-field compression. */
  fromDisplay(v, expand = false) {
    if (!expand) return [v.x, -v.z, v.y];
    const L = this.listenerWorld;
    const dx = v.x - L[0];
    const dy = -v.z - L[1];
    const r = Math.hypot(dx, dy);
    const k = r > NEAR_R ? expandDist(r) / r : 1;
    return [L[0] + dx * k, L[1] + dy * k, v.y];
  }

  strikeDisplay(km, bearingDeg) {
    const L = this.listenerWorld;
    const d = km * 1000;
    const b = bearingDeg * DEG;
    return this.toDisplay([L[0] + Math.sin(b) * d, L[1] + Math.cos(b) * d, 0], true);
  }

  listenerEarDisplay() {
    const s = this.store.scene.listener;
    return new THREE.Vector3(s.position[0], s.position[2] + s.height_m, -s.position[1]);
  }

  getFocus() {
    if (this.mode === 'walk' || this.mode === 'demo') return this.camera.position.clone();
    return this.orbit.target.clone();
  }

  /* ------------------------------------------------------------------ store sync */
  _bindStore() {
    this.store.on('change', (e) => {
      if (e.kind === 'load' || e.kind === 'restore') {
        this.syncAll();
        if (e.kind === 'load' && this.mode !== 'demo') this.frameListener();
        return;
      }
      let weather = false;
      let listener = false;
      for (const p of e.paths) {
        if (p.startsWith('place.objects.')) this._syncObject(p.split('.')[2]);
        else if (p.startsWith('listener')) listener = true;
        else if (p.startsWith('weather.season')) {
          weather = true;
          this.syncAll();
        } else if (p.startsWith('weather') || p === 'vibe' || p.startsWith('place.')) weather = true;
      }
      if (listener) this._syncListener();
      if (weather || listener) this.fx.setWeather(this.store.scene);
    });
    this.store.on('select', ({ id }) => this._onSelect(id));
  }

  syncAll() {
    const objs = this.store.scene.place.objects;
    const ids = new Set(objs.map((o) => o.id));
    for (const id of [...this.nodes.keys()]) if (!ids.has(id)) this._removeNode(id);
    for (const id of [...this.stormNodes.keys()]) if (id !== '__virtual' && !ids.has(id)) this._removeStorm(id);
    for (const o of objs) this._syncObject(o.id, true);
    this._syncListener();
    this.fx.setWeather(this.store.scene);
    this._onSelect(this.store.selection);
  }

  _sig(o) {
    const s = { t: o.type, m: o.material, s: o.size, p: o.params };
    if (o.type === 'tree') s.season = this.store.scene.weather.season;
    return JSON.stringify(s);
  }

  _syncObject(id, force = false) {
    const o = this.store.getObject(id);
    if (!o) {
      this._removeNode(id);
      this._removeStorm(id);
      return;
    }
    if (o.type === 'storm') {
      if (!this.stormNodes.has(id)) this._buildStorm(o);
      return;
    }
    let node = this.nodes.get(id);
    const sig = this._sig(o);
    if (!node || node.sig !== sig || force) {
      if (node && node.sig === sig && force) {
        this._placeNode(node, o);
        return;
      }
      const wasAttached = this.transform.object && this.transform.object === node?.group;
      if (node) this._removeNode(id);
      node = this._buildNode(o);
      node.sig = sig;
      this.nodes.set(id, node);
      this.objectsGroup.add(node.group);
      if (wasAttached || this.store.selection === id) this._onSelect(id);
    }
    this._placeNode(node, o);
  }

  _placeNode(node, o) {
    if (this._tcDragging && this.transform.object === node.group) return;
    node.group.position.copy(this.toDisplay(o.position));
    node.group.rotation.set(0, -(o.rotation_deg || 0) * DEG, 0);
    node.group.scale.set(1, 1, 1);
    if (node.label) node.label.element.textContent = objectLabel(o);
  }

  _removeNode(id) {
    const node = this.nodes.get(id);
    if (!node) return;
    if (this.transform.object === node.group) this.transform.detach();
    this.objectsGroup.remove(node.group);
    node.group.traverse((c) => {
      if (c.geometry) c.geometry.dispose();
      if (c.isCSS2DObject) c.element.remove();
      if (c.material && c.userData.ownMaterial) c.material.dispose();
    });
    this.nodes.delete(id);
  }

  _syncListener() {
    const s = this.store.scene.listener;
    const L = this.listener;
    const h = Math.max(0.2, s.height_m);
    L.group.position.set(s.position[0], s.position[2], -s.position[1]);
    L.group.rotation.y = -(s.heading_deg || 0) * DEG;
    L.headPivot.position.y = h;
    L.body.scale.set(1, Math.max(0.2, h - 0.2), 1);
    L.body.position.y = (h - 0.2) / 2;
    L.pick.scale.set(1, h + 0.2, 1);
    L.pick.position.y = (h + 0.2) / 2;
    L.ring50.scale.setScalar(h);
    L.ring90.scale.setScalar(3 * h);
    L.l50.position.set(h * 0.71, 0.1, h * 0.71);
    L.l90.position.set(3 * h * 0.71, 0.1, 3 * h * 0.71);
    L.nameLabel.position.set(0, h + 0.45, 0);
    const visible = this.mode !== 'walk' && this.mode !== 'demo';
    L.headPivot.visible = visible;
    L.body.visible = visible;
    L.nameLabel.visible = visible && this.layers.labels;
    L.ring50.visible = L.ring90.visible = L.l50.visible = L.l90.visible = this.layers.footprint;
  }

  /* ------------------------------------------------------------------ object meshes */
  _mat(key, { double = false, color = null, opacity = null } = {}) {
    const ck = `${key}|${double}|${color}|${opacity}`;
    if (this.matCache.has(ck)) return this.matCache.get(ck);
    const m = MATERIALS[key] || { color: '#8a8a8a', roughness: 0.8, metalness: 0 };
    const op = opacity ?? m.opacity;
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(color || m.color),
      roughness: m.roughness ?? 0.8,
      metalness: m.metalness ?? 0,
      flatShading: true,
      transparent: op !== undefined && op < 1,
      opacity: op ?? 1,
      side: double ? THREE.DoubleSide : THREE.FrontSide,
      depthWrite: op === undefined || op > 0.6,
    });
    this.matCache.set(ck, mat);
    return mat;
  }

  _buildNode(o) {
    const g = new THREE.Group();
    g.userData.objectId = o.id;
    const [w, d, hRaw] = o.size;
    const h = hRaw || 0;
    const mat = this._mat(o.material);
    const add = (geo, m, x = 0, y = 0, z = 0) => {
      const mesh = new THREE.Mesh(geo, m);
      mesh.position.set(x, y, z);
      mesh.userData.objectId = o.id;
      g.add(mesh);
      return mesh;
    };
    const node = { group: g, anim: null, top: 0, water: false, rot: 0 };
    const rand = mulberry32(hashStr(o.id));

    switch (o.type) {
      case 'surface': {
        const t = Math.max(0.03, h);
        add(new THREE.BoxGeometry(w, t, d), mat, 0, t / 2, 0);
        node.top = t;
        node.water = o.material === 'puddle';
        node.small = true;
        break;
      }
      case 'water': {
        add(new THREE.BoxGeometry(w, 0.05, d), this._mat('water'), 0, 0.025, 0);
        add(new THREE.BoxGeometry(w + 0.3, 0.02, d + 0.3), this._mat('soil'), 0, 0.005, 0);
        node.top = 0.05;
        node.water = true;
        break;
      }
      case 'roof': {
        const style = o.params?.style || 'gable';
        if (style === 'porch' || style === 'shed') {
          const pitch = (o.params?.pitch_deg ?? 8) * DEG;
          const slab = add(new THREE.BoxGeometry(w, 0.08, d), this._mat(o.material, { double: true }), 0, Math.sin(pitch) * d * 0.5, 0);
          slab.rotation.x = pitch;
          const eave = o.position[2];
          if (o.params?.posts !== false) {
            const postGeo = new THREE.CylinderGeometry(0.06, 0.07, eave, 8);
            for (const sx of [-1, 1]) {
              add(postGeo, this._mat('wood'), sx * (w / 2 - 0.15), -eave / 2, d / 2 - 0.15);
            }
          }
          node.top = Math.sin(pitch) * d;
        } else {
          add(prismGeometry(w, d, h, 0.25), this._mat(o.material, { double: true }));
          node.top = h;
        }
        break;
      }
      case 'wall': {
        add(new THREE.BoxGeometry(w, h || 2.5, d || 0.25), mat, 0, (h || 2.5) / 2, 0);
        node.top = h;
        break;
      }
      case 'window': {
        const H = h || 1.2;
        const frameMat = this._mat('wood', { color: '#d9d2c3' });
        const f = 0.05;
        add(new THREE.BoxGeometry(w, f, 0.1), frameMat, 0, f / 2, 0);
        add(new THREE.BoxGeometry(w, f, 0.1), frameMat, 0, H - f / 2, 0);
        add(new THREE.BoxGeometry(f, H, 0.1), frameMat, -w / 2 + f / 2, H / 2, 0);
        add(new THREE.BoxGeometry(f, H, 0.1), frameMat, w / 2 - f / 2, H / 2, 0);
        const hinge = new THREE.Group();
        hinge.position.set(-w / 2 + f, 0, 0);
        const pane = new THREE.Mesh(new THREE.BoxGeometry(w - 2 * f, H - 2 * f, 0.02), this._mat('glass'));
        pane.position.set((w - 2 * f) / 2, H / 2, 0);
        pane.userData.objectId = o.id;
        hinge.add(pane);
        g.add(hinge);
        node.hinge = hinge;
        node.anim = () => {
          const cur = this.store.getObject(o.id);
          const open = clamp(cur?.params?.open ?? 0, 0, 1);
          hinge.rotation.y += (-open * 75 * DEG - hinge.rotation.y) * 0.2;
        };
        node.top = H;
        break;
      }
      case 'tree': {
        const H = o.params?.height || h || 9;
        const conifer = o.params?.species === 'conifer';
        const season = this.store.scene.weather.season;
        const trunkMat = this._mat('wood', { color: '#4a3424' });
        if (conifer) {
          const trunkH = H * 0.25;
          add(new THREE.CylinderGeometry(0.12 + H * 0.008, 0.2 + H * 0.012, trunkH, 7), trunkMat, 0, trunkH / 2, 0);
          const needle = this._mat('needles', { color: season === 'winter' ? '#3d5236' : MATERIALS.needles.color });
          for (let i = 0; i < 4; i++) {
            const r = (w / 2) * (1 - i * 0.22);
            const ch = H * 0.34;
            add(new THREE.ConeGeometry(r, ch, 7), needle, 0, H * 0.18 + i * H * 0.19 + ch / 2, 0);
          }
        } else {
          const trunkH = H * 0.38;
          add(new THREE.CylinderGeometry(0.12 + H * 0.01, 0.2 + H * 0.016, trunkH, 7), trunkMat, 0, trunkH / 2, 0);
          const ry = (H - trunkH) * 0.55;
          const winter = season === 'winter';
          const leafMat = this._mat('leaves', { color: SEASON_LEAF[season] || MATERIALS.leaves.color, opacity: winter ? 0.28 : undefined });
          const canopy = add(new THREE.IcosahedronGeometry(1, 1), leafMat, 0, trunkH + ry * 0.92, 0);
          canopy.scale.set(w / 2, ry, w / 2);
          const lobe = add(new THREE.IcosahedronGeometry(1, 0), leafMat, w * 0.18, trunkH + ry * 1.3, -w * 0.1);
          lobe.scale.set(w * 0.28, ry * 0.55, w * 0.28);
          if (winter) {
            for (let i = 0; i < 6; i++) {
              const b = add(new THREE.CylinderGeometry(0.03, 0.06, ry * 1.2, 4), trunkMat, 0, trunkH + ry * 0.5, 0);
              b.rotation.set((rand() - 0.5) * 1.6, rand() * Math.PI * 2, (rand() - 0.5) * 1.6);
            }
          }
          node.canopy = canopy;
        }
        const phase = rand() * 10;
        node.anim = (t, wind) => {
          const sway = Math.min(0.06, 0.004 * wind.speed) * Math.sin(t * (0.6 + wind.speed * 0.05) + phase);
          g.rotation.z = sway * Math.cos(wind.toYaw);
          g.rotation.x = sway * Math.sin(wind.toYaw);
        };
        node.top = H;
        node.canopyR = w / 2;
        break;
      }
      case 'grass': {
        const base = add(new THREE.PlaneGeometry(w, d), this._mat('grass'), 0, 0.012, 0);
        base.rotation.x = -Math.PI / 2;
        const n = Math.round(clamp(w * d * 1.2, 30, 700));
        const blades = new THREE.InstancedMesh(new THREE.ConeGeometry(0.06, Math.max(0.2, h || 0.35), 3), this._mat('grass', { color: '#5f8d3e' }), n);
        const m4 = new THREE.Matrix4();
        const q = new THREE.Quaternion();
        const pos = new THREE.Vector3();
        const sc = new THREE.Vector3();
        for (let i = 0; i < n; i++) {
          pos.set((rand() - 0.5) * w, (h || 0.35) / 2, (rand() - 0.5) * d);
          q.setFromEuler(new THREE.Euler((rand() - 0.5) * 0.4, rand() * 6.28, (rand() - 0.5) * 0.4));
          const s = 0.6 + rand() * 0.9;
          sc.set(s, s, s);
          m4.compose(pos, q, sc);
          blades.setMatrixAt(i, m4);
        }
        blades.userData.objectId = o.id;
        g.add(blades);
        node.top = h || 0.35;
        break;
      }
      case 'gutter': {
        const gut = add(new THREE.CylinderGeometry(0.075, 0.06, w, 10, 1, true, Math.PI / 2, Math.PI), this._mat('metal', { double: true }), 0, 0, 0);
        gut.rotation.z = Math.PI / 2;
        node.top = 0.05;
        break;
      }
      case 'downspout': {
        const H = h || 2.6;
        add(new THREE.CylinderGeometry(0.05, 0.05, H, 10), this._mat('metal'), 0, H / 2, 0);
        add(new THREE.BoxGeometry(0.12, 0.08, 0.3), this._mat('metal'), 0, 0.06, 0.12);
        node.top = H;
        break;
      }
      case 'eave': {
        add(new THREE.BoxGeometry(w, 0.06, d || 0.3), this._mat(o.material || 'wood'), 0, 0, 0);
        const n = Math.round(clamp(w * (o.params?.drips_per_m ?? 3) * 0.5, 2, 14));
        const dropMat = this._mat('water', { color: '#9cc8e6', opacity: 0.85 });
        const dropGeo = new THREE.SphereGeometry(0.03, 8, 6);
        const drops = [];
        for (let i = 0; i < n; i++) {
          const m = add(dropGeo, dropMat, (i / Math.max(1, n - 1) - 0.5) * w * 0.9, -0.05, 0.12);
          drops.push({ mesh: m, phase: rand() * 10, period: 0.8 + rand() * 1.6 });
        }
        const fallH = Math.max(0.3, o.position[2]);
        node.anim = (t) => {
          const R = this.store.scene.weather.precipitation.rate_mm_h;
          const wet = this.store.scene.weather.precipitation.type === 'none' ? 0.15 : 1;
          const rateScale = clamp(5 / Math.max(0.2, R), 0.25, 8) / wet;
          for (const dr of drops) {
            const period = dr.period * rateScale;
            const tt = ((t + dr.phase) % period);
            const hang = period * 0.6;
            if (tt < hang) {
              dr.mesh.position.y = -0.05;
              dr.mesh.scale.setScalar(0.4 + 0.6 * (tt / hang));
            } else {
              const ft = tt - hang;
              dr.mesh.position.y = -0.05 - 0.5 * 9.81 * ft * ft;
              dr.mesh.scale.setScalar(1);
              if (-dr.mesh.position.y > fallH) dr.mesh.scale.setScalar(0.001);
            }
          }
        };
        node.top = 0.03;
        break;
      }
      case 'wire': {
        const z = o.position[2];
        const poleGeo = new THREE.CylinderGeometry(0.06, 0.08, z, 6);
        add(poleGeo, this._mat('wood', { color: '#5b4632' }), -w / 2, -z / 2, 0);
        add(poleGeo, this._mat('wood', { color: '#5b4632' }), w / 2, -z / 2, 0);
        const sag = w * 0.02;
        const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(-w / 2, 0, 0), new THREE.Vector3(0, -sag * 2, 0), new THREE.Vector3(w / 2, 0, 0));
        const r = Math.max(0.008, (o.params?.diameter_m ?? 0.01) * 0.8);
        const wire = add(new THREE.TubeGeometry(curve, 24, r, 5, false), this._mat('metal', { color: '#2b2f33' }));
        const phase = rand() * 10;
        node.anim = (t, wind) => {
          const f = 0.2 * Math.max(0.5, wind.speed) / Math.max(0.002, o.params?.diameter_m ?? 0.01);
          const amp = wind.speed > 5 ? Math.min(0.08, (wind.speed - 5) * 0.006) : 0;
          wire.rotation.x = amp * Math.sin(t * Math.min(40, f * 0.02) + phase);
        };
        node.top = 0.02;
        break;
      }
      case 'chime': {
        add(new THREE.CylinderGeometry(0.13, 0.13, 0.02, 14), this._mat('wood'), 0, 0, 0);
        const tubes = [];
        const n = o.params?.tubes ?? 5;
        for (let i = 0; i < n; i++) {
          const pivot = new THREE.Group();
          const a = (i / n) * Math.PI * 2;
          pivot.position.set(Math.cos(a) * 0.08, -0.02, Math.sin(a) * 0.08);
          const len = 0.25 + (i / Math.max(1, n - 1)) * 0.3;
          const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, len, 8), this._mat('metal'));
          tube.position.y = -len / 2 - 0.05;
          tube.userData.objectId = o.id;
          pivot.add(tube);
          g.add(pivot);
          tubes.push({ pivot, phase: rand() * 10 });
        }
        node.anim = (t, wind) => {
          const amp = Math.min(0.5, 0.02 + wind.gust * 0.025);
          for (const tb of tubes) {
            tb.pivot.rotation.z = amp * Math.sin(t * 2.2 + tb.phase) * Math.cos(wind.toYaw);
            tb.pivot.rotation.x = amp * Math.sin(t * 1.7 + tb.phase * 1.3) * Math.sin(wind.toYaw);
          }
        };
        node.top = 0;
        break;
      }
      case 'flag': {
        const H = h || 6;
        add(new THREE.CylinderGeometry(0.035, 0.05, H, 8), this._mat('metal', { color: '#d0d4d8' }), 0, H / 2, 0);
        const fw = w || 1.5;
        const geo = new THREE.PlaneGeometry(fw, fw * 0.62, 18, 6);
        geo.translate(fw / 2, 0, 0);
        const flagMat = this._mat('fabric', { double: true, color: OKABE_ITO.blue });
        const pivot = new THREE.Group();
        pivot.position.set(0, H - fw * 0.35, 0);
        const flag = new THREE.Mesh(geo, flagMat);
        flag.userData.objectId = o.id;
        pivot.add(flag);
        g.add(pivot);
        const basePos = geo.attributes.position.array.slice();
        node.anim = (t, wind) => {
          const arr = geo.attributes.position.array;
          const amp = Math.min(0.35, 0.04 + wind.gust * 0.02);
          const speed = 3 + wind.gust * 0.6;
          for (let i = 0; i < arr.length; i += 3) {
            const x = basePos[i];
            arr[i + 2] = Math.sin(x * 3.2 - t * speed) * amp * (x / fw) + Math.sin(x * 7.1 - t * speed * 1.7) * amp * 0.25 * (x / fw);
            arr[i + 1] = basePos[i + 1] - (x / fw) * Math.max(0, 0.3 - wind.speed * 0.03);
          }
          geo.attributes.position.needsUpdate = true;
          geo.computeVertexNormals();
          pivot.rotation.y = wind.toYaw - g.rotation.y;
        };
        node.top = H;
        break;
      }
      case 'car': {
        const L = w || 4.4;
        const Wd = d || 1.8;
        const H = h || 1.45;
        add(new THREE.BoxGeometry(L, H * 0.45, Wd), mat, 0, H * 0.42, 0);
        add(new THREE.BoxGeometry(L * 0.5, H * 0.4, Wd * 0.9), this._mat('glass', { color: '#2a3946', opacity: 0.75 }), -L * 0.05, H * 0.82, 0);
        add(new THREE.BoxGeometry(L * 0.46, 0.04, Wd * 0.86), mat, -L * 0.05, H * 1.02, 0);
        const wheel = new THREE.CylinderGeometry(0.33, 0.33, 0.22, 14);
        wheel.rotateX(Math.PI / 2);
        const tyre = this._mat('asphalt', { color: '#16181b' });
        for (const sx of [-1, 1]) for (const sz of [-1, 1]) add(wheel, tyre, sx * L * 0.32, 0.33, sz * Wd * 0.46);
        const wiperMat = this._mat('metal', { color: '#111' });
        const wipers = [];
        for (const sz of [-0.25, 0.25]) {
          const piv = new THREE.Group();
          piv.position.set(L * 0.21, H * 0.66, sz * Wd);
          const blade = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.02, 0.55), wiperMat);
          blade.position.z = 0.27;
          piv.add(blade);
          g.add(piv);
          wipers.push(piv);
        }
        node.anim = (t) => {
          const mode = this.store.scene.place.wipers;
          const cpm = mode === 'fast' ? 60 : mode === 'slow' ? 25 : 0;
          const ang = cpm ? (Math.sin(t * (cpm / 60) * Math.PI * 2) * 0.5 + 0.5) * 1.6 : 0;
          for (const p of wipers) p.rotation.y = ang;
        };
        node.top = H * 1.02;
        break;
      }
      case 'tent': {
        add(prismGeometry(w, d, h || 1.3, 0.05), this._mat(o.material || 'tent_fabric', { double: true }));
        node.top = h || 1.3;
        break;
      }
      case 'umbrella': {
        const H = h || 2;
        const r = (w || 1.1) / 2;
        add(new THREE.ConeGeometry(r, 0.34, 16, 1, true), this._mat(o.material || 'umbrella_fabric', { double: true }), 0, H - 0.17, 0);
        add(new THREE.CylinderGeometry(0.012, 0.012, H, 6), this._mat('metal'), 0, H / 2, 0);
        node.top = H;
        node.canopyR = r;
        break;
      }
      default: {
        add(new THREE.BoxGeometry(Math.max(0.2, w), Math.max(0.2, h), Math.max(0.2, d)), mat, 0, Math.max(0.2, h) / 2, 0);
        node.top = h;
      }
    }
    node.rot = -(o.rotation_deg || 0) * DEG;

    const heat = new THREE.Mesh(new THREE.CircleGeometry(1, 28), new THREE.MeshBasicMaterial({ color: OKABE_ITO.sky, transparent: true, opacity: 0.45, depthWrite: false, side: THREE.DoubleSide }));
    heat.userData.ownMaterial = true;
    heat.raycast = () => {};
    heat.geometry.rotateX(-Math.PI / 2);
    const hr = Math.max(0.4, Math.max(w, o.type === 'wall' || o.type === 'window' ? 0.6 : d) * 0.6);
    heat.scale.setScalar(hr);
    heat.position.y = (o.type === 'roof' ? node.top : node.top) + 0.04;
    heat.visible = this.layers.heatmap;
    g.add(heat);
    node.heat = heat;

    const label = this._label(objectLabel(o), 'object-label');
    label.position.set(0, Math.max(node.top, 0.5) + 0.6, 0);
    label.visible = this.layers.labels;
    g.add(label);
    node.label = label;
    return node;
  }

  _footprints() {
    const out = [];
    for (const [id, node] of this.nodes) {
      const o = this.store.getObject(id);
      if (!o) continue;
      if (['wire', 'chime', 'flag', 'downspout', 'wall', 'window'].includes(o.type)) continue;
      const c = node.group.position;
      let top = c.y + node.top;
      let w = o.size[0];
      let d = o.size[1];
      if (o.type === 'tree' || o.type === 'umbrella') {
        w = d = (node.canopyR || 1) * 1.6;
        top = c.y + (o.type === 'tree' ? node.top * 0.75 : node.top);
      } else if (o.type === 'car') {
        w = o.size[0] * 0.9;
        d = o.size[1] * 0.9;
      }
      out.push({ id, center: c, w, d, top, water: node.water, small: node.small, rot: -node.group.rotation.y });
    }
    return out;
  }

  /* ------------------------------------------------------------------ storms */
  _buildStorm(o) {
    const id = o.id;
    const group = new THREE.Group();
    group.userData.objectId = id;
    const tex = getCloudTexture();
    const rand = mulberry32(hashStr(id));
    const sprites = [];
    for (let i = 0; i < 18; i++) {
      const m = new THREE.SpriteMaterial({ map: tex, color: new THREE.Color('#596171'), transparent: true, opacity: 0.85, depthWrite: false, fog: false });
      const s = new THREE.Sprite(m);
      s.userData.objectId = id;
      s.userData.offset = new THREE.Vector3((rand() - 0.5) * 1.8, (rand() - 0.3) * 0.5, (rand() - 0.5) * 1.8);
      s.userData.size = 0.7 + rand() * 0.8;
      group.add(s);
      sprites.push(s);
    }
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(1, 1.15, 1, 18, 1, true),
      new THREE.MeshBasicMaterial({ color: '#7d8ba0', transparent: true, opacity: 0.12, depthWrite: false, side: THREE.DoubleSide, fog: false }),
    );
    shaft.userData.objectId = id;
    group.add(shaft);
    const markerMat = new THREE.MeshBasicMaterial({ color: OKABE_ITO.orange, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide });
    const marker = new THREE.Mesh(new THREE.RingGeometry(0.8, 1, 48), markerMat);
    marker.geometry.rotateX(-Math.PI / 2);
    marker.userData.objectId = id;
    const label = this._label('', 'storm-label');
    const trackMat = new THREE.LineBasicMaterial({ color: OKABE_ITO.orange, transparent: true, opacity: 0.9 });
    const track = new THREE.Line(new THREE.BufferGeometry(), trackMat);
    track.frustumCulled = false;
    const handleGroup = new THREE.Group();
    this.stormGroup.add(group, marker, label, track, handleGroup);
    const node = { id, group, sprites, shaft, marker, markerMat, label, track, trackMat, handleGroup, handles: [], trackSig: '', flash: 0, virtual: false };
    this.stormNodes.set(id, node);
    return node;
  }

  _removeStorm(id) {
    const node = this.stormNodes.get(id);
    if (!node) return;
    this.stormGroup.remove(node.group, node.marker, node.label, node.track, node.handleGroup);
    node.sprites.forEach((s) => s.material.dispose());
    node.shaft.geometry.dispose();
    node.marker.geometry.dispose();
    node.track.geometry.dispose();
    node.label.element.remove();
    node.handles.forEach((h) => h.geometry.dispose());
    this.stormNodes.delete(id);
  }

  _updateStorms(dt) {
    const s = this.store.scene;
    const T = s.weather.temperature_c;
    const storms = s.place.objects.filter((o) => o.type === 'storm');
    const wantVirtual = storms.length === 0 && s.weather.thunder.enabled;
    if (wantVirtual && !this.stormNodes.has('__virtual')) {
      const n = this._buildStorm({ id: '__virtual' });
      n.virtual = true;
      n.marker.userData.kind = 'thunder-marker';
      n.marker.userData.objectId = undefined;
      n.sprites.forEach((sp) => { sp.userData.objectId = undefined; });
      n.shaft.userData.objectId = undefined;
      n.group.userData.objectId = undefined;
    } else if (!wantVirtual && this.stormNodes.has('__virtual')) {
      this._removeStorm('__virtual');
    }
    const camDist = this.camera.position.distanceTo(this.orbit.target);
    for (const [id, node] of this.stormNodes) {
      let wx;
      let wy;
      let radiusKm = 4;
      let name = 'Thunder';
      let o = null;
      if (node.virtual) {
        const L = s.listener.position;
        const d = s.weather.thunder.distance_km * 1000;
        const b = s.weather.thunder.bearing_deg * DEG;
        wx = L[0] + Math.sin(b) * d;
        wy = L[1] + Math.cos(b) * d;
        radiusKm = 3;
      } else {
        o = this.store.getObject(id);
        if (!o) continue;
        [wx, wy] = stormPosition(o, this.timelineTime);
        radiusKm = o.params?.radius_km ?? 4;
        name = objectLabel(o);
      }
      const ground = this.toDisplay([wx, wy, 0], true);
      const dReal = Math.hypot(wx - s.listener.position[0], wy - s.listener.position[1]);
      const k = dReal > NEAR_R ? compressDist(dReal) / dReal : 1;
      const R = clamp(radiusKm * 1000 * k * 0.5, 18, 150);
      const alt = clamp(1500 * k, 25, 160);
      node.flash *= Math.exp(-dt * 4);
      const sel = this.store.selection === id;
      node.sprites.forEach((sp) => {
        const off = sp.userData.offset;
        sp.position.set(ground.x + off.x * R, alt + off.y * R * 0.5, ground.z + off.z * R);
        sp.scale.setScalar(R * sp.userData.size);
        sp.material.color.set('#596171').lerp(new THREE.Color('#e6ecff'), Math.min(1, node.flash));
        sp.material.opacity = this.visualsMode === 'calm' ? 0.6 : 0.85;
      });
      node.shaft.position.set(ground.x, alt / 2, ground.z);
      node.shaft.scale.set(R * 0.7, alt, R * 0.7);
      node.shaft.material.opacity = 0.05 + 0.12 * clamp(s.weather.precipitation.rate_mm_h / 30, 0, 1);
      const ms = clamp(camDist * 0.02, 3, 30);
      node.marker.position.set(ground.x, 0.08, ground.z);
      node.marker.scale.setScalar(ms * (sel ? 1.4 : 1));
      node.markerMat.color.set(sel ? OKABE_ITO.yellow : OKABE_ITO.orange);
      const km = dReal / 1000;
      node.label.position.set(ground.x, alt + R * 0.7, ground.z);
      const rb = rangeBearing(s, wx, wy);
      node.label.element.textContent = `${name} · ${fmtNum(km, km < 10 ? 1 : 0)} km ${compassWord(rb.bearing).toUpperCase().replace(/-/g, '')} · ${fmtNum(flashBang(km, T), 1)} s`;

      // track spline + handles
      const track = node.virtual ? null : o.params?.track;
      const lsig = `${JSON.stringify(track)}|${s.listener.position.join(',')}|${Math.round(ms)}`;
      if (track && lsig !== node.trackSig) {
        node.trackSig = lsig;
        const pts = track.map((p) => this.toDisplay([p[0], p[1], 0], true).setY(0.3));
        let linePts = pts;
        if (pts.length > 2) linePts = new THREE.CatmullRomCurve3(pts, false, 'centripetal').getPoints(pts.length * 16);
        node.track.geometry.dispose();
        node.track.geometry = new THREE.BufferGeometry().setFromPoints(linePts);
        while (node.handles.length > pts.length) {
          const h = node.handles.pop();
          node.handleGroup.remove(h);
          h.geometry.dispose();
        }
        while (node.handles.length < pts.length) {
          const h = new THREE.Mesh(new THREE.SphereGeometry(1, 14, 10), new THREE.MeshBasicMaterial({ color: OKABE_ITO.orange, depthTest: false, transparent: true, opacity: 0.95 }));
          h.renderOrder = 20;
          h.userData.kind = 'track-handle';
          h.userData.stormId = id;
          node.handleGroup.add(h);
          node.handles.push(h);
        }
        node.handles.forEach((h, i) => {
          h.userData.index = i;
          h.position.copy(pts[i]);
        });
      }
      node.handles.forEach((h, i) => {
        h.scale.setScalar(ms * (sel ? 0.45 : 0.32));
        h.material.color.set(i === 0 ? OKABE_ITO.green : i === node.handles.length - 1 ? OKABE_ITO.vermillion : OKABE_ITO.orange);
      });
      node.trackMat.opacity = sel ? 1 : 0.55;
      node.track.visible = !!track;
    }
  }

  /* ------------------------------------------------------------------ selection & gizmo */
  _onSelect(id) {
    if (this.selHelper) {
      this.scene.remove(this.selHelper);
      this.selHelper.geometry.dispose();
      this.selHelper = null;
    }
    this.transform.detach();
    for (const n of this.nodes.values()) if (n.label) n.label.visible = this.layers.labels;
    if (!id) return;
    const node = this.nodes.get(id);
    if (node) {
      if (this.mode !== 'walk' && this.mode !== 'demo') this.transform.attach(node.group);
      this._configureGizmo();
      this.selHelper = new THREE.BoxHelper(node.group, new THREE.Color(OKABE_ITO.yellow));
      this.scene.add(this.selHelper);
      if (node.label) node.label.visible = true;
    }
  }

  setGizmoMode(m) {
    this.gizmoMode = m;
    this.transform.setMode(m);
    this._configureGizmo();
  }

  _configureGizmo() {
    const m = this.transform.mode;
    this.transform.showX = m !== 'rotate';
    this.transform.showZ = m !== 'rotate';
    this.transform.showY = true;
    this.transform.setTranslationSnap(this.snap ? 0.5 : null);
    this.transform.setRotationSnap(this.snap ? 15 * DEG : null);
    this.transform.setScaleSnap(this.snap ? 0.1 : null);
  }

  toggleSnap(on = !this.snap) {
    this.snap = on;
    this._configureGizmo();
    return this.snap;
  }

  _onGizmoChange(final) {
    const g = this.transform.object;
    if (!g) return;
    const id = g.userData.objectId;
    const o = this.store.getObject(id);
    if (!o) return;
    const now = performance.now();
    if (!final && now - (this._lastGizmoPush || 0) < 50) return;
    this._lastGizmoPush = now;
    const pos = this.fromDisplay(g.position);
    const patch = {
      position: [round(pos[0]), round(pos[1]), round(Math.max(0, pos[2]))],
      rotation_deg: round(((-g.rotation.y / DEG) % 360 + 360) % 360, 1),
    };
    if (final && this.transform.mode === 'scale') {
      const s = g.scale;
      patch.size = [round(o.size[0] * s.x), round(o.size[1] * s.z), round(o.size[2] * s.y)];
      if (o.type === 'tree') patch.params = { height: round(o.size[2] * s.y) };
    }
    this.store.updateObject(id, patch, { coalesce: `gizmo:${id}` });
  }

  frameSelected() {
    const id = this.store.selection;
    if (!id) return this.frameListener();
    const node = this.nodes.get(id);
    let target;
    let r = 6;
    if (node) {
      const box = new THREE.Box3().setFromObject(node.group);
      target = box.getCenter(new THREE.Vector3());
      r = Math.max(3, box.getSize(new THREE.Vector3()).length());
    } else if (this.stormNodes.has(id)) {
      target = this.stormNodes.get(id).marker.position.clone();
      r = 120;
    } else return;
    this._flyTo(target, r);
  }

  frameListener() {
    if (this.mode === 'walk' || this.mode === 'demo') return;
    const ear = this.listenerEarDisplay();
    this._flyTo(ear, this.mode === 'top' ? 60 : 18);
  }

  _flyTo(target, r) {
    const dir = this.camera.position.clone().sub(this.orbit.target).normalize();
    if (this.mode === 'top') dir.set(0, 1, 0.0001).normalize();
    this._fly = { from: this.camera.position.clone(), to: target.clone().add(dir.multiplyScalar(r * 1.6)), tFrom: this.orbit.target.clone(), tTo: target.clone(), t: 0 };
  }

  /* ------------------------------------------------------------------ camera modes */
  setCameraMode(mode) {
    if (mode === this.mode) return;
    const prev = this.mode;
    if (prev === 'director') this.directorPose = { pos: this.camera.position.clone(), target: this.orbit.target.clone() };
    if (prev === 'walk') {
      this.walk.unlock();
      this.keys.clear();
    }
    if (prev === 'demo') this.demo = null;
    this.mode = mode;
    this.container.dataset.camera = mode;
    const ear = this.listenerEarDisplay();
    if (mode === 'director') {
      this.orbit.enabled = true;
      this.orbit.enableRotate = true;
      this.orbit.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
      this.orbit.screenSpacePanning = false;
      const p = this.directorPose;
      this.camera.position.copy(p.pos);
      this.orbit.target.copy(p.target);
    } else if (mode === 'top') {
      this.orbit.enabled = true;
      this.orbit.enableRotate = false;
      this.orbit.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
      this.orbit.screenSpacePanning = true;
      this.orbit.target.set(ear.x, 0, ear.z);
      this.camera.position.set(ear.x, 140, ear.z + 0.001);
    } else if (mode === 'walk') {
      this.orbit.enabled = false;
      this.camera.position.copy(ear);
      const h = (this.store.scene.listener.heading_deg || 0) * DEG;
      this.camera.rotation.set(0, -h, 0, 'YXZ');
      this.transform.detach();
    } else if (mode === 'demo') {
      this.orbit.enabled = false;
      this.transform.detach();
    }
    this.camera.updateProjectionMatrix();
    this._syncListener();
    if (mode === 'director' || mode === 'top') this._onSelect(this.store.selection);
    this.onModeChange(mode, prev);
  }

  get cameraMode() {
    return this.mode;
  }

  exitWalk() {
    if (this.mode === 'walk' || this.mode === 'demo') this.setCameraMode('director');
  }

  /* ------------------------------------------------------------------ guided demo */
  startDemo() {
    const wps = DEMO_WALK.waypoints;
    const pts = wps.map((w) => new THREE.Vector3(w.pos[0], 1.7, -w.pos[1]));
    this.setCameraMode('demo');
    this.demo = { t: 0, curve: new THREE.CatmullRomCurve3(pts, false, 'centripetal'), wps, idx: -1, windowTween: null };
    this.camera.position.copy(pts[0]);
    this.onCaption('Guided demo: walk in from the storm. Press Escape to stop.');
  }

  stopDemo() {
    if (this.mode === 'demo') this.setCameraMode('director');
    this.demo = null;
  }

  _updateDemo(dt) {
    const D = this.demo;
    if (!D) return;
    D.t += dt;
    const wps = D.wps;
    const n = wps.length;
    let i = 0;
    while (i < n - 2 && D.t >= wps[i + 1].t) i++;
    const seg = wps[i + 1].t - wps[i].t || 1;
    const local = clamp((D.t - wps[i].t) / seg, 0, 1);
    const ease = local * local * (3 - 2 * local);
    const u = (i + ease) / (n - 1);
    const pos = D.curve.getPoint(Math.min(1, u));
    this.camera.position.copy(pos);
    const la = wps[i].look;
    const lb = wps[i + 1].look;
    const look = new THREE.Vector3(la[0] + (lb[0] - la[0]) * ease, 1.6, -(la[1] + (lb[1] - la[1]) * ease));
    const m = new THREE.Matrix4().lookAt(pos, look, new THREE.Vector3(0, 1, 0));
    const q = new THREE.Quaternion().setFromRotationMatrix(m);
    this.camera.quaternion.slerp(q, Math.min(1, dt * 3));

    for (let k = D.idx + 1; k < n; k++) {
      if (D.t < wps[k].t) break;
      D.idx = k;
      const wp = wps[k];
      if (wp.caption) this.onCaption(wp.caption);
      if (wp.windowOpen !== undefined) {
        const from = this.store.scene.place.window_open;
        if (Math.abs(from - wp.windowOpen) > 0.01) D.windowTween = { from, to: wp.windowOpen, start: D.t, dur: 3, shelter: wp.shelter };
        else if (wp.shelter) this.store.set('place.shelter', wp.shelter, { source: 'demo', history: false });
      } else if (wp.shelter) {
        this.store.set('place.shelter', wp.shelter, { source: 'demo', history: false });
      }
    }
    if (D.windowTween) {
      const tw = D.windowTween;
      const k = clamp((D.t - tw.start) / tw.dur, 0, 1);
      const v = tw.from + (tw.to - tw.from) * k;
      this.store.set('place.window_open', v, { source: 'demo', history: false });
      for (const o of this.store.scene.place.objects) {
        if (o.type === 'window') this.store.updateObject(o.id, { params: { open: v } }, { history: false, source: 'demo' });
      }
      if (k >= 1) {
        if (tw.shelter) this.store.set('place.shelter', tw.shelter, { source: 'demo', history: false });
        D.windowTween = null;
      }
    }
    this._pushListenerFromCamera(D.idx !== D.lastPushedIdx);
    D.lastPushedIdx = D.idx;
    if (D.t >= wps[n - 1].t) {
      this.demo = null;
      this.mode = 'walk';
      this.setCameraMode('director');
      this.onDemoEnd();
    }
  }

  /* ------------------------------------------------------------------ walk */
  _updateWalk(dt) {
    if (this.mode !== 'walk') return;
    const run = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    const sp = (run ? 4.2 : 1.4) * dt;
    let f = 0;
    let r = 0;
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) f += 1;
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) f -= 1;
    if (this.keys.has('KeyD')) r += 1;
    if (this.keys.has('KeyA')) r -= 1;
    if (this.keys.has('ArrowLeft')) this.camera.rotation.y += 1.6 * dt;
    if (this.keys.has('ArrowRight')) this.camera.rotation.y -= 1.6 * dt;
    if (f) this.walk.moveForward(f * sp);
    if (r) this.walk.moveRight(r * sp);
    this.camera.position.y = this.store.scene.listener.position[2] + this.store.scene.listener.height_m;
    this._pushListenerFromCamera(false);
  }

  _pushListenerFromCamera(force) {
    const fwd = new THREE.Vector3();
    this.camera.getWorldDirection(fwd);
    const heading = ((Math.atan2(fwd.x, -fwd.z) / DEG) + 360) % 360;
    const pitch = Math.asin(clamp(fwd.y, -1, 1)) / DEG;
    const z = this.store.scene.listener.position[2];
    const pos = [round(this.camera.position.x, 2), round(-this.camera.position.z, 2), z];
    const sig = `${pos[0]},${pos[1]},${Math.round(heading)}`;
    if (sig !== this._lastLiveSig) {
      this._lastLiveSig = sig;
      this.onListenerLive({ position: pos, headingDeg: heading, pitchDeg: pitch });
    }
    const now = performance.now();
    if (force || now - this._lastListenerPush > 200) {
      this._lastListenerPush = now;
      this.store.setMany({ 'listener.position': pos, 'listener.heading_deg': round(heading, 1) }, { source: 'walk', history: false });
    }
    if (this.mode === 'walk' && this.autoShelter && now - this._lastShelterCheck > 250) {
      this._lastShelterCheck = now;
      const sh = detectShelter(pos[0], pos[1], this.store.scene);
      if (sh !== this.store.scene.place.shelter) {
        this.store.set('place.shelter', sh, { source: 'walk', history: false });
        this.onCaption(`Shelter: ${sh.replace('_', ', window ').replace('room, window open', 'indoors, window open').replace('room, window closed', 'indoors, window shut')}`);
      }
    }
  }

  /* ------------------------------------------------------------------ pointer */
  _ndc(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  _groundAt(e) {
    this.raycaster.setFromCamera(this._ndc(e), this.camera);
    const p = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(this.groundPlane, p) ? p : null;
  }

  _pick(e) {
    this.raycaster.setFromCamera(this._ndc(e), this.camera);
    const targets = [this.listener.pick, this.objectsGroup, this.stormGroup];
    const hits = this.raycaster.intersectObjects(targets, true);
    for (const h of hits) {
      let o = h.object;
      while (o) {
        if (o.userData.kind) return { kind: o.userData.kind, object: o, point: h.point };
        if (o.userData.objectId) return { kind: 'object', id: o.userData.objectId, point: h.point, object: o };
        o = o.parent;
      }
    }
    return null;
  }

  _bindPointer() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener('pointerdown', (e) => {
      el.focus({ preventScroll: true });
      if (this.mode === 'demo') return;
      if (this.mode === 'walk') {
        if (!this.walk.isLocked && e.pointerType === 'mouse') {
          try { this.walk.lock(); } catch (_) { /* pointer lock unavailable: drag to look */ }
        }
        down = { x: e.clientX, y: e.clientY, look: true, yaw: this.camera.rotation.y, pitch: this.camera.rotation.x };
        return;
      }
      if (this.transform.axis) return;
      const hit = this._pick(e);
      down = { x: e.clientX, y: e.clientY, hit, drag: null };
      if (e.button !== 0 || !hit) return;
      if (hit.kind === 'track-handle') {
        const { stormId, index } = hit.object.userData;
        if (e.altKey) {
          const o = this.store.getObject(stormId);
          if (o && o.params.track.length > 2) {
            const track = o.params.track.filter((_, i) => i !== index);
            this.store.updateObject(stormId, { params: { track } });
          }
          down = null;
          return;
        }
        down.drag = { kind: 'track', stormId, index };
      } else if (hit.kind === 'listener') {
        down.drag = { kind: 'listener' };
      } else if (hit.kind === 'thunder-marker') {
        down.drag = { kind: 'thunder' };
      }
      if (down.drag) {
        this.orbit.enabled = false;
        this.store.beginGesture(down.drag.kind);
        el.setPointerCapture(e.pointerId);
      }
    });
    el.addEventListener('pointermove', (e) => {
      if (!down) return;
      if (down.look && !this.walk.isLocked && this.mode === 'walk' && (e.buttons & 1)) {
        this.camera.rotation.y = down.yaw - (e.clientX - down.x) * 0.004;
        this.camera.rotation.x = clamp(down.pitch - (e.clientY - down.y) * 0.004, -1.4, 1.4);
        return;
      }
      if (!down.drag) return;
      const g = this._groundAt(e);
      if (!g) return;
      const dr = down.drag;
      if (dr.kind === 'track') {
        const o = this.store.getObject(dr.stormId);
        if (!o) return;
        const w = this.fromDisplay(g, true);
        const track = o.params.track.map((p, i) => (i === dr.index ? [Math.round(w[0]), Math.round(w[1])] : p));
        this.store.updateObject(dr.stormId, { params: { track }, position: [track[0][0], track[0][1], o.position[2]] });
      } else if (dr.kind === 'listener') {
        const w = this.fromDisplay(g);
        const z = this.store.scene.listener.position[2];
        const pos = [round(w[0]), round(w[1]), z];
        this.store.set('listener.position', pos, { coalesce: 'listener-drag' });
        this.onListenerLive({ position: pos, headingDeg: this.store.scene.listener.heading_deg, pitchDeg: 0 });
        if (this.autoShelter) {
          const sh = detectShelter(pos[0], pos[1], this.store.scene);
          if (sh !== this.store.scene.place.shelter) this.store.set('place.shelter', sh);
        }
      } else if (dr.kind === 'thunder') {
        const w = this.fromDisplay(g, true);
        const rb = rangeBearing(this.store.scene, w[0], w[1]);
        this.store.setMany({
          'weather.thunder.distance_km': round(clamp(rb.km, 0.3, 25), 2),
          'weather.thunder.bearing_deg': round(rb.bearing, 0),
        }, { coalesce: 'thunder-drag' });
      }
    });
    el.addEventListener('pointerup', (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      if (down.drag) {
        this.store.endGesture();
        this.orbit.enabled = this.mode === 'director' || this.mode === 'top';
        try { el.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
      } else if (!down.look && moved < 5 && e.button === 0 && !this.transform.dragging) {
        const hit = down.hit;
        if (hit && hit.kind === 'object') this.store.select(hit.id);
        else if (hit && hit.kind === 'track-handle') this.store.select(hit.object.userData.stormId);
        else if (!hit) this.store.select(null);
      }
      down = null;
    });
    el.addEventListener('dblclick', (e) => {
      const sel = this.store.selection;
      const o = sel && this.store.getObject(sel);
      if (!o || o.type !== 'storm') return;
      const g = this._groundAt(e);
      if (!g) return;
      const w = this.fromDisplay(g, true);
      this.store.updateObject(sel, { params: { track: [...o.params.track, [Math.round(w[0]), Math.round(w[1])]] } });
    });
  }

  _bindDrop() {
    const el = this.container;
    el.addEventListener('dragover', (e) => {
      if ([...e.dataTransfer.types].includes('application/x-raingen-template')) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
        el.classList.add('drop-target');
      }
    });
    el.addEventListener('dragleave', () => el.classList.remove('drop-target'));
    el.addEventListener('drop', (e) => {
      el.classList.remove('drop-target');
      const key = e.dataTransfer.getData('application/x-raingen-template');
      if (!key) return;
      e.preventDefault();
      const g = this._groundAt(e);
      if (!g) return;
      const isStorm = key === 'storm';
      let w = this.fromDisplay(g, isStorm);
      if (!isStorm) {
        const hit = this._pick(e);
        if (hit && hit.kind === 'object' && ['tin-roof', 'glass', 'puddle', 'leaves', 'concrete', 'gravel', 'grass-patch'].includes(key)) {
          w = [w[0], w[1], Math.max(0, hit.point.y)];
        } else w = [w[0], w[1], 0];
      }
      this.addTemplateAt(key, [round(w[0]), round(w[1]), round(w[2] || 0)]);
    });
  }

  addTemplateAt(key, pos) {
    const obj = instantiate(key, pos, this.store.scene);
    if (!obj) return null;
    return this.store.addObject(obj);
  }

  /** Place a library template 4 m in front of the listener (keyboard path); storms 6 km upwind. */
  placeInFront(key) {
    const s = this.store.scene;
    const L = s.listener.position;
    if (key === 'storm') {
      const b = s.weather.wind.from_deg * DEG;
      return this.addTemplateAt(key, [Math.round(L[0] + Math.sin(b) * 6000), Math.round(L[1] + Math.cos(b) * 6000), 0]);
    }
    const h = (s.listener.heading_deg || 0) * DEG;
    return this.addTemplateAt(key, [round(L[0] + Math.sin(h) * 4), round(L[1] + Math.cos(h) * 4), 0]);
  }

  _bindKeys() {
    this._keyDown = (e) => {
      if (this.mode !== 'walk') return;
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return;
      if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ShiftLeft', 'ShiftRight'].includes(e.code)) {
        this.keys.add(e.code);
        e.preventDefault();
        e.stopPropagation();
      }
    };
    this._keyUp = (e) => this.keys.delete(e.code);
    window.addEventListener('keydown', this._keyDown, true);
    window.addEventListener('keyup', this._keyUp, true);
    window.addEventListener('blur', () => this.keys.clear());
  }

  /** Arrow-key nudging from the main shortcut handler (Director/Top). */
  nudge(dx, dy, rotate = 0) {
    const id = this.store.selection;
    if (id) {
      const o = this.store.getObject(id);
      if (!o) return;
      if (o.type === 'storm') {
        const track = o.params.track.map(([x, y]) => [x + dx * 100, y + dy * 100]);
        this.store.updateObject(id, { params: { track } }, { coalesce: `nudge:${id}` });
      } else {
        this.store.updateObject(id, {
          position: [round(o.position[0] + dx), round(o.position[1] + dy), o.position[2]],
          rotation_deg: ((o.rotation_deg + rotate) % 360 + 360) % 360,
        }, { coalesce: `nudge:${id}` });
      }
      return;
    }
    const L = this.store.scene.listener;
    const pos = [round(L.position[0] + dx), round(L.position[1] + dy), L.position[2]];
    this.store.setMany({ 'listener.position': pos, 'listener.heading_deg': ((L.heading_deg + rotate) % 360 + 360) % 360 }, { coalesce: 'nudge:listener' });
    this.onListenerLive({ position: pos, headingDeg: this.store.scene.listener.heading_deg, pitchDeg: 0 });
  }

  /* ------------------------------------------------------------------ engine events */
  onFlash(evt) {
    const v = this.fx.flash(evt);
    if (!v) return null;
    const s = this.store.scene;
    const b = evt.bearingDeg ?? s.weather.thunder.bearing_deg;
    let best = null;
    let bestD = Infinity;
    for (const [id, node] of this.stormNodes) {
      if (evt.stormId && evt.stormId === id) {
        best = node;
        break;
      }
      const o = this.store.getObject(id);
      const p = o ? stormPosition(o, this.timelineTime) : null;
      const rb = p ? rangeBearing(s, p[0], p[1]) : { bearing: s.weather.thunder.bearing_deg };
      const d = Math.abs(((rb.bearing - b + 540) % 360) - 180);
      if (d < bestD) {
        bestD = d;
        best = node;
      }
    }
    if (best) best.flash = v.mode === 'flash' ? 1.2 : 0.5;
    return v;
  }

  onThunder() {
    this.pulse = 1;
  }

  onGust(evt) {
    this.fx.gust(evt);
  }

  onImpactStats({ objectId, ratePerSec }) {
    this.liveRates.set(objectId, ratePerSec);
    this.fx.setImpactRate(objectId, ratePerSec);
  }

  setTime(t) {
    this.timelineTime = Math.max(0, t || 0);
  }

  /* ------------------------------------------------------------------ layers & modes */
  setLayer(name, on) {
    this.layers[name] = on;
    if (name === 'grid') this.grid.visible = on;
    if (name === 'footprint') this._syncListener();
    if (name === 'wind' || name === 'wavefront') this.fx.setLayerVisible(name, on);
    if (name === 'rays') this.raysGroup.visible = on;
    if (name === 'heatmap') for (const n of this.nodes.values()) n.heat.visible = on;
    if (name === 'labels') {
      for (const [id, n] of this.nodes) n.label.visible = on || id === this.store.selection;
      this._syncListener();
    }
  }

  setVisualsMode(mode) {
    this.visualsMode = mode;
    this.fx.setMode(mode);
    if (mode === 'off') {
      this.renderer.setAnimationLoop(null);
      this.container.classList.add('visuals-off');
    } else {
      this.container.classList.remove('visuals-off');
      this.clock.getDelta();
      this.renderer.setAnimationLoop(() => this._frame());
    }
  }

  setReducedMotion(on) {
    this.fx.setReducedMotion(on);
  }

  resize() {
    const r = this.container.getBoundingClientRect();
    const w = Math.max(1, Math.floor(r.width));
    const h = Math.max(1, Math.floor(r.height));
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = `${w}px`;
    this.renderer.domElement.style.height = `${h}px`;
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /* ------------------------------------------------------------------ frame */
  _windState() {
    const w = this.store.scene.weather.wind;
    const to = (w.from_deg + 180) * DEG;
    const dx = Math.sin(to);
    const dz = -Math.cos(to);
    return { speed: w.speed_mps, gust: Math.max(w.speed_mps, this.fx.gustSpeed || 0), toYaw: Math.atan2(-dz, dx) };
  }

  _updateHeat(dt) {
    this._heatTimer -= dt;
    if (this._heatTimer > 0 || !this.layers.heatmap) return;
    this._heatTimer = 0.4;
    for (const [id, n] of this.nodes) {
      const o = this.store.getObject(id);
      if (!o) continue;
      const r = this.liveRates.has(id) ? this.liveRates.get(id) : estimateImpactRate(o, this.store.scene);
      const x = clamp(Math.log10(1 + r) / 4.5, 0, 1);
      n.heat.material.color.copy(heatColor(x));
      n.heat.material.opacity = 0.15 + 0.45 * x;
    }
  }

  _updateRays(dt) {
    this._raysTimer -= dt;
    if (this._raysTimer > 0 || !this.layers.rays) return;
    this._raysTimer = 0.3;
    this.raysGroup.children.forEach((c) => {
      c.geometry.dispose();
      c.material.dispose();
    });
    this.raysGroup.clear();
    const ear = this.listenerEarDisplay();
    const add = (p, dist) => {
      const color = dist < 5 ? OKABE_ITO.green : dist < 25 ? OKABE_ITO.sky : dist < 1000 ? OKABE_ITO.yellow : OKABE_ITO.vermillion;
      const geo = new THREE.BufferGeometry().setFromPoints([p, ear]);
      this.raysGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 })));
    };
    for (const [id, n] of this.nodes) {
      const p = n.group.position.clone();
      p.y += n.top;
      const o = this.store.getObject(id);
      const L = this.store.scene.listener.position;
      add(p, o ? Math.hypot(o.position[0] - L[0], o.position[1] - L[1]) : 10);
    }
    for (const n of this.stormNodes.values()) add(n.marker.position.clone().setY(2), 5000);
  }

  _frame() {
    const dt = Math.min(0.1, this.clock.getDelta());
    const t = this.clock.elapsedTime;
    if (this._fly) {
      const f = this._fly;
      f.t = Math.min(1, f.t + dt * 2.2);
      const k = f.t * f.t * (3 - 2 * f.t);
      this.camera.position.lerpVectors(f.from, f.to, k);
      this.orbit.target.lerpVectors(f.tFrom, f.tTo, k);
      if (f.t >= 1) this._fly = null;
    }
    if (this.mode === 'demo') this._updateDemo(dt);
    else if (this.mode === 'walk') this._updateWalk(dt);
    else this.orbit.update();

    const wind = this._windState();
    for (const n of this.nodes.values()) if (n.anim) n.anim(t, wind);
    this._updateStorms(dt);
    this.fx.update(dt);
    this._updateHeat(dt);
    this._updateRays(dt);

    if (this.pulse > 0.001) {
      this.pulse *= Math.exp(-dt * 2.5);
      const L = this.listener;
      const s = 1 + 0.25 * this.pulse;
      const h = Math.max(0.2, this.store.scene.listener.height_m);
      L.ring50.scale.setScalar(h * s);
      L.ring90.scale.setScalar(3 * h * s);
      L.ringMat50.opacity = 0.75 + 0.25 * this.pulse;
      L.ringMat90.opacity = 0.55 + 0.45 * this.pulse;
    }
    if (this.selHelper) this.selHelper.update();
    this.renderer.render(this.scene, this.camera);
    this.labelRenderer.render(this.scene, this.camera);
  }

  /** Small PNG snapshot of the current view (for share cards). */
  snapshot() {
    this.renderer.render(this.scene, this.camera);
    return this.renderer.domElement.toDataURL('image/png');
  }

  dispose() {
    this.renderer.setAnimationLoop(null);
    this._resizeObserver.disconnect();
    window.removeEventListener('keydown', this._keyDown, true);
    window.removeEventListener('keyup', this._keyUp, true);
    this.fx.dispose();
    this.transform.dispose();
    this.orbit.dispose();
    this.walk.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.labelRenderer.domElement.remove();
  }
}

function round(v, digits = 2) {
  const k = 10 ** digits;
  return Math.round(v * k) / k;
}

export async function createViewport(container, opts) {
  return new Viewport(container, opts);
}
