/**
 * Weather visual effects for the 4D viewport (Three.js r160).
 *
 * Physics shown, not decorated:
 * - Rain streaks: count ~ rate^0.5, fall speed from Atlas et al. (1973)
 *   v = 9.65 - 10.3 e^(-0.6 D) at D = Dm of the chosen drop character, slant from
 *   wind at ear height via the log law u(z) = u10 ln(z/z0)/ln(10/z0).
 * - Hail fall speed ~ 4.5 sqrt(D_mm) m/s (NOAA-order values), snow ~ 1 m/s.
 * - Lightning bolt drawn from the engine's channel geometry (same geometry the
 *   thunder synthesis uses), gated by a WCAG 2.3.1 flash limiter.
 * - Thunder wavefront: a sphere expanding at c(T) = 331.3 sqrt(1 + T/273.15) m/s,
 *   reaching the listener exactly when the engine delivers the sound.
 * - Fog from rain (Koschmieder-style visibility), sky from sun elevation and cloud.
 */
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { FlashLimiter, clamp, fmtNum } from './a11y.js';

const MAX_DROPS = 26000;
const MAX_SPLASH = 2400;
const MAX_RIPPLES = 480;
const N_ARROWS = 140;
const RAIN_BOX = 44;

function mulberry32(a) {
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const atlasFall = (dmm) => Math.max(0.5, 9.65 - 10.3 * Math.exp(-0.6 * dmm));
const smooth = (e0, e1, x) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

/* ------------------------------------------------------------------ shaders */

const SKY_VS = /* glsl */ `
varying vec3 vDir;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vDir = wp.xyz - cameraPosition;
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

const SKY_FS = /* glsl */ `
uniform vec3 uTop;
uniform vec3 uHorizon;
uniform vec3 uGround;
uniform vec3 uSunDir;
uniform vec3 uSunColor;
uniform vec3 uFlashDir;
uniform vec3 uFlashColor;
uniform float uFlash;
uniform float uOvercast;
uniform float uTime;
varying vec3 vDir;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p); vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0; float a = 0.5;
  for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; }
  return v;
}
void main() {
  vec3 d = normalize(vDir);
  float h = d.y;
  vec3 col = h > 0.0 ? mix(uHorizon, uTop, pow(clamp(h, 0.0, 1.0), 0.55)) : mix(uHorizon, uGround, clamp(-h * 5.0, 0.0, 1.0));
  float sun = max(dot(d, uSunDir), 0.0);
  col += uSunColor * (pow(sun, 256.0) * 4.0 * (1.0 - uOvercast) + pow(sun, 6.0) * 0.35 * (1.0 - 0.6 * uOvercast));
  if (h > 0.0) {
    vec2 uv = d.xz / max(h, 0.06) * 0.9 + vec2(uTime * 0.004, uTime * 0.002);
    float c = fbm(uv);
    float band = smoothstep(0.0, 0.25, h);
    col = mix(col, col * (0.62 + 0.7 * c), uOvercast * band);
  }
  float f = uFlash * (0.25 + 0.75 * pow(max(dot(d, uFlashDir), 0.0), 2.5));
  col += uFlashColor * f;
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}`;

const RAIN_VS = /* glsl */ `
attribute vec3 aSeed;
uniform float uTime;
uniform float uBox;
uniform float uLen;
uniform float uFall;
uniform float uSnow;
uniform float uOpacity;
uniform vec3 uCenter;
uniform vec3 uWind;
varying float vAlpha;
void main() {
  float tail = position.y;
  float sp = 0.8 + 0.4 * aSeed.y;
  vec3 vel = vec3(uWind.x * (0.85 + 0.3 * aSeed.z), -uFall * sp, uWind.z * (0.85 + 0.3 * aSeed.x));
  vec3 p = aSeed * uBox + vel * uTime;
  p.x += uSnow * sin(uTime * 1.3 + aSeed.z * 40.0) * 0.5;
  p.z += uSnow * cos(uTime * 1.1 + aSeed.x * 40.0) * 0.5;
  vec3 lo = uCenter - vec3(0.5 * uBox);
  p = lo + mod(p - lo, uBox);
  vec3 dir = normalize(vel);
  p -= dir * uLen * tail * (0.7 + 0.6 * aSeed.x);
  float fadeGround = smoothstep(0.0, 0.3, p.y);
  float dist = length(p.xz - uCenter.xz);
  float fadeEdge = 1.0 - smoothstep(uBox * 0.3, uBox * 0.5, dist);
  vAlpha = uOpacity * (1.0 - 0.8 * tail) * fadeGround * fadeEdge;
  gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
}`;

const RAIN_FS = /* glsl */ `
uniform vec3 uColor;
varying float vAlpha;
void main() {
  if (vAlpha < 0.004) discard;
  gl_FragColor = vec4(uColor, vAlpha);
  #include <colorspace_fragment>
}`;

const SPLASH_VS = /* glsl */ `
attribute float aBirth;
attribute float aSize;
uniform float uNow;
uniform float uScale;
uniform float uLife;
varying float vA;
void main() {
  float age = uNow - aBirth;
  float k = clamp(age / uLife, 0.0, 1.0);
  vA = (age < 0.0 || age > uLife) ? 0.0 : (1.0 - k);
  vec4 mv = modelViewMatrix * vec4(position + vec3(0.0, k * 0.06, 0.0), 1.0);
  gl_PointSize = aSize * (0.35 + k) * uScale / max(0.2, -mv.z);
  gl_Position = projectionMatrix * mv;
}`;

const SPLASH_FS = /* glsl */ `
uniform vec3 uColor;
varying float vA;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c) * 2.0;
  float a = smoothstep(1.0, 0.75, r) * (0.3 + 0.7 * smoothstep(0.25, 0.8, r)) * vA;
  if (a < 0.01) discard;
  gl_FragColor = vec4(uColor, a);
  #include <colorspace_fragment>
}`;

const WAVE_VS = /* glsl */ `
varying vec3 vN;
varying vec3 vV;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vN = normalize(mat3(modelMatrix) * normal);
  vV = normalize(cameraPosition - wp.xyz);
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

const WAVE_FS = /* glsl */ `
uniform vec3 uColor;
uniform float uOpacity;
varying vec3 vN;
varying vec3 vV;
void main() {
  float rim = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
  float a = uOpacity * (0.06 + 0.94 * rim);
  gl_FragColor = vec4(uColor, a);
  #include <colorspace_fragment>
}`;

/* ------------------------------------------------------------------ class */

export class WeatherFX {
  /**
   * ctx: {
   *   scene: THREE.Scene, renderer, ground: THREE.Mesh,
   *   getFocus(): THREE.Vector3            display-space point the rain box follows
   *   getListener(): THREE.Vector3         display-space ear position
   *   channelToDisplay(points[[x,y,z]]): THREE.Vector3[]
   *   strikeToDisplay(distanceKm, bearingDeg): THREE.Vector3   ground point of a strike
   *   footprints(): Array<{id, center: THREE.Vector3, w, d, top, water}>
   *   estimateRate(id): number              impacts/s when the engine has no stats
   * }
   */
  constructor(ctx) {
    this.ctx = ctx;
    this.scene = ctx.scene;
    this.mode = 'full';
    this.reducedMotion = false;
    this.limiter = new FlashLimiter({ maxPerSec: 3 });
    this.rand = mulberry32(0x5eed);
    this.rates = new Map();
    this.time = 0;
    this.flashLevel = 0;
    this.glowLevel = 0;
    this.flashDir = new THREE.Vector3(0, 1, 0);
    this.wetness = 0;
    this.snowCover = 0;
    this.weather = null;
    this.windVec = new THREE.Vector3();
    this.windSpeed = 0;
    this.gustSpeed = 0;
    this.gustPulse = 0;
    this.bolts = [];
    this.waves = [];
    this.seenFlashes = new Set();
    this.onWaveArrive = null;

    this._makeLights();
    this._makeSky();
    this._makeRain();
    this._makeSplashes();
    this._makeRipples();
    this._makeWind();
    this.scene.fog = new THREE.FogExp2(0x10141d, 0.004);

    this.groundDry = new THREE.Color('#2f3a2c');
    this.groundWet = new THREE.Color('#161c17');
    this.groundSnow = new THREE.Color('#dfe6ee');
  }

  /* ---------------- construction */
  _makeLights() {
    this.hemi = new THREE.HemisphereLight(0x8090a0, 0x202418, 0.6);
    this.sun = new THREE.DirectionalLight(0xffffff, 1);
    this.sun.position.set(30, 60, 20);
    this.sun.castShadow = false;
    this.flashLight = new THREE.DirectionalLight(0xcfdcff, 0);
    this.flashLight.position.set(0, 100, 0);
    this.scene.add(this.hemi, this.sun, this.sun.target, this.flashLight, this.flashLight.target);
  }

  _makeSky() {
    this.skyUniforms = {
      uTop: { value: new THREE.Color('#0b1220') },
      uHorizon: { value: new THREE.Color('#1c2533') },
      uGround: { value: new THREE.Color('#0b0e12') },
      uSunDir: { value: new THREE.Vector3(0, 1, 0) },
      uSunColor: { value: new THREE.Color('#fff2d6') },
      uFlashDir: { value: new THREE.Vector3(0, 1, 0) },
      uFlashColor: { value: new THREE.Color('#c9d6ff') },
      uFlash: { value: 0 },
      uOvercast: { value: 0.8 },
      uTime: { value: 0 },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.skyUniforms,
      vertexShader: SKY_VS,
      fragmentShader: SKY_FS,
      side: THREE.BackSide,
      depthWrite: false,
      fog: false,
      toneMapped: false,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(2400, 48, 24), mat);
    this.sky.renderOrder = -10;
    this.sky.frustumCulled = false;
    this.scene.add(this.sky);
  }

  _makeRain() {
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 1, 0], 3));
    const seeds = new Float32Array(MAX_DROPS * 3);
    for (let i = 0; i < seeds.length; i++) seeds[i] = this.rand();
    geo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seeds, 3));
    geo.instanceCount = 0;
    this.rainUniforms = {
      uTime: { value: 0 },
      uBox: { value: RAIN_BOX },
      uLen: { value: 0.25 },
      uFall: { value: 6 },
      uSnow: { value: 0 },
      uOpacity: { value: 0.45 },
      uCenter: { value: new THREE.Vector3(0, RAIN_BOX / 2, 0) },
      uWind: { value: new THREE.Vector3() },
      uColor: { value: new THREE.Color('#b8c8da') },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.rainUniforms,
      vertexShader: RAIN_VS,
      fragmentShader: RAIN_FS,
      transparent: true,
      depthWrite: false,
      fog: false,
    });
    this.rain = new THREE.LineSegments(geo, mat);
    this.rain.frustumCulled = false;
    this.rain.renderOrder = 5;
    this.scene.add(this.rain);
  }

  _makeSplashes() {
    const geo = new THREE.BufferGeometry();
    this.splashPos = new Float32Array(MAX_SPLASH * 3);
    this.splashBirth = new Float32Array(MAX_SPLASH).fill(-100);
    this.splashSize = new Float32Array(MAX_SPLASH).fill(1);
    geo.setAttribute('position', new THREE.BufferAttribute(this.splashPos, 3));
    geo.setAttribute('aBirth', new THREE.BufferAttribute(this.splashBirth, 1));
    geo.setAttribute('aSize', new THREE.BufferAttribute(this.splashSize, 1));
    this.splashUniforms = {
      uNow: { value: 0 },
      uScale: { value: 400 },
      uLife: { value: 0.32 },
      uColor: { value: new THREE.Color('#d6e2ee') },
    };
    const mat = new THREE.ShaderMaterial({
      uniforms: this.splashUniforms,
      vertexShader: SPLASH_VS,
      fragmentShader: SPLASH_FS,
      transparent: true,
      depthWrite: false,
      fog: false,
    });
    this.splashes = new THREE.Points(geo, mat);
    this.splashes.frustumCulled = false;
    this.splashes.renderOrder = 6;
    this.splashIdx = 0;
    this.splashAcc = new Map();
    this.scene.add(this.splashes);
  }

  _makeRipples() {
    const ring = new THREE.RingGeometry(0.82, 1, 28);
    ring.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false });
    this.ripples = new THREE.InstancedMesh(ring, mat, MAX_RIPPLES);
    this.ripples.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.ripples.frustumCulled = false;
    this.rippleData = new Array(MAX_RIPPLES).fill(null).map(() => ({ birth: -100, x: 0, y: 0, z: 0, size: 0.2 }));
    this.rippleIdx = 0;
    this.rippleColor = new THREE.Color('#7fa8c4');
    const zero = new THREE.Matrix4().makeScale(0, 0, 0);
    const black = new THREE.Color(0, 0, 0);
    for (let i = 0; i < MAX_RIPPLES; i++) {
      this.ripples.setMatrixAt(i, zero);
      this.ripples.setColorAt(i, black);
    }
    this.scene.add(this.ripples);
  }

  _makeWind() {
    const geo = new THREE.ConeGeometry(0.05, 1.5, 5, 1, true);
    geo.rotateZ(-Math.PI / 2);
    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#56B4E9'), transparent: true, opacity: 0.3, depthWrite: false, fog: true });
    this.arrows = new THREE.InstancedMesh(geo, mat, N_ARROWS);
    this.arrows.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.arrows.frustumCulled = false;
    this.arrowPos = new Float32Array(N_ARROWS * 3);
    for (let i = 0; i < N_ARROWS; i++) {
      this.arrowPos[i * 3] = (this.rand() - 0.5) * 70;
      this.arrowPos[i * 3 + 1] = 0.6 + this.rand() * 11;
      this.arrowPos[i * 3 + 2] = (this.rand() - 0.5) * 70;
    }
    this.scene.add(this.arrows);
    this._m4 = new THREE.Matrix4();
    this._q = new THREE.Quaternion();
    this._v = new THREE.Vector3();
    this._s = new THREE.Vector3();
  }

  /* ---------------- inputs */
  setWeather(s) {
    this.weather = s;
    const p = s.weather.precipitation;
    const type = p.type;
    const rate = type === 'none' ? 0 : p.rate_mm_h;
    const dm = p.dm_mm ?? { fine: 1.0, normal: 1.6, fat: 2.4 }[p.drops] ?? 1.6;
    let fall = atlasFall(type === 'drizzle' ? Math.min(dm, 0.6) : dm);
    let len = fall / 26;
    let snow = 0;
    let n = 0;
    let opacity = 0.42;
    const color = new THREE.Color('#b8c8da');
    if (type === 'snow') {
      fall = 1.0; len = 0.05; snow = 1; n = 2600 * Math.sqrt(rate); opacity = 0.85; color.set('#f1f5fa');
    } else if (type === 'graupel') {
      fall = 2.5; len = 0.06; n = 1600 * Math.sqrt(rate); opacity = 0.8; color.set('#eef2f6');
    } else if (type === 'sleet' || type === 'freezing_rain') {
      fall = type === 'sleet' ? 4.5 : fall; len = 0.12; n = 1100 * Math.sqrt(rate); opacity = 0.6; color.set('#dfe8f1');
    } else if (type === 'hail') {
      fall = 4.5 * Math.sqrt(p.hail_size_mm || 13); len = 0.25; n = 320 * Math.sqrt(rate); opacity = 0.95; color.set('#f7fbff');
    } else if (type === 'drizzle') {
      n = 2400 * Math.pow(rate, 0.35); opacity = 0.18; len = Math.max(0.05, len * 0.6);
    } else if (rate > 0) {
      n = 900 * Math.sqrt(rate); opacity = 0.32 + 0.18 * clamp(Math.log10(rate + 1) / 2, 0, 1);
    }
    this.baseDropCount = Math.min(MAX_DROPS, Math.round(n));
    this.rainUniforms.uFall.value = fall;
    this.rainUniforms.uLen.value = len;
    this.rainUniforms.uSnow.value = snow;
    this.baseOpacity = opacity;
    this.rainUniforms.uColor.value.copy(color);
    this.splashUniforms.uLife.value = type === 'snow' ? 0.0001 : type === 'hail' ? 0.18 : 0.32;

    const w = s.weather.wind;
    const to = ((w.from_deg + 180) * Math.PI) / 180;
    const earFactor = Math.log(Math.max(0.5, s.listener.height_m) / 0.05) / Math.log(10 / 0.05);
    this.windSpeed = w.speed_mps;
    if (!this.gustSpeed) this.gustSpeed = w.speed_mps;
    const u = w.speed_mps * clamp(earFactor, 0.35, 1.2) * 0.9;
    this.windVec.set(Math.sin(to), 0, -Math.cos(to));
    this.rainUniforms.uWind.value.set(this.windVec.x * u, 0, this.windVec.z * u);
    this.rainRate = rate;
    this.precipType = type;
    this._updateSky();
    this._applyMode();
  }

  setMode(mode) {
    this.mode = mode;
    this._applyMode();
  }

  setReducedMotion(on) {
    this.reducedMotion = !!on;
    this._applyMode();
  }

  _applyMode() {
    const calm = this.mode === 'calm' || this.reducedMotion;
    const scale = this.mode === 'off' ? 0 : calm ? 0.4 : 1;
    this.rain.geometry.instanceCount = Math.round((this.baseDropCount || 0) * scale);
    this.rainUniforms.uOpacity.value = (this.baseOpacity || 0.4) * (calm ? 0.75 : 1);
    this.rain.visible = scale > 0 && this.rain.geometry.instanceCount > 0;
    this.arrows.visible = this.mode !== 'off' && this.showWind !== false;
    this.splashes.visible = this.mode !== 'off';
    this.ripples.visible = this.mode !== 'off';
  }

  setImpactRate(objectId, rate) {
    this.rates.set(objectId, Math.max(0, rate || 0));
  }

  gust({ speedMps }) {
    if (!Number.isFinite(speedMps)) return;
    if (speedMps > this.gustSpeed * 1.08) this.gustPulse = Math.min(1, this.gustPulse + 0.6);
    this.gustSpeed = speedMps;
  }

  /* ---------------- sky */
  _updateSky() {
    const s = this.weather;
    if (!s) return;
    const hour = s.weather.hour;
    const maxEl = { summer: 62, spring: 42, autumn: 34, winter: 15 }[s.weather.season] ?? 40;
    const el = Math.sin(((hour - 6) / 12) * Math.PI) * maxEl;
    const azDeg = 90 + ((hour - 6) / 12) * 180;
    const elR = (el * Math.PI) / 180;
    const azR = (azDeg * Math.PI) / 180;
    const sunDir = new THREE.Vector3(Math.sin(azR) * Math.cos(elR), Math.sin(elR), -Math.cos(azR) * Math.cos(elR));
    const day = smooth(-6, 10, el);
    const twi = Math.exp(-Math.pow(el / 7, 2));
    const rate = this.rainRate || 0;
    const thunder = s.weather.thunder.enabled ? 1 : 0;
    const oc = clamp((rate > 0.03 ? 0.55 + 0.25 * clamp(Math.log10(rate * 2 + 1) / 2, 0, 1) : 0.35) + 0.15 * thunder, 0, 1);
    this.overcast = oc;

    const nightTop = new THREE.Color('#05070d');
    const nightHor = new THREE.Color('#0f1420');
    const dayTop = new THREE.Color('#3a6ea8').lerp(new THREE.Color('#58616c'), oc);
    const dayHor = new THREE.Color('#a9c4dc').lerp(new THREE.Color('#8b939c'), oc);
    const dusk = new THREE.Color('#b8754a').lerp(new THREE.Color('#5d5560'), oc * 0.8);
    const top = nightTop.clone().lerp(dayTop, day);
    const hor = nightHor.clone().lerp(dayHor, day).lerp(dusk, twi * 0.55);
    const dark = 1 - 0.35 * thunder * day;
    top.multiplyScalar(dark);
    hor.multiplyScalar(dark);
    this.skyUniforms.uTop.value.copy(top);
    this.skyUniforms.uHorizon.value.copy(hor);
    this.skyUniforms.uGround.value.copy(hor.clone().multiplyScalar(0.35));
    this.skyUniforms.uSunDir.value.copy(sunDir.y > -0.05 ? sunDir : sunDir.clone().negate());
    this.skyUniforms.uSunColor.value.set(day > 0.05 ? '#fff0d0' : '#8fa6c8').multiplyScalar(day > 0.05 ? 1 : 0.25);
    this.skyUniforms.uOvercast.value = oc;

    this.hemi.color.copy(hor).lerp(new THREE.Color('#9fb2c8'), 0.3);
    this.hemi.groundColor.set('#1c2017');
    this.hemi.intensity = (0.25 + 1.25 * day) * (1 - 0.35 * oc) + 0.12;
    const sunVis = sunDir.y > 0 ? sunDir : sunDir.clone().negate().setY(Math.abs(sunDir.y) + 0.3).normalize();
    this.sunDirVec = sunVis;
    this.sun.color.set(day > 0.05 ? '#fff1dc' : '#9db3d6');
    this.sun.intensity = day > 0.05 ? 1.9 * day * (1 - 0.7 * oc) + 0.1 : 0.18 * (1 - 0.6 * oc);

    const fogColor = hor.clone();
    this.scene.fog.color.copy(fogColor);
    const p = s.weather.precipitation;
    const fogBoost = p.type === 'snow' ? 1.8 : p.type === 'drizzle' ? 1.4 : 1;
    this.baseFog = (0.0025 + 0.012 * clamp(Math.log10(rate + 1) / 2.2, 0, 1) * fogBoost) * (s.weather.humidity_pct > 97 ? 1.3 : 1);
    this.scene.fog.density = this.baseFog;
  }

  /* ---------------- lightning */
  /**
   * Show a flash from the engine 'flash' event. Returns the limiter verdict.
   * evt: {id, t, channel: [[x,y,z]...], distanceKm, delaySec, brightness, bearingDeg?}
   */
  flash(evt) {
    if (!evt) return null;
    if (evt.id !== undefined) {
      if (this.seenFlashes.has(evt.id)) return null;
      this.seenFlashes.add(evt.id);
      if (this.seenFlashes.size > 200) this.seenFlashes.delete(this.seenFlashes.values().next().value);
    }
    const now = this.time;
    const verdict = this.limiter.admit(performance.now(), evt.brightness ?? 1, this.reducedMotion || this.mode === 'calm');
    let points = Array.isArray(evt.channel) && evt.channel.length > 1 ? this.ctx.channelToDisplay(evt.channel) : null;
    const strike = points ? this._lowestPoint(points) : this.ctx.strikeToDisplay(evt.distanceKm ?? 8, evt.bearingDeg ?? this.weather?.weather?.thunder?.bearing_deg ?? 270);
    if (!points && this.mode !== 'off') points = this._syntheticChannel(strike, evt.id);

    const listener = this.ctx.getListener();
    this.flashDir.copy(strike).setY(strike.y + 120).sub(listener).normalize();
    this.skyUniforms.uFlashDir.value.copy(this.flashDir);
    this.flashLight.position.copy(strike).setY(300);
    this.flashLight.target.position.copy(listener);

    if (verdict.mode === 'flash') {
      this.flashLevel = Math.max(this.flashLevel, 0.9 * verdict.brightness);
      this._addBolt(points, verdict.brightness);
    } else if (verdict.mode === 'merge') {
      this.flashLevel = Math.max(this.flashLevel, 0.7 * verdict.brightness);
      if (this.bolts.length) this.bolts[this.bolts.length - 1].restrike = now;
    } else {
      this.glowTarget = Math.max(this.glowTarget || 0, 0.35 * verdict.brightness);
      if (!this.reducedMotion && this.mode !== 'calm') this._addBolt(points, 0.25, true);
    }
    if (evt.delaySec > 0 && this.mode !== 'off') this._addWave(strike, evt);
    return verdict;
  }

  _lowestPoint(points) {
    let low = points[0];
    for (const p of points) if (p.y < low.y) low = p;
    return low.clone().setY(0);
  }

  _syntheticChannel(ground, id = 1) {
    const r = mulberry32(typeof id === 'number' ? id : String(id).split('').reduce((a, c) => (a * 31 + c.charCodeAt(0)) | 0, 7));
    const pts = [];
    const top = 140 + r() * 60;
    let x = ground.x + (r() - 0.5) * 40;
    let z = ground.z + (r() - 0.5) * 40;
    const n = 18;
    for (let i = 0; i <= n; i++) {
      const y = top * (1 - i / n);
      x += (r() - 0.5) * 14;
      z += (r() - 0.5) * 14;
      const t = i / n;
      pts.push(new THREE.Vector3(x * (1 - t) + ground.x * t, y, z * (1 - t) + ground.z * t));
    }
    return pts;
  }

  _splitBranches(points) {
    if (points.length < 3) return [points];
    const seg = [];
    for (let i = 1; i < points.length; i++) seg.push(points[i].distanceTo(points[i - 1]));
    const sorted = [...seg].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] || 1;
    const out = [];
    let cur = [points[0]];
    for (let i = 1; i < points.length; i++) {
      if (seg[i - 1] > median * 6) {
        if (cur.length > 1) out.push(cur);
        cur = [points[i]];
      } else cur.push(points[i]);
    }
    if (cur.length > 1) out.push(cur);
    return out.length ? out : [points];
  }

  _addBolt(points, brightness, dim = false) {
    if (!points || points.length < 2) return;
    const group = new THREE.Group();
    const branches = this._splitBranches(points);
    const coreMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#f4f7ff'), transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
    const glowMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#7f9cff'), transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false, fog: false });
    branches.forEach((pts, bi) => {
      const path = new THREE.CurvePath();
      for (let i = 1; i < pts.length; i++) path.add(new THREE.LineCurve3(pts[i - 1], pts[i]));
      const segs = Math.min(400, pts.length * 2);
      const scale = bi === 0 ? 1 : 0.55;
      const span = pts[0].distanceTo(pts[pts.length - 1]);
      const r = Math.max(0.15, span * 0.0035) * scale;
      group.add(new THREE.Mesh(new THREE.TubeGeometry(path, segs, r, 4, false), coreMat));
      group.add(new THREE.Mesh(new THREE.TubeGeometry(path, segs, r * 5, 6, false), glowMat));
    });
    group.renderOrder = 8;
    this.scene.add(group);
    this.bolts.push({ group, born: this.time, brightness: dim ? 0.25 : brightness, coreMat, glowMat, restrike: -1, life: dim ? 0.5 : 0.75 });
    while (this.bolts.length > 6) this._removeBolt(this.bolts[0]);
  }

  _removeBolt(b) {
    this.scene.remove(b.group);
    b.group.traverse((o) => o.geometry && o.geometry.dispose());
    b.coreMat.dispose();
    b.glowMat.dispose();
    this.bolts.splice(this.bolts.indexOf(b), 1);
  }

  _addWave(center, evt) {
    const listener = this.ctx.getListener();
    const dispDist = Math.max(1, Math.hypot(listener.x - center.x, listener.z - center.z));
    const mat = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color('#9fd3ff') }, uOpacity: { value: 0.0 } },
      vertexShader: WAVE_VS,
      fragmentShader: WAVE_FS,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      fog: false,
    });
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), mat);
    sphere.position.copy(center);
    sphere.renderOrder = 7;
    const ringMat = new THREE.MeshBasicMaterial({ color: new THREE.Color('#56B4E9'), transparent: true, opacity: 0.6, depthWrite: false, side: THREE.DoubleSide, fog: false });
    const ringGeo = new THREE.RingGeometry(0.985, 1, 128);
    ringGeo.rotateX(-Math.PI / 2);
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.copy(center).setY(0.06);
    const labelEl = document.createElement('div');
    labelEl.className = 'vp-label wave-label';
    labelEl.setAttribute('aria-hidden', 'true');
    const label = new CSS2DObject(labelEl);
    const vis = this.showWaves !== false;
    sphere.visible = vis;
    ring.visible = vis;
    label.visible = vis;
    this.scene.add(sphere, ring, label);
    this.waves.push({
      sphere, ring, label, labelEl, mat, ringMat, center: center.clone(), dispDist,
      born: this.time, delay: Math.max(0.05, evt.delaySec), km: evt.distanceKm ?? dispDist / 1000, arrived: false, id: evt.id,
    });
    while (this.waves.length > 5) this._removeWave(this.waves[0]);
  }

  _removeWave(w) {
    this.scene.remove(w.sphere, w.ring, w.label);
    w.sphere.geometry.dispose();
    w.ring.geometry.dispose();
    w.mat.dispose();
    w.ringMat.dispose();
    w.labelEl.remove();
    this.waves.splice(this.waves.indexOf(w), 1);
  }

  /* ---------------- per frame */
  update(dt) {
    this.time += dt;
    const t = this.time;
    const focus = this.ctx.getFocus();
    const cam = this.ctx.camera;

    this.sky.position.copy(cam.position);
    this.skyUniforms.uTime.value = t;
    this.rainUniforms.uTime.value = t;
    this.rainUniforms.uCenter.value.set(focus.x, RAIN_BOX * 0.5, focus.z);
    this.splashUniforms.uNow.value = t;
    this.splashUniforms.uScale.value = this.ctx.renderer.domElement.height * 0.5;

    // flash envelope and glow
    const glowTarget = this.glowTarget || 0;
    this.glowLevel += (glowTarget - this.glowLevel) * Math.min(1, dt * 3);
    this.glowTarget = Math.max(0, glowTarget - dt * 0.6);
    this.flashLevel *= Math.exp(-dt * 7);
    const flicker = this.reducedMotion ? 1 : 0.75 + 0.25 * Math.sin(t * 90) * Math.sin(t * 37);
    const f = this.flashLevel * flicker + this.glowLevel;
    this.skyUniforms.uFlash.value = this.mode === 'off' ? 0 : f * 1.4;
    this.flashLight.intensity = this.mode === 'off' ? 0 : f * 6;
    this.hemiBase = this.hemiBase ?? this.hemi.intensity;
    this.scene.fog.density = (this.baseFog || 0.004) * (1 - 0.3 * Math.min(1, f));

    // bolts
    for (const b of [...this.bolts]) {
      const age = t - b.born;
      const sinceRe = b.restrike >= 0 ? t - b.restrike : Infinity;
      let a = Math.max(0, 1 - age / b.life);
      if (sinceRe < 0.2) a = Math.max(a, 0.8 * (1 - sinceRe / 0.2));
      a *= b.brightness;
      if (!this.reducedMotion) a *= 0.7 + 0.3 * Math.sin(age * 70) * Math.sin(age * 23);
      b.coreMat.opacity = a;
      b.glowMat.opacity = a * 0.35;
      if (age > b.life && sinceRe > 0.25) this._removeBolt(b);
    }

    // wavefronts: radius = displayDistance * elapsed / delay (exact arrival at the listener)
    const listener = this.ctx.getListener();
    for (const w of [...this.waves]) {
      const age = t - w.born;
      const prog = age / w.delay;
      const r = Math.max(0.5, w.dispDist * prog);
      w.sphere.scale.setScalar(r);
      w.ring.scale.setScalar(r);
      const fadeIn = clamp(age / 0.4, 0, 1);
      const fadeOut = prog > 1 ? clamp(1 - (prog - 1) / 0.25, 0, 1) : 1;
      const op = (this.reducedMotion ? 0.35 : 0.6) * fadeIn * fadeOut;
      w.mat.uniforms.uOpacity.value = op;
      w.ringMat.opacity = op * 0.9;
      const toL = new THREE.Vector3(listener.x - w.center.x, 0, listener.z - w.center.z).normalize();
      w.label.position.set(w.center.x + toL.x * r, Math.min(r * 0.6, 30) + 2, w.center.z + toL.z * r);
      const remainKm = Math.max(0, w.km * (1 - prog));
      w.labelEl.textContent = prog < 1 ? `sound: ${fmtNum(remainKm, remainKm < 1 ? 2 : 1)} km to go · ${fmtNum(Math.max(0, w.delay - age), 1)} s` : 'arrived';
      w.labelEl.style.opacity = String(op > 0.05 ? 1 : 0);
      if (!w.arrived && prog >= 1) {
        w.arrived = true;
        this.onWaveArrive?.(w);
      }
      if (prog > 1.25) this._removeWave(w);
    }

    // wetness store (visual): fills with rain, drains when dry
    const rate = this.rainRate || 0;
    const liquid = ['rain', 'showers', 'drizzle', 'freezing_rain', 'sleet', 'hail'].includes(this.precipType);
    const fill = liquid ? clamp(Math.log10(rate * 4 + 1) / 2, 0, 1) : 0;
    const target = Math.max(fill, 0);
    const tau = target > this.wetness ? 12 : 90;
    this.wetness += (target - this.wetness) * Math.min(1, dt / tau);
    const snowing = this.precipType === 'snow' || this.precipType === 'graupel';
    this.snowCover += ((snowing ? clamp(rate / 2, 0.3, 1) : 0) - this.snowCover) * Math.min(1, dt / (snowing ? 30 : 120));
    if (this.ctx.ground) {
      const m = this.ctx.ground.material;
      m.color.copy(this.groundDry).lerp(this.groundWet, this.wetness).lerp(this.groundSnow, this.snowCover);
      m.roughness = 0.95 - 0.55 * this.wetness * (1 - this.snowCover);
      m.metalness = 0.08 * this.wetness;
    }

    if (this.mode === 'off') return;
    this._updateSplashes(dt, focus);
    this._updateRipples();
    this._updateWind(dt, focus);
  }

  _spawnSplash(x, y, z, size) {
    const i = this.splashIdx;
    this.splashIdx = (i + 1) % MAX_SPLASH;
    this.splashPos[i * 3] = x;
    this.splashPos[i * 3 + 1] = y;
    this.splashPos[i * 3 + 2] = z;
    this.splashBirth[i] = this.time + this.rand() * 0.02;
    this.splashSize[i] = size;
  }

  _spawnRipple(x, y, z, size) {
    const r = this.rippleData[this.rippleIdx];
    this.rippleIdx = (this.rippleIdx + 1) % MAX_RIPPLES;
    r.birth = this.time;
    r.x = x;
    r.y = y;
    r.z = z;
    r.size = size;
  }

  _updateSplashes(dt, focus) {
    const rate = this.rainRate || 0;
    const type = this.precipType;
    const calm = this.mode === 'calm' || this.reducedMotion;
    const modeScale = calm ? 0.35 : 1;
    if (rate > 0 && type !== 'snow' && type !== 'none') {
      const big = type === 'hail' ? 1.6 : this.weather?.weather?.precipitation?.drops === 'fat' ? 1.3 : 1;
      // ground splashes around the focus (visual density, not physical flux)
      const groundRate = Math.min(260, 30 * Math.sqrt(rate)) * modeScale;
      let acc = (this.splashAcc.get('_ground') || 0) + groundRate * dt;
      while (acc >= 1) {
        acc -= 1;
        const a = this.rand() * Math.PI * 2;
        const rr = Math.sqrt(this.rand()) * 16;
        this._spawnSplash(focus.x + Math.cos(a) * rr, 0.03, focus.z + Math.sin(a) * rr, 2.2 * big);
      }
      this.splashAcc.set('_ground', acc);

      for (const fp of this.ctx.footprints()) {
        const live = this.rates.get(fp.id);
        const r = live !== undefined ? live : this.ctx.estimateRate(fp.id);
        if (!(r > 0)) continue;
        const vis = clamp(6 * Math.pow(Math.log10(1 + r), 1.6), 0, 90) * modeScale;
        let a2 = (this.splashAcc.get(fp.id) || 0) + vis * dt;
        while (a2 >= 1) {
          a2 -= 1;
          const ux = (this.rand() - 0.5) * fp.w;
          const uz = (this.rand() - 0.5) * fp.d;
          const cs = Math.cos(fp.rot || 0);
          const sn = Math.sin(fp.rot || 0);
          const x = fp.center.x + ux * cs - uz * sn;
          const z = fp.center.z + ux * sn + uz * cs;
          if (fp.water) this._spawnRipple(x, fp.top + 0.02, z, fp.small ? 0.12 : 0.25);
          else this._spawnSplash(x, fp.top + 0.02, z, 2.4 * big);
        }
        this.splashAcc.set(fp.id, a2);
      }
    }
    const g = this.splashes.geometry;
    g.attributes.position.needsUpdate = true;
    g.attributes.aBirth.needsUpdate = true;
    g.attributes.aSize.needsUpdate = true;
  }

  _updateRipples() {
    const life = 1.1;
    const c = new THREE.Color();
    for (let i = 0; i < MAX_RIPPLES; i++) {
      const r = this.rippleData[i];
      const age = this.time - r.birth;
      if (age < 0 || age > life) {
        this._m4.makeScale(0, 0, 0);
        this.ripples.setMatrixAt(i, this._m4);
        continue;
      }
      const k = age / life;
      const s = r.size * (0.15 + 1.6 * k);
      this._m4.makeScale(s, 1, s).setPosition(r.x, r.y, r.z);
      this.ripples.setMatrixAt(i, this._m4);
      c.copy(this.rippleColor).multiplyScalar((1 - k) * 0.9);
      this.ripples.setColorAt(i, c);
    }
    this.ripples.instanceMatrix.needsUpdate = true;
    if (this.ripples.instanceColor) this.ripples.instanceColor.needsUpdate = true;
  }

  _updateWind(dt, focus) {
    const mean = this.windSpeed || 0;
    this.gustSpeed += (mean - this.gustSpeed) * Math.min(1, dt * 0.25);
    this.gustPulse *= Math.exp(-dt * 1.5);
    const speed = Math.max(mean, this.gustSpeed);
    const calm = this.mode === 'calm' || this.reducedMotion;
    const visSpeed = speed * (calm ? 0.3 : 0.8);
    const yaw = Math.atan2(-this.windVec.z, this.windVec.x);
    this._q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
    const half = 35;
    const len = 0.6 + Math.min(2.5, speed / 6);
    for (let i = 0; i < N_ARROWS; i++) {
      let x = this.arrowPos[i * 3] + this.windVec.x * visSpeed * dt * (0.8 + 0.4 * ((i * 37) % 10) / 10);
      const y = this.arrowPos[i * 3 + 1];
      let z = this.arrowPos[i * 3 + 2] + this.windVec.z * visSpeed * dt * (0.8 + 0.4 * ((i * 53) % 10) / 10);
      if (x < -half) x += 2 * half;
      if (x > half) x -= 2 * half;
      if (z < -half) z += 2 * half;
      if (z > half) z -= 2 * half;
      this.arrowPos[i * 3] = x;
      this.arrowPos[i * 3 + 2] = z;
      this._v.set(focus.x + x, y, focus.z + z);
      this._s.set(len, 1, 1);
      this._m4.compose(this._v, this._q, this._s);
      this.arrows.setMatrixAt(i, this._m4);
    }
    this.arrows.instanceMatrix.needsUpdate = true;
    const op = clamp((speed - 0.8) / 14, 0.04, 0.5) + 0.3 * this.gustPulse;
    this.arrows.material.opacity = op * (calm ? 0.6 : 1) * (this.showWind === false ? 0 : 1);
  }

  setLayerVisible(name, on) {
    if (name === 'wind') {
      this.showWind = on;
      this.arrows.visible = on && this.mode !== 'off';
    } else if (name === 'rain') {
      this.rain.visible = on;
    } else if (name === 'wavefront') {
      this.showWaves = on;
      for (const w of this.waves) {
        w.sphere.visible = on;
        w.ring.visible = on;
        w.label.visible = on;
      }
    }
  }

  dispose() {
    for (const b of [...this.bolts]) this._removeBolt(b);
    for (const w of [...this.waves]) this._removeWave(w);
    [this.sky, this.rain, this.splashes, this.ripples, this.arrows].forEach((m) => {
      this.scene.remove(m);
      m.geometry.dispose();
      m.material.dispose();
    });
    this.scene.remove(this.hemi, this.sun, this.flashLight);
  }
}
