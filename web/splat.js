// WebGL2 3D Gaussian Splatting viewer for DreamGaussian PLY output.
// Per Gaussian: world covariance S = R diag(s)^2 R^T (precomputed on CPU), projected per frame in the
// vertex shader with the EWA Jacobian (Zwicker et al. / Kerbl et al. 2023), drawn as an instanced quad,
// alpha-composited back-to-front after a depth sort in a Web Worker.

const SH_C0 = 0.28209479177387814;
const COV_SCALE = 1000.0; // covariances are stored as half floats; pre-scale to stay in the normal range
const TEX_W = 2048; // texels per row; 2 texels per Gaussian -> 1024 Gaussians per row

// ------------------------------------------------------------------ PLY -> Gaussian arrays
export async function loadPly(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  return parsePly(await res.arrayBuffer());
}

export function parsePly(buf) {
  const bytes = new Uint8Array(buf);
  const head = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 8192)));
  const end = head.indexOf("end_header\n");
  if (end < 0) throw new Error("not a PLY file");
  const lines = head.slice(0, end).split("\n");
  if (!lines.some((l) => l.startsWith("format binary_little_endian"))) throw new Error("only binary_little_endian PLY supported");
  let count = 0, inVertex = false;
  const props = [];
  const size = { float: 4, float32: 4, double: 8, uchar: 1, uint8: 1, int: 4, uint: 4 };
  for (const l of lines) {
    const t = l.trim().split(/\s+/);
    if (t[0] === "element") { inVertex = t[1] === "vertex"; if (inVertex) count = +t[2]; }
    else if (t[0] === "property" && inVertex) props.push({ type: t[1], name: t[2] });
  }
  let stride = 0;
  const off = {};
  for (const p of props) { off[p.name] = { o: stride, t: p.type }; stride += size[p.type]; }
  const dv = new DataView(buf, end + "end_header\n".length);
  const rd = (i, name) => {
    const p = off[name];
    const at = i * stride + p.o;
    return p.t === "double" ? dv.getFloat64(at, true) : p.t === "uchar" || p.t === "uint8" ? dv.getUint8(at) / 255 : dv.getFloat32(at, true);
  };
  const g = { count, pos: new Float32Array(count * 3), col: new Uint8Array(count * 4), cov: new Float32Array(count * 6),
              scale: new Float32Array(count * 3), nrm: new Float32Array(count * 3) };
  for (let i = 0; i < count; i++) {
    g.pos[3 * i] = rd(i, "x"); g.pos[3 * i + 1] = rd(i, "y"); g.pos[3 * i + 2] = rd(i, "z");
    const sig = 1 / (1 + Math.exp(-rd(i, "opacity")));
    for (let c = 0; c < 3; c++) g.col[4 * i + c] = clamp255((0.5 + SH_C0 * rd(i, "f_dc_" + c)) * 255);
    g.col[4 * i + 3] = clamp255(sig * 255);
    const s = [Math.exp(rd(i, "scale_0")), Math.exp(rd(i, "scale_1")), Math.exp(rd(i, "scale_2"))];
    g.scale.set(s, 3 * i);
    const q = [rd(i, "rot_0"), rd(i, "rot_1"), rd(i, "rot_2"), rd(i, "rot_3")];
    covariance(q, s, g.cov, 6 * i, g.nrm, 3 * i);
  }
  orientNormals(g);
  return g;
}

// Normals point away from the object's centre (a splat's flattest axis has no inherent sign)
function orientNormals(g) {
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < g.count; i++) { cx += g.pos[3 * i]; cy += g.pos[3 * i + 1]; cz += g.pos[3 * i + 2]; }
  cx /= g.count || 1; cy /= g.count || 1; cz /= g.count || 1;
  for (let i = 0; i < g.count; i++) {
    const o = 3 * i;
    const d = (g.pos[o] - cx) * g.nrm[o] + (g.pos[o + 1] - cy) * g.nrm[o + 1] + (g.pos[o + 2] - cz) * g.nrm[o + 2];
    if (d < 0) { g.nrm[o] = -g.nrm[o]; g.nrm[o + 1] = -g.nrm[o + 1]; g.nrm[o + 2] = -g.nrm[o + 2]; }
  }
}

function clamp255(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

function covariance(q, s, out, o, nrm, ni) {
  let [w, x, y, z] = q;
  const n = Math.hypot(w, x, y, z) || 1; w /= n; x /= n; y /= n; z /= n;
  const R = [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
  const M = R.map((v, k) => v * s[k % 3]); // M = R * diag(s)
  const dot = (i, j) => M[3 * i] * M[3 * j] + M[3 * i + 1] * M[3 * j + 1] + M[3 * i + 2] * M[3 * j + 2];
  out[o] = dot(0, 0); out[o + 1] = dot(0, 1); out[o + 2] = dot(0, 2);
  out[o + 3] = dot(1, 1); out[o + 4] = dot(1, 2); out[o + 5] = dot(2, 2);
  if (nrm) {  // surface normal = rotation axis of the smallest scale (the splat's flat direction)
    const k = s[0] <= s[1] && s[0] <= s[2] ? 0 : s[1] <= s[2] ? 1 : 2;
    nrm[ni] = R[k]; nrm[ni + 1] = R[3 + k]; nrm[ni + 2] = R[6 + k];
  }
}

// Stress-test helper: k jittered copies of every Gaussian (same as scripts/benchmark_renderer.py)
export function replicate(g, k, seed = 1) {
  if (k <= 1) return g;
  let st = seed >>> 0;
  const rand = () => { st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
  const randn = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const N = g.count * k;
  const r = { count: N, pos: new Float32Array(N * 3), col: new Uint8Array(N * 4), cov: new Float32Array(N * 6), scale: null,
              nrm: g.nrm ? new Float32Array(N * 3) : null };
  for (let c = 0; c < k; c++) {
    r.col.set(g.col, c * g.count * 4);
    r.cov.set(g.cov, c * g.count * 6);
    if (g.nrm) r.nrm.set(g.nrm, c * g.count * 3);
    for (let i = 0; i < g.count; i++) {
      const d = 3 * (c * g.count + i);
      for (let a = 0; a < 3; a++) r.pos[d + a] = g.pos[3 * i + a] + (c ? randn() * g.scale[3 * i + a] : 0);
    }
  }
  return r;
}

// Exhibition: several models side by side in one scene, sorted and blended together as one splat set.
// items = [{ g, yaw }]; yaw (degrees) turns each model so its front faces the viewer.
// LOD: each object keeps its most opaque Gaussians so the total stays within `budget`.
export function compose(items, budget = 1.2e6, spacing = 1.45) {
  const objs = items.map((it) => it.g);
  const k = objs.length;
  const per = Math.max(1, Math.floor(budget / Math.max(1, k)));
  const picks = objs.map((g) => {
    if (g.count <= per) return null;
    const idx = Array.from({ length: g.count }, (_, i) => i);
    idx.sort((a, b) => g.col[4 * b + 3] - g.col[4 * a + 3]);
    return idx.slice(0, per);
  });
  const N = objs.reduce((a, g, j) => a + (picks[j] ? picks[j].length : g.count), 0);
  const r = { count: N, pos: new Float32Array(N * 3), col: new Uint8Array(N * 4), cov: new Float32Array(N * 6),
              nrm: new Float32Array(N * 3), scale: null, objects: [] };
  let w = 0;
  // one row up to 4 models, otherwise two rows (back row staggered by half a slot so it shows between the front ones)
  const cols = k <= 4 ? k : Math.ceil(k / 2), rows = Math.ceil(k / cols);
  r.center = [0, 0, rows > 1 ? -0.8 : 0];
  r.halfWidth = ((cols - 1) / 2) * spacing + 0.8;
  objs.forEach((g, j) => {
    const row = Math.floor(j / cols), col = j % cols, inRow = Math.min(cols, k - row * cols);
    const dx = (col - (inRow - 1) / 2) * spacing + (row % 2) * spacing * 0.5 * (inRow === cols ? 1 : 0), dz = -row * 1.6;
    const ids = picks[j] || null, n = ids ? ids.length : g.count;
    const th = ((items[j].yaw || 0) * Math.PI) / 180, co = Math.cos(th), si = Math.sin(th);
    // rotation about y: x' = c x + s z, z' = -s x + c z; covariance C' = R C R^T
    const R = [co, 0, si, 0, 1, 0, -si, 0, co];
    for (let t = 0; t < n; t++) {
      const i = ids ? ids[t] : t;
      const x = g.pos[3 * i], z = g.pos[3 * i + 2];
      r.pos[3 * w] = co * x + si * z + dx; r.pos[3 * w + 1] = g.pos[3 * i + 1]; r.pos[3 * w + 2] = -si * x + co * z + dz;
      for (let c = 0; c < 4; c++) r.col[4 * w + c] = g.col[4 * i + c];
      const q = 6 * i, C = [g.cov[q], g.cov[q + 1], g.cov[q + 2], g.cov[q + 1], g.cov[q + 3], g.cov[q + 4], g.cov[q + 2], g.cov[q + 4], g.cov[q + 5]];
      const RC = new Array(9);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) RC[3 * a + b] = R[3 * a] * C[b] + R[3 * a + 1] * C[3 + b] + R[3 * a + 2] * C[6 + b];
      const M = (a, b) => RC[3 * a] * R[3 * b] + RC[3 * a + 1] * R[3 * b + 1] + RC[3 * a + 2] * R[3 * b + 2];
      const o = 6 * w;
      r.cov[o] = M(0, 0); r.cov[o + 1] = M(0, 1); r.cov[o + 2] = M(0, 2); r.cov[o + 3] = M(1, 1); r.cov[o + 4] = M(1, 2); r.cov[o + 5] = M(2, 2);
      if (g.nrm) {
        const nx = g.nrm[3 * i], nz = g.nrm[3 * i + 2];
        r.nrm[3 * w] = co * nx + si * nz; r.nrm[3 * w + 1] = g.nrm[3 * i + 1]; r.nrm[3 * w + 2] = -si * nx + co * nz;
      }
      w++;
    }
    r.objects.push({ dx, dz, kept: n, of: g.count });
  });
  return r;
}

// ------------------------------------------------------------------ float -> half
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
function toHalf(v) {
  f32[0] = v;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    m = (m | 0x800000) >> (1 - e);
    return sign | ((m + 0x1000) >> 13);
  }
  return sign | (e << 10) | ((m + 0x1000) >> 13);
}

function packTexture(g) {
  const rows = Math.ceil(g.count / (TEX_W / 2));
  const data = new Uint32Array(TEX_W * rows * 4);
  const fbits = new Float32Array(data.buffer);
  for (let i = 0; i < g.count; i++) {
    const o = 8 * i;
    fbits[o] = g.pos[3 * i]; fbits[o + 1] = g.pos[3 * i + 1]; fbits[o + 2] = g.pos[3 * i + 2];
    if (g.nrm) {
      let nx = g.nrm[3 * i], ny = g.nrm[3 * i + 1], nz = g.nrm[3 * i + 2];
      const l1 = Math.abs(nx) + Math.abs(ny) + Math.abs(nz) || 1;
      nx /= l1; ny /= l1;
      if (nz < 0) { const ox = (1 - Math.abs(ny)) * (nx >= 0 ? 1 : -1), oy = (1 - Math.abs(nx)) * (ny >= 0 ? 1 : -1); nx = ox; ny = oy; }
      data[o + 3] = (toHalf(nx) | (toHalf(ny) << 16)) >>> 0;
    }
    const c = 6 * i;
    data[o + 4] = toHalf(g.cov[c] * COV_SCALE) | (toHalf(g.cov[c + 1] * COV_SCALE) << 16);
    data[o + 5] = toHalf(g.cov[c + 2] * COV_SCALE) | (toHalf(g.cov[c + 3] * COV_SCALE) << 16);
    data[o + 6] = toHalf(g.cov[c + 4] * COV_SCALE) | (toHalf(g.cov[c + 5] * COV_SCALE) << 16);
    data[o + 7] = (g.col[4 * i] | (g.col[4 * i + 1] << 8) | (g.col[4 * i + 2] << 16) | (g.col[4 * i + 3] << 24)) >>> 0;
  }
  return { data, rows };
}

// ------------------------------------------------------------------ shaders
const VS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform usampler2D u_tex;
uniform mat4 u_view;
uniform vec2 u_viewport;   // pixels
uniform float u_focal;     // pixels
uniform float u_scaleMod;  // like upstream GUI "gaussian scale"
uniform int u_mode;        // 0 colour, 1 depth, 2 Gaussian centres
uniform vec2 u_depth;      // near/far for the depth colour map
uniform int u_lit;         // 1 = studio lighting from per-Gaussian normals
uniform vec3 u_light;      // unit direction towards the key light (world)
uniform vec3 u_eye;        // camera position (world)
in vec2 a_corner;          // quad corner in [-2,2]^2
in uint a_index;           // sorted Gaussian index (instanced)
out vec4 v_color;
out vec2 v_pos;
vec3 octDecode(vec2 e) {
  vec3 v = vec3(e, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0.0) v.xy = (1.0 - abs(v.yx)) * vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
  return normalize(v);
}
vec3 depthColor(float k) {   // near = warm yellow -> teal -> far = deep purple
  vec3 a = vec3(0.99, 0.91, 0.14), b = vec3(0.13, 0.57, 0.55), c = vec3(0.27, 0.0, 0.33);
  return k < 0.5 ? mix(a, b, k * 2.0) : mix(b, c, k * 2.0 - 1.0);
}
void main() {
  ivec2 t0 = ivec2((a_index & 1023u) << 1, a_index >> 10);
  uvec4 c0 = texelFetch(u_tex, t0, 0);
  vec4 cam = u_view * vec4(uintBitsToFloat(c0.xyz), 1.0);
  float z = cam.z;
  vec2 ndc = vec2(u_focal * cam.x / z, -u_focal * cam.y / z) * 2.0 / u_viewport;
  if (z < 0.05 || abs(ndc.x) > 1.3 || abs(ndc.y) > 1.3) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  uvec4 c1 = texelFetch(u_tex, t0 + ivec2(1, 0), 0);
  vec2 h0 = unpackHalf2x16(c1.x), h1 = unpackHalf2x16(c1.y), h2 = unpackHalf2x16(c1.z);
  mat3 S = mat3(h0.x, h0.y, h1.x,  h0.y, h1.y, h2.x,  h1.x, h2.x, h2.y) * (u_scaleMod * u_scaleMod / ${COV_SCALE.toFixed(1)});
  mat3 W = mat3(u_view);
  mat3 Sc = W * S * transpose(W);                      // camera-space covariance
  float lim = 1.3 * 0.5 * u_viewport.x / u_focal;
  float tx = clamp(cam.x / z, -lim, lim) * z, ty = clamp(cam.y / z, -lim, lim) * z;
  vec3 j0 = vec3(u_focal / z, 0.0, -u_focal * tx / (z * z));   // EWA affine approximation
  vec3 j1 = vec3(0.0, u_focal / z, -u_focal * ty / (z * z));
  float a = dot(j0, Sc * j0) + 0.3, b = dot(j0, Sc * j1), c = dot(j1, Sc * j1) + 0.3;
  float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b));
  float l1 = mid + rad, l2 = mid - rad;
  if (l2 <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec2 e = vec2(b, l1 - a);
  vec2 e1 = dot(e, e) < 1e-20 ? (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(e);
  vec2 e2 = vec2(-e1.y, e1.x);
  vec2 d = a_corner.x * min(sqrt(2.0 * l1), 1024.0) * e1 + a_corner.y * min(sqrt(2.0 * l2), 1024.0) * e2;
  uint rgba = c1.w;
  v_color = vec4(float(rgba & 255u), float((rgba >> 8) & 255u), float((rgba >> 16) & 255u), float(rgba >> 24)) / 255.0;
  if (u_lit == 1 && u_mode == 0) {   // studio lighting: soft wrap diffuse + specular + cool rim
    vec3 P = uintBitsToFloat(c0.xyz);
    vec3 n = octDecode(unpackHalf2x16(c0.w));
    vec3 V = normalize(u_eye - P);
    float nl = dot(n, u_light);
    float wrap = clamp((nl + 0.4) / 1.4, 0.0, 1.0);
    float spec = pow(max(dot(n, normalize(u_light + V)), 0.0), 28.0) * step(0.0, nl);
    float rim = pow(1.0 - abs(dot(n, V)), 3.0);
    v_color.rgb = v_color.rgb * (0.42 + 0.78 * wrap) + vec3(0.20) * spec + vec3(0.10, 0.13, 0.18) * rim;
  }
  if (u_mode == 1) v_color.rgb = depthColor(clamp((z - u_depth.x) / (u_depth.y - u_depth.x), 0.0, 1.0));
  if (u_mode == 2) { d = a_corner * 0.9; v_color.a = 1.0; }   // ~2 px dot at each Gaussian centre
  v_pos = a_corner;
  gl_Position = vec4(ndc + vec2(d.x, -d.y) * 2.0 / u_viewport, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
in vec4 v_color;
in vec2 v_pos;
out vec4 fragColor;
void main() {
  float A = -dot(v_pos, v_pos);
  if (A < -4.0) discard;
  float alpha = min(0.99, exp(A) * v_color.a);
  if (alpha < 1.0 / 255.0) discard;
  fragColor = vec4(v_color.rgb * alpha, alpha);   // premultiplied, composited back-to-front
}`;

const BG_VS = `#version 300 es
in vec2 a_xy; out vec2 v_uv;
void main() { v_uv = a_xy * 0.5 + 0.5; gl_Position = vec4(a_xy, 0.0, 1.0); }`;
const BG_FS = `#version 300 es
precision highp float; in vec2 v_uv; out vec4 o;
uniform vec3 u_top, u_bottom;
void main() {
  float vig = 1.0 - 0.35 * pow(length(v_uv - vec2(0.5, 0.45)), 2.0);
  o = vec4(mix(u_bottom, u_top, smoothstep(0.0, 1.0, v_uv.y)) * vig, 1.0);
}`;
const GROUND_VS = `#version 300 es
precision highp float;
in vec2 a_xy; out vec2 v_uv;
uniform mat4 u_view; uniform vec2 u_viewport; uniform float u_focal;
uniform vec3 u_center; uniform float u_radius;
void main() {
  v_uv = a_xy;
  vec4 cam = u_view * vec4(u_center + vec3(a_xy.x * u_radius, 0.0, a_xy.y * u_radius), 1.0);
  // perspective-correct: clip.w = camera depth
  gl_Position = vec4(u_focal * cam.x * 2.0 / u_viewport.x, -u_focal * cam.y * 2.0 / u_viewport.y, 0.0, max(cam.z, 0.05));
}`;
const GROUND_FS = `#version 300 es
precision highp float; in vec2 v_uv; out vec4 o;
uniform sampler2D u_shadow; uniform vec3 u_floor;
void main() {
  float r = length(v_uv);
  if (r > 1.0) discard;
  float a = 1.0 - smoothstep(0.55, 1.0, r);           // floor fades into the backdrop
  float sh = texture(u_shadow, v_uv * 0.5 + 0.5).r;     // soft shadow from the object's Gaussians
  vec3 c = u_floor * (1.0 - 0.62 * sh) * (1.05 - 0.25 * r);
  o = vec4(c * a, a);
}`;

function link(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}

// ------------------------------------------------------------------ viewer
export class SplatViewer {
  constructor(canvas, { onStats } = {}) {
    this.canvas = canvas;
    this.onStats = onStats || (() => {});
    this.initGL();
    // a lost WebGL context (driver reset, GPU pressure, tab backgrounded) is restored instead of leaving a blank canvas
    canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); this.lost = true; });
    canvas.addEventListener("webglcontextrestored", () => {
      this.initGL();
      this.lost = false;
      if (this.lastG) this.setGaussians(this.lastG);
    });
    this.mode = 0;
    this.lit = true;            // studio lighting (Rendering pillar), on by default
    this.stage = true;          // backdrop + ground + soft shadow
    this.light = { az: 35, el: 50 };
    this.theme = { top: [0.17, 0.19, 0.24], bottom: [0.055, 0.062, 0.078], floor: [0.36, 0.39, 0.46] };

    // camera (orbit around origin, DreamGaussian convention: y up, front view looks from +z)
    this.cam = { az: 0, el: 15, r: 2.5, fovy: 49.1 };
    this.bg = [1, 1, 1];
    this.scaleMod = 1;
    this.resScale = 1;
    this.autoRotate = 0; // deg/s
    this.count = 0;
    this.sortedCount = 0;
    this.sortMs = 0;
    this.needsSort = false;
    this.sorting = false;
    this.frameTimes = [];
    this.worker = new Worker(new URL("./sort-worker.js", import.meta.url));
    this.worker.onmessage = (e) => this.onSorted(e.data);
    this.installControls();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
    this.resize();
    this.last = performance.now();
    requestAnimationFrame((t) => this.frame(t));
  }

  initGL() {
    const canvas = this.canvas;
    const gl = (this.gl = this.gl || canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, preserveDrawingBuffer: true }));
    if (!gl) throw new Error("WebGL2 not available");
    const prog = (this.prog = gl.createProgram());
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    this.u = {};
    for (const n of ["u_tex", "u_view", "u_viewport", "u_focal", "u_scaleMod", "u_mode", "u_depth", "u_lit", "u_light", "u_eye"]) this.u[n] = gl.getUniformLocation(prog, n);

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
    const aCorner = gl.getAttribLocation(prog, "a_corner");
    gl.enableVertexAttribArray(aCorner);
    gl.vertexAttribPointer(aCorner, 2, gl.FLOAT, false, 0, 0);
    this.indexBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    const aIndex = gl.getAttribLocation(prog, "a_index");
    gl.enableVertexAttribArray(aIndex);
    gl.vertexAttribIPointer(aIndex, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(aIndex, 1);
    const ebo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);

    // stage: backdrop + ground programs share one quad
    this.bgProg = link(gl, BG_VS, BG_FS);
    this.groundProg = link(gl, GROUND_VS, GROUND_FS);
    this.ug = {};
    for (const n of ["u_view", "u_viewport", "u_focal", "u_center", "u_radius", "u_shadow", "u_floor"]) this.ug[n] = gl.getUniformLocation(this.groundProg, n);
    this.ub = { u_top: gl.getUniformLocation(this.bgProg, "u_top"), u_bottom: gl.getUniformLocation(this.bgProg, "u_bottom") };
    this.stageVao = gl.createVertexArray();
    gl.bindVertexArray(this.stageVao);
    const sq = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, sq);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    for (const prog2 of [this.bgProg, this.groundProg]) {
      const loc = gl.getAttribLocation(prog2, "a_xy");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    }
    this.shadowTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindVertexArray(this.vao);

    this.tex = gl.createTexture();
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    this.timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.queries = [];
    this.sortedCount = 0;
    this.gpuMs = null;
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    this.glRenderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);

  }

  // PNG of the current frame (preserveDrawingBuffer is on)
  screenshot() { return this.canvas.toDataURL("image/png"); }

  lightDir() {
    const a = (this.light.az * Math.PI) / 180, e = (Math.max(12, Math.min(85, this.light.el)) * Math.PI) / 180;
    return [Math.cos(e) * Math.sin(a), Math.sin(e), Math.cos(e) * Math.cos(a)];
  }

  // Ground under the object(s) + soft shadow: project every Gaussian centre along the light onto the floor,
  // accumulate opacity in a 160x160 grid, blur, upload as a texture. Re-run only when the model or light changes.
  updateStage() {
    const g = this.lastG;
    if (!g || !g.count || !this.gl) return;
    const N = g.count, ys = [];
    let cx = 0, cz = 0;
    for (let i = 0; i < N; i += Math.max(1, Math.floor(N / 4000))) ys.push(g.pos[3 * i + 1]);
    ys.sort((a, b) => a - b);
    const y0 = ys[Math.floor(ys.length * 0.01)] - 0.01;
    let ext = 0;
    for (let i = 0; i < N; i++) { cx += g.pos[3 * i]; cz += g.pos[3 * i + 2]; }
    cx /= N; cz /= N;
    for (let i = 0; i < N; i += 7) ext = Math.max(ext, Math.abs(g.pos[3 * i] - cx), Math.abs(g.pos[3 * i + 2] - cz));
    const R = ext * 1.6 + 0.45;
    const L = this.lightDir(), S = g.objects ? 256 : 160, acc = new Float32Array(S * S);
    for (let i = 0; i < N; i++) {
      const y = g.pos[3 * i + 1] - y0;
      const hx = g.pos[3 * i] - (L[0] / L[1]) * y - cx, hz = g.pos[3 * i + 2] - (L[2] / L[1]) * y - cz;
      const ix = Math.floor((hx / R * 0.5 + 0.5) * S), iz = Math.floor((hz / R * 0.5 + 0.5) * S);
      if (ix >= 0 && ix < S && iz >= 0 && iz < S) acc[iz * S + ix] += g.col[4 * i + 3] / 255;
    }
    const blur = (a) => {  // separable box blur, radius 2
      const t = new Float32Array(S * S);
      for (let z = 0; z < S; z++) for (let x = 0; x < S; x++) { let v = 0; for (let k = -2; k <= 2; k++) v += a[z * S + Math.min(S - 1, Math.max(0, x + k))]; t[z * S + x] = v / 5; }
      for (let z = 0; z < S; z++) for (let x = 0; x < S; x++) { let v = 0; for (let k = -2; k <= 2; k++) v += t[Math.min(S - 1, Math.max(0, z + k)) * S + x]; a[z * S + x] = v / 5; }
    };
    blur(acc); blur(acc);
    let mx = 0;
    for (let i = 0; i < S * S; i++) mx = Math.max(mx, acc[i]);
    const tex = new Uint8Array(S * S);
    for (let i = 0; i < S * S; i++) tex[i] = Math.round(255 * (1 - Math.exp(-4 * acc[i] / (mx || 1))));
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, S, S, 0, gl.RED, gl.UNSIGNED_BYTE, tex);
    this.ground = { center: [cx, y0, cz], radius: R };
  }

  setLight(az, el) {
    this.light = { az, el };
    clearTimeout(this._stageTimer);
    this._stageTimer = setTimeout(() => this.updateStage(), 60);
  }

  setGaussians(g) {
    this.lastG = g;
    const gl = this.gl;
    const { data, rows } = packTexture(g);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, TEX_W, rows, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, data);
    this.count = g.count;
    this.sortedCount = 0;
    this.worker.postMessage({ type: "positions", pos: g.pos.slice(), count: g.count });
    this.needsSort = true;
    this.sorting = false;
    this.lastSortView = null;
    this.updateStage();
  }

  resize() {
    const dpr = (window.devicePixelRatio || 1) * this.resScale;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  setResolutionScale(s) { this.resScale = s; this.resize(); }

  viewMatrix() {
    const { az, el, r } = this.cam;
    const a = (az * Math.PI) / 180, e = (el * Math.PI) / 180;
    const t = this.target || [0, 0, 0];
    const eye = [t[0] + r * Math.cos(e) * Math.sin(a), t[1] + r * Math.sin(e), t[2] + r * Math.cos(e) * Math.cos(a)];
    this.eye = eye;
    const f = norm([t[0] - eye[0], t[1] - eye[1], t[2] - eye[2]]);
    const right = norm(cross(f, [0, 1, 0]));
    const down = cross(f, right);
    // rows: right, down, forward (OpenCV-style camera: +z into the screen)
    const m = new Float32Array(16);
    const rows = [right, down, f];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) m[j * 4 + i] = rows[i][j];
      m[12 + i] = -(rows[i][0] * eye[0] + rows[i][1] * eye[1] + rows[i][2] * eye[2]);
    }
    m[15] = 1;
    return m;
  }

  requestSort(view) {
    if (this.sorting || !this.count) { this.needsSort = true; return; }
    const lv = this.lastSortView;
    if (lv && !this.needsSort) {
      const d = Math.abs(lv[2] - view[2]) + Math.abs(lv[6] - view[6]) + Math.abs(lv[10] - view[10]) + Math.abs(lv[14] - view[14]);
      if (d < 0.01) return;
    }
    this.sorting = true;
    this.needsSort = false;
    this.lastSortView = view;
    this.sortStart = performance.now();
    this.worker.postMessage({ type: "sort", view });
  }

  onSorted({ indices, count }) {
    this.sorting = false;
    if (count !== this.count) { this.needsSort = true; return; }
    this.sortMs = performance.now() - this.sortStart;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
    this.sortedCount = count;
  }

  frame(t) {
    if (this.lost) { requestAnimationFrame((tt) => this.frame(tt)); return; }
    const dt = Math.min(0.1, (t - this.last) / 1000);
    this.last = t;
    if (this.autoRotate && !this.dragging) this.cam.az = (this.cam.az + this.autoRotate * dt) % 360;
    this.resize();
    const gl = this.gl;
    const view = this.viewMatrix();
    this.requestSort(view);

    // collect GPU timer results
    if (this.timerExt) {
      while (this.queries.length && gl.getQueryParameter(this.queries[0], gl.QUERY_RESULT_AVAILABLE)) {
        const q = this.queries.shift();
        if (!gl.getParameter(this.timerExt.GPU_DISJOINT_EXT)) this.gpuMs = gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6;
        gl.deleteQuery(q);
      }
    }
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const focal = (0.5 * this.canvas.height) / Math.tan((this.cam.fovy * Math.PI) / 360);
    if (this.stage) {   // backdrop gradient, then the ground disc with the object's soft shadow
      gl.bindVertexArray(this.stageVao);
      gl.disable(gl.BLEND);
      gl.useProgram(this.bgProg);
      gl.uniform3fv(this.ub.u_top, this.theme.top);
      gl.uniform3fv(this.ub.u_bottom, this.theme.bottom);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.enable(gl.BLEND);
      if (this.ground) {
        gl.useProgram(this.groundProg);
        gl.uniformMatrix4fv(this.ug.u_view, false, view);
        gl.uniform2f(this.ug.u_viewport, this.canvas.width, this.canvas.height);
        gl.uniform1f(this.ug.u_focal, focal);
        gl.uniform3fv(this.ug.u_center, this.ground.center);
        gl.uniform1f(this.ug.u_radius, this.ground.radius);
        gl.uniform3fv(this.ug.u_floor, this.theme.floor);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.shadowTex);
        gl.uniform1i(this.ug.u_shadow, 1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
    }
    if (this.sortedCount) {
      let q = null;
      if (this.timerExt && this.queries.length < 4) { q = gl.createQuery(); gl.beginQuery(this.timerExt.TIME_ELAPSED_EXT, q); }
      gl.useProgram(this.prog);
      gl.bindVertexArray(this.vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.uniform1i(this.u.u_tex, 0);
      gl.uniformMatrix4fv(this.u.u_view, false, view);
      gl.uniform2f(this.u.u_viewport, this.canvas.width, this.canvas.height);
      gl.uniform1f(this.u.u_focal, (0.5 * this.canvas.height) / Math.tan((this.cam.fovy * Math.PI) / 360));
      gl.uniform1f(this.u.u_scaleMod, this.scaleMod);
      gl.uniform1i(this.u.u_mode, this.mode | 0);
      gl.uniform1i(this.u.u_lit, this.lit && this.lastG?.nrm ? 1 : 0);
      gl.uniform3fv(this.u.u_light, this.lightDir());
      gl.uniform3fv(this.u.u_eye, this.eye || [0, 0, 2.5]);
      gl.uniform2f(this.u.u_depth, this.cam.r - 1.2, this.cam.r + 1.2);
      gl.drawElementsInstanced(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0, this.sortedCount);
      if (q) { gl.endQuery(this.timerExt.TIME_ELAPSED_EXT); this.queries.push(q); }
    }
    this.frameTimes.push(t);
    while (this.frameTimes.length && t - this.frameTimes[0] > 1000) this.frameTimes.shift();
    this.onStats({
      fps: this.frameTimes.length, gpuMs: this.gpuMs, sortMs: this.sortMs, count: this.count,
      width: this.canvas.width, height: this.canvas.height, renderer: this.glRenderer,
    });
    requestAnimationFrame((tt) => this.frame(tt));
  }

  installControls() {
    const c = this.canvas;
    let px = 0, py = 0;
    c.addEventListener("pointerdown", (e) => { this.dragging = true; px = e.clientX; py = e.clientY; c.setPointerCapture(e.pointerId); });
    c.addEventListener("pointerup", (e) => { this.dragging = false; c.releasePointerCapture(e.pointerId); });
    c.addEventListener("pointermove", (e) => {
      if (!this.dragging) return;
      this.cam.az -= (e.clientX - px) * 0.4;
      this.cam.el = Math.max(-85, Math.min(85, this.cam.el + (e.clientY - py) * 0.4));
      px = e.clientX; py = e.clientY;
    });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.cam.r = Math.max(0.6, Math.min(14, this.cam.r * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  }
}

function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(v) { const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
