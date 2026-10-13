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
  const g = { count, pos: new Float32Array(count * 3), col: new Uint8Array(count * 4), cov: new Float32Array(count * 6), scale: new Float32Array(count * 3) };
  for (let i = 0; i < count; i++) {
    g.pos[3 * i] = rd(i, "x"); g.pos[3 * i + 1] = rd(i, "y"); g.pos[3 * i + 2] = rd(i, "z");
    const sig = 1 / (1 + Math.exp(-rd(i, "opacity")));
    for (let c = 0; c < 3; c++) g.col[4 * i + c] = clamp255((0.5 + SH_C0 * rd(i, "f_dc_" + c)) * 255);
    g.col[4 * i + 3] = clamp255(sig * 255);
    const s = [Math.exp(rd(i, "scale_0")), Math.exp(rd(i, "scale_1")), Math.exp(rd(i, "scale_2"))];
    g.scale.set(s, 3 * i);
    const q = [rd(i, "rot_0"), rd(i, "rot_1"), rd(i, "rot_2"), rd(i, "rot_3")];
    covariance(q, s, g.cov, 6 * i);
  }
  return g;
}

function clamp255(v) { return v < 0 ? 0 : v > 255 ? 255 : v; }

function covariance(q, s, out, o) {
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
}

// Stress-test helper: k jittered copies of every Gaussian (same as scripts/benchmark_renderer.py)
export function replicate(g, k, seed = 1) {
  if (k <= 1) return g;
  let st = seed >>> 0;
  const rand = () => { st = (st * 1664525 + 1013904223) >>> 0; return st / 4294967296; };
  const randn = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const N = g.count * k;
  const r = { count: N, pos: new Float32Array(N * 3), col: new Uint8Array(N * 4), cov: new Float32Array(N * 6), scale: null };
  for (let c = 0; c < k; c++) {
    r.col.set(g.col, c * g.count * 4);
    r.cov.set(g.cov, c * g.count * 6);
    for (let i = 0; i < g.count; i++) {
      const d = 3 * (c * g.count + i);
      for (let a = 0; a < 3; a++) r.pos[d + a] = g.pos[3 * i + a] + (c ? randn() * g.scale[3 * i + a] : 0);
    }
  }
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
in vec2 a_corner;          // quad corner in [-2,2]^2
in uint a_index;           // sorted Gaussian index (instanced)
out vec4 v_color;
out vec2 v_pos;
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
    const gl = (this.gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, preserveDrawingBuffer: true }));
    if (!gl) throw new Error("WebGL2 not available");
    const prog = (this.prog = gl.createProgram());
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    this.u = {};
    for (const n of ["u_tex", "u_view", "u_viewport", "u_focal", "u_scaleMod"]) this.u[n] = gl.getUniformLocation(prog, n);

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

    this.tex = gl.createTexture();
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    this.timerExt = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.queries = [];
    this.gpuMs = null;
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    this.glRenderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);

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

  setGaussians(g) {
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
    const eye = [r * Math.cos(e) * Math.sin(a), r * Math.sin(e), r * Math.cos(e) * Math.cos(a)];
    const f = norm([-eye[0], -eye[1], -eye[2]]);
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
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.cam.r = Math.max(0.6, Math.min(8, this.cam.r * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  }
}

function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm(v) { const l = Math.hypot(...v) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
