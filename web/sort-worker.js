// Depth sort for back-to-front splat compositing: 16-bit counting sort on camera-space depth.
let pos = null, count = 0;

onmessage = (e) => {
  const m = e.data;
  if (m.type === "positions") { pos = m.pos; count = m.count; return; }
  if (m.type !== "sort" || !pos) return;
  const v = m.view; // column-major world->camera; camera z = row 2
  const depth = new Float32Array(count);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < count; i++) {
    const d = v[2] * pos[3 * i] + v[6] * pos[3 * i + 1] + v[10] * pos[3 * i + 2] + v[14];
    depth[i] = d;
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  const B = 65536, scale = (B - 1) / Math.max(1e-9, hi - lo);
  const key = new Uint16Array(count);
  const counts = new Uint32Array(B);
  for (let i = 0; i < count; i++) { const k = ((hi - depth[i]) * scale) | 0; key[i] = k; counts[k]++; } // far first
  for (let i = 1; i < B; i++) counts[i] += counts[i - 1];
  const indices = new Uint32Array(count);
  for (let i = count - 1; i >= 0; i--) indices[--counts[key[i]]] = i;
  postMessage({ indices, count }, [indices.buffer]);
};
