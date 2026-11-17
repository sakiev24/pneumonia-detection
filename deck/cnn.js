// Forward pass of PneumoniaCNN in plain JS (BatchNorm already folded into the convs).
// Works in the browser (window.CNN) and in node (module.exports) for checking against PyTorch.
(function (root) {
  function decode(weights) {
    const bin = typeof atob === "function"
      ? Uint8Array.from(atob(weights.data), c => c.charCodeAt(0))
      : Uint8Array.from(Buffer.from(weights.data, "base64"));
    const all = new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / 4);
    const out = [];
    let offset = 0;
    for (const shape of weights.shapes) {
      const n = shape.reduce((a, b) => a * b, 1);
      out.push({ shape, data: all.subarray(offset, offset + n) });
      offset += n;
    }
    return out;
  }

  // conv 3x3, padding 1, then ReLU
  function convRelu(x, C, H, W, w, b) {
    const O = w.shape[0];
    const out = new Float32Array(O * H * W);
    const k = w.data;
    for (let o = 0; o < O; o++) {
      const plane = out.subarray(o * H * W, (o + 1) * H * W);
      plane.fill(b.data[o]);
      for (let c = 0; c < C; c++) {
        const src = x.subarray(c * H * W, (c + 1) * H * W);
        const kb = (o * C + c) * 9;
        for (let ky = -1; ky <= 1; ky++) {
          for (let kx = -1; kx <= 1; kx++) {
            const kv = k[kb + (ky + 1) * 3 + (kx + 1)];
            const y0 = Math.max(0, -ky), y1 = Math.min(H, H - ky);
            const x0 = Math.max(0, -kx), x1 = Math.min(W, W - kx);
            for (let y = y0; y < y1; y++) {
              const row = y * W, srow = (y + ky) * W + kx;
              for (let xx = x0; xx < x1; xx++) plane[row + xx] += kv * src[srow + xx];
            }
          }
        }
      }
      for (let i = 0; i < plane.length; i++) if (plane[i] < 0) plane[i] = 0;
    }
    return out;
  }

  function maxPool(x, C, H, W) {
    const h = H >> 1, w = W >> 1;
    const out = new Float32Array(C * h * w);
    for (let c = 0; c < C; c++)
      for (let y = 0; y < h; y++)
        for (let xx = 0; xx < w; xx++) {
          const i = c * H * W + 2 * y * W + 2 * xx;
          out[c * h * w + y * w + xx] = Math.max(x[i], x[i + 1], x[i + W], x[i + W + 1]);
        }
    return out;
  }

  function linear(x, w, b, relu) {
    const [O, I] = w.shape;
    const out = new Float32Array(O);
    for (let o = 0; o < O; o++) {
      let s = b.data[o];
      const row = o * I;
      for (let i = 0; i < I; i++) s += w.data[row + i] * x[i];
      out[o] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  // pixels: Float32Array of 128*128 values in 0..1
  function predict(params, pixels) {
    let x = pixels, C = 1, H = 128, W = 128;
    const maps = [];
    for (let blk = 0; blk < 4; blk++) {
      const w = params[blk * 2], b = params[blk * 2 + 1];
      x = convRelu(x, C, H, W, w, b);
      C = w.shape[0];
      maps.push({ data: x, C, H, W });
      x = maxPool(x, C, H, W);
      H >>= 1; W >>= 1;
    }
    const h = linear(x, params[8], params[9], true);
    const logits = linear(h, params[10], params[11], false);
    const m = Math.max(...logits);
    const e = Array.from(logits, v => Math.exp(v - m));
    const s = e.reduce((a, v) => a + v, 0);
    return { probs: e.map(v => v / s), maps };
  }

  const api = { decode, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.CNN = api;
})(typeof window !== "undefined" ? window : globalThis);
