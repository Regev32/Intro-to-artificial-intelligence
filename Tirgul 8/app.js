const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const scaleCanvas = document.getElementById("colorScale");
const sctx = scaleCanvas.getContext("2d");

const W = canvas.width;
const H = canvas.height;

let scale = 80;
const SCALE_MIN = 20;
const SCALE_MAX = 300;

const START_POINT = [2.5, 2.0];
let running = false;
let startedOptimizers = new Set();

// --------------------------------------------------
// Colors
// --------------------------------------------------
const COLORS = {
  SGD: "#ff5555",
  Momentum: "#ffaa00",
  Nesterov: "#ff00ff",
  AdaGrad: "#7b3fe4",   // deep violet
  RMSProp: "#00ff00",
  Adam: "#ffffff"
};

// --------------------------------------------------
// Math
// --------------------------------------------------
function f(x, y) {
  return x * x + 0.3 * y * y - 0.5 * x * y;
}

function grad(x, y) {
  return [2 * x - 0.5 * y, 0.6 * y - 0.5 * x];
}

function toCanvas(x, y) {
  return [W / 2 + x * scale, H / 2 - y * scale];
}

// --------------------------------------------------
// Colormap (calm, no bright yellow)
// --------------------------------------------------
function colormap(t) {
  const stops = [
    [0.0, 30, 60, 170],
    [0.35, 40, 180, 120],
    [0.65, 230, 140, 60],
    [1.0, 200, 50, 50]
  ];

  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    if (t >= t0 && t <= t1) {
      const a = (t - t0) / (t1 - t0);
      return `rgb(${(r0 + a * (r1 - r0)) | 0},
                  ${(g0 + a * (g1 - g0)) | 0},
                  ${(b0 + a * (b1 - b0)) | 0})`;
    }
  }
}

// --------------------------------------------------
// Draw surface + colorbar
// --------------------------------------------------
function drawSurface() {
  const zmin = 0;
  const zmax = 6;

  for (let i = 0; i < W; i++) {
    for (let j = 0; j < H; j++) {
      const x = (i - W / 2) / scale;
      const y = (H / 2 - j) / scale;
      const z = f(x, y);
      const t = Math.min(1, Math.max(0, (z - zmin) / (zmax - zmin)));
      ctx.fillStyle = colormap(t);
      ctx.fillRect(i, j, 1, 1);
    }
  }
  drawColorbar(zmin, zmax);
}

function drawColorbar(zmin, zmax) {
  sctx.clearRect(0, 0, scaleCanvas.width, scaleCanvas.height);

  for (let j = 0; j < scaleCanvas.height; j++) {
    const t = 1 - j / scaleCanvas.height;
    sctx.fillStyle = colormap(t);
    sctx.fillRect(0, j, scaleCanvas.width, 1);
  }

  const ticks = document.getElementById("ticks");
  ticks.innerHTML = "";
  for (let i = 0; i <= 4; i++) {
    const v = (zmax - i * (zmax - zmin) / 4).toFixed(2);
    const div = document.createElement("div");
    div.textContent = v;
    ticks.appendChild(div);
  }
}

// --------------------------------------------------
// Optimizers
// --------------------------------------------------
const optimizers = {
  SGD: { x: [...START_POINT], lr: 0.08 },

  Momentum: { x: [...START_POINT], v: [0, 0], lr: 0.08, b: 0.9 },

  Nesterov: { x: [...START_POINT], v: [0, 0], lr: 0.08, b: 0.9 },

  AdaGrad: { x: [...START_POINT], G: [0, 0], lr: 0.6 },

  RMSProp: { x: [...START_POINT], G: [0, 0], lr: 0.08, b: 0.9 },

  Adam: { x: [...START_POINT], m: [0, 0], v: [0, 0], t: 0, lr: 0.12 }
};

const paths = {};
Object.keys(optimizers).forEach(k => paths[k] = [[...START_POINT]]);

function step(name) {
  const o = optimizers[name];
  const [gx, gy] = grad(o.x[0], o.x[1]);

  if (name === "SGD") {
    o.x[0] -= o.lr * gx;
    o.x[1] -= o.lr * gy;
  }

  if (name === "Momentum") {
    o.v[0] = o.b * o.v[0] - o.lr * gx;
    o.v[1] = o.b * o.v[1] - o.lr * gy;
    o.x[0] += o.v[0];
    o.x[1] += o.v[1];
  }

  if (name === "Nesterov") {
    const nx = o.x[0] + o.b * o.v[0];
    const ny = o.x[1] + o.b * o.v[1];
    const [ngx, ngy] = grad(nx, ny);
    o.v[0] = o.b * o.v[0] - o.lr * ngx;
    o.v[1] = o.b * o.v[1] - o.lr * ngy;
    o.x[0] += o.v[0];
    o.x[1] += o.v[1];
  }

  if (name === "AdaGrad") {
    o.G[0] += gx * gx;
    o.G[1] += gy * gy;
    o.x[0] -= o.lr * gx / Math.sqrt(o.G[0] + 1e-8);
    o.x[1] -= o.lr * gy / Math.sqrt(o.G[1] + 1e-8);
  }

  if (name === "RMSProp") {
    o.G[0] = o.b * o.G[0] + (1 - o.b) * gx * gx;
    o.G[1] = o.b * o.G[1] + (1 - o.b) * gy * gy;
    o.x[0] -= o.lr * gx / Math.sqrt(o.G[0] + 1e-8);
    o.x[1] -= o.lr * gy / Math.sqrt(o.G[1] + 1e-8);
  }

  if (name === "Adam") {
    o.t++;
    o.m[0] = 0.9 * o.m[0] + 0.1 * gx;
    o.m[1] = 0.9 * o.m[1] + 0.1 * gy;
    o.v[0] = 0.999 * o.v[0] + 0.001 * gx * gx;
    o.v[1] = 0.999 * o.v[1] + 0.001 * gy * gy;
    o.x[0] -= o.lr * o.m[0] / Math.sqrt(o.v[0] + 1e-8);
    o.x[1] -= o.lr * o.m[1] / Math.sqrt(o.v[1] + 1e-8);
  }
}

// --------------------------------------------------
// Animation + zoom
// --------------------------------------------------
function drawAllPaths() {
  startedOptimizers.forEach(name => {
    ctx.strokeStyle = COLORS[name];
    ctx.beginPath();
    paths[name].forEach((p, i) => {
      const [cx, cy] = toCanvas(p[0], p[1]);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();
  });

  const [mx, my] = toCanvas(0, 0);
  ctx.fillStyle = "red";
  ctx.beginPath();
  ctx.arc(mx, my, 5, 0, 2 * Math.PI);
  ctx.fill();
}

function animate() {
  if (!running) return;

  ctx.clearRect(0, 0, W, H);
  drawSurface();

  startedOptimizers.forEach(name => {
    step(name);
    paths[name].push([...optimizers[name].x]);
  });

  drawAllPaths();
  requestAnimationFrame(animate);
}

function start() {
  reset();
  document.querySelectorAll("input[data-opt]").forEach(cb => {
    if (cb.checked) startedOptimizers.add(cb.dataset.opt);
  });
  running = true;
  animate();
}

function reset() {
  running = false;
  startedOptimizers.clear();

  Object.keys(optimizers).forEach(k => {
    optimizers[k].x = [...START_POINT];
    optimizers[k].v = [0, 0];
    optimizers[k].m = [0, 0];
    optimizers[k].G = [0, 0];
    optimizers[k].t = 0;
    paths[k] = [[...START_POINT]];
  });

  ctx.clearRect(0, 0, W, H);
  drawSurface();
}

// Zoom
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.1 : 0.9;
  scale = Math.min(SCALE_MAX, Math.max(SCALE_MIN, scale * factor));
  ctx.clearRect(0, 0, W, H);
  drawSurface();
  drawAllPaths();
});

drawSurface();
