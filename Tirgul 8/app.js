const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const title = document.getElementById("title");

const W = canvas.width;
const H = canvas.height;

let mode = "2D"; // "2D" or "1D"
let running = false;
let started = new Set();

const COLORS = {
  SGD: "#ff5555",
  Momentum: "#ffaa00",
  Nesterov: "#ff00ff",
  AdaGrad: "#7b3fe4",
  RMSProp: "#00ff00",
  Adam: "#ffffff"
};

/* -------------------- FUNCTIONS -------------------- */

// 2D
function f2(x, y) { return x*x + 0.3*y*y - 0.5*x*y; }
function grad2(x, y) { return [2*x - 0.5*y, 0.6*y - 0.5*x]; }

// 1D
function f1(x) { return x**5 - 3*x**3 + x**2 + Math.sin(x); }
function grad1(x) {
  return 5*x**4 - 9*x**2 + 2*x + Math.cos(x);
}

/* -------------------- OPTIMIZERS -------------------- */

const optimizers = {};
const paths = {};

function initOptimizers() {
  const start2D = [2.5, 2.0];
  const start1D = -1.4;

  ["SGD","Momentum","Nesterov","AdaGrad","RMSProp","Adam"].forEach(k => {
    optimizers[k] = {
      x2: [...start2D],
      x1: start1D,
      v: [0,0],
      v1: 0,
      m: [0,0],
      m1: 0,
      G: [0,0],
      G1: 0,
      t: 0,
      lr: k === "AdaGrad" ? 0.4 : 0.08
    };
    paths[k] = [];
  });
}

initOptimizers();

/* -------------------- DRAWING -------------------- */

function draw2DSurface() {
  for (let i=0;i<W;i++) {
    for (let j=0;j<H;j++) {
      const x=(i-W/2)/80, y=(H/2-j)/80;
      const z=f2(x,y);
      const t=Math.min(1,Math.max(0,(z+2)/6));
      ctx.fillStyle=`hsl(${220-200*t},70%,50%)`;
      ctx.fillRect(i,j,1,1);
    }
  }
}

function draw1DCurve() {
  ctx.strokeStyle="#888";
  ctx.beginPath();
  for(let i=0;i<W;i++){
    const x=(i-W/2)/60;
    const y=f1(x);
    const py=H/2 - y*40;
    if(i===0) ctx.moveTo(i,py);
    else ctx.lineTo(i,py);
  }
  ctx.stroke();
}

function draw() {
  ctx.clearRect(0,0,W,H);

  if(mode==="2D") {
    title.innerHTML = "Optimizer Dynamics on z(x,y)=x²+0.3y²−0.5xy";
    draw2DSurface();

    started.forEach(k=>{
      ctx.strokeStyle=COLORS[k];
      ctx.beginPath();
      paths[k].forEach((p,i)=>{
        const cx=W/2+p[0]*80, cy=H/2-p[1]*80;
        if(i===0) ctx.moveTo(cx,cy);
        else ctx.lineTo(cx,cy);
      });
      ctx.stroke();
    });
  }

  else {
    title.innerHTML = "Optimizer Dynamics on f(x)=x⁵−3x³+x²+sin(x)";
    draw1DCurve();

    started.forEach(k=>{
      ctx.fillStyle=COLORS[k];
      paths[k].forEach(p=>{
        const cx=W/2+p*60;
        const cy=H/2 - f1(p)*40;
        ctx.beginPath();
        ctx.arc(cx,cy,3,0,2*Math.PI);
        ctx.fill();
      });
    });
  }
}

/* -------------------- STEP -------------------- */

function step(k) {
  const o=optimizers[k];

  if(mode==="2D") {
    const [gx,gy]=grad2(o.x2[0],o.x2[1]);
    o.x2[0]-=o.lr*gx;
    o.x2[1]-=o.lr*gy;
    paths[k].push([...o.x2]);
  }

  else {
    const g=grad1(o.x1);
    o.x1-=o.lr*g;
    paths[k].push(o.x1);
  }
}

/* -------------------- CONTROL -------------------- */

function animate() {
  if(!running) return;
  started.forEach(step);
  draw();
  requestAnimationFrame(animate);
}

function start() {
  reset();
  document.querySelectorAll("input[data-opt]").forEach(cb=>{
    if(cb.checked) started.add(cb.dataset.opt);
  });
  running=true;
  animate();
}

function reset() {
  running=false;
  started.clear();
  initOptimizers();
  draw();
}

function toggleMode() {
  reset();
  mode = mode==="2D" ? "1D" : "2D";
}

/* -------------------- INIT -------------------- */

draw();
