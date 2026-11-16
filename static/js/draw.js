/* canvas 绘制：波形、log-mel 频谱、分段疑似合成度叠加 */
function fit(cv, cssH) {
  const w = cv.clientWidth, dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(cssH * dpr);
  cv.style.height = cssH + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h: cssH };
}

function drawWave(env) {
  const { ctx, w, h } = fit($("#wave"), 64);
  ctx.clearRect(0, 0, w, h);
  const n = env.length, mid = h / 2, step = w / (n - 1);
  ctx.fillStyle = "rgba(0,47,167,.72)"; // 克莱因蓝
  ctx.beginPath();
  ctx.moveTo(0, mid);
  for (let i = 0; i < n; i++) {
    const a = Math.min(1, env[i] * 1.8) * (mid - 4);
    ctx.lineTo(i * step, mid - a);
  }
  for (let i = n - 1; i >= 0; i--) {
    const a = Math.min(1, env[i] * 1.8) * (mid - 4);
    ctx.lineTo(i * step, mid + a);
  }
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = "#dfe5ea";
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
}

/* 在波形上叠加：分段疑似合成度（红色面积，0~1 映射到底部高度）+ 深度分块虚线。
   注意：不能调 fit()（会重置 canvas 尺寸、清掉已画的波形），
   直接复用 drawWave 设定好的尺寸。 */
function drawScoreOverlay(env, points, chunks, duration) {
  if (!points || !points.length) return;
  const cv = $("#wave");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.width / dpr, h = cv.height / dpr;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const n = env.length, step = w / (n - 1);

  // 深度分块边界虚线
  if (chunks && chunks.length > 1) {
    ctx.strokeStyle = "rgba(0,47,167,.55)";
    ctx.setLineDash([4, 4]);
    chunks.forEach((c, i) => {
      if (i === 0) return;
      const x = (c.start / duration) * w;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    });
    ctx.setLineDash([]);
  }

  // 疑似合成度面积：把滑窗点插值到 500 点轴
  const score = new Float32Array(n).fill(NaN);
  points.forEach(p => {
    const i = Math.min(n - 1, Math.max(0, Math.round((p.t / duration) * (n - 1))));
    score[i] = p.score;
  });
  let last = NaN;
  for (let i = 0; i < n; i++) {
    if (!isNaN(score[i])) last = score[i];
    else score[i] = last;
  }
  for (let i = n - 1; i >= 0; i--) {
    if (isNaN(score[i])) score[i] = last; else last = score[i];
  }
  if (isNaN(score[0])) return;

  ctx.fillStyle = "rgba(180,35,24,.40)";
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = 0; i < n; i++) {
    ctx.lineTo(i * step, h - Math.min(1, score[i]) * (h - 6));
  }
  ctx.lineTo(w, h);
  ctx.closePath();
  ctx.fill();
}

/* 克莱因蓝系 colormap：深海军蓝 → 克莱因蓝 → 亮蓝 → 浅蓝 → 淡黄 */
const STOPS = [[8,14,44],[0,47,167],[38,99,235],[96,165,250],[255,235,150]];
function cmap(t) {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
}

function drawMel(mel) {
  const cv = $("#mel");
  const { ctx, w, h } = fit(cv, 150);
  const rows = mel.length, cols = mel[0].length;
  const off = document.createElement("canvas");
  off.width = cols; off.height = rows;
  const octx = off.getContext("2d");
  const img = octx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const [R, G, B] = cmap(mel[rows - 1 - r][c]);
      const idx = (r * cols + c) * 4;
      img.data[idx] = R; img.data[idx+1] = G; img.data[idx+2] = B; img.data[idx+3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, w, h);
}
