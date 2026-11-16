/* 上传 / 样例 / 录音 / 模型状态 / 结果渲染 */
const $ = s => document.querySelector(s);
const LIGHT_LABELS = ["真人", "AI合成", "变声"];
const DEEP_LABELS = ["AI合成", "真人"];

const drop = $("#drop"), file = $("#file"), meta = $("#meta");
let lastStatus = null;
let lastResult = null;

/* ---------- 上传 ---------- */

drop.onclick = () => file.click();
file.onchange = () => file.files[0] && sendAudio(file.files[0], file.files[0].name);
["dragenter","dragover"].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.add("over");
}));
["dragleave","drop"].forEach(e => drop.addEventListener(e, ev => {
  ev.preventDefault(); drop.classList.remove("over");
}));
drop.addEventListener("drop", ev => {
  const f = ev.dataTransfer.files[0]; if (f) sendAudio(f, f.name);
});

document.querySelectorAll("[data-demo]").forEach(b => {
  b.onclick = () => {
    if (!ensureIdle()) return;
    setLoading(true, "正在加载样例并分析…");
    meta.textContent = "正在加载样例…";
    fetch("/api/demo/" + b.dataset.demo).then(r => r.json()).then(d => {
      if (d.error) return fail(d.error);
      meta.textContent = "样例音频";
      render(d);
    }).catch(() => fail("无法连接服务"));
  };
});

function ensureIdle() {
  if (lastStatus && lastStatus.deep && lastStatus.deep.busy) {
    fail("深度模型正在处理另一段音频，请稍候几秒再试");
    return false;
  }
  return true;
}

function sendAudio(blobOrFile, name) {
  if (!ensureIdle()) return;
  const deepReady = lastStatus && (lastStatus.deep.state === "ready" || lastStatus.deep.state === "loading");
  setLoading(true, deepReady
    ? "分析中…（深度模型 CPU 推理，长音频约 10~30 秒，请勿关闭页面）"
    : "分析中…（轻量模型）");
  meta.textContent = `正在分析 ${name}（${(blobOrFile.size/1024/1024).toFixed(2)} MB）…`;
  const fd = new FormData(); fd.append("audio", blobOrFile, name);
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), 180000); // 深度推理保底超时 3 分钟
  fetch("/api/predict", { method: "POST", body: fd, signal: ctl.signal })
    .then(r => r.json())
    .then(d => { clearTimeout(timer); d.error ? fail(d.error) : render(d); })
    .catch(e => { clearTimeout(timer); fail(e.name === "AbortError" ? "分析超时（CPU 过慢？），请尝试更短的音频" : "无法连接服务，请确认 app.py 正在运行"); });
}

function fail(msg) {
  setLoading(false);
  $("#empty").hidden = false;
  $("#empty").textContent = msg;
  $("#empty").style.color = "#b42318";
  meta.textContent = "";
}

function setLoading(on, text) {
  $("#loading").hidden = !on;
  $("#loading").textContent = text || "分析中…";
  $("#empty").hidden = on || !$("#resultBox").hidden;
  if (on) { $("#resultBox").hidden = true; }
}

/* ---------- 结果渲染 ---------- */

function render(d) {
  lastResult = d;
  $("#loading").hidden = true;
  $("#empty").hidden = true;
  $("#resultBox").hidden = false;

  const v = d.verdict;
  const badge = $("#badge");
  badge.textContent = v.label;
  badge.className = "badge " + (v.label === "真人" ? "b0" : v.label === "AI合成" ? "b1" : "b2");
  $("#conf").textContent = ((v.conf || 0) * 100).toFixed(1) + "%";
  $("#advice").textContent = v.advice;

  const total = (d.latency_ms / 1000).toFixed(1);
  $("#modeLine").textContent = d.mode === "deep+light"
    ? `本次分析：深度 + 轻量模型 · 总耗时 ${total}s`
    : `本次分析：仅轻量模型 · 总耗时 ${total}s`;

  // 提示与警告
  const hints = $("#hints");
  if (v.hints && v.hints.length) {
    hints.hidden = false;
    hints.innerHTML = v.hints.map(h => `<div>⚠ ${h}</div>`).join("");
  } else hints.hidden = true;
  const warn = $("#warnings");
  if (d.warnings && d.warnings.length) {
    warn.hidden = false;
    warn.innerHTML = d.warnings.map(w => `<div>· ${w}</div>`).join("");
  } else warn.hidden = true;

  // 概率条：深度（二分）
  const bg = $("#barGroupDeep");
  if (d.deep && d.deep.available) {
    bg.hidden = false;
    bg.querySelector(".group-label").textContent =
      `深度模型 · wav2vec2 真伪二分类（耗时 ${(d.deep.latency_ms/1000).toFixed(1)}s）`;
    $("#barsDeep").innerHTML = d.deep.probs.map((p, i) => barRow(DEEP_LABELS[i], p, i)).join("");
  } else {
    bg.hidden = false;
    bg.querySelector(".group-label").textContent = "深度模型 · 未参与本次分析";
    $("#barsDeep").innerHTML = "";
  }
  // 概率条：轻量（三分类）
  if (d.light && d.light.available) {
    $("#lightGroupLabel").textContent = `轻量模型 · SVM 三分类（离线兜底，耗时 ${d.light.latency_ms}ms）`;
    $("#barsLight").innerHTML = d.light.probs.map((p, i) => barRow(LIGHT_LABELS[i], p, i)).join("");
  } else {
    $("#lightGroupLabel").textContent = "轻量模型 · 不可用";
    $("#barsLight").innerHTML = `<div class="bar-label" style="color:#b42318">${d.light && d.light.error || "加载失败"}</div>`;
  }

  // 可视化
  drawWave(d.wave);
  if (d.segments && d.segments.available && d.segments.points.length) {
    drawScoreOverlay(d.wave, d.segments.points, d.deep && d.deep.chunks, d.duration);
  }
  drawMel(d.mel);

  const dur = d.original_duration != null ? d.original_duration : d.duration;
  meta.textContent = `时长 ${dur}s · 分段 ${d.segments && d.segments.points ? d.segments.points.length : 0} 点`;
  refreshStatus();
}

function barRow(label, p, i) {
  return `<div class="bar-row">
    <span class="bar-label">${label}</span>
    <span class="track"><span class="fill f${i}" style="width:${(p*100).toFixed(1)}%"></span></span>
    <span class="bar-val">${(p*100).toFixed(1)}%</span>
  </div>`;
}

/* ---------- 模型状态与下载 ---------- */

async function refreshStatus() {
  try {
    const r = await fetch("/api/model/status");
    const d = await r.json();
    lastStatus = d;
    renderStatus(d);
    if (d.deep && d.deep.state === "downloading") {
      setTimeout(refreshStatus, 2000); // 下载中每 2s 轮询进度
    }
  } catch (_e) { /* 服务未起时不打扰 */ }
}

function renderStatus(d) {
  const s = d.deep || {};
  const dot = $("#deepDot"), st = $("#deepStatus"), btn = $("#deepBtn"),
        wrap = $("#dlWrap"), fill = $("#dlFill");
  dot.className = "status-dot " +
    (s.state === "ready" || s.state === "loading" ? "ok"
     : s.state === "failed" || s.state === "disabled" ? "err"
     : s.state === "downloading" ? "" : "");
  const busy = s.busy ? " · 推理中…" : "";
  const texts = {
    not_downloaded: "未下载（结果仅来自轻量模型）",
    downloading: `下载中 ${s.progress != null ? s.progress + "%" : ""}`,
    loading: "加载到内存中…",
    ready: "就绪" + busy,
    failed: "不可用（可重试）",
    disabled: "已停用",
  };
  st.textContent = texts[s.state] || "未知状态";
  btn.hidden = !(s.state === "not_downloaded" || s.state === "failed");
  btn.textContent = s.state === "failed" ? "重试" : "下载";
  if (s.state === "downloading") {
    wrap.hidden = false;
    fill.style.width = (s.progress != null ? s.progress : 5) + "%";
  } else {
    wrap.hidden = true;
    fill.style.width = "0";
  }

  const l = d.light || {};
  $("#lightDot").className = "status-dot " + (l.available ? "ok" : "err");
  $("#lightStatus").textContent = l.available ? "就绪" : "不可用";
}

$("#deepBtn").onclick = async () => {
  $("#deepBtn").disabled = true;
  try {
    await fetch("/api/model/download", { method: "POST" });
  } catch (_e) { /* 状态轮询会显示失败 */ }
  $("#deepBtn").disabled = false;
  refreshStatus();
};

/* ---------- 录音 ---------- */

const recBtn = $("#recBtn"), recTime = $("#recTime"), recLive = $("#recLive"),
      recDone = $("#recDone"), recAudio = $("#recAudio"),
      levelTrack = $("#levelTrack"), levelFill = $("#levelFill"),
      recDeviceWrap = $("#recDeviceWrap"), recDevice = $("#recDevice");
let chosenDeviceId = null;

function setInputsDisabled(on) {
  drop.classList.toggle("disabled", on);
  document.querySelectorAll("[data-demo]").forEach(b => b.disabled = on);
  file.disabled = on;
}

recBtn.onclick = async () => {
  if (recBtn.classList.contains("recording")) {
    Recorder.stop();
    onRecEnd(Recorder.getResult());
    return;
  }
  try {
    await Recorder.start(
      (s, t) => { recTime.textContent = t; },
      onRecEnd,
      peak => {  // 实时音量条：峰值 0.25 映射满格
        levelFill.style.width = Math.min(100, peak / 0.25 * 100).toFixed(0) + "%";
      },
      { deviceId: chosenDeviceId }
    );
    recBtn.textContent = "■ 停止录音";
    recBtn.classList.add("recording");
    recLive.hidden = false;
    levelTrack.hidden = false;
    levelFill.style.width = "0";
    recDone.hidden = true;
    recDevice.disabled = true;  // 录音中不切设备
    setInputsDisabled(true);
    $("#empty").hidden = true;
    meta.textContent = "";
    await refreshDevices();   // 拿到权限后设备列表才有标签
  } catch (e) {
    fail(e.message);
  }
};

/* 填充输入设备下拉框；无设备时给出提示 */
async function refreshDevices() {
  try {
    const devices = await Recorder.listDevices();
    if (!devices.length) {
      recDeviceWrap.hidden = false;
      recDevice.innerHTML = "<option>未检测到麦克风设备</option>";
      recDevice.disabled = true;
      return;
    }
    const prev = chosenDeviceId || recDevice.value;
    recDevice.innerHTML = devices.map((d, i) =>
      `<option value="${d.deviceId}">${d.label || "麦克风 " + (i + 1)}${d.deviceId === "default" || (i === 0 && !prev) ? "（默认）" : ""}</option>`
    ).join("");
    recDevice.disabled = false;
    if (prev && devices.some(d => d.deviceId === prev)) recDevice.value = prev;
    recDeviceWrap.hidden = false;
  } catch (_e) {
    recDeviceWrap.hidden = true;
  }
}

recDevice.onchange = () => {
  chosenDeviceId = recDevice.value;
  if (chosenDeviceId === "default") chosenDeviceId = null;
  meta.textContent = "已切换输入设备，点击「开始录音」生效";
};

function onRecEnd(r) {
  recBtn.textContent = "● 开始录音";
  recBtn.classList.remove("recording");
  recLive.hidden = true;
  levelTrack.hidden = true;
  recDevice.disabled = false;
  setInputsDisabled(false);
  if (!r || !r.seconds) return;
  recTime.textContent = Recorder.fmt(r.seconds);
  recAudio.src = URL.createObjectURL(r.blob);
  recDone.hidden = false;
  window.__rec = r;
  if (r.silent) {
    meta.textContent = "⚠ 没有录到声音：请用上方的「输入设备」下拉框换一个设备（常见原因是选了立体声混音/显示器音频），或检查系统声音设置里的输入设备";
    meta.style.color = "#b42318";
  } else {
    meta.textContent = r.quiet
      ? "⚠ 录音音量偏低，可能影响判断（建议靠近麦克风重录）"
      : "";
    meta.style.color = "";
  }
}

$("#recUse").onclick = () => {
  if (window.__rec) sendAudio(window.__rec.blob, `录音_${window.__rec.seconds}s.wav`);
};
$("#recRedo").onclick = () => {
  recDone.hidden = true;
  if (recAudio.src) URL.revokeObjectURL(recAudio.src);
  recAudio.removeAttribute("src");
  window.__rec = null;
  recTime.textContent = "00:00";
};

/* ---------- 窗口缩放重绘 ---------- */

window.addEventListener("resize", () => {
  if (lastResult && !$("#resultBox").hidden) {
    drawWave(lastResult.wave);
    if (lastResult.segments && lastResult.segments.available && lastResult.segments.points.length) {
      drawScoreOverlay(lastResult.wave, lastResult.segments.points,
                       lastResult.deep && lastResult.deep.chunks, lastResult.duration);
    }
    drawMel(lastResult.mel);
  }
});

refreshStatus();

/* ---------- 打包版：退出程序（源码模式不显示） ---------- */
fetch("/api/app/ping").then(r => r.json()).then(d => {
  const btn = $("#quitBtn");
  if (!btn || !d || !d.frozen) return;
  btn.hidden = false;
  btn.onclick = () => {
    if (!confirm("确定退出语音鉴别吗？服务停止后需手动关闭此页面。")) return;
    btn.disabled = true;
    btn.textContent = "正在退出…";
    fetch("/api/app/quit", { method: "POST" }).then(() => {
      btn.textContent = "已退出，可关闭此页";
    }).catch(() => { btn.textContent = "已退出，可关闭此页"; });
  };
}).catch(() => {});
