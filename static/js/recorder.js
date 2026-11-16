/* 在线录音：getUserMedia + AudioContext(16k) + ScriptProcessor 收 PCM，
   停止时手写 WAV 编码（纯前端，不依赖 ffmpeg / 服务器）。
   说明：ScriptProcessor 已标记废弃但全浏览器可用；AudioWorklet 需额外模块文件，
   这里取前者（现代浏览器仍长期支持）。 */
const Recorder = (() => {
  let ctx = null, node = null, srcNode = null, mute = null, stream = null;
  let chunks = [], sr = 16000, secs = 0, timer = null, levelCb = null;
  let peakTotal = 0, peakCount = 0;
  const MAX_SECONDS = 60;

  function fmt(s) {
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" +
           String(Math.floor(s % 60)).padStart(2, "0");
  }

  async function start(onTick, onEnd, onLevel, opts) {
    stop(); // 兜底清理
    let media;
    const audioCfg = { channelCount: 1, echoCancellation: false,
                       noiseSuppression: false, autoGainControl: false };
    if (opts && opts.deviceId) audioCfg.deviceId = { exact: opts.deviceId };
    try {
      media = await navigator.mediaDevices.getUserMedia({ audio: audioCfg });
    } catch (e) {
      const msg = e.name === "NotAllowedError" || e.name === "SecurityError"
        ? "麦克风权限被拒绝：请点击地址栏左侧的锁形图标允许麦克风访问后重试"
        : e.name === "NotFoundError"
          ? "未检测到麦克风设备（请确认已插入麦克风，并在系统声音设置里启用）"
          : e.name === "OverconstrainedError"
            ? "所选输入设备不可用，请在下拉框换一个设备再试"
            : `无法访问麦克风（${e.name}）`;
      throw new Error(msg);
    }
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    } catch (_e) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
    }
    sr = ctx.sampleRate;
    stream = media;
    chunks = [];
    secs = 0;
    levelCb = onLevel || null;
    peakTotal = 0; peakCount = 0;
    const src = ctx.createMediaStreamSource(media);
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = ev => {
      const data = new Float32Array(ev.inputBuffer.getChannelData(0));
      chunks.push(data);
      // 实时音量：块内峰值
      let peak = 0;
      for (let i = 0; i < data.length; i++) {
        const a = Math.abs(data[i]);
        if (a > peak) peak = a;
      }
      peakTotal += peak; peakCount++;
      if (levelCb) levelCb(peak);
    };
    src.connect(node);
    // 必须连接到 destination 才会触发回调；接零增益节点，避免扬声器回授
    mute = ctx.createGain();
    mute.gain.value = 0;
    node.connect(mute);
    mute.connect(ctx.destination);
    srcNode = src;
    timer = setInterval(() => {
      secs++;
      onTick && onTick(secs, fmt(secs));
      if (secs >= MAX_SECONDS) { stop(); onEnd && onEnd(getResult()); }
    }, 1000);
    onTick && onTick(0, fmt(0));
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    if (srcNode) { try { srcNode.disconnect(); } catch (_e) {} srcNode = null; }
    if (node) { try { node.disconnect(); } catch (_e) {} node = null; }
    if (mute) { try { mute.disconnect(); } catch (_e) {} mute = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (ctx) { try { ctx.close(); } catch (_e) {} ctx = null; }
    levelCb = null;
  }

  function concat() {
    let total = 0;
    chunks.forEach(c => total += c.length);
    const out = new Float32Array(total);
    let o = 0;
    chunks.forEach(c => { out.set(c, o); o += c.length; });
    return out;
  }

  /* 线性重采样（源采样率 sr → 16000），浏览器给什么采样率都能对齐 */
  function resample(pcm, from, to) {
    if (from === to) return pcm;
    const n = Math.floor(pcm.length * to / from);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const pos = i * from / to;
      const j = Math.floor(pos), f = pos - j;
      out[i] = pcm[j] * (1 - f) + (j + 1 < pcm.length ? pcm[j + 1] : pcm[j]) * f;
    }
    return out;
  }

  function encodeWav(pcm) {
    const data = resample(pcm, sr, 16000);
    const buf = new ArrayBuffer(44 + data.length * 2);
    const v = new DataView(buf);
    const writeStr = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    writeStr(0, "RIFF"); v.setUint32(4, 36 + data.length * 2, true); writeStr(8, "WAVE");
    writeStr(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, 16000, true);
    v.setUint32(28, 16000 * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    writeStr(36, "data"); v.setUint32(40, data.length * 2, true);
    for (let i = 0; i < data.length; i++) {
      const s = Math.max(-1, Math.min(1, data[i]));
      v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([buf], { type: "audio/wav" });
  }

  function getResult() {
    const blob = encodeWav(concat());
    const avgPeak = peakCount ? peakTotal / peakCount : 0;
    return {
      blob,
      seconds: secs || (chunks.reduce((s, c) => s + c.length, 0) / sr),
      silent: !chunks.length || avgPeak < 1e-4,   // 没录到声音
      quiet: avgPeak < 0.02 && avgPeak >= 1e-4,   // 音量偏低
    };
  }

  /* 列出输入设备。注意：未授予麦克风权限时拿不到设备标签（label 为空） */
  async function listDevices() {
    const all = await navigator.mediaDevices.enumerateDevices();
    return all.filter(d => d.kind === "audioinput");
  }

  return { start, stop, getResult, listDevices, MAX_SECONDS, fmt };
})();
