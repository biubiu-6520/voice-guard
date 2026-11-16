"""深度检测器：mo-thecreator/Deepfake-audio-detection（wav2vec2-base 二分类）。

设计原则：
- 本模块顶层不 import torch/transformers/huggingface_hub（惰性 import），
  app 启动零 torch 成本；只有首次下载/推理才加载。
- HF_ENDPOINT/HF_HOME 在模块顶部设置（早于任何 HF 库 import），
  setdefault 允许用系统环境变量覆盖。
- 一切失败都可降级：下载失败可重试、推理失败返回 DeepUnavailable，
  由 pipeline 捕获后降级为纯轻量模型，应用永远可用。
"""
import os
import threading
import time

from config import (DEEP_CHUNK_SECONDS, DEEP_ENABLED, DEEP_LOCAL_DIR,
                    DEEP_MAX_SECONDS, DEEP_MODEL_ID, HF_ENDPOINT, HF_HOME, SR)

# 必须早于 huggingface_hub/transformers 的任何 import；
# setdefault 允许用户用系统环境变量覆盖
os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
os.environ.setdefault("HF_HOME", str(HF_HOME))

# 下载所需文件（按仓库实际内容；safetensors 缺失时回退 pytorch_model.bin）
_REQUIRED_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


class DeepUnavailable(Exception):
    """深度模型不可用（未下载/下载中/加载失败/未装 torch），message 面向用户。"""


class DeepDetector:
    """状态机：not_downloaded → downloading → loading → ready / failed。"""

    def __init__(self, model_id=DEEP_MODEL_ID, local_dir=DEEP_LOCAL_DIR):
        self.model_id = model_id
        self.local_dir = local_dir
        self._pipe = None
        self._lock = threading.Lock()  # 推理串行锁，防并发打满 CPU
        self._load_lock = threading.Lock()  # 加载互斥，防预热与首次点击并发加载两份
        self._dl = {"state": "not_downloaded", "progress": None,
                    "detail": "", "error": None}
        self._dl_thread = None
        self.last_chunk_latency = 0.0
        self._refresh_local_state()

    # ---------- 状态与下载 ----------

    def _files_ready(self):
        if not (self.local_dir / "config.json").exists():
            return False
        if not (self.local_dir / "preprocessor_config.json").exists():
            return False
        if (self.local_dir / "model.safetensors").exists():
            return True
        return (self.local_dir / "pytorch_model.bin").exists()

    def _refresh_local_state(self):
        """纯本地探测：文件齐全即 ready，离线可用。
        下载中/加载中不覆盖状态（避免并发时状态回跳）。"""
        if self._dl["state"] in ("downloading", "loading"):
            return
        if not DEEP_ENABLED:
            self._dl["state"] = "disabled"
        elif self._files_ready():
            self._dl["state"] = "ready"
            self._dl["detail"] = "已下载"

    def status(self):
        s = dict(self._dl)
        s["busy"] = self._lock.locked()
        s["model_id"] = self.model_id
        return s

    def start_download(self):
        """幂等：已在下载/已就绪直接返回状态。"""
        self._refresh_local_state()
        if not DEEP_ENABLED:
            return self.status()
        if self._dl["state"] in ("ready", "loading"):
            return self.status()
        if self._dl["state"] == "downloading":
            return self.status()
        self._dl.update({"state": "downloading", "progress": 0,
                         "detail": "正在下载深度模型（约 360MB，国内镜像）", "error": None})
        self._dl_thread = threading.Thread(target=self._download_worker, daemon=True)
        self._dl_thread.start()
        return self.status()

    @staticmethod
    def _resolve_cdn(url):
        """手动跟随 302，拿到支持 Range 的 S3 签名 CDN URL。

        hf-mirror 的 resolve 本身不支持 Range（返回 200 全文），
        但其 302 跳转的 cas-bridge S3 URL 支持（206 + Content-Range）。
        """
        import urllib.parse
        import urllib.request

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        op = urllib.request.build_opener(_NoRedirect)
        try:
            op.open(urllib.request.Request(url, headers={"User-Agent": "voiceguard/1.0"}),
                    timeout=30)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                loc = e.headers.get("Location")
                return urllib.parse.urljoin(url, loc) if loc else None
            raise
        return None  # 无重定向（小文件可能直接返回）

    def _download_file(self, fname, idx, n_files):
        """多线程分块下载单个文件（每块独立断点续传），实时回写进度。

        镜像入口限速明显（单连接约 30-50KB/s）且不支持 Range，
        所以手动跟随 302 到 S3 CDN，用 N 个并行 Range 连接分块下载
        （实测每连接 ~30KB/s，6 并发合计 ~190KB/s）。签名 URL 约 1 小时
        过期，403 时自动重新签名续传。
        """
        import concurrent.futures
        import shutil
        import urllib.error
        import urllib.request

        resolve = f"{HF_ENDPOINT}/{self.model_id}/resolve/main/{fname}"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        dest = self.local_dir / fname
        if dest.exists():
            return
        # 只清理旧版单流式下载的残留文件；.parts 目录是分块续传状态，必须保留
        leftover = self.local_dir / (fname + ".part")
        if leftover.exists():
            try:
                leftover.unlink()
            except OSError:
                pass
        parts_dir = self.local_dir / (fname + ".parts")

        cdn = self._resolve_cdn(resolve) or resolve

        # 先拿总大小：CDN 支持 Range（Content-Range 带总大小）；
        # 拿不到（小文件走 chunked 传输）则退化为单流下载
        total = 0
        req = urllib.request.Request(cdn, headers={"User-Agent": "voiceguard/1.0",
                                                   "Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                cr = r.headers.get("Content-Range")
                if cr:
                    total = int(cr.split("/")[-1])
                elif r.headers.get("Content-Length"):
                    total = int(r.headers["Content-Length"])
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):  # 签名失效，重签一次
                cdn = self._resolve_cdn(resolve) or resolve
                with urllib.request.urlopen(
                        urllib.request.Request(cdn, headers={"User-Agent": "voiceguard/1.0",
                                                             "Range": "bytes=0-0"}),
                        timeout=30) as r:
                    cr = r.headers.get("Content-Range")
                    if cr:
                        total = int(cr.split("/")[-1])
        if not total:
            # 小文件：单流下载即可
            with urllib.request.urlopen(
                    urllib.request.Request(resolve,
                                           headers={"User-Agent": "voiceguard/1.0"}),
                    timeout=120) as r:
                with open(dest, "wb") as f:
                    while True:
                        c = r.read(256 * 1024)
                        if not c:
                            break
                        f.write(c)
            self._dl["progress"] = min(99, int((idx + 1) / n_files * 100))
            return

        try:
            parts_dir.mkdir()
        except FileExistsError:
            pass  # 复用旧分块（断点续传）
        except OSError:
            # 旧目录被占用（上次进程残留句柄）：换个名字
            parts_dir = self.local_dir / f"{fname}.parts.{os.getpid()}"
            parts_dir.mkdir()
        n_parts = 12 if total > 100 * 1024 * 1024 else 2
        chunk = (total + n_parts - 1) // n_parts

        def progress():
            done = sum(p.stat().st_size for p in parts_dir.iterdir())
            frac = (idx + min(1.0, done / total)) / n_files
            self._dl["progress"] = min(99, int(frac * 100))

        def dl_part(k, url):
            part = parts_dir / f"{k:02d}"
            a, b = k * chunk, min(total, (k + 1) * chunk)
            while True:
                have = part.stat().st_size if part.exists() else 0
                if have >= b - a:
                    return
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "voiceguard/1.0",
                                      "Range": f"bytes={a + have}-{b - 1}"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        if r.status != 206 and have > 0:
                            # 服务器忽略了 Range：分块数据会错位，直接失败
                            raise RuntimeError("镜像不支持断点续传")
                        with open(part, "ab") as f:
                            while have < b - a:
                                c = r.read(min(256 * 1024, b - a - have))
                                if not c:
                                    break
                                f.write(c)
                                have += len(c)
                                progress()
                    if have >= b - a:
                        return
                except urllib.error.HTTPError as e:
                    if e.code in (401, 403):
                        url = self._resolve_cdn(resolve) or resolve  # 重新签名
                    else:
                        raise
                except RuntimeError:
                    raise  # 致命错误（如镜像不支持 Range），交给上层报错
                except Exception:
                    pass  # 网络抖动：稍候重试（断点续传）
                time.sleep(2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parts) as ex:
            list(ex.map(dl_part, range(n_parts), [cdn] * n_parts))
        with open(dest, "wb") as out:
            for k in range(n_parts):
                out.write((parts_dir / f"{k:02d}").read_bytes())
        shutil.rmtree(parts_dir, ignore_errors=True)

    def _download_worker(self):
        import urllib.error

        for i, fname in enumerate(_REQUIRED_FILES):
            try:
                self._dl["detail"] = f"下载 {fname}（{i + 1}/{len(_REQUIRED_FILES)}）"
                self._download_file(fname, i, len(_REQUIRED_FILES))
            except Exception as e:
                # 仅当 safetensors 不存在（404）时尝试 pytorch_model.bin
                if fname == "model.safetensors" and isinstance(e, urllib.error.HTTPError) \
                        and e.code == 404:
                    try:
                        self._download_file("pytorch_model.bin", i, len(_REQUIRED_FILES))
                        continue
                    except Exception as e2:
                        e = e2
                self._dl.update({"state": "failed",
                                 "error": f"下载失败（{fname}）：{e}",
                                 "detail": "可点击重试，或参考 README 手动放置模型文件"})
                return
        self._dl.update({"state": "ready", "progress": 100,
                         "detail": "深度模型就绪", "error": None})

    # ---------- 推理 ----------

    def _load(self):
        """首次推理时加载 pipeline（torch import 3-8s + 权重 ~400MB 内存）。

        加锁：后台预热与用户首次点击并发时只加载一份，后来者等前者完成。
        """
        with self._load_lock:
            if self._pipe is not None:      # 另一线程已加载完成
                return
            self._dl["state"] = "loading"
            self._dl["detail"] = "正在加载深度模型到内存…"
            try:
                import torch
                from transformers import pipeline
            except ImportError as e:
                self._dl.update({"state": "failed",
                                 "error": f"未安装 PyTorch/transformers（{e}），"
                                          f"请重新运行 start.bat 安装依赖"})
                raise DeepUnavailable(self._dl["error"])
            try:
                torch.set_num_threads(min(4, os.cpu_count() or 1))
                self._pipe = pipeline("audio-classification", model=str(self.local_dir),
                                      top_k=None)
            except Exception as e:
                self._dl.update({"state": "failed",
                                 "error": f"模型加载失败：{e}"})
                raise DeepUnavailable(self._dl["error"])
            self._dl["state"] = "ready"
            self._dl["detail"] = "深度模型就绪"

    def preload(self):
        """后台预热（打包版启动后调用）：把首次加载的十几秒挪到用户点击之前。"""
        if self._pipe is not None:
            return False
        self._refresh_local_state()
        if self._dl["state"] != "ready":
            return False
        try:
            self._load()
            return True
        except DeepUnavailable:
            return False

    def predict(self, y, sr=SR):
        """真/假二分类。返回 {label_name, probs:[p_fake,p_real], chunks, truncated,
        latency_ms}。状态非 ready 抛 DeepUnavailable（pipeline 捕获后降级）。"""
        if not DEEP_ENABLED:
            raise DeepUnavailable("深度模型已停用")
        self._refresh_local_state()
        if self._dl["state"] != "ready":
            if self._dl["state"] == "downloading":
                raise DeepUnavailable(
                    f"深度模型正在下载（{self._dl['progress'] or 0}%），本次仅用轻量模型")
            if self._dl["state"] in ("not_downloaded", "failed"):
                raise DeepUnavailable(
                    self._dl["error"] or "深度模型未下载，本次仅用轻量模型")
        if self._pipe is None:
            self._load()

        t0 = time.time()
        n = len(y)
        truncated = n > sr * DEEP_MAX_SECONDS
        y = y[: sr * DEEP_MAX_SECONDS]
        n = len(y)
        chunk_len = DEEP_CHUNK_SECONDS * sr
        n_chunks = min(3, max(1, -(-n // chunk_len)))  # 等分块数：≤10s→1，≤20s→2，>20s→3
        boundaries = [int(round(n * k / n_chunks)) for k in range(n_chunks + 1)]

        chunks = []
        with self._lock:
            for k in range(n_chunks):
                a, b = boundaries[k], boundaries[k + 1]
                c0 = time.time()
                res = self._pipe({"array": y[a:b], "sampling_rate": int(sr)})
                self.last_chunk_latency = time.time() - c0
                probs = self._map_scores(res)
                chunks.append({
                    "start": round(a / sr, 2), "end": round(b / sr, 2),
                    "probs": probs, "latency_ms": int(self.last_chunk_latency * 1000),
                })

        # 按时长加权合成整体概率
        total = sum(c["end"] - c["start"] for c in chunks)
        probs = [sum(c["probs"][i] * (c["end"] - c["start"]) for c in chunks) / total
                 for i in range(2)]
        k = 0 if probs[0] >= probs[1] else 1
        return {
            "available": True,
            "label": k,
            "label_name": ("真人" if k == 1 else "AI合成"),
            "probs": [round(float(p), 4) for p in probs],
            "chunks": chunks,
            "truncated": truncated,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    @staticmethod
    def _map_scores(res):
        """pipeline 输出 [{label, score}] → [p_fake, p_real]。"""
        p_fake = p_real = 0.0
        for item in res:
            if str(item["label"]).lower() == "real":
                p_real = float(item["score"])
            elif str(item["label"]).lower() == "fake":
                p_fake = float(item["score"])
        if p_fake + p_real == 0:
            # 未知标签时按顺序兜底
            for item in res:
                if item["label"] in ("LABEL_1", "1", "real"):
                    p_real = float(item["score"])
            p_fake = 1.0 - p_real
        return [round(p_fake, 4), round(p_real, 4)]
