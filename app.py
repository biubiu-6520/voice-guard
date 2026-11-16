"""语音真伪鉴别 · 混合模型版（轻量 SVC + 深度 wav2vec2）

运行：python app.py
然后浏览器打开 http://127.0.0.1:8000
"""
from flask import Flask, jsonify, render_template, request

from backend import audio, pipeline
from backend.audio import AudioError
from backend.deep_model import DeepDetector
from backend.light_model import LightModel
from config import (DEEP_ENABLED, MAX_FILE_MB, MAX_LIGHT_SECONDS, SAMPLES_DIR,
                    STATIC_DIR, TEMPLATES_DIR)

# 显式指定模板/静态目录：打包后 Flask 的 root_path 不再等于项目目录
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024  # 超限自动 413

_light = LightModel()          # 启动即载（joblib 秒级；失败降级为不可用）
_deep = DeepDetector()         # 只建状态机，不 import torch

# 样例白名单：名称 → 文件（samples 目录缺失时不崩，仅无内置样例）
DEMO_FILES = {p.stem: p.name for p in SAMPLES_DIR.iterdir()
              if p.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg")} \
    if SAMPLES_DIR.is_dir() else {}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    f = request.files.get("audio")
    if f is None:
        return jsonify({"ok": False, "error": "没有收到音频文件"}), 400
    try:
        y, info = audio.read_audio(f.read(), max_seconds=MAX_LIGHT_SECONDS)
    except AudioError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    res = pipeline.run_full_analysis(
        y, _light, _deep,
        original_duration=info["original_duration"],
        extra_warnings=info["warnings"],
    )。
    if res["verdict"]["label"] == "模型不可用":
        return jsonify(res), 503
    return jsonify(res)


@app.get("/api/demo/<name>")
def demo(name):
    fname = DEMO_FILES.get(name)
    if fname is None:
        return jsonify({"ok": False, "error": "未知样例"}), 404
    try:
        y, info = audio.read_audio((SAMPLES_DIR / fname).read_bytes(),
                                   max_seconds=MAX_LIGHT_SECONDS)
    except AudioError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    res = pipeline.run_full_analysis(y, _light, _deep,
                                     original_duration=info["original_duration"],
                                     extra_warnings=info["warnings"])
    return jsonify(res)


def warmup_deep():
    """启动器在后台线程调用：预加载深度模型 + 跑一遍完整链路。

    只加载模型还不够：首次推理还要付 torch 首帧 kernel、numba JIT 编译的
    十几秒成本。用 3 秒底噪跑一次全链路，把这些开销挪到用户点击之前。
    """
    if not _deep.preload():
        return False
    try:
        import numpy as np
        y = (np.random.standard_normal(48000) * 0.02).astype(np.float32)
        pipeline.run_full_analysis(y, _light, _deep)  # 特征/sklearn/深度/torch 全预热
    except Exception as e:  # 预热失败不影响正常使用
        app.logger.warning("预热推理失败：%s", e)
    return True


@app.get("/api/app/ping")
def app_ping():
    """启动器用来判断端口上是不是本程序（避免误判其它服务）。"""
    import sys as _sys
    return jsonify({"app": "voiceguard", "frozen": bool(getattr(_sys, "frozen", False))})


@app.get("/api/model/status")
def model_status():
    return jsonify({
        "deep": _deep.status(),
        "light": {"available": _light.available, "error": _light.error},
        "deep_enabled": DEEP_ENABLED,
    })


@app.post("/api/model/download")
def model_download():
    if not DEEP_ENABLED:
        return jsonify({"ok": False, "error": "深度模型已停用"}), 403
    return jsonify({"ok": True, "status": _deep.start_download()}), 202


@app.post("/api/app/quit")
def app_quit():
    """打包版专用：页面上一键停服务（源码模式请用 Ctrl+C，避免误杀开发服务）。"""
    import os
    import sys as _sys
    import threading
    import time
    if not getattr(_sys, "frozen", False):
        return jsonify({"ok": False, "error": "源码模式请用 Ctrl+C 停止"}), 403

    def _die():
        time.sleep(0.5)     # 先把响应发出去
        os._exit(0)

    threading.Thread(target=_die, daemon=True).start()
    return jsonify({"ok": True})


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"ok": False, "error": f"文件超过 {MAX_FILE_MB}MB 上限"}), 413


@app.errorhandler(500)
def server_error(e):
    app.logger.exception(e)
    return jsonify({"ok": False, "error": "服务器内部错误，请查看启动窗口日志"}), 500


if __name__ == "__main__":
    print("模型与服务已就绪，浏览器打开 http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)
