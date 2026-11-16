"""打包入口 / 一键启动器。

职责：起本地 Flask 服务 → 自动打开浏览器 → 日志落盘 → 崩溃弹窗。
开发/源码模式仍可用 `python app.py`（行为不变）。

要点：
- windowed 模式无控制台，sys.stdout/stderr 为 None，必须重定向到文件，
  否则第三方库（transformers/tqdm/werkzeug）一 print 就抛异常。
- 8000 被占用自动顺延；已在运行的实例直接复用（打开浏览器后退出）。
- 打包后资源在 _MEIPASS，模型在 exe 同级 models/（见 config.py）。
"""
import multiprocessing
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

APP_NAME = "语音鉴别"

# PyInstaller 打包后无 mp 支持时 freeze_support 是空操作，但有 torch 时必须有
multiprocessing.freeze_support()


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = _base_dir()
LOG_DIR = BASE_DIR / "logs"


def _setup_streams():
    """无控制台时把 stdout/stderr 写进日志文件，避免 None.write 崩溃。"""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    if sys.stdout is None or sys.stderr is None:
        f = open(LOG_DIR / "runtime.log", "a", encoding="utf-8", errors="ignore")
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "app.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    try:
        print(line, flush=True)
    except Exception:
        pass


def msgbox(text: str, title: str = APP_NAME, icon: int = 0x10):
    """Windows 原生弹窗（不依赖 tkinter）。"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, title, icon)
    except Exception:
        pass


def _is_our_app(port: int, timeout: float = 0.35) -> bool:
    """探测该端口是否已跑着本程序（用专属 ping 接口，不误判其它服务）。"""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/app/ping",
                                    timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "{}").get("app") == "voiceguard"
    except Exception:
        return False


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pick_port(start: int = 8000, tries: int = 20) -> int:
    """复用已运行实例 → 否则找第一个空闲端口；全被占用交给系统分配。"""
    for p in range(start, start + tries):
        if not _port_free(p):
            if _is_our_app(p):
                log(f"检测到已在运行的实例：{p} 端口，直接打开页面")
                webbrowser.open(f"http://127.0.0.1:{p}")
                sys.exit(0)
            continue
        return p
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _open_when_ready(port: int, url: str, timeout: float = 40.0):
    """等服务真正可连再开浏览器，避免打开空白页。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                time.sleep(0.4)  # 给模板首字节留一点余量
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
                return
        time.sleep(0.25)
    log("等待服务就绪超时，请手动打开 " + url)


def _warmup(delay: float = 1.5):
    """后台预热深度模型：让用户第一次点击就是秒回，而不是干等 20 秒。"""
    time.sleep(delay)
    try:
        from app import warmup_deep
        t0 = time.time()
        ok = warmup_deep()
        log(f"深度模型预热{'完成' if ok else '跳过'}（{time.time() - t0:.1f}s）")
    except Exception:
        log("预热失败（不影响使用）：\n" + traceback.format_exc())


def main():
    _setup_streams()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # 打包后子进程（如 torch 的 DataLoader）需要知道入口
    os.environ.setdefault("PYTHONUTF8", "1")

    if not (BASE_DIR / "models").is_dir():
        log("警告：未找到 models 目录，模型将不可用，请把 models 文件夹放在 exe 同级目录")

    log(f"启动 {APP_NAME}｜BASE={BASE_DIR}｜frozen={getattr(sys, 'frozen', False)}")

    try:
        from app import app
    except Exception:
        err = traceback.format_exc()
        log("导入 app 失败：\n" + err)
        msgbox("程序启动失败（模块导入错误）：\n\n" + err[-1200:] +
               "\n\n详情见 logs/app.log")
        sys.exit(1)

    port = _pick_port()
    url = f"http://127.0.0.1:{port}"

    if os.environ.get("VOICEGUARD_NO_WARMUP") != "1":
        threading.Thread(target=_warmup, daemon=True).start()

    threading.Thread(target=_open_when_ready, args=(port, url), daemon=True).start()
    log(f"服务地址 {url}（关闭本窗口即停止服务）")

    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True,
                use_reloader=False)
    except Exception:
        err = traceback.format_exc()
        log("服务异常退出：\n" + err)
        msgbox("服务异常退出：\n\n" + err[-1200:] + "\n\n详情见 logs/app.log")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        err = traceback.format_exc()
        log("未捕获异常：\n" + err)
        msgbox("程序异常：\n\n" + err[-1200:] + "\n\n详情见 logs/app.log")
        sys.exit(1)
