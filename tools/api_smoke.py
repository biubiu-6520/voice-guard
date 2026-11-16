"""API 冒烟测试脚本（curl 在本机证书不可用，用 urllib 替代）。

用法（先启动 app.py，另开窗口）：
  python tools/api_smoke.py status            查看模型状态
  python tools/api_smoke.py download          触发深度模型下载
  python tools/api_smoke.py predict 文件路径   上传音频分析
  python tools/api_smoke.py demo real         分析内置样例（real/ai/vc）
"""
import json
import sys
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000"


def call(method, path, data=None, headers=None, raw=False):
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            body = r.read()
            return r.status, (body if raw else json.loads(body))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def multipart(fields):
    """构造 multipart/form-data 请求体。fields: {name: (filename, bytes, ctype)}"""
    b = uuid.uuid4().hex
    parts = []
    for name, (fname, content, ctype) in fields.items():
        parts.append(
            f"--{b}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
            + content + b"\r\n")
    body = b"".join(parts) + f"--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "status":
        code, d = call("GET", "/api/model/status")
        print(code, json.dumps(d, ensure_ascii=False, indent=1))
    elif cmd == "download":
        code, d = call("POST", "/api/model/download")
        print(code, json.dumps(d, ensure_ascii=False, indent=1)[:500])
    elif cmd in ("predict", "demo"):
        if cmd == "demo":
            code, d = call("GET", f"/api/demo/{sys.argv[2] if len(sys.argv) > 2 else 'real'}")
        else:
            path = sys.argv[2]
            body, headers = multipart(
                {"audio": (path.replace("\\", "/").rsplit("/", 1)[-1],
                           open(path, "rb").read(), "application/octet-stream")})
            code, d = call("POST", "/api/predict", data=body, headers=headers)
        if code != 200:
            print("HTTP", code, json.dumps(d, ensure_ascii=False))
            return
        out = {k: d[k] for k in ("mode", "duration", "original_duration", "latency_ms",
                                 "verdict", "warnings")}
        out["light"] = {k: d["light"].get(k) for k in ("available", "label_name", "probs")}
        if d.get("deep"):
            out["deep"] = {k: d["deep"].get(k) for k in
                           ("label_name", "probs", "truncated", "latency_ms")}
        out["segments"] = {"available": d["segments"].get("available"),
                           "n_points": len(d["segments"].get("points", [])),
                           "first_scores": [p["score"] for p in d["segments"].get("points", [])[:5]]}
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        print("未知子命令:", cmd)


if __name__ == "__main__":
    main()
