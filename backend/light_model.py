"""轻量模型：SVC 三分类（真人/AI合成/变声）。

离线兜底 + 「变声」细分提示。加载失败只标记不可用，绝不阻止应用启动。
"""
import time
import warnings

import joblib

from config import ADVICE, LABELS, LIGHT_MODEL_PATH, SEG_HOP, SEG_WINDOW
from backend import features


class LightModel:
    def __init__(self, path=LIGHT_MODEL_PATH):
        self.available = False
        self.error = None
        try:
            # 模型已用当前 sklearn 版本重新序列化，这里再抑制一次以防环境变动
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.clf = joblib.load(path)
            self.available = True
        except Exception as e:
            self.error = str(e)

    def _predict_proba(self, x):
        return self.clf.predict_proba([x])[0]

    def predict(self, y, frames=None):
        """整段三分类。返回 {available, label, label_name, probs, advice, latency_ms}。
        frames 传入则复用帧级特征。"""
        t0 = time.time()
        if not self.available:
            return {"available": False, "error": self.error,
                    "label_name": "", "probs": [], "advice": "", "latency_ms": 0}
        probs = self._predict_proba(features.extract(y, frames))
        k = int(probs.argmax())
        return {
            "available": True,
            "label": k,
            "label_name": LABELS[k],
            "probs": [float(p) for p in probs],
            "advice": ADVICE[LABELS[k]],
            "latency_ms": int((time.time() - t0) * 1000),
        }

    def predict_segments(self, y, frames=None):
        """2s 窗 / 0.5s 步滑窗打分。返回 {window, hop, points:[{t, score, probs}]}。

        score = P(AI合成) + P(变声)，即该段「疑似伪造度」0~1。
        帧级特征全局一次计算 + 按帧切片聚合（60s 音频 <3s）。
        frames 传入则复用帧级特征。
        """
        t0 = time.time()
        if not self.available:
            return {"available": False, "points": []}
        frames = features.frame_features(y) if frames is None else frames
        contour = features._f0_contour(y)
        win, hop = int(SEG_WINDOW * 16000), int(SEG_HOP * 16000)
        n = len(y)
        points = []

        def make(a, b):
            x = features.aggregate(frames, y, a, b, f0_src=contour)
            probs = self._predict_proba(x)
            points.append({
                "t": round((a + b) / 2 / 16000, 2),
                "start": round(a / 16000, 2),
                "end": round(b / 16000, 2),
                "score": round(float(probs[1] + probs[2]), 4),
                "probs": [round(float(p), 4) for p in probs],
            })

        if n < win + hop:      # 音频太短不足一窗，整段作一个点
            if n >= 8000:
                make(0, n)
        else:
            start = 0
            while start + win <= n:
                make(start, start + win)
                start += hop
            if n - start >= 8000:  # 尾部不足一窗但 >= 0.5s，并入最后一窗
                make(start, n)
        return {"available": True, "window": SEG_WINDOW, "hop": SEG_HOP,
                "points": points, "latency_ms": int((time.time() - t0) * 1000)}
