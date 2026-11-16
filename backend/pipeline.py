"""双模型编排：轻量（SVC 三分类）+ 深度（wav2vec2 二分类）→ 统一响应。

主结论规则：
- 深度模型可用 → 主结论取深度二分（真人/AI合成）；轻量检出变声痕迹则附加提示。
- 深度不可用 → 主结论取轻量三分类，warnings 标注精度有限。
"""
import time

from config import MAX_LIGHT_SECONDS, SR
from backend import features
from backend.deep_model import DeepDetector, DeepUnavailable
from backend.light_model import LightModel


def run_full_analysis(y, light: LightModel, deep: DeepDetector,
                      with_segments=True, original_duration=None,
                      extra_warnings=None):
    """返回统一响应结构（见前端 main.js 消费格式）。"""
    t0 = time.time()
    warnings = list(extra_warnings or [])

    frames = features.frame_features(y)  # 全局一次，extract/segments/visualize 共享

    # ---- 轻量模型：整段三分类 ----
    light_res = light.predict(y, frames) if light.available else {
        "available": False, "label_name": "", "probs": [], "advice": "",
        "error": light.error, "latency_ms": 0,
    }

    # ---- 深度模型：整段二分类（不可用则降级）----
    deep_res = None
    mode = "light_only"
    if deep.status()["state"] in ("ready", "loading"):
        try:
            deep_res = deep.predict(y)
            deep_res["used"] = True
            mode = "deep+light"
            if deep_res.get("truncated"):
                warnings.append("音频超过 30 秒，深度模型仅分析前 30 秒")
        except DeepUnavailable as e:
            warnings.append(f"深度模型不可用：{e}")
    else:
        s = deep.status()
        if s["state"] == "disabled":
            warnings.append("深度模型已停用，本次仅用轻量模型")
        elif s["state"] == "downloading":
            warnings.append(f"深度模型正在下载（{s['progress'] or 0}%），本次仅用轻量模型")
        elif s["state"] == "failed":
            warnings.append(f"深度模型不可用（{s['error']}），本次仅用轻量模型")
        else:
            warnings.append("深度模型未下载，本次仅用轻量模型（精度有限）")

    # ---- 分段滑窗（轻量模型，可视化用）----
    segments = light.predict_segments(y, frames) if (with_segments and light.available) \
        else {"available": False, "points": []}

    # ---- 主结论 ----
    verdict = _verdict_rule(light_res, deep_res)

    mel, wave = features.visualize(y, frames=frames)
    return {
        "ok": True,
        "mode": mode,
        "duration": round(len(y) / SR, 2),
        "original_duration": original_duration,
        "latency_ms": int((time.time() - t0) * 1000),
        "verdict": verdict,
        "light": light_res,
        "deep": deep_res,
        "segments": segments,
        "warnings": warnings,
        "mel": mel,
        "wave": wave,
    }


def _verdict_rule(light, deep):
    """主结论合成。deep 可用取 deep 二分，否则取 light 三分类。"""
    hints = []
    if deep is not None:
        is_real = deep["label"] == 1
        label = "真人" if is_real else "AI合成"
        conf = deep["probs"][1 if is_real else 0]
        advice = ("未发现明显的合成痕迹。" if is_real
                  else "疑似语音合成/克隆，注意声码器伪造迹象。")
        if light.get("available") and light["probs"][2] > 0.5:
            hints.append("轻量模型检出变声痕迹，建议人工复核")
    else:
        if not light.get("available"):
            return {"label": "模型不可用", "is_real": None, "conf": 0,
                    "hints": ["轻量模型加载失败，请检查 models/model.joblib"],
                    "advice": "两个模型均不可用，无法鉴别。"}
        label = light["label_name"]
        is_real = label == "真人"
        k = light["label"]
        conf = light["probs"][k]
        advice = light["advice"]
    return {"label": label, "is_real": is_real, "conf": round(float(conf), 4),
            "hints": hints, "advice": advice}
