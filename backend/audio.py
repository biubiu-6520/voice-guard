"""音频读取与预处理。

read_audio 统一入口，三级解码链：
1. soundfile（wav/flac/ogg，新版 libsndfile 也支持 mp3）
2. PyAV（ffmpeg 库，m4a/aac/mp4 等压缩格式）
3. librosa/audioread（最后兜底）
错误分类为 AudioError（→ HTTP 400），截断不再静默、通过 info 回传。
"""
import io

import librosa
import numpy as np
import soundfile as sf

from config import SR


class AudioError(Exception):
    """可预期的音频错误，message 面向用户（中文）。"""


def _read_with_av(data):
    """PyAV（自带 ffmpeg 库）解码：支持 m4a/aac/mp4 等 soundfile 不认的格式。"""
    import av

    container = av.open(io.BytesIO(data))
    try:
        stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        raise AudioError("文件里没有音频轨道")
    try:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in container.decode(stream):
            out = resampler.resample(frame)
            if out is None:
                continue
            outs = out if isinstance(out, list) else [out]  # av 15+ 返回 list
            for o in outs:
                chunks.append(o.to_ndarray().reshape(-1))
        if not chunks:
            raise AudioError("音频流为空（可能是损坏或不支持的编码）")
        y = np.concatenate(chunks)
    finally:
        container.close()
    return np.ascontiguousarray(y, dtype=np.float32)


def read_audio(data: bytes, max_seconds: int = 60):
    """解码音频 → (y, info)。

    y: 16k 单声道 float32 numpy 数组
    info: {original_duration, truncated, warnings[]}
    """
    y, errors = None, []
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        y = np.ascontiguousarray(y, dtype=np.float32)
    except Exception as e:
        errors.append(f"soundfile: {e}")

    if y is None:
        try:
            y = _read_with_av(data)
        except Exception as e:
            errors.append(f"ffmpeg: {e}")

    if y is None:
        try:
            y, _sr = librosa.load(io.BytesIO(data), sr=SR, mono=True)
            y = np.ascontiguousarray(y, dtype=np.float32)
        except Exception as e:
            errors.append(f"librosa: {e}")

    if y is None:
        raise AudioError(
            "无法识别音频格式（支持 wav / mp3 / flac / ogg / m4a / aac；"
            + "；".join(errors)[:300] + "）"
        )

    if len(y) < SR // 2:
        raise AudioError("音频太短或没有录到声音，至少需要 0.5 秒")

    info = {"original_duration": round(len(y) / SR, 2), "truncated": False, "warnings": []}
    peak = float(np.abs(y).max())
    if peak < 1e-4:
        # 数字级静音（麦克风没录到音 / 文件本身无声）
        raise AudioError("音频几乎没有声音：麦克风可能没有录到音，"
                         "请检查麦克风权限、输入设备选择后重试")
    if peak < 0.02:
        info["warnings"].append("录音音量较低，可能影响判断（建议靠近麦克风重录）")
    if len(y) > SR * max_seconds:
        y = y[: SR * max_seconds]
        info["truncated"] = True
        info["warnings"].append(f"音频超过 {max_seconds} 秒，已截取前 {max_seconds} 秒")
    return y, info
