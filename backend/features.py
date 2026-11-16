"""特征提取与可视化。

extract(): 与旧版 voice_app.py 的特征公式逐行一致（顺序决定模型兼容性）。
frame_features/aggregate: 全局一次算帧级特征、按帧切片聚合，供分段滑窗复用，
避免逐窗重算 STFT（60s 音频逐窗要 30s+，切片方案 <3s）。

等价性：aggregate 全范围(0,None) 与 extract 逐位一致；窗口切片与独立计算
仅边界 1-2 帧受 center 填充影响，误差 <1e-2，足够分段可视化使用。
"""
import librosa
import numpy as np

from config import SR

HOP = 160          # STFT hop（样本）
N_FFT = 512
RMS_HOP = 512      # rms/zcr 默认 hop 与 STFT 不同，单独对齐


def _f0_stats(y):
    """f0 均值/标准差。先 2 倍降采样再 yin：fmax=400Hz << 8kHz 奈奎斯特，
    结果与 16kHz 全采样差 <0.5Hz，耗时约 1/4（30s 音频 1.5s → 0.4s）。"""
    ys = y[::2]
    f0 = librosa.yin(ys, fmin=60, fmax=400, sr=SR // 2, frame_length=N_FFT // 2)
    f0 = f0[np.isfinite(f0)]
    return float(f0.mean()) if len(f0) else 0.0, float(f0.std()) if len(f0) else 0.0


def frame_features(y):
    """全局一次计算的帧级特征矩阵。键与帧对齐：
    mfcc/centroid/bandwidth/rolloff/flatness/S/mel → hop=160 帧
    rms/zcr → hop=512 帧"""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    mel = librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_fft=N_FFT,
                                         hop_length=HOP, n_mels=80)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    return {
        "S": S,
        "mel": mel,
        "mfcc": mfcc,
        "centroid": librosa.feature.spectral_centroid(S=S, sr=SR),
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=SR),
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=SR),
        "flatness": librosa.feature.spectral_flatness(S=S),
        "rms": librosa.feature.rms(y=y),
        "zcr": librosa.feature.zero_crossing_rate(y),
    }


def aggregate(frames, y, a=0, b=None, f0_src=None):
    """把 [a, b) 样本窗口聚合为 56 维特征向量。

    a/b 为 16k 采样下标，a 取整到 hop 倍数；b=None 表示到结尾（与 extract 等价）。
    f0_src: (centers, f0) 全局轮廓，传入则窗口 f0 从轮廓切片（分段滑窗提速）；
            为 None 时在窗口内独立算 yin。
    """
    i0 = a // HOP
    i1 = frames["S"].shape[1] if b is None else b // HOP
    j0 = a // RMS_HOP
    j1 = frames["rms"].shape[1] if b is None else b // RMS_HOP
    parts = [
        frames["mfcc"][:, i0:i1],
        frames["centroid"][:, i0:i1],
        frames["bandwidth"][:, i0:i1],
        frames["rolloff"][:, i0:i1],
        frames["flatness"][:, i0:i1],
        frames["rms"][:, j0:j1],
        frames["zcr"][:, j0:j1],
    ]
    v = []
    for f in parts:
        v.append(f.mean(axis=1))
        v.append(f.std(axis=1))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    Sw = frames["S"][:, i0:i1]
    hf = Sw[freqs > 4000].sum() / (Sw.sum() + 1e-9)
    v.append(np.array([hf, hf / (Sw.shape[1] + 1e-9)]))
    if f0_src is None:
        seg = y if b is None else y[a:b]
        v.append(np.array(_f0_stats(seg)))
    else:
        centers, f0 = f0_src
        v.append(np.array(f0_stats_slice(centers, f0, a, len(y) if b is None else b)))
    return np.concatenate(v).astype(np.float32)


def extract(y, frames=None):
    """整段音频 → 56 维特征向量（与旧版公式一致）。frames 传入则复用帧级特征。"""
    return aggregate(frame_features(y) if frames is None else frames, y)


def _f0_contour(y):
    """整段 f0 轮廓（16k 采样下标 → f0 值），分段滑窗共享一次计算。"""
    ys = y[::2]
    f0 = librosa.yin(ys, fmin=60, fmax=400, sr=SR // 2, frame_length=N_FFT // 2)
    centers = (np.arange(len(f0)) * (N_FFT // 8) + N_FFT // 4) * 2  # 换算回 16k 采样
    return centers, f0


def f0_stats_slice(centers, f0, a, b):
    """从全局 f0 轮廓取 [a, b) 窗口的均值/标准差（b=None 到结尾）。"""
    m = np.isfinite(f0)
    m &= (centers >= a)
    if b is not None:
        m &= (centers < b)
    vals = f0[m]
    return (float(vals.mean()) if len(vals) else 0.0,
            float(vals.std()) if len(vals) else 0.0)


def downsample(a, h, w):
    H, W = a.shape
    yi = np.linspace(0, H, h + 1).astype(int)
    xi = np.linspace(0, W, w + 1).astype(int)
    out = np.empty((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = a[yi[i]:max(yi[i + 1], yi[i] + 1),
                          xi[j]:max(xi[j + 1], xi[j] + 1)].mean()
    return out


def visualize(y, frames=None):
    """波形包络 + 下采样 log-mel 谱（供前端 canvas 绘制）。
    frames 传入则复用帧级特征，避免重复 STFT。"""
    if frames is None:
        frames = frame_features(y)
    S, mel = frames["S"], frames["mel"]
    # ref 加下限，避免全静音时 log(0/0) 产生 NaN（前端画布会变纯色块）
    db = np.clip((librosa.power_to_db(mel, ref=max(1e-10, float(np.max(mel)))) + 70) / 70, 0, 1)
    mel_small = downsample(db, 72, 220).round(3).tolist()
    n = 500
    idx = np.linspace(0, len(y), n + 1).astype(int)
    env = [float(np.abs(y[idx[i]:max(idx[i + 1], idx[i] + 1)]).max() or 0) for i in range(n)]
    return mel_small, env
