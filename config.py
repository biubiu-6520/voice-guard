"""全局常量与路径。改上限、换模型都来这里。

路径双模式：
- 源码运行：BASE = 项目目录
- 打包运行（PyInstaller）：RES_DIR = 打包内资源（templates/static 等，只读）；
  EXE_DIR = exe 所在目录。models/ samples/ 优先取 exe 同级目录（可替换、可写），
  缺失时回退到打包内资源。
"""
import sys
from pathlib import Path

SR = 16000

FROZEN = bool(getattr(sys, "frozen", False))

if FROZEN:
    # onefile: _MEIPASS 是临时解压目录；onedir: _MEIPASS 指向 _internal
    RES_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    EXE_DIR = Path(sys.executable).resolve().parent
else:
    RES_DIR = EXE_DIR = Path(__file__).resolve().parent

BASE = EXE_DIR  # 兼容旧引用：可写数据默认落在程序目录


def _data_dir(name):
    """exe 同级目录优先（用户可替换模型），缺失回退打包内资源。"""
    p = EXE_DIR / name
    if p.is_dir():
        return p
    return RES_DIR / name


TEMPLATES_DIR = RES_DIR / "templates"
STATIC_DIR = RES_DIR / "static"

MODELS_DIR = _data_dir("models")
SAMPLES_DIR = _data_dir("samples")
LIGHT_MODEL_PATH = MODELS_DIR / "model.joblib"

# ---- 深度模型 ----
DEEP_ENABLED = True  # 总开关：出问题时改 False 一键回退纯轻量模型
DEEP_MODEL_ID = "mo-thecreator/Deepfake-audio-detection"
DEEP_LOCAL_DIR = MODELS_DIR / "deepfake_audio"
HF_ENDPOINT = "https://hf-mirror.com"  # 国内镜像，实测可达
HF_HOME = MODELS_DIR / "hf_cache"      # 缓存放项目内，随文件夹备份/迁移

# ---- 上限 ----
MAX_FILE_MB = 20           # 上传上限（Flask MAX_CONTENT_LENGTH）
MAX_LIGHT_SECONDS = 60     # 轻量模型分析上限（分段曲线需要更长的轴）
DEEP_MAX_SECONDS = 30      # 深度模型截断上限（wav2vec2 最大输入 30s）
DEEP_CHUNK_SECONDS = 10    # 深度分块长度
SEG_WINDOW = 2.0           # 轻量分段窗口（秒）
SEG_HOP = 0.5              # 分段步长（秒）
RECORD_MAX_SECONDS = 60    # 浏览器录音上限

LABELS = ["真人", "AI合成", "变声"]
ADVICE = {
    "真人": "未发现明显的合成或变声痕迹。",
    "AI合成": "疑似语音合成/克隆，注意声码器伪造迹象。",
    "变声": "疑似经过变声处理，音色与韵律不一致。",
}
