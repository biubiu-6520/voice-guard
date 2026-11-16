# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（目录模式）。

用法： .venv\\Scripts\\python.exe -m PyInstaller --noconfirm voiceguard.spec
或直接双击 build_exe.bat

关键点：
- torch / transformers / librosa / sklearn 大量使用惰性导入，静态分析抓不全，
  必须显式 collect_submodules，否则打包后必报 ModuleNotFoundError。
- models/ 不打进包内（361MB 且在 exe 同级可替换），由构建脚本复制到 dist。
- 禁用 UPX：压缩 torch 的 DLL 会导致运行时崩溃。
"""
import os

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = SPECPATH
APP_NAME = "语音鉴别"


def _subs(pkg):
    """整包收集子模块，剔掉测试模块减体积。"""
    return [m for m in collect_submodules(pkg)
            if ".tests." not in m and not m.rsplit(".", 1)[-1].startswith("test_")]


_hidden = []
for _p in ("sklearn", "librosa", "transformers", "av", "numba"):
    _hidden += _subs(_p)

_hidden += [
    "soundfile", "joblib", "lazy_loader", "soxr", "pooch", "audioread",
    "llvmlite.binding", "scipy.special._cdflib",
    "sklearn.svm._classes", "sklearn.utils._typedefs",
    "transformers.models.wav2vec2", "transformers.models.auto",
    "av.audio.resampler", "av.container",
]

_datas = [
    (os.path.join(ROOT, "templates"), "templates"),
    (os.path.join(ROOT, "static"), "static"),
    (os.path.join(ROOT, "samples"), "samples"),
]
_datas += collect_data_files("librosa")
_datas += collect_data_files("transformers", includes=["**/*.json", "**/*.txt"])
_datas += collect_data_files("numba", includes=["**/*.py"])  # numba 需要源码做 JIT 类型检查

_excludes = [
    "torchvision", "torchaudio", "tensorflow", "keras", "jax", "flax",
    "onnx", "onnxruntime", "tensorboard", "matplotlib", "IPython",
    "notebook", "pytest", "torch.test", "torch.distributed.elastic",
]

a = Analysis(
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=_datas,
    hiddenimports=_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # 无黑窗
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
