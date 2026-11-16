@echo off
setlocal
cd /d "%~dp0"
title VoiceGuard

echo ==========================================
echo    Voice Guard - starting
echo ==========================================
echo    Folder: %cd%
echo.

where python >nul 2>nul
if errorlevel 1 goto NOPYTHON

if exist "app.py" goto HASAPP
echo [ERROR] app.py not found in this folder.
echo Extract the whole zip first, then run start.bat inside the folder.
pause
exit /b 1
:HASAPP

if exist ".venv\Scripts\python.exe" goto HAVEVENV
echo [1/5] Creating local environment (first run only, ~20s)...
python -m venv .venv
if errorlevel 1 goto VENVFAIL
:HAVEVENV

echo [2/5] Installing base dependencies (first run only)...
echo       Keep this window open.
".venv\Scripts\python.exe" -m pip install --upgrade pip -q --disable-pip-version-check
".venv\Scripts\python.exe" -m pip install numpy scipy scikit-learn soundfile librosa joblib flask transformers huggingface_hub av -q -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto TORCH
echo       mirror 1 failed, trying mirror 2 ...
".venv\Scripts\python.exe" -m pip install numpy scipy scikit-learn soundfile librosa joblib flask transformers huggingface_hub av -q -i https://mirrors.aliyun.com/pypi/simple/
if not errorlevel 1 goto TORCH
echo       mirror 2 failed, trying default index ...
".venv\Scripts\python.exe" -m pip install numpy scipy scikit-learn soundfile librosa joblib flask transformers huggingface_hub av -q
if errorlevel 1 goto PIPFAIL

:TORCH
echo [3/5] Installing PyTorch CPU (~230MB, for deep model) ...
".venv\Scripts\python.exe" -c "import torch" 2>nul
if not errorlevel 1 goto CHECK
echo       installing small deps first ...
".venv\Scripts\python.exe" -m pip install filelock typing-extensions sympy networkx jinja2 fsspec -q -i https://mirrors.aliyun.com/pypi/simple/
echo       downloading torch from aliyun mirror (no CUDA extras) ...
".venv\Scripts\python.exe" -m pip install torch==2.8.0 --no-deps -q -i https://mirrors.aliyun.com/pypi/simple/
if not errorlevel 1 goto CHECK
echo       aliyun failed, trying SJTU pytorch mirror ...
".venv\Scripts\python.exe" -m pip install torch==2.8.0+cpu --no-deps -q --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu -i https://mirrors.aliyun.com/pypi/simple/
if not errorlevel 1 goto CHECK
echo       SJTU failed, trying official pytorch index ...
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q
if not errorlevel 1 goto CHECK
echo.
echo [WARN] PyTorch install failed. Deep model will be unavailable,
echo        light model still works. Check network and retry later with:
echo        ".venv\Scripts\python.exe" -m pip install torch==2.8.0 --no-deps -i https://mirrors.aliyun.com/pypi/simple/

:CHECK
".venv\Scripts\python.exe" -c "import flask, joblib, librosa, sklearn, soundfile, transformers, huggingface_hub"
if errorlevel 1 goto PIPFAIL

echo [4/5] Starting server ...
echo       Browser will open http://127.0.0.1:8000
echo.
echo NOTE: first time using the deep model, the page will download it
echo       (~360MB via hf-mirror.com). Light model works immediately.
echo       Close this window to stop.
echo.
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:NOPYTHON
echo.
echo [ERROR] Python not found.
echo Install Python 3.10+ from https://www.python.org/downloads/
echo IMPORTANT: tick "Add python.exe to PATH" during installation.
echo Then double-click start.bat again.
echo.
pause
exit /b 1

:VENVFAIL
echo.
echo [ERROR] Failed to create .venv. Try running as administrator.
echo.
pause
exit /b 1

:PIPFAIL
echo.
echo [ERROR] Dependency installation failed. Check network and retry.
echo.
pause
exit /b 1
