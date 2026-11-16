@echo off
REM ============================================================
REM  Build VoiceGuard.exe  (PyInstaller onedir)
REM  Requires: .venv with deps installed (created by start.bat)
REM  Output:   dist\<app>\<app>.exe  + models\ next to exe
REM ============================================================
setlocal
cd /d "%~dp0"
title VoiceGuard build

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run start.bat once first to create it.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
  echo [1/4] Installing PyInstaller...
  ".venv\Scripts\python.exe" -m pip install pyinstaller pyinstaller-hooks-contrib -q -i https://pypi.tuna.tsinghua.edu.cn/simple
  if errorlevel 1 goto FAIL
)

echo [2/4] Cleaning old build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/4] Running PyInstaller ^(3-15 min, torch is big^)...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm voiceguard.spec
if errorlevel 1 goto FAIL

echo [4/4] Copying models next to exe...
for /d %%D in ("dist\*") do (
  robocopy "models" "%%~D\models" /E /NFL /NDL /NJH /NJS /NP >nul
  robocopy ".venv\Lib\site-packages\numba" "%%~D\_internal\numba" *.py /S /NFL /NDL /NJH /NJS /NP >nul
)

echo.
echo ============================================================
echo  BUILD OK
echo  Folder: dist\  ^(ship the whole folder, do not split it^)
echo ============================================================
pause
exit /b 0

:FAIL
echo.
echo [ERROR] Build failed. See messages above.
pause
exit /b 1
