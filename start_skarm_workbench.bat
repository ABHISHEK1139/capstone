@echo off
title skarm Research Workbench Launcher
setlocal
pushd "%~dp0"

set "NODE_FOUND=0"
where node >nul 2>nul
if not errorlevel 1 (
    set "NODE_FOUND=1"
) else if exist "C:\Program Files\nodejs\node.exe" (
    set "PATH=C:\Program Files\nodejs;%PATH%"
    set "NODE_FOUND=1"
)
if "%NODE_FOUND%"=="0" (
    echo [ERROR] Node.js was not found on PATH or at C:\Program Files\nodejs\node.exe.
    pause
    exit /b 1
)

powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue; if ($c) { exit 0 } else { exit 1 }"
if %errorlevel%==0 (
    echo skarm Research Workbench is already running on http://localhost:3000
    start "" "http://localhost:3000"
    exit /b 0
)

echo Starting skarm Conference Research Workbench on http://localhost:3000 ...
start "skarm Server (Port 3000 - Close Window to Stop)" /D "%~dp0" cmd /k "node server.mjs || pause"

powershell -NoProfile -Command "for ($i=0; $i -lt 30; $i++) { try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:3000/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep -Milliseconds 300 }; exit 1"
if errorlevel 1 (
    echo [WARNING] Server is still starting or failed to bind port 3000. Check the server console window.
    pause
    exit /b 1
)
start "" "http://localhost:3000"
exit /b 0
