@echo off
title skarm Research Workbench Launcher
setlocal
pushd "%~dp0"

set "NODE_BIN=node"
where node >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files\nodejs\node.exe" (
        set "NODE_BIN=C:\Program Files\nodejs\node.exe"
    ) else (
        echo [ERROR] Node.js was not found on PATH or at C:\Program Files\nodejs\node.exe.
        pause
        exit /b 1
    )
)

powershell -NoProfile -Command "$c = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue; if ($c) { exit 0 } else { exit 1 }"
if %errorlevel%==0 (
    echo skarm Research Workbench is already running on http://localhost:3000
    start "" "http://localhost:3000"
    timeout /t 2 >nul
    exit /b 0
)

echo Starting skarm Conference Research Workbench on http://localhost:3000 ...
start "skarm Server (Port 3000 - Close Window to Stop)" cmd /k "cd /d ""%~dp0"" && ""%NODE_BIN%"" server.mjs"

powershell -NoProfile -Command "for ($i=0; $i -lt 25; $i++) { try { $r = Invoke-WebRequest -Uri 'http://localhost:3000/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { break } } catch {}; Start-Sleep -Milliseconds 300 }"
start "" "http://localhost:3000"
exit /b 0
