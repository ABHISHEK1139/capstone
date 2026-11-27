@echo off
title Stop skarm Research Workbench
powershell -NoProfile -Command "$conns = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue; if ($conns) { $conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue; Write-Host ('Stopped skarm server process PID ' + $_) } } else { Write-Host 'No server running on port 3000.' }"
timeout /t 2 >nul
