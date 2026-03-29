@echo off
setlocal

set "PORT=8501"
set "ROOT=%~dp0..\\"
set "DESKTOP_PID_FILE=%ROOT%.nebula_lens.pid"
set "STREAMLIT_PID_FILE=%ROOT%.nebula_lens_streamlit.pid"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Add-Type -AssemblyName System.Windows.Forms; " ^
  "$pids = @(); " ^
  "foreach ($pidFile in @('%DESKTOP_PID_FILE%', '%STREAMLIT_PID_FILE%')) { if (Test-Path $pidFile) { $pidText = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1; if ($pidText -match '^\d+$') { $pids += [int]$pidText } } }; " ^
  "$connections = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; " ^
  "if ($connections) { $pids += $connections | Select-Object -ExpandProperty OwningProcess -Unique }; " ^
  "$pids = $pids | Sort-Object -Unique; " ^
  "if (-not $pids) { Remove-Item '%DESKTOP_PID_FILE%', '%STREAMLIT_PID_FILE%' -ErrorAction SilentlyContinue; [System.Windows.Forms.MessageBox]::Show('Nebula Lens is not running.','Nebula Lens'); exit 0 }; " ^
  "foreach ($proc in $pids) { Stop-Process -Id $proc -Force -ErrorAction SilentlyContinue }; " ^
  "Remove-Item '%DESKTOP_PID_FILE%', '%STREAMLIT_PID_FILE%' -ErrorAction SilentlyContinue; " ^
  "[System.Windows.Forms.MessageBox]::Show('Nebula Lens was stopped.','Nebula Lens')" ^
  >nul 2>&1

exit /b 0
