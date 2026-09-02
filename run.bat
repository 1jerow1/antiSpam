@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Proverka Ollama...
ollama list >nul 2>&1

set tries=0
:waitloop
curl -s -o nul -m 2 http://localhost:11434/api/tags
if not errorlevel 1 goto ready
set /a tries+=1
if %tries% geq 15 goto skipwait
timeout /t 2 >nul
goto waitloop

:skipwait
echo VNIMANIE: Ollama ne otvechaet na localhost:11434. Zapuskayu bota vse ravno.
goto rundone

:ready
echo Ollama gotova.

:rundone
echo Zapusk antiSpam bota...
python antiSpam.py

echo.
echo Bot ostanovlen. Nazhmite lyubuyu klavishu, chtoby zakryt okno.
pause >nul
