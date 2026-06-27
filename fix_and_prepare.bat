@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: Kök dizinde Python scriptini çalıştırır.
python fix_pipeline.py

echo.
echo Kapatmak icin bir tusa basin...
pause >nul