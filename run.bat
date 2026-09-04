@echo off
title HyperVector — Facial Identity Engine
echo ========================================================
echo  Starting HyperVector Face Recognition Web Application...
echo ========================================================
echo.
echo Opening browser at http://localhost:3000 ...
timeout /t 2 /nobreak >nul
start "" "http://localhost:3000"
echo.
echo Starting Node server...
node server.js
pause
