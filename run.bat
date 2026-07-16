@echo off
REM ============================================
REM  Knowledge Graph Pipeline - Launcher
REM  Usage: run.bat [input_dir]
REM ============================================

setlocal

set "PROJECTROOT=%~dp0"
set "PYTHON=C:\Users\kirak\AppData\Local\Programs\Python\Python313\python.exe"
set "INPUTDIR=%~1"
if "%INPUTDIR%"=="" set "INPUTDIR=input"

echo.
echo ========================================
echo   Knowledge Graph Pipeline
echo ========================================
echo.

REM Check Python
echo [1/5] Python...
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
    echo   NOT FOUND at %PYTHON%
    exit /b 1
)
echo   OK

REM Check Docker
echo [2/5] Docker...
docker info >nul 2>&1
if errorlevel 1 (
    echo   Docker not running
    exit /b 1
)
echo   OK

REM Start containers
echo [3/5] Docker containers...
docker compose -f "%PROJECTROOT%src\docker-compose.yml" up -d 2>nul
timeout /t 8 /nobreak >nul
echo   Started

REM Check model
echo [4/5] Ollama model...
docker exec kg_ollama ollama list 2>nul | findstr "qwen2.5:3b" >nul
if errorlevel 1 (
    echo   Pulling qwen2.5:3b...
    docker exec kg_ollama ollama pull qwen2.5:3b
) else (
    echo   Ready
)

REM Run pipeline
echo [5/5] Running pipeline...
echo.
cd /d "%PROJECTROOT%"
"%PYTHON%" -m src.main --input-dir "%INPUTDIR%"

echo.
echo ========================================
if %errorlevel%==0 (
    echo   Pipeline completed!
) else (
    echo   Pipeline failed with code %errorlevel%
)
echo ========================================
echo.
echo   Qdrant:   http://localhost:6343/dashboard
echo   Memgraph: bolt://localhost:7697
echo.

endlocal
