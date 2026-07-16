<#
.SYNOPSIS
    Knowledge Graph Pipeline - Единый скрипт запуска
.DESCRIPTION
    Проверяет Docker, запускает контейнеры, ждёт готовности,
    проверяет модель Ollama, запускает pipeline.
.PARAMETER InputDir
    Директория с входными файлами (.txt, .pdf, .docx). По умолчанию: input
.PARAMETER Interactive
    После pipeline запускает интерактивный Q&A режим
.EXAMPLE
    .\run.ps1
    .\run.ps1 -InputDir "C:\Documents"
    .\run.ps1 -InputDir "C:\Documents" -Interactive
#>

param(
    [string]$InputDir = "input",
    [switch]$Interactive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = "C:\Users\kirak\AppData\Local\Programs\Python\Python313\python.exe"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Knowledge Graph Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
Write-Host "[1/5] Python..." -NoNewline
if (-not (Test-Path $PythonExe)) {
    Write-Host " NOT FOUND at $PythonExe" -ForegroundColor Red
    exit 1
}
$pyVer = & $PythonExe --version 2>&1
Write-Host " OK ($pyVer)" -ForegroundColor Green

# 2. Check Docker
Write-Host "[2/5] Docker..." -NoNewline
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host " NOT FOUND" -ForegroundColor Red
    exit 1
}
$dockerOk = docker info 2>&1 | Select-String "Server Version"
if (-not $dockerOk) {
    Write-Host " Docker daemon not running" -ForegroundColor Red
    exit 1
}
Write-Host " OK" -ForegroundColor Green

# 3. Start containers
Write-Host "[3/5] Docker containers..."
$composePath = Join-Path $ProjectRoot "src\docker-compose.yml"
$running = docker ps --format "{{.Names}}" 2>&1
$allRunning = $true
foreach ($name in @("kg_ollama", "kg_qdrant", "kg_memgraph")) {
    if ($running -match $name) {
        Write-Host "  $name - already running" -ForegroundColor Green
    } else {
        $allRunning = $false
    }
}
if (-not $allRunning) {
    Write-Host "  Starting containers..." -ForegroundColor Yellow
    docker compose -f $composePath up -d 2>&1 | Out-Null
    Start-Sleep -Seconds 5
}

# Wait for health
Write-Host "  Waiting for services..."
$services = @(
    @{ Name="Ollama"; Check={ (Invoke-WebRequest -Uri "http://localhost:8010/" -TimeoutSec 3 -ErrorAction SilentlyContinue).StatusCode -eq 200 } },
    @{ Name="Qdrant"; Check={ (Invoke-WebRequest -Uri "http://localhost:6343/healthz" -TimeoutSec 3 -ErrorAction SilentlyContinue).StatusCode -eq 200 } },
    @{ Name="Memgraph"; Check={ try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect("localhost", 7697); $c.Close(); $true } catch { $false } } }
)
foreach ($svc in $services) {
    Write-Host "  $($svc.Name)..." -NoNewline
    $ok = $false
    for ($i = 0; $i -lt 30; $i++) {
        try { if (& $svc.Check) { $ok = $true; break } } catch { }
        Start-Sleep -Seconds 2
        Write-Host "." -NoNewline
    }
    if ($ok) { Write-Host " healthy" -ForegroundColor Green } else { Write-Host " TIMEOUT" -ForegroundColor Red }
}

# 4. Check Ollama model
Write-Host "[4/5] Ollama model..."
$modelList = docker exec kg_ollama ollama list 2>&1
if ($modelList -match "qwen2.5:3b") {
    Write-Host "  qwen2.5:3b - ready" -ForegroundColor Green
} else {
    Write-Host "  Pulling qwen2.5:3b (~2 GB)..." -ForegroundColor Yellow
    docker exec kg_ollama ollama pull qwen2.5:3b
}

# 5. Run pipeline
Write-Host "[5/5] Running pipeline..."
Write-Host "  Input: $InputDir" -ForegroundColor Yellow

$pyArgs = @("-m", "src.main", "--input-dir", $InputDir)
if ($Interactive) {
    $pyArgs += "--interactive"
}

Write-Host ""
& $PythonExe @pyArgs
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Pipeline completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Qdrant:  http://localhost:6343/dashboard" -ForegroundColor Cyan
    Write-Host "  Memgraph Lab: http://localhost:7444" -ForegroundColor Cyan
    Write-Host "  Memgraph connection: bolt://localhost:7697" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  Opening browsers..." -ForegroundColor Yellow
    Start-Process "http://localhost:6343/dashboard"
    Start-Process "http://localhost:7444"
} else {
    Write-Host "Pipeline failed with exit code $exitCode" -ForegroundColor Red
}

Write-Host ""
