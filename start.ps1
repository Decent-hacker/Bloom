param()

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$backend = Join-Path $root 'backend'
$frontend = Join-Path $root 'frontend'
$python = Join-Path $backend '.venv\Scripts\python.exe'
$logs = Join-Path $root '.run-logs'
$backendProcess = $null
$frontendProcess = $null

function Test-LocalPort([int]$port) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $result = $client.BeginConnect('127.0.0.1', $port, $null, $null)
        return ($result.AsyncWaitHandle.WaitOne(200) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Get-FreePort([int]$preferred) {
    for ($port = $preferred; $port -lt ($preferred + 20); $port++) {
        if (-not (Test-LocalPort $port)) { return $port }
    }
    throw "No free port found near $preferred. Stop another local server and try again."
}

function Wait-ForUrl([string]$url, [System.Diagnostics.Process]$process) {
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $process.Refresh()
        if ($process.HasExited) { throw "A server exited during startup. See $logs" }
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Seconds 1
    }
    throw "Timed out waiting for $url. See $logs"
}

try {
    if (-not (Test-Path -LiteralPath $python)) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw 'Python 3.11 is required. Install it, then run this script again.'
        }
        Write-Host 'Creating Python 3.11 environment...'
        & py -3.11 -m venv (Join-Path $backend '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python 3.11 environment.' }
    }

    & $python -c 'import fastapi, cv2, mediapipe, slowapi' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing backend dependencies...'
        & $python -m pip install -r (Join-Path $backend 'requirements.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
    }

    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) { throw 'Node.js and npm are required. Install Node.js, then run this script again.' }
    if (-not (Test-Path -LiteralPath (Join-Path $frontend 'node_modules'))) {
        Write-Host 'Installing frontend dependencies...'
        & $npm.Source ci --prefix $frontend
        if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
    }

    $backendPort = Get-FreePort 8000
    $frontendPort = Get-FreePort 5173
    if ($backendPort -ne 8000 -or $frontendPort -ne 5173) {
        Write-Host "Existing servers occupy the default ports. Using backend $backendPort and frontend $frontendPort."
    }
    $env:ALLOWED_ORIGINS = "http://localhost:$frontendPort,http://127.0.0.1:$frontendPort"
    $env:BLOOM_BACKEND_URL = "http://127.0.0.1:$backendPort"
    $env:VITE_API_URL = ''

    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    Write-Host 'Starting Bloom backend and frontend...'
    $backendProcess = Start-Process -FilePath $python -ArgumentList @('-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', "$backendPort") -WorkingDirectory $backend -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'backend.log') -RedirectStandardError (Join-Path $logs 'backend-error.log')
    Wait-ForUrl "http://127.0.0.1:$backendPort/api/health" $backendProcess
    $frontendProcess = Start-Process -FilePath $npm.Source -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1', '--port', "$frontendPort") -WorkingDirectory $frontend -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logs 'frontend.log') -RedirectStandardError (Join-Path $logs 'frontend-error.log')
    Wait-ForUrl "http://127.0.0.1:$frontendPort" $frontendProcess

    Write-Host ''
    Write-Host "Bloom is ready: http://localhost:$frontendPort"
    Write-Host "Logs: $logs"
    Write-Host 'Press Ctrl+C to stop both servers.'
    while ($true) {
        $backendProcess.Refresh()
        $frontendProcess.Refresh()
        if ($backendProcess.HasExited -or $frontendProcess.HasExited) {
            throw "A server stopped unexpectedly. See $logs"
        }
        Start-Sleep -Seconds 1
    }
} finally {
    foreach ($process in @($frontendProcess, $backendProcess)) {
        if ($process -and -not $process.HasExited) {
            & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
        }
    }
}
