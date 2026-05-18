param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    $fallbackPython = Join-Path $projectRoot ".venv312\Scripts\python.exe"
    if (Test-Path -LiteralPath $fallbackPython) {
        $python = $fallbackPython
    } else {
        throw "No virtual environment was found. Expected $python or $fallbackPython"
    }
}

$env:HF_HOME = Join-Path $projectRoot "data\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$openBrowserCommand = @"
$ErrorActionPreference = 'SilentlyContinue'
for ($i = 0; $i -lt 40; $i++) {
    try {
        Invoke-RestMethod -Uri 'http://127.0.0.1:$Port/api/health' -TimeoutSec 3 | Out-Null
        Start-Process 'http://127.0.0.1:$Port/app/'
        break
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
"@

$browserWaiter = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-Command",
    $openBrowserCommand
) -PassThru -WindowStyle Hidden

try {
    & $python -X utf8 -m uvicorn app.api.server:app --host 127.0.0.1 --port $Port --reload
} finally {
    if ($browserWaiter -and -not $browserWaiter.HasExited) {
        Stop-Process -Id $browserWaiter.Id -Force
    }
}
