param(
    [int]$DemoPort = 8013,
    [string]$PythonPath = $env:PROJECT_PYTHON,
    [switch]$SkipDemo,
    [switch]$SkipRetrievalBenchmark
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot

function Add-ProjectPythonPaths {
    param([string]$Root)

    $paths = @($Root)
    $venvSitePackages = Join-Path $Root ".venv312\Lib\site-packages"
    if (Test-Path -LiteralPath $venvSitePackages) {
        $paths += $venvSitePackages
        $paths += Join-Path $venvSitePackages "win32"
        $paths += Join-Path $venvSitePackages "win32\lib"

        $pywin32Dlls = Join-Path $venvSitePackages "pywin32_system32"
        if (Test-Path -LiteralPath $pywin32Dlls) {
            $env:PATH = "$pywin32Dlls;$env:PATH"
        }
    }

    $existing = @()
    if ($env:PYTHONPATH) {
        $existing = $env:PYTHONPATH -split ";"
    }
    $env:PYTHONPATH = (($paths + $existing) | Where-Object { $_ } | Select-Object -Unique) -join ";"
}

function Resolve-PythonCommand {
    param(
        [string]$Root,
        [string]$RequestedPython
    )

    $candidates = @()
    if ($RequestedPython) {
        $candidates += $RequestedPython
    }

    $candidates += Join-Path $Root ".venv312\Scripts\python.exe"
    $candidates += Join-Path $Root ".venv\Scripts\python.exe"

    if ($env:USERPROFILE) {
        $candidates += Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    }

    $pathPython = Get-Command "python" -ErrorAction SilentlyContinue
    if ($pathPython) {
        $candidates += $pathPython.Source
    }

    $probe = @'
import importlib.util
import os
import sys

if sys.version_info < (3, 11):
    raise SystemExit('Python 3.11+ is required')

for module_name in ('fastapi', 'httpx', 'pytest', 'qdrant_client', 'sentence_transformers', 'uvicorn'):
    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(f'missing dependency: {module_name}')

if os.name == 'nt':
    import pywintypes

print(sys.executable)
'@

    $failures = @()
    foreach ($candidate in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        $resolved = $candidate
        if (-not (Test-Path -LiteralPath $resolved)) {
            $command = Get-Command $candidate -ErrorAction SilentlyContinue
            if ($command) {
                $resolved = $command.Source
            } else {
                $failures += "$candidate : not found"
                continue
            }
        }

        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            $output = & $resolved -X utf8 -c $probe 2>&1
            $exitCode = $LASTEXITCODE
        } catch {
            $output = @($_.Exception.Message)
            $exitCode = 1
        } finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        if ($exitCode -eq 0) {
            return $resolved
        }
        $failures += "$resolved : $($output -join ' ')"
    }

    throw "No usable Python runtime found. Tried: $($failures -join ' | ')"
}

function Invoke-BaselineCommand {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Name"
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE."
    }
}

Add-ProjectPythonPaths -Root $projectRoot
$python = Resolve-PythonCommand -Root $projectRoot -RequestedPython $PythonPath

Write-Host "Using Python: $python"

Invoke-BaselineCommand `
    -Name "pytest" `
    -FilePath $python `
    -Arguments @("-m", "pytest", "-q", "-p", "no:cacheprovider")

if (-not $SkipDemo) {
    Invoke-BaselineCommand `
        -Name "local embeddings demo" `
        -FilePath "powershell" `
        -Arguments @(
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            (Join-Path $PSScriptRoot "verify-local-embeddings.ps1"),
            "-Port",
            "$DemoPort",
            "-PythonPath",
            $python
        )
}

if (-not $SkipRetrievalBenchmark) {
    Invoke-BaselineCommand `
        -Name "retrieval benchmark" `
        -FilePath $python `
        -Arguments @("-X", "utf8", (Join-Path $PSScriptRoot "evaluate_retrieval.py"))
}

Write-Host ""
Write-Host "Baseline check completed."
