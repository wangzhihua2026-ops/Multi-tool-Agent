param(
    [int]$Port = 8010,
    [string]$PythonPath = $env:PROJECT_PYTHON
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

for module_name in ('fastapi', 'httpx', 'qdrant_client', 'sentence_transformers', 'uvicorn'):
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

Add-ProjectPythonPaths -Root $projectRoot
$python = Resolve-PythonCommand -Root $projectRoot -RequestedPython $PythonPath

$env:HF_HOME = Join-Path $projectRoot "data\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:LLM_PROVIDER = "mock"
$env:EMBEDDING_PROVIDER = "sentence_transformers"
$env:EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
$env:EMBEDDING_DIMENSIONS = "384"
$env:EMBEDDING_DEVICE = "cpu"
$env:KNOWLEDGE_STORE_PROVIDER = "sqlite"
$env:KNOWLEDGE_STORE_PATH = Join-Path $projectRoot "data\knowledge_base_312_live.db"
$env:VECTOR_STORE_PROVIDER = "qdrant_local"
$env:VECTOR_STORE_PATH = Join-Path $projectRoot "data\qdrant-312-live"
$env:VECTOR_STORE_COLLECTION = "multi_tool_agent_chunks_e5_small_live"
$env:RUN_STORAGE_PATH = Join-Path $projectRoot "data\multi_tool_agent_312_live.db"

$server = Start-Process -FilePath $python -ArgumentList @(
    "-X",
    "utf8",
    "-m",
    "uvicorn",
    "app.api.server:app",
    "--host",
    "127.0.0.1",
    "--port",
    "$Port"
) -WorkingDirectory $projectRoot -PassThru -WindowStyle Hidden

try {
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 5 | Out-Null
            $ready = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $ready) {
        throw "Server did not become ready on port $Port."
    }

    @"
import json
import httpx
import time

BASE = "http://127.0.0.1:$Port/api"

with httpx.Client(timeout=120.0) as client:
    upload = client.post(
        f"{BASE}/documents",
        json={
            "title": "verify-local-embeddings",
            "content": "Deployment steps: configure environment variables, start the FastAPI service, then check the health endpoint.",
            "metadata": {"source": "verify-script"},
        },
    )
    upload.raise_for_status()

    search = client.get(
        f"{BASE}/documents/search",
        params={"query": "How do I deploy the FastAPI service?", "top_k": 3},
    )
    search.raise_for_status()

    reindex = client.post(
        f"{BASE}/documents/reindex",
        json={"clear_vector_store": True},
    )
    reindex.raise_for_status()
    reindex_job = reindex.json()
    job_payload = reindex_job
    for _ in range(40):
        job = client.get(f"{BASE}/documents/reindex/{reindex_job['job_id']}")
        job.raise_for_status()
        job_payload = job.json()
        if job_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.25)

    if job_payload["status"] != "completed":
        raise RuntimeError(f"Reindex job did not complete: {job_payload}")

    search_hits = search.json()
    if not search_hits:
        raise RuntimeError("Search returned no hits.")

    chat = client.post(
        f"{BASE}/chat/stream",
        json={
            "session_id": "verify-local-embeddings-session",
            "message": "Please search the knowledge base for deployment steps and summarize the result.",
        },
    )
    chat.raise_for_status()

    events = []
    for line in chat.text.splitlines():
        if line.startswith("data: "):
            payload = line[6:].strip()
            if payload:
                events.append(json.loads(payload))

    assistant_message = next(
        (item["data"].get("content") for item in events if item["type"] == "assistant.message"),
        "",
    )
    planner = next(
        (item["data"] for item in events if item["type"] == "planner.completed"),
        {},
    )

    if not assistant_message:
        raise RuntimeError("Chat stream did not include an assistant.message event.")

    result = {
        "document_id": upload.json()["document_id"],
        "search_hit_count": len(search_hits),
        "reindex_job_id": reindex_job["job_id"],
        "reindex_status": job_payload["status"],
        "reindex_chunk_count": job_payload["summary"]["chunk_count"],
        "planner_provider": planner.get("provider"),
        "planner_model": planner.get("model"),
        "assistant_preview": assistant_message[:240],
    }
    print(json.dumps(result, ensure_ascii=False))
"@ | & $python -X utf8 -
} finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
