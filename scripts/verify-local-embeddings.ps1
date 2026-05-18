param(
    [int]$Port = 8010
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv312\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python 3.12 virtual environment was not found at $python"
}

$env:HF_HOME = Join-Path $projectRoot "data\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
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

    result = {
        "document_id": upload.json()["document_id"],
        "search_hit_count": len(search.json()),
        "reindex_chunk_count": reindex.json()["chunk_count"],
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
