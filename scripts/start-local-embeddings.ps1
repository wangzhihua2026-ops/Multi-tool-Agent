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

& $python -X utf8 -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000 --reload
