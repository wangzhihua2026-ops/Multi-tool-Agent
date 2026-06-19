# 2-Minute Demo Script

## Demo Goal

Show that this is not only an LLM API wrapper. It is a runnable knowledge-base Agent system with document ingestion, RAG retrieval, tool calling, human approval, trace persistence, evaluation, and delivery assets.

## Before the Demo

Run the baseline evidence chain:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-baseline.ps1 -DemoPort 8016
```

Expected evidence:

- `129 passed`
- upload/search/reindex/SSE chat passed
- retrieval benchmark script can compare flat and parent-child strategies

For an interactive UI demo:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-agent-app.ps1
```

Open:

```text
http://127.0.0.1:8000/app/
```

## Talk Track

### 0:00-0:20 Project Context

“This project is a local-first enterprise knowledge-base Agent. The goal is to connect private documents, RAG retrieval, controlled tool calling, human approval, and trace persistence into one runnable system.”

Point to:

- FastAPI + SSE chat
- built-in web UI
- local SQLite and Qdrant Local modes

### 0:20-0:50 Architecture

“I split the system into UI, API, Service, Agent/Tool, RAG, and Storage layers. FastAPI routes stay thin; services own business flow; AgentRuntime owns the bounded plan-tool-answer loop; storage is abstracted behind SQLite/Qdrant interfaces.”

Mention:

- max 3 tool-planning rounds
- ToolRegistry and ToolExecutor
- high-risk tool approval
- run timeline persistence

### 0:50-1:20 RAG Flow

“The document path supports upload, parsing, chunking, embedding, vector write, and metadata persistence. Retrieval combines lexical and vector ranking with weighted RRF, because exact terms and semantic similarity both matter.”

Show or mention:

- multilingual-e5-small, 384 dimensions
- Qdrant Local
- embedding signature to avoid mixing incompatible vector versions
- parent-child retrieval can be demonstrated with `strategy=parent_child` or `strategy=parent_child_rerank`

### 1:20-1:45 Evidence

“I added a reproducible baseline script so the project can prove it runs. It executes tests, verifies upload/search/reindex/SSE chat, and refreshes the retrieval benchmark.”

Use verified local evidence:

- `129 passed`
- Hybrid Hit@1 `93.3%`
- Recall@3 `100.0%`
- MRR@3 `96.1%`
- P95 latency `14.12 ms`
- Parent-child/reranker benchmark support is implemented, but do not quote an improvement number until the updated benchmark has been run.

### 1:45-2:00 Delivery and Honest Boundary

“For delivery, I added Dockerfile, docker-compose, and a GitHub Actions workflow that runs tests, builds the image, starts the container, and checks `/api/health`. The current limitation is that the evaluation set is still small and Docker has not been locally verified on this machine, so my next steps are larger RAG evaluation, answer groundedness, screenshots, and CI/cloud verification.”

## Demo Checklist

- Open README and show current status table.
- Open built-in UI at `/app/`.
- Upload or use sample document.
- Search knowledge base.
- Ask Agent a question that triggers retrieval.
- Show persisted run or SSE event timeline.
- Show benchmark report under `evaluation/results/`.
- Show Docker/CI files.
- End with one honest next step.

## Good Interview Line

“The main value of this project is not that it calls a model. The value is that it wraps the model in an engineered system: retrievable knowledge, bounded tools, approval gates, traces, tests, evaluation, and reproducible startup.”
