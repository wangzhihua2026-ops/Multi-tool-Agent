# Multi Tool Agent 项目求职准备

## 0. 一句话定位

这是一个面向本地知识库的多工具智能体系统：用户可以上传文档，系统完成文本解析、切分、向量化、持久化索引，并通过 FastAPI + SSE 提供流式对话；Agent 可以根据问题自动规划是否调用工具，例如知识库检索、文档全量抽取、计算器、MCP 外部工具，并对高风险工具调用加入人工审批。

适合投递方向：

- 后端开发实习：FastAPI、分层架构、SQLite/Qdrant、接口设计、测试。
- AI 应用开发实习：LLM Gateway、Agent runtime、工具调用、RAG、embedding、SSE。
- Python 工程实习：抽象接口、异常处理、服务封装、pytest、配置管理。
- 大模型应用/RAG 实习：文档解析、chunking、向量检索、混合检索、重建索引。

## 1. 技术路线

### 1.1 项目目标

项目解决的是“本地资料如何被 Agent 安全、可追踪地调用”的问题。普通聊天机器人只能回答模型已有知识；这个项目把本地文档、工具系统、审批机制和会话持久化接入 Agent，让它能围绕用户上传的知识库做检索、抽取和工具增强回答。

核心需求拆成六块：

- 文档接入：支持直接文本和文件上传，解析 TXT、Markdown、CSV、JSON、HTML、XML、PDF、DOCX。
- 知识库索引：文档切分、token 化、embedding、写入向量库和元数据存储。
- Agent 对话：支持 SSE 流式事件，前端可以实时看到运行状态、工具调用、最终答案。
- 工具调用：内置 calculator、knowledge base search、document extractor、CCF catalog extractor、send email 等工具。
- 安全治理：高风险工具进入人工审批流程，远程 API 访问需要 token 或显式开关。
- 可扩展性：LLM、Embedding、VectorStore、KnowledgeStore、MCP 工具都通过抽象层接入。

### 1.2 总体架构

可以按五层来讲：

- UI 层：`app/ui/index.html`、`app/ui/app.js`、`app/ui/styles.css`，提供 Documents、Agent Chat、Paused Actions 三个工作区。
- API 层：`app/api/server.py` 挂载 FastAPI 路由，包括 chat、documents、approvals、runs、messages、tools、health。
- Service 层：`ChatService`、`DocumentService`、`ApprovalService`、`ReindexJobService` 等封装业务流程。
- Agent/Tool 层：`AgentRuntime` 负责 plan -> tool -> answer 的循环；`ToolRegistry` 管理工具定义；`ToolExecutor` 负责参数校验和同步/异步执行。
- Storage/RAG 层：SQLite 保存 runs/messages/documents/reindex jobs，Qdrant Local 保存向量，Retriever 做 lexical + vector 混合检索。

典型依赖注入入口在 `app/api/dependencies.py`：通过 `lru_cache` 构建单例式组件，避免每个请求重复创建 vector store、embedding provider、tool registry。

### 1.3 文档入库流程

流程：

1. 用户调用 `POST /api/documents` 或 `POST /api/documents/upload`。
2. 如果是文件上传，`DocumentFileParser` 先做 base64 解码、文件类型识别和文本抽取。
3. `DocumentService.create_document` 创建 `DocumentRecord`。
4. 使用 `chunk_text` 按默认 `chunk_size=500`、`chunk_overlap=80` 切分文档。
5. 每个 chunk 做 tokenize，形成 `ChunkRecord`。
6. 元数据先写入 `KnowledgeStore`，索引状态为 `pending`。
7. 调用 embedding provider 对 chunk 批量向量化。
8. 向量写入 `VectorStore`，chunk 上记录实际 embedding signature。
9. 成功后文档状态变为 `ready`；失败则删除已写入向量，状态标记为 `failed` 并保存错误信息。

这里可以强调一个工程细节：系统不是盲目相信 embedding 一定成功，而是对异常做回滚，避免“数据库里文档 ready，但向量库没有对应向量”的不一致状态。

### 1.4 Embedding 技术路线

当前项目已经从 hash embedding 切换到开源免费的 `sentence-transformers`：

- 模型：`intfloat/multilingual-e5-small`
- 维度：384
- 设备：CPU
- 缓存路径：`./data/hf-cache`
- 离线加载：`EMBEDDING_LOCAL_FILES_ONLY=true`
- 当前验证签名：`sentence_transformers:intfloat/multilingual-e5-small:384`

系统支持三种 provider：

- `hash`：离线开发兜底，基于 token 哈希生成确定性向量。
- `openai`：兼容 OpenAI embedding API。
- `sentence_transformers`：本地开源模型，适合免费和隐私敏感场景。

关键设计是 embedding signature。每个 chunk 会记录“当时由哪个 provider/model/dimensions 生成”，检索时 query vector 也会带当前 signature。只有签名兼容的 chunk 才参与 vector score，避免模型切换后旧向量和新向量混用导致检索质量异常。

对于 E5 模型，项目自动区分 query/passages 前缀：

- 查询：`query: 用户问题`
- 文档段落：`passage: 文档内容`

这是 E5 系列推荐的语义对齐方式，可以提升 query-document 检索效果。

### 1.5 向量库技术路线

VectorStore 抽象有三种实现：

- `memory`：纯内存，适合测试。
- `qdrant_local`：嵌入式 Qdrant，数据存在本地目录，不需要 Docker 或外部服务。
- `qdrant`：HTTP Qdrant 服务，支持 API key 和超时配置。

当前配置：

- `VECTOR_STORE_PROVIDER=qdrant_local`
- `VECTOR_STORE_PATH=./data/qdrant-312-live`
- `VECTOR_STORE_COLLECTION=multi_tool_agent_chunks_e5_small_live`

Qdrant 集合使用 cosine distance，集合维度来自 `EMBEDDING_DIMENSIONS=384`。本地 Qdrant 的 point id 使用 `uuid5(namespace, chunk_id)`，避免 chunk_id 不是合法 point id 时写入失败。

`replace_all` 做了一个比较稳的设计：先 snapshot 当前点位，删除集合并重建；如果重建失败，则 restore snapshot。这比直接清空后写入更安全。

### 1.6 检索技术路线

Retriever 是混合检索：

- lexical score：基于 token overlap 和文本匹配。
- vector score：embedding 后到 Qdrant 检索。
- 融合策略：默认以 `vector_weight=0.7`、`lexical_weight=0.3` 做加权 Reciprocal Rank Fusion。

流程：

1. 对 query 归一化和 tokenize。
2. 遍历 chunk 计算 lexical score。
3. 对 query 生成 embedding。
4. 向量库取 `top_k * 4` 候选，增加召回。
5. 根据 embedding signature 过滤不兼容旧向量。
6. 合并 lexical/vector 候选。
7. 对 lexical 与 vector 排名做加权 Reciprocal Rank Fusion 后返回结果。
8. 返回 `SearchHit`，包含 retrieval_mode、lexical_score、vector_score。

面试中可以说：我没有只做纯向量检索，因为短关键词、专有名词、编号、期刊名这类查询仍需要 lexical 信号；融合阶段使用加权 Reciprocal Rank Fusion，而不是直接混合两类异量纲分数。

### 1.7 Agent Runtime 技术路线

Agent 核心在 `AgentRuntime.stream`，它不是简单的一次 LLM 调用，而是受控循环：

1. 初始化 run state，发出 `run.started` 事件。
2. 调用 LLMGateway 进行 planning。
3. 发出 `planner.completed` 事件，告诉前端选择了回答还是调用工具。
4. 如果不需要工具，直接进入 answer。
5. 如果需要工具，先解析工具定义。
6. 如果工具高风险，发出 `approval.required` 和 `run.waiting_approval`，暂停运行。
7. 如果工具低风险，调用 ToolExecutor 执行。
8. 工具结果加入上下文，再进入下一轮 plan。
9. 最多执行 `MAX_TOOL_STEPS` 轮，避免无限工具循环。
10. 生成最终 `assistant.message` 和 `run.completed`。

这个设计的亮点是“有边界的 Agent”：能用工具，但不会无限调用；能执行高风险动作，但必须审批；前端可以通过事件看到整个推理和执行过程。

### 1.8 LLM Gateway 技术路线

项目不是把 DeepSeek/OpenAI 调用写死，而是抽象成 `LLMGateway`：

- `MockLLMGateway`：无网络、无 key 时也能测试基本流程。
- `OpenAICompatibleGateway`：通过 `/chat/completions` 调 OpenAI-compatible API，当前配置指向 DeepSeek。
- `FallbackLLMGateway`：真实模型失败时回退 mock，保证系统可用性。

LLM 调用分两类：

- `plan`：决定是否调用工具，使用 tool schema/function calling。
- `answer`：根据用户问题、历史上下文、工具结果生成最终回答。

项目还加了 deterministic extraction plan：对于“列出所有/完整抽取/继续下一页”等任务，系统可以绕过不稳定的 LLM 判断，直接调用文档抽取工具。这是一个很好的面试点：关键业务路径不能完全依赖模型随机性，适合用规则增强确定性。

### 1.9 工具系统技术路线

工具由三部分组成：

- `ToolDefinition`：名称、描述、JSON Schema、风险等级、是否需要审批。
- `ToolRegistry`：注册和查询工具，防止同名工具被覆盖。
- `ToolExecutor`：校验参数，支持同步函数放到线程执行，也支持 async handler。

内置工具：

- `calculator`：用 Python AST 安全解析算术表达式，不使用 `eval`，限制表达式长度、指数大小和结果范围。
- `search_knowledge_base`：调用 Retriever 查询知识库。
- `extract_document_items`：对文档做全量规则抽取，适合“列出所有符合条件项”。
- `extract_ccf_c_journals`：面向 CCF 目录的专用抽取工具。
- `send_email`：模拟邮件发送，高风险，必须人工审批。
- MCP tools：从配置文件发现 mock/http/streamable_http 工具。

### 1.10 审批机制

审批机制用于解决 Agent 安全问题。比如 send_email 工具带：

- `risk_level="high"`
- `approval_required=True`

Runtime 发现工具需要审批后，不会执行，而是：

1. 保存 pending tool name 和 arguments。
2. run 状态进入 `waiting_approval`。
3. 前端 Paused Actions 展示待审批动作。
4. 用户调用 `/api/approvals/{run_id}` approve/reject。
5. `ApprovalService` 使用 `claim_pending_approval` 防止重复审批。
6. approve 则 resume 执行工具；reject 则把拒绝原因交给 LLM 生成回复。

面试中可以说：这是从“能跑”到“可控”的关键一步，尤其 Agent 一旦接入邮件、支付、数据库写入、部署脚本等工具，必须有人类确认或策略治理。

### 1.11 MCP 扩展路线

项目支持从 `config/mcp_servers.json` 读取 MCP 工具：

- mock transport：本地模板返回，适合开发调试。
- http/streamable_http：通过 JSON-RPC 调用远端 MCP server。
- 支持 initialize：如果服务端要求初始化，会先发 `initialize`。
- 支持 tools/list：可动态发现 server 暴露的工具。
- 支持 mcp-session-id：初始化后把 session id 带到后续请求头。

目前 stdio transport 还没有实现，README 里也明确列为后续里程碑。这一点面试时要诚实：HTTP MCP 已接入，stdio 是下一阶段。

### 1.12 API 与前端

主要 API：

- `POST /api/chat/stream`：SSE 聊天入口。
- `POST /api/documents`：文本入库。
- `POST /api/documents/upload`：文件上传入库。
- `GET /api/documents/search`：直接搜索知识库。
- `POST /api/documents/reindex`：后台重建索引。
- `GET /api/runs/{run_id}`：查看 run timeline。
- `GET /api/sessions/{session_id}/messages`：查看会话历史。
- `GET /api/tools`：查看工具清单。
- `POST /api/approvals/{run_id}`：审批高风险工具。
- `GET /api/health/deep`：检查知识库、向量库、embedding、LLM 配置。

前端重点：

- Documents：上传/查看/搜索/重建索引。
- Agent Chat：发送消息，查看用户友好的 transcript。
- Advanced SSE panel：折叠展示技术事件。
- Paused Actions：处理等待审批的高风险动作。

### 1.13 持久化设计

项目有几类持久化：

- 文档和 chunk 元数据：KnowledgeStore，支持 memory/sqlite/postgres。
- 向量：VectorStore，支持 memory/qdrant_local/qdrant。
- 运行记录：SqliteRunRepository，保存 run、event timeline、pending approval。
- 消息历史：SqliteMessageRepository，按 session_id 保存 user/assistant message。
- reindex job：SQLite 表保存 job 状态、summary 和 error。

Service 层依赖抽象接口，所以从 SQLite 切到 Postgres、从 qdrant_local 切到远端 Qdrant，不需要大改业务逻辑。

### 1.14 安全与稳定性

可以重点讲这些设计：

- API local-first：除 `/api/health` 外，远程 API 默认禁止；远程访问需要 `API_AUTH_TOKEN` 或显式允许。
- 请求追踪：每个请求带 `x-request-id`，日志记录 request_id；chat run 记录 run_id。
- 文件上传限制：限制 decoded bytes、抽取文本长度、PDF 页数。
- 工具参数校验：ToolExecutor 根据 JSON Schema 检查必填字段和类型。
- calculator 不用 eval：使用 AST 白名单和数值范围限制。
- LLM/Embedding fallback：外部 provider 失败时可回退 mock/hash。
- reindex 持久化：服务重启后仍能查询上次 job 状态。
- vector replace_all 回滚：避免重建索引失败造成向量库空掉。

### 1.15 测试与工程状态

当前基线验收：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-baseline.ps1 -DemoPort 8016
```

已验证结果：

```text
78 passed
Demo verification passed: upload, search, reindex, SSE chat
Hybrid retrieval: Hit@1 93.3%, Recall@3 100.0%, MRR@3 96.1%, P95 14.12 ms
```

测试覆盖方向：

- health/security/middleware
- documents upload/parser/service
- embedding provider 和 fallback
- vector store / hybrid RAG
- runtime / tool executor / registry
- approvals 审批流
- MCP discovery / adapter
- messages/runs 持久化
- UI 静态资源
- exports API

工程交付状态：

- 已补 `Dockerfile`：默认 mock/hash，本地优先，适合离线 demo。
- 已补 `docker-compose.yml`：一条命令启动容器化 demo，SQLite 数据使用 named volume。
- 已补 `.github/workflows/ci.yml`：安装依赖、运行 `pytest -q`、构建 Docker image、启动容器并检查 `/api/health`。
- 已补 `scripts/check-baseline.ps1`：串联测试、端到端 demo 和 retrieval benchmark。
- 当前机器未安装 Docker，因此 Docker build 需在 GitHub Actions 或有 Docker 的环境中最终验证。
- 根目录存在本地 `.env`，但 `.gitignore` 已忽略；投递前仍需确认不要提交密钥、本地缓存和运行数据。

## 2. 简历项目经历

### 2.1 简历标题

多工具知识库智能体系统 Multi Tool Agent

### 2.2 简历一句话版

基于 FastAPI、SSE、Qdrant Local、sentence-transformers 和 OpenAI-compatible LLM API，设计并实现一个支持文档上传、RAG 检索、工具调用、人工审批和运行轨迹持久化的本地知识库 Agent 系统。

### 2.3 简历项目描述版

个人项目：多工具知识库智能体系统
技术栈：Python, FastAPI, Pydantic, SQLite, Qdrant, sentence-transformers, HTTPX, SSE, pytest, Docker, GitHub Actions, HTML/CSS/JavaScript

- 设计分层式 Agent 应用架构，基于 FastAPI 暴露 chat/documents/tools/approvals/runs 等接口，通过 Service 层解耦 API、Agent Runtime、RAG、工具执行和持久化模块。
- 实现 SSE 流式对话链路，Agent 运行过程以 `run.started`、`planner.completed`、`tool.requested`、`tool.completed`、`assistant.message` 等事件输出，并将 run timeline 持久化到 SQLite，便于调试和审计。
- 构建本地 RAG 知识库，支持文本/PDF/DOCX/Markdown/CSV/JSON/HTML/XML 上传，完成文档解析、chunk 切分、token 化、embedding 生成、Qdrant Local 向量写入和 SQLite 元数据管理。
- 将 embedding 方案从 hash fallback 升级为开源 `sentence-transformers` 模型 `intfloat/multilingual-e5-small`，支持本地 Hugging Face 缓存和离线加载，并通过 embedding signature 解决模型切换后的向量兼容问题。
- 实现 lexical + vector 混合检索，向量检索使用 cosine similarity，并以加权 Reciprocal Rank Fusion 融合关键词与语义候选；通过 30 条人工标注查询进行可复现检索消融评测。
- 设计可扩展工具系统，抽象 ToolDefinition/ToolRegistry/ToolExecutor，支持 JSON Schema 参数校验、同步/异步工具执行、内置知识库搜索、文档全量抽取、计算器和 MCP HTTP 工具接入。
- 引入高风险工具人工审批机制，对发送邮件等操作在执行前暂停 run，保存 pending tool call，用户 approve/reject 后再恢复执行，降低 Agent 自动执行外部动作的风险。
- 实现本地优先安全策略，请求中间件默认禁止非本地 API 访问，远程访问需配置 token；同时加入 request_id/run_id 日志追踪、文件大小限制、PDF 页数限制和工具参数校验。
- 编写 pytest 测试覆盖 Agent runtime、RAG、embedding fallback、文档解析、工具执行、MCP discovery、审批流、API 安全和 UI 静态资源等模块，当前全量测试 78 个用例通过。
- 补充一键基线验收脚本，串联 pytest、文档上传、检索、reindex、SSE chat 与 retrieval benchmark；配置 Dockerfile、docker-compose 和 GitHub Actions workflow，形成可复现交付链路。

### 2.4 更短的简历 bullet 版

- 基于 FastAPI + SSE 实现本地知识库 Agent，支持文档上传、RAG 检索、工具调用、审批流和运行轨迹持久化。
- 接入 `sentence-transformers/intfloat-multilingual-e5-small` 与 Qdrant Local，设计 embedding signature 机制，避免模型切换后新旧向量混用。
- 实现 lexical + vector 混合检索，以加权 Reciprocal Rank Fusion 合并关键词与向量排序，并建立 30 条人工标注查询的可复现消融评测。
- 抽象 ToolRegistry/ToolExecutor，支持 JSON Schema 参数校验、同步/异步工具执行和 MCP HTTP 工具扩展。
- 设计高风险工具审批机制，Agent 调用邮件类工具前进入 waiting approval 状态，用户确认后恢复执行。
- 使用 SQLite 持久化文档元数据、会话消息、run timeline 和 reindex job；通过一键基线脚本验证 78 个测试、端到端 demo 和 retrieval benchmark。
- 配置 Dockerfile、docker-compose 与 GitHub Actions workflow，使项目具备容器化启动和 CI smoke test 的工程交付基础。

### 2.5 面试自我介绍项目口径

我做了一个本地知识库多工具 Agent 项目，目标是把本地文档、RAG 检索、工具调用和安全审批串成一个可运行系统。后端用 FastAPI，聊天接口用 SSE 返回结构化事件，前端能看到 Agent 从 planning、工具调用到最终回答的完整过程。知识库部分支持多格式文件上传，解析后做 chunking，用 sentence-transformers 的 multilingual-e5-small 生成 384 维向量，写入 Qdrant Local，同时用 SQLite 保存文档和 chunk 元数据。检索时以加权 Reciprocal Rank Fusion 融合 lexical 与 vector 排名，并用 embedding signature 避免切换模型后向量不兼容。我建立了 12 段知识样本和 30 条人工标注查询的本地消融评测，Hybrid Hit@1 为 93.3%，Recall@3 为 100%，MRR@3 为 96.1%，P95 延迟约 14.12 ms。Agent 侧我抽象了 LLMGateway、ToolRegistry、ToolExecutor，支持 OpenAI-compatible 模型、mock fallback、内置工具和 MCP HTTP 工具。对邮件这类高风险工具，我做了人工审批流程，Agent 会暂停等待用户确认。工程交付上补了 Dockerfile、docker-compose、GitHub Actions 和一键基线脚本，目前 78 个 pytest 测试通过，并能自动验证上传、检索、reindex 和 SSE chat。

### 2.6 可以诚实补充的不足

- 当前主要是本地运行和本地 Qdrant，已有 Docker/Compose 配置，但还没有完整云部署。
- HTTP MCP 已支持，stdio MCP 还在后续规划。
- RAG 目前没有 reranker，召回后排序只做 lexical/vector 融合。
- 前端偏工程工具型，不是复杂 React/Vue 项目。
- 没有大规模真实用户数据压测，性能指标主要来自功能测试、本地 demo 和小规模检索 benchmark。

## 3. 面试可能问题与参考答案

### Q1：这个项目解决什么问题？

答：它解决的是本地资料如何被大模型 Agent 安全调用的问题。用户上传文档后，系统会解析、切分、embedding、写入向量库；聊天时 Agent 可以根据问题决定是否检索知识库或调用其它工具，并通过 SSE 返回运行过程。相比普通聊天，它能基于用户自己的资料回答；相比简单 RAG，它还加入了工具注册、审批、运行轨迹持久化和 MCP 扩展。

### Q2：整体架构怎么设计？

答：我按 UI、API、Service、Agent/Tool、Storage/RAG 五层设计。FastAPI 路由只负责请求响应；Service 层处理业务流程；AgentRuntime 负责 plan-tool-answer 循环；ToolRegistry/Executor 管理工具定义和执行；底层用 SQLite 存文档、消息和运行事件，用 Qdrant Local 存向量。这样各层职责比较清晰，后面替换 LLM、embedding、vector store 或数据库都比较容易。

### Q3：为什么用 SSE，而不是普通 HTTP 或 WebSocket？

答：这个项目的聊天是服务端持续推送事件，客户端主要消费流，不需要强双向实时通信，所以 SSE 更轻量。它基于 HTTP，前端处理也简单，适合输出 `run.started`、`planner.completed`、`tool.completed`、`assistant.message` 这种有顺序的事件流。如果后续要做多人协作或客户端频繁实时控制，再考虑 WebSocket。

### Q4：Agent Runtime 是怎么避免无限工具调用的？

答：Runtime 有 `MAX_TOOL_STEPS` 配置，每轮先让 LLMGateway plan，最多执行固定轮数。每轮只能执行一个工具，工具结果会传回下一轮 planning。如果到上限还没有最终答案，就记录错误并进入 answer 阶段。这样 Agent 能用工具，但执行边界是可控的。

### Q5：RAG 入库流程是什么？

答：文档上传后先解析文本，然后按 500 字符左右、80 overlap 切 chunk，每个 chunk tokenize 后保存到 KnowledgeStore。之后用 embedding provider 批量生成向量，写入 Qdrant Local，并把实际 embedding signature 写回 chunk 元数据。如果 embedding 或向量写入失败，会回滚已写入向量，并把文档状态设为 failed。

### Q6：为什么做混合检索？

答：纯向量检索对语义问题很好，但对短关键词、缩写、编号、期刊名、工具名这类查询仍需要 lexical 信号。两类原始分数的量纲不同，直接加权会造成排序退化，因此项目以 0.7/0.3 的权重做 Reciprocal Rank Fusion，将各自排名而非原始分数合并。

### Q7：embedding signature 是什么？为什么需要？

答：embedding signature 是 provider、model、dimension 的组合，例如 `sentence_transformers:intfloat/multilingual-e5-small:384`。每个 chunk 写入向量时记录这个签名，查询时也记录当前 query 的签名。检索结果只有签名兼容才参与 vector score，避免从 hash embedding 切到 sentence-transformers 后，新 query 去匹配旧向量，导致分数不可信。

### Q8：为什么选择 `intfloat/multilingual-e5-small`？

答：它是开源免费的 sentence-transformers 模型，支持多语言，适合中文和英文混合知识库；维度 384，资源占用比大模型低，CPU 上也能跑。项目是本地知识库工具，免费、可离线、隐私友好比追求最大模型更重要。

### Q9：E5 模型为什么要加 query/passsage 前缀？

答：E5 系列训练时区分 query 和 passage 输入形式。查询侧加 `query: `，文档侧加 `passage: `，可以让模型更好地学习“问题”和“文档段落”的匹配关系。项目里在 provider 内部自动处理，不需要业务层关心。

### Q10：Qdrant Local 和普通 Qdrant 服务有什么区别？

答：Qdrant Local 是嵌入式本地存储，不需要 Docker 或额外服务，适合个人项目和本地开发；远程 Qdrant 更适合多人、服务化和云部署。项目通过 VectorStore 抽象支持两者切换，业务代码不依赖具体后端。

### Q11：你怎么处理向量库重建失败？

答：`replace_all` 不是直接清空就结束，而是先 snapshot 当前 points；删除集合并重建，如果过程中失败，会 restore snapshot。这样 reindex 失败时尽量不破坏原有可用索引。

### Q12：LLM Gateway 为什么要抽象？

答：因为模型供应商、接口协议、是否有 key、是否离线都可能变化。抽象 Gateway 后，业务层只调用 `plan` 和 `answer`。当前支持 mock 和 OpenAI-compatible，真实模型失败时可以 fallback 到 mock，测试也不会依赖外部网络。

### Q13：工具系统怎么设计？

答：每个工具都有 ToolDefinition，包括名称、描述、输入 JSON Schema、风险等级和是否需要审批。注册时 ToolRegistry 防止同名覆盖；执行时 ToolExecutor 先按 schema 校验参数，再判断 handler 是同步还是异步，同步函数会放到线程里执行，避免阻塞 async 主流程。

### Q14：为什么 calculator 不直接 eval？

答：`eval` 有代码执行风险。项目用 Python AST 解析表达式，只允许数字常量、基础算术操作和有限的 unary/bin ops，并限制表达式长度、指数大小和结果范围。这样既能完成计算器功能，又避免任意代码执行。

### Q15：人工审批是怎么实现的？

答：工具定义里可以标记 `approval_required=True`。Runtime 发现高风险工具后，不执行工具，而是发出 approval.required 事件，把 run 状态设为 waiting approval，并保存 pending tool name 和 arguments。用户 approve 后 ApprovalService 调用 runtime.resume 继续执行；reject 则生成拒绝后的回答。审批还用了 claim 机制避免重复处理。

### Q16：如何保证会话连续性？

答：ChatService 会按 session_id 从 SQLite 加载最近 `SESSION_HISTORY_LIMIT` 条消息，转换成 ConversationMessage 传给 runtime。每次用户请求和 assistant 回复都会写入 message repository，所以后续对话能带上近期上下文。

### Q17：运行轨迹怎么保存？

答：每个 chat run 会创建 run record，所有 AgentEvent 按 sequence 追加到 SQLite。这样可以通过 `/api/runs/{run_id}` 查看 planner、tool、approval、assistant message 等完整 timeline，方便调试和审计。

### Q18：文件上传安全怎么做？

答：上传内容必须是 base64，会先检查编码长度和解码后大小；PDF 有最大页数限制，抽取文本有最大字符数限制；文件名会清理路径，只保留文件名；不支持的类型直接拒绝。这样可以避免超大文件、路径注入和不可控解析。

### Q19：PDF/DOCX 是怎么解析的？

Answer: DOCX is parsed from `word/document.xml` plus header/footer XML with ElementTree. PDF parsing uses `pdfplumber` for page text and table extraction, serializing tables into searchable Markdown-style text. If native PDF text is too sparse, the optional OCR extra can use PaddleOCR for scanned PDFs and image files. OCR dependencies are intentionally optional because they are heavy.

### Q20：MCP 支持到什么程度？

答：项目支持从配置文件加载 MCP server，mock transport 可以返回模板结果，HTTP/streamable_http transport 可以通过 JSON-RPC 调 tools/call；如果服务端要求 initialize，会先初始化并保存 mcp-session-id。还支持 tools/list 动态发现工具。stdio transport 目前还没实现，是后续计划。

### Q21：项目有哪些测试？

答：当前有 78 个 pytest 用例，覆盖文档服务、文件解析、embedding provider、vector store、hybrid RAG、Agent runtime、工具执行器、审批流、MCP discovery、API 安全、消息和运行记录等。测试里也用 mock/fake provider 降低外部依赖。

### Q22：如果真实 LLM API 挂了怎么办？

答：LLMGateway 外面包了一层 FallbackLLMGateway。plan 或 answer 调真实 provider 失败时，会记录 warning 并回退到 mock gateway。mock 不能替代真实推理，但至少保证系统流程、工具调用和错误返回还能运行。

### Q23：如果 sentence-transformers 本地模型加载失败怎么办？

答：embedding provider 支持 fallback。如果启用 `EMBEDDING_FALLBACK_ENABLED=true`，主 provider 失败会回退 hash embedding，并把实际使用的 signature 写入 chunk。这样系统不会完全不可用。但生产上我会更倾向在 health check 里明确暴露 provider 状态，避免无感降级影响检索质量。

### Q24：这个项目最有技术含量的地方是什么？

答：不是单点调用大模型，而是把 Agent 工具调用、RAG、持久化、审批和可观测事件串起来。尤其是 bounded tool loop、embedding signature、混合检索、高风险工具审批和运行事件持久化，这些都是让大模型应用从 demo 走向工程系统的关键。

### Q25：你会怎么继续优化？

答：我会优先做五件事：第一，将当前 30 条检索查询扩展为更大规模且包含困难负样本的评测集，并补充回答 groundedness；第二，引入 reranker 提升检索排序；第三，在 GitHub Actions 或云环境中完成 Docker build 与部署验证；第四，实现 stdio MCP transport；第五，给 reindex 加真正的后台任务队列。

### Q26：如果面试官问“这是个人项目，有没有真实价值？”

答：我会说它的价值在于覆盖了大模型应用落地的完整工程链路：文档接入、RAG、Agent、工具、审批、持久化、测试和本地部署。它不只是调一个模型 API，而是把很多实际项目会遇到的问题提前做了工程化处理，比如模型失败回退、向量兼容、工具风险、运行审计和上传安全。

### Q27：如果面试官问“你负责了哪些部分？”

答：如果按事实讲，可以说我主导了项目的后端架构设计和核心模块实现，包括 FastAPI 路由、Agent runtime、LLM gateway、RAG pipeline、embedding provider、Qdrant vector store、工具注册执行、审批流和 pytest 测试。同时也实现了一个基础前端用于文档管理、聊天和审批。

### Q28：如果面试官让你现场画流程图，怎么说？

答：用户发消息到 `/api/chat/stream`，ChatService 加载历史并创建 run，AgentRuntime 调 LLMGateway 做 planning。如果需要工具，先从 ToolRegistry 找定义，再判断是否审批；低风险工具由 ToolExecutor 执行，高风险工具进入 ApprovalService。工具结果返回给 Runtime，Runtime 再让 LLM 生成答案。整个过程中事件通过 SSE 给前端，同时写入 SQLite。

### Q29：如果问“为什么不用 LangChain/LlamaIndex？”

答：这个项目我更想展示底层工程能力，所以没有直接套完整框架，而是自己实现 Agent runtime、tool registry、retriever 和 provider abstraction。好处是可控性强，能清楚处理审批、事件流、持久化和 fallback。实际团队项目中，如果需求更偏业务落地，也可以基于 LangChain/LlamaIndex 提速，但底层原理是一致的。

### Q30：如果问“项目目前最大短板是什么？”

答：最大短板是还没有真实生产部署和大规模评测。当前只有 12 段知识样本、30 条标注查询的小规模本地检索评测；Docker、Compose 和 CI workflow 已经配置，但还需要在真实 GitHub Actions/云环境中完成验证。下一步我会扩展 RAG benchmark，补端到端回答质量评估、并发压测和更细的日志指标。

## 4. 面试前建议准备的补充材料

- 准备一张架构图：UI/API/Service/Agent/RAG/Storage 五层。
- 准备一次 2 分钟 demo：上传文档 -> 搜索 -> chat 问答 -> 高风险工具审批。
- 准备一次问题排查故事：embedding 从 hash 切到 sentence-transformers 后，如何用 signature 和 reindex 避免旧向量污染。
- 准备一段代码讲解：`AgentRuntime.stream` 或 `KnowledgeRetriever.search`。
- 准备诚实边界：stdio MCP 未实现、未云部署、无 reranker、无大规模评测。
