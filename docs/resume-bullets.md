# 简历表述与面试要点

## 项目名称

企业内部智能知识问答系统

名称无需修改；增强后把技术定位升级为“可恢复的企业知识 Agent 执行平台”。

## 推荐简历描述

- 基于 FastAPI、PostgreSQL、Redis 与异步 Worker，将进程内知识问答 Agent 升级为可恢复执行平台，持久化 Run/Step/Event 状态，支持幂等重试、租约恢复、人工审批续跑和 SSE 断线续传。
- 设计 transactional outbox 与 Redis ARQ 队列解耦提交和执行，通过步骤幂等键、审批 compare-and-set、过期租约重排避免重复副作用；Docker Compose 实测 API、Worker、PostgreSQL、Redis 四服务健康运行。
- 构建 lexical/vector/RRF 混合检索、父子分块及可选 reranker，保留文档上传、PDF/DOCX/OCR、知识检索和受控工具调用能力。
- 建立 30 个确定性 Agent 场景、10 类故障注入检查、Prometheus 指标与 OpenTelemetry step spans；本地全量 `182 passed`，真实 PostgreSQL/Redis 集成测试通过。
- 完成 50 个 mock Run 并发提交与执行：50/50 到达 completed，提交延迟 P50 `510.69 ms`、P95 `532.97 ms`，整批完成 `5018.41 ms`（Windows Docker Desktop，2026-07-12）。

## 一句话介绍

这是一个面向企业内部知识问答的 Agent 平台：不仅完成 RAG 与工具调用，还把执行状态、审批、重试、恢复、实时事件、评测和容器化交付做成了可验证的后端工程闭环。

## 面试边界

可以声明：本地 Docker Compose 四服务部署、182 项自动化测试、50 Run mock 并发实测、PostgreSQL/Redis 集成与恢复验证。

暂不声明：生产流量、真实大模型质量提升、云端公开部署 SLA。公开 URL 完成验证后再补入简历。

证据来源：`docs/agent-platform-verification.md` 与 `evaluation/results/agent_platform_latest.md`。
