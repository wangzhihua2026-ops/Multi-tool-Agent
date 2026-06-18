# Resume Bullets and Interview Notes

## Project Name

企业知识库多工具 Agent 系统

Alternative English title:

Multi-Tool Knowledge Base Agent System

## One-Line Summary

基于 FastAPI + SSE 构建本地优先的企业知识库 Agent，支持文档上传、RAG 检索、多工具调用、人工审批、运行轨迹持久化、可复现评测和容器化交付。

## Recommended Resume Bullets

- 基于 FastAPI + SSE 设计企业知识库 Agent，支持文档上传、RAG 检索、多工具调用、人工审批和运行轨迹持久化。
- 构建 lexical + vector + RRF 混合检索链路，接入 Qdrant Local 与 `multilingual-e5-small`，并通过 embedding signature 避免向量版本混用。
- 建立可复现 RAG 评测集，对 lexical、vector、hybrid 策略进行消融实验，输出 Hit@1、Recall@3、MRR@3 与 P95 延迟指标；Hybrid Hit@1 `93.3%`、Recall@3 `100.0%`、MRR@3 `96.1%`。
- 设计受控 Agent Runtime，限制最多 3 轮工具调用，支持高风险工具暂停审批、恢复执行和事件级 trace。
- 编写一键基线验收脚本，串联 `pytest`、端到端 demo 与 retrieval benchmark；当前 `78` 个测试通过，并补充 Dockerfile、docker-compose 和 GitHub Actions workflow。

- Enhanced the document parsing path with `pdfplumber` PDF text/table extraction and optional PaddleOCR support for scanned PDFs and image OCR while keeping the parsed output connected to the existing chunking, embedding, and Qdrant Local retrieval flow.

## Short Version

- 实现 FastAPI + SSE 知识库 Agent，支持 RAG、工具调用、审批流和 trace 持久化。
- 使用 Qdrant Local + `multilingual-e5-small` 构建 lexical/vector/RRF 混合检索，并以 embedding signature 避免向量版本混用。
- 构建 30 条人工标注查询的检索评测，Hybrid Hit@1 `93.3%`、Recall@3 `100.0%`、MRR@3 `96.1%`。
- 补齐一键基线、Docker/Compose、GitHub Actions 和项目文档，使项目具备可复现演示能力。

## Interview Q&A Anchors

### What Makes This More Than an API Demo?

它把 LLM 放进了完整工程链路：文档接入、RAG、工具注册、审批、trace、持久化、测试、评测和容器化交付。核心不是“调用模型”，而是让模型调用变得可控、可观测、可复现。

### Why Hybrid Retrieval?

纯向量检索适合语义问题，但对短关键词、缩写、编号和工具名不稳定；lexical 对精确匹配强，但语义泛化弱。项目用加权 RRF 合并排序，避免直接混合不同量纲的原始分数。

### What Is the Biggest Current Limitation?

目前评测集仍偏小，只有 `12` 段语料和 `30` 条标注查询；还没有端到端回答 groundedness 评测、reranker、真实云部署和大规模压测。Docker/CI 配置已经补齐，但仍需在 GitHub Actions 或 Docker 环境中完成最终验证。

## Metric Source of Truth

- Latest benchmark report: `evaluation/results/retrieval_benchmark_latest.md`
- Baseline audit: `docs/baseline-audit-2026-06-15.md`
- Project status: `docs/project-status-2026-06-15.md`
