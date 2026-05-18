from app.rag.retriever import KnowledgeRetriever
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolDefinition, ToolExecutionResult


def build_knowledge_base_tool(retriever: KnowledgeRetriever):
    def knowledge_base_tool(arguments: dict) -> ToolExecutionResult:
        query = str(arguments.get("query", "")).strip()
        top_k = int(arguments.get("top_k", 3))
        if not query:
            return ToolExecutionResult(
                tool_name="search_knowledge_base",
                content="Knowledge base query was empty.",
            )

        hits = retriever.search(query=query, top_k=top_k)
        if not hits:
            return ToolExecutionResult(
                tool_name="search_knowledge_base",
                content="No relevant knowledge base passages were found.",
                metadata={"hits": []},
            )

        lines = [
            f"[{index}] {hit.document_title} (score={hit.score}): {hit.content}"
            for index, hit in enumerate(hits, start=1)
        ]
        return ToolExecutionResult(
            tool_name="search_knowledge_base",
            content="\n\n".join(lines),
            metadata={"hits": [hit.model_dump(mode='json') for hit in hits]},
        )

    return knowledge_base_tool


def register_knowledge_base_tool(registry: ToolRegistry, retriever: KnowledgeRetriever) -> None:
    registry.register(
        ToolDefinition(
            name="search_knowledge_base",
            description="Search the internal knowledge base for relevant passages.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        ),
        build_knowledge_base_tool(retriever),
    )
