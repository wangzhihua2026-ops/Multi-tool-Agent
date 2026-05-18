from app.tools.executor import ToolExecutor


async def retrieve_knowledge_context(executor: ToolExecutor, query: str) -> str:
    result = await executor.execute("search_knowledge_base", {"query": query})
    return result.content
