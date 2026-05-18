def should_search_knowledge_base(message: str) -> bool:
    lowered = message.lower()
    keywords = (
        "知识库",
        "文档",
        "资料",
        "kb",
        "knowledge base",
        "manual",
    )
    return any(keyword in lowered for keyword in keywords)
