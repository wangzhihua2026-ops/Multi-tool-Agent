from app.agent.nodes.plan import should_search_knowledge_base


class AgentGraph:
    def choose_retrieval_path(self, message: str) -> bool:
        return should_search_knowledge_base(message)
