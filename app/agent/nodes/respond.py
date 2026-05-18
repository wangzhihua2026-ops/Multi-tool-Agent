def build_response_text(user_message: str, knowledge_context: str) -> str:
    if knowledge_context:
        return (
            "当前骨架已经走通了检索工具链路。\n\n"
            f"用户问题：{user_message}\n"
            f"检索结果：{knowledge_context}\n\n"
            "下一步可以把这里替换成真实的 LLM 总结与引用生成。"
        )

    return (
        "当前骨架已经走通了基础对话链路，但还没有接入真实模型规划。\n\n"
        f"收到的问题：{user_message}\n\n"
        "如果你希望它具备真实问答能力，下一步应该接入 LLM Gateway 和持久化会话状态。"
    )
