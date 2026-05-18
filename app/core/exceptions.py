class ToolExecutionError(Exception):
    def __init__(self, tool_name: str, detail: str) -> None:
        self.tool_name = tool_name
        self.detail = detail
        super().__init__(f"Tool '{tool_name}' failed: {detail}")


class ApprovalRequiredError(Exception):
    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' requires human approval.")


class DocumentNotFoundError(Exception):
    def __init__(self, document_id: str) -> None:
        super().__init__(f"Document '{document_id}' was not found.")


class RunNotFoundError(Exception):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run '{run_id}' was not found.")


class ApprovalStateError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
