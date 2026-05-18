import logging
from contextvars import ContextVar


request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
run_id_context: ContextVar[str] = ContextVar("run_id", default="-")


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        record.run_id = run_id_context.get()
        return True


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s run_id=%(run_id)s %(message)s",
    )
    context_filter = RequestContextFilter()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not any(isinstance(item, RequestContextFilter) for item in handler.filters):
            handler.addFilter(context_filter)


def set_request_id(request_id: str):
    return request_id_context.set(request_id)


def reset_request_id(token) -> None:
    request_id_context.reset(token)


def set_run_id(run_id: str):
    return run_id_context.set(run_id)


def reset_run_id(token) -> None:
    run_id_context.reset(token)
