import json
import logging
from pathlib import Path

from app.tools.mcp.models import MCPCatalog

logger = logging.getLogger(__name__)


def load_mcp_catalog(config_path: str) -> MCPCatalog:
    path = Path(config_path)
    if not path.exists():
        logger.info("MCP config file not found at %s; continuing without discovered MCP tools.", path)
        return MCPCatalog()

    content = json.loads(path.read_text(encoding="utf-8"))
    return MCPCatalog.model_validate(content)
