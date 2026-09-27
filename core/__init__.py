"""Shared foundation for the stock-earnings-agent.

Exposes application settings and a preconfigured logger so every other
package (``rag_pipeline``, ``agents``, ``mcp_server``, ``app``,
``evaluation``) depends on a single source of truth for configuration
and logging.
"""

from core.config import Settings, get_settings
from core.logging import get_logger, setup_logging

__all__ = ["Settings", "get_settings", "get_logger", "setup_logging"]
