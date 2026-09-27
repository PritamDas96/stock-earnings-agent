"""Project-wide structured logging built on Loguru.

Critical constraint: the MCP server communicates over **stdout** using the
JSON-RPC protocol. Any stray write to stdout corrupts that stream, so all logs
are emitted to **stderr**. Configuration is idempotent — importing modules may
call :func:`get_logger` freely without duplicating sinks.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

_CONFIGURED = False

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logging(level: str | None = None) -> None:
    """Configure the global logger to write to stderr.

    Args:
        level: Log level override. Falls back to the ``LOG_LEVEL`` environment
            variable, then ``INFO``. Read directly from the environment to keep
            logging usable even before settings validation runs.
    """
    global _CONFIGURED
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=resolved,
        format=_LOG_FORMAT,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None):
    """Return the shared logger, configuring it on first use.

    Args:
        name: Optional component name bound to log records via ``extra``.
    """
    if not _CONFIGURED:
        setup_logging()
    return logger.bind(component=name) if name else logger
