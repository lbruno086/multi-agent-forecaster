from __future__ import annotations

import logging
import os
import sys

import structlog

_LEVEL = getattr(logging, os.environ.get("SYSTEM_LOG_LEVEL", "INFO").upper(), logging.INFO)


def _configure() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_LEVEL),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


_configure()


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
