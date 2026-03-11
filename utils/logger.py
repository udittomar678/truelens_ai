"""
utils/logger.py
---------------
Centralised structured logger for TrueLens AI.
Uses structlog for JSON-formatted, machine-parseable output to both
stdout and a rotating JSONL file.  Every analysis event is persisted
with a timestamp for audit and retraining use.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from core.config import settings


# ── stdlib logging setup (sink for structlog) ─────────────────────

def _configure_stdlib_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    fmt = logging.Formatter("%(message)s")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler (rotating, max 50 MB × 5 backups)
    if settings.log_to_file:
        log_path = settings.log_dir / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def _configure_structlog() -> None:
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.env == "development" and not settings.debug is False:
        # Human-friendly in dev
        renderer: Any = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ── initialise once at import time ────────────────────────────────
_configure_stdlib_logging()
_configure_structlog()


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a named structlog logger.

    Usage::

        log = get_logger(__name__)
        log.info("image_analysed", file="photo.jpg", ai_probability=0.91)
    """
    return structlog.get_logger(name)


# ── Analysis event persistence ────────────────────────────────────

class AnalysisLogger:
    """Appends structured analysis results to a JSONL audit log file."""

    def __init__(self) -> None:
        self._path: Path = settings.log_dir / settings.structured_log_file
        self._log = get_logger(self.__class__.__name__)

    def record(self, event: dict[str, Any]) -> None:
        """Persist a single analysis event as a JSON line."""
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **event,
        }
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:
            self._log.error("audit_log_write_failed", error=str(exc))


# ── Module-level convenience ──────────────────────────────────────
analysis_logger = AnalysisLogger()
