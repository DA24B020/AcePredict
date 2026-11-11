"""Structured JSON logging shared by the pipeline, the backend and the services.

Every log line is one JSON object. The backend puts a ``request_id`` into a context
variable so that all lines written while serving a request carry it.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Format a log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or request_id_var.get()
        if rid:
            payload["request_id"] = rid
        for key in ("stage", "event", "duration_s", "rows"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger once: JSON to stdout and optionally to a rotating file."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not _CONFIGURED:
        for h in list(root.handlers):
            root.removeHandler(h)
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(JsonFormatter())
        root.addHandler(stream)
        # Quieten chatty third-party loggers
        for noisy in ("py4j", "urllib3", "matplotlib", "filelock", "alembic", "git"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _CONFIGURED = True
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(
            isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == log_file.resolve()
            for h in root.handlers
        ):
            fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
            fh.setFormatter(JsonFormatter())
            root.addHandler(fh)
