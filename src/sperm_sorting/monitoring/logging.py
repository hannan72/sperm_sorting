"""Structured logging setup.

Two output modes: human-readable for a console, and JSON lines for a log
collector. The JSON mode matters because this device runs unattended -- when a
run produces an unexpected sort, the log is the only account of what happened,
and grepping prose is not a plan.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

#: Attributes present on every LogRecord, which must not be copied into the
#: JSON payload as if they were caller-supplied context.
_STANDARD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": record.created,
            "iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(record.created)
            )
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    value = repr(value)
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


class ElapsedFormatter(logging.Formatter):
    """Human-readable, with seconds since process start.

    Relative time is more useful than a timestamp when reading a real-time
    log: "the queue backed up 4.2 s in" is the question being asked, not "the
    queue backed up at 14:07:33".
    """

    def __init__(self) -> None:
        super().__init__("%(elapsed)8.3f  %(levelname)-7s %(name)-34s %(message)s")
        self._start = time.monotonic()

    def format(self, record: logging.LogRecord) -> str:
        record.elapsed = time.monotonic() - self._start
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    *,
    json_logs: bool = False,
    stream: Any = None,
) -> None:
    """Configure the root logger. Idempotent -- safe to call more than once."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else ElapsedFormatter())
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # These are chatty at DEBUG and never say anything we need.
    for noisy in ("matplotlib", "PIL", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
