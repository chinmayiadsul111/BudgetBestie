"""
Structured logging.

Every log line is written as JSON to a rotating file (machine-readable, one
object per line — easy to grep/ship to a log aggregator), and optionally
pretty-printed to the console for local dev. Warnings/errors from anywhere in
the agent pipeline (LLM timeouts, low-confidence retrieval, SQL failures,
critic rejections) all flow through this so there's one place to look.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config_loader import get_config, PROJECT_ROOT

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Renders each LogRecord as a single JSON line with structured fields."""

    RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Pull in any `extra={...}` fields the caller attached.
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)  # ensure serializable, else stringify
                    payload[key] = value
                except TypeError:
                    payload[key] = str(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class PrettyConsoleFormatter(logging.Formatter):
    """Human-readable console output for local development."""

    COLORS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        base = f"{color}[{record.levelname:<8}]{self.RESET} {record.name}: {record.getMessage()}"
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in JsonFormatter.RESERVED and not k.startswith("_") and k != "message"
        }
        if extras:
            base += f"  {extras}"
        return base


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = get_config()
    level_name = cfg.get("observability.logging.level", "INFO")
    json_dir = PROJECT_ROOT / cfg.get("observability.logging.json_logs_dir", "logs")
    json_filename = cfg.get("observability.logging.json_log_filename", "app.log.json")
    console_pretty = cfg.get("observability.logging.console_pretty", True)
    max_bytes = cfg.get("observability.logging.rotate_max_bytes", 5_000_000)
    backup_count = cfg.get("observability.logging.rotate_backup_count", 5)

    json_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level_name)
    root.handlers.clear()

    file_handler = logging.handlers.RotatingFileHandler(
        json_dir / json_filename, maxBytes=max_bytes, backupCount=backup_count,
    )
    file_handler.setFormatter(JsonFormatter())
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(PrettyConsoleFormatter() if console_pretty else JsonFormatter())
    root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
