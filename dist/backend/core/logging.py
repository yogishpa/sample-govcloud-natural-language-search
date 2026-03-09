"""Structured logging module with JSON formatting and request tracing.

Provides JSON-structured log output with request trace ID support.
User query content is excluded from default log output (Req 11.3).
Includes a helper for logging Bedrock API calls with region, model_id,
and latency_ms fields (Req 9.4).

Requirements: 8.6, 9.4, 11.3
"""

import json
import logging
import time
from contextvars import ContextVar
from typing import Any

# Context variable holding the current request's trace ID.
request_trace_id: ContextVar[str] = ContextVar("request_trace_id", default="")

# Keys whose values must be stripped from log records to protect user privacy.
_SENSITIVE_KEYS = frozenset({"query", "user_query", "raw_query"})


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects.

    Every entry includes timestamp, level, logger name, message, and the
    current request trace ID (if set).  Extra fields attached to the record
    are merged into the JSON payload.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": request_trace_id.get(""),
        }

        # Merge any extra fields the caller attached via the `extra` kwarg.
        if hasattr(record, "_extra"):
            entry.update(record._extra)

        # Include exception info when present.
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


class SensitiveFieldFilter(logging.Filter):
    """Strips user query content from log records.

    Inspects the ``_extra`` dict attached to each record and removes any
    key found in ``_SENSITIVE_KEYS``.  This ensures raw user queries never
    appear in default log output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "_extra") and isinstance(record._extra, dict):
            for key in _SENSITIVE_KEYS:
                record._extra.pop(key, None)
        # Also scrub the message args in case query was interpolated.
        if record.args and isinstance(record.args, dict):
            record.args = {
                k: v for k, v in record.args.items() if k not in _SENSITIVE_KEYS
            }
        return True


class _ExtraAdapter(logging.LoggerAdapter):
    """Adapter that stores extra fields in a ``_extra`` attribute on the record.

    This keeps extra data separate from built-in LogRecord attributes so the
    ``JsonFormatter`` and ``SensitiveFieldFilter`` can process them cleanly.
    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra = kwargs.get("extra", {})
        # Merge adapter-level extras with call-site extras.
        merged = {**self.extra, **extra}  # type: ignore[arg-type]
        kwargs["extra"] = {"_extra": merged}
        return msg, kwargs


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with JSON formatting and the sensitive-field filter."""
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on repeated calls (e.g. tests).
    if not any(isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler.addFilter(SensitiveFieldFilter())
        root.addHandler(handler)


def get_logger(name: str) -> _ExtraAdapter:
    """Return a logger adapter that supports structured extra fields."""
    return _ExtraAdapter(logging.getLogger(name), {})


def log_bedrock_call(
    logger: _ExtraAdapter,
    *,
    inference_region: str,
    model_id: str,
    latency_ms: float,
    **extra: Any,
) -> None:
    """Log a Bedrock API call with the required operational fields (Req 9.4).

    Parameters
    ----------
    logger:
        Logger instance (from ``get_logger``).
    inference_region:
        AWS region where the inference was routed.
    model_id:
        Bedrock model identifier or inference profile ARN.
    latency_ms:
        Request round-trip time in milliseconds.
    **extra:
        Any additional fields to include in the log entry.
    """
    logger.info(
        "Bedrock API call completed",
        extra={
            "inference_region": inference_region,
            "model_id": model_id,
            "latency_ms": latency_ms,
            "event": "bedrock_api_call",
            **extra,
        },
    )
