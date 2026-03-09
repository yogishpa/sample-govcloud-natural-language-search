"""Tests for core structured logging module.

Validates JSON formatting, request trace ID propagation, sensitive field
filtering, and Bedrock API call logging.

Requirements: 8.6, 9.4, 11.3
"""

import json
import logging

import pytest

from core.logging import (
    JsonFormatter,
    SensitiveFieldFilter,
    _ExtraAdapter,
    get_logger,
    log_bedrock_call,
    request_trace_id,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    message: str = "test",
    level: int = logging.INFO,
    extra: dict | None = None,
) -> logging.LogRecord:
    """Create a minimal LogRecord, optionally with _extra attached."""
    record = logging.LogRecord(
        name="test_logger",
        level=level,
        pathname="",
        lineno=0,
        msg=message,
        args=None,
        exc_info=None,
    )
    if extra is not None:
        record._extra = extra  # type: ignore[attr-defined]
    return record


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------

class TestJsonFormatter:
    def test_output_is_valid_json(self):
        fmt = JsonFormatter()
        record = _make_record("hello")
        output = fmt.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello"

    def test_includes_standard_fields(self):
        fmt = JsonFormatter()
        record = _make_record("msg")
        parsed = json.loads(fmt.format(record))
        for key in ("timestamp", "level", "logger", "message", "trace_id"):
            assert key in parsed

    def test_includes_trace_id_from_contextvar(self):
        token = request_trace_id.set("abc-123")
        try:
            fmt = JsonFormatter()
            record = _make_record()
            parsed = json.loads(fmt.format(record))
            assert parsed["trace_id"] == "abc-123"
        finally:
            request_trace_id.reset(token)

    def test_trace_id_empty_when_not_set(self):
        fmt = JsonFormatter()
        record = _make_record()
        parsed = json.loads(fmt.format(record))
        assert parsed["trace_id"] == ""

    def test_extra_fields_merged(self):
        fmt = JsonFormatter()
        record = _make_record(extra={"model_id": "claude", "latency_ms": 42.5})
        parsed = json.loads(fmt.format(record))
        assert parsed["model_id"] == "claude"
        assert parsed["latency_ms"] == 42.5

    def test_exception_info_included(self):
        fmt = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            record = _make_record()
            record.exc_info = __import__("sys").exc_info()
        parsed = json.loads(fmt.format(record))
        assert "exception" in parsed
        assert "boom" in parsed["exception"]


# ---------------------------------------------------------------------------
# SensitiveFieldFilter
# ---------------------------------------------------------------------------

class TestSensitiveFieldFilter:
    def test_strips_query_from_extra(self):
        f = SensitiveFieldFilter()
        record = _make_record(extra={"query": "secret question", "model_id": "claude"})
        f.filter(record)
        assert "query" not in record._extra  # type: ignore[attr-defined]
        assert record._extra["model_id"] == "claude"  # type: ignore[attr-defined]

    def test_strips_user_query_from_extra(self):
        f = SensitiveFieldFilter()
        record = _make_record(extra={"user_query": "tell me secrets"})
        f.filter(record)
        assert "user_query" not in record._extra  # type: ignore[attr-defined]

    def test_strips_raw_query_from_extra(self):
        f = SensitiveFieldFilter()
        record = _make_record(extra={"raw_query": "my question"})
        f.filter(record)
        assert "raw_query" not in record._extra  # type: ignore[attr-defined]

    def test_passes_record_through(self):
        f = SensitiveFieldFilter()
        record = _make_record()
        assert f.filter(record) is True

    def test_strips_query_from_args_dict(self):
        f = SensitiveFieldFilter()
        record = _make_record()
        record.args = {"query": "secret", "safe_key": "ok"}
        f.filter(record)
        assert "query" not in record.args
        assert record.args["safe_key"] == "ok"


# ---------------------------------------------------------------------------
# get_logger / _ExtraAdapter
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_adapter(self):
        logger = get_logger("mymodule")
        assert isinstance(logger, _ExtraAdapter)

    def test_adapter_attaches_extra_to_record(self):
        """Verify that extra fields flow through to the formatter."""
        lines: list[str] = []

        class _Collector(logging.Handler):
            def __init__(self, dest: list[str]):
                super().__init__()
                self.dest = dest

            def emit(self, record: logging.LogRecord) -> None:
                self.dest.append(self.format(record))

        handler = _Collector(lines)
        handler.setFormatter(JsonFormatter())
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        try:
            logger = get_logger("test.adapter")
            logger.info("hi", extra={"foo": "bar"})
            parsed = json.loads(lines[-1])
            assert parsed["foo"] == "bar"
        finally:
            root.removeHandler(handler)


# ---------------------------------------------------------------------------
# log_bedrock_call
# ---------------------------------------------------------------------------

class TestLogBedrockCall:
    """Use a dedicated handler to capture JSON output reliably in tests."""

    @pytest.fixture(autouse=True)
    def _capture_handler(self):
        """Install a handler that collects formatted log lines."""
        self.lines: list[str] = []

        class _Collector(logging.Handler):
            def __init__(self, dest: list[str]):
                super().__init__()
                self.dest = dest

            def emit(self, record: logging.LogRecord) -> None:
                self.dest.append(self.format(record))

        handler = _Collector(self.lines)
        handler.setFormatter(JsonFormatter())
        handler.addFilter(SensitiveFieldFilter())
        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
        yield
        root.removeHandler(handler)

    def test_logs_required_fields(self):
        logger = get_logger("test.bedrock")
        log_bedrock_call(
            logger,
            inference_region="us-east-1",
            model_id="anthropic.claude-3-5-sonnet",
            latency_ms=123.4,
        )
        parsed = json.loads(self.lines[-1])
        assert parsed["inference_region"] == "us-east-1"
        assert parsed["model_id"] == "anthropic.claude-3-5-sonnet"
        assert parsed["latency_ms"] == 123.4
        assert parsed["event"] == "bedrock_api_call"

    def test_accepts_additional_extra_fields(self):
        logger = get_logger("test.bedrock.extra")
        log_bedrock_call(
            logger,
            inference_region="us-gov-east-1",
            model_id="titan-embed",
            latency_ms=55.0,
            status_code=200,
        )
        parsed = json.loads(self.lines[-1])
        assert parsed["status_code"] == 200

    def test_query_content_excluded_from_bedrock_log(self):
        """Even if someone accidentally passes query as extra, it gets stripped."""
        logger = get_logger("test.bedrock.sensitive")
        log_bedrock_call(
            logger,
            inference_region="us-east-1",
            model_id="claude",
            latency_ms=10.0,
            query="user secret question",
        )
        parsed = json.loads(self.lines[-1])
        # The SensitiveFieldFilter should have removed 'query'
        assert "query" not in parsed


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_idempotent(self):
        """Calling setup_logging twice should not duplicate handlers."""
        root = logging.getLogger()
        before = len(root.handlers)
        setup_logging()
        setup_logging()
        json_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and isinstance(h.formatter, JsonFormatter)
        ]
        assert len(json_handlers) == 1
