"""Property-based test for user query content exclusion from default logs.

Feature: nl-search-chatbot, Property 16: User query content excluded from default logs

Validates: Requirements 11.3

For any log entry produced during request processing with default logging
configuration, the log entry must not contain the raw user query text.
"""

import json
import logging

from hypothesis import given, settings
from hypothesis import strategies as st

from core.logging import (
    JsonFormatter,
    SensitiveFieldFilter,
    get_logger,
)


# Strategy: non-empty printable text that won't be a substring of standard
# log scaffolding (level names, logger names, etc.).  min_size=2 avoids
# single-char strings that could coincidentally match JSON punctuation.
_query_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=2,
    max_size=200,
).filter(lambda s: s.strip() != "")


def _make_capturing_logger(name: str, lines: list[str]) -> logging.Logger:
    """Return a stdlib logger wired with JsonFormatter + SensitiveFieldFilter.

    Captured formatted output is appended to *lines*.
    """

    class _Collector(logging.Handler):
        def __init__(self, dest: list[str]) -> None:
            super().__init__()
            self.dest = dest

        def emit(self, record: logging.LogRecord) -> None:
            self.dest.append(self.format(record))

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Remove any pre-existing handlers to isolate the test.
    logger.handlers.clear()
    logger.propagate = False

    handler = _Collector(lines)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(SensitiveFieldFilter())
    logger.addHandler(handler)
    return logger


@settings(max_examples=100)
@given(query=_query_text)
def test_query_excluded_from_extra_fields(query: str) -> None:
    """**Validates: Requirements 11.3**

    When a query is passed via the ``query``, ``user_query``, or
    ``raw_query`` extra fields, the formatted log output must not
    contain the sensitive key or its value in the parsed JSON.
    """
    lines: list[str] = []
    raw_logger = _make_capturing_logger("prop16.extra", lines)
    adapter = get_logger("prop16.extra")
    # Point the adapter at our isolated logger.
    adapter.logger = raw_logger

    for key in ("query", "user_query", "raw_query"):
        lines.clear()
        adapter.info("processing request", extra={key: query, "safe": "ok"})
        assert len(lines) == 1, f"Expected exactly one log line, got {len(lines)}"
        output = lines[0]
        # Verify the output is valid JSON and the sensitive key was stripped.
        parsed = json.loads(output)
        assert key not in parsed, (
            f"Sensitive key '{key}' leaked in parsed log output: {parsed}"
        )
        # Verify the query value does not appear as any JSON value.
        for k, v in parsed.items():
            assert v != query, (
                f"Raw query value leaked as value of key '{k}' in log output"
            )
        # Non-sensitive data must be retained.
        assert parsed["safe"] == "ok"


@settings(max_examples=100)
@given(query=_query_text)
def test_query_excluded_when_all_sensitive_keys_present(query: str) -> None:
    """**Validates: Requirements 11.3**

    When all three sensitive keys are present simultaneously, none of
    them leak into the formatted log output.
    """
    lines: list[str] = []
    raw_logger = _make_capturing_logger("prop16.all_keys", lines)
    adapter = get_logger("prop16.all_keys")
    adapter.logger = raw_logger

    adapter.info(
        "multi-field request",
        extra={
            "query": query,
            "user_query": query,
            "raw_query": query,
            "model_id": "claude",
        },
    )

    assert len(lines) == 1
    output = lines[0]
    parsed = json.loads(output)
    # Sensitive keys must be absent from the parsed JSON.
    assert "query" not in parsed, f"'query' key leaked in parsed log output: {parsed}"
    assert "user_query" not in parsed, f"'user_query' key leaked in parsed log output: {parsed}"
    assert "raw_query" not in parsed, f"'raw_query' key leaked in parsed log output: {parsed}"
    # The query value must not appear as any JSON value.
    for k, v in parsed.items():
        assert v != query, (
            f"Raw query value leaked as value of key '{k}' in log output"
        )
    assert parsed["model_id"] == "claude"
