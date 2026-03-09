"""Property-based test for Bedrock API call logging completeness.

Feature: nl-search-chatbot, Property 15: Bedrock API call logging completeness

Validates: Requirements 9.4

For any Bedrock API call (LLM or embedding), the resulting log entry must
contain the inference region, model ID, and request latency in milliseconds.
"""

import json
import logging

from hypothesis import given, settings
from hypothesis import strategies as st

from core.logging import (
    JsonFormatter,
    SensitiveFieldFilter,
    get_logger,
    log_bedrock_call,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_region = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-"),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "")

_model_id = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-.:/_"),
    min_size=1,
    max_size=120,
).filter(lambda s: s.strip() != "")

_latency = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_capturing_logger(name: str, lines: list[str]) -> logging.Logger:
    """Return a stdlib logger wired with JsonFormatter + SensitiveFieldFilter."""

    class _Collector(logging.Handler):
        def __init__(self, dest: list[str]) -> None:
            super().__init__()
            self.dest = dest

        def emit(self, record: logging.LogRecord) -> None:
            self.dest.append(self.format(record))

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    handler = _Collector(lines)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(SensitiveFieldFilter())
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

@settings(max_examples=100)
@given(region=_region, model=_model_id, latency=_latency)
def test_bedrock_log_contains_required_fields(
    region: str, model: str, latency: float
) -> None:
    """**Validates: Requirements 9.4**

    For any Bedrock API call, the resulting log entry must contain the
    inference_region, model_id, and latency_ms fields matching the inputs.
    """
    lines: list[str] = []
    raw_logger = _make_capturing_logger("prop15.bedrock", lines)
    adapter = get_logger("prop15.bedrock")
    adapter.logger = raw_logger

    log_bedrock_call(
        adapter,
        inference_region=region,
        model_id=model,
        latency_ms=latency,
    )

    assert len(lines) == 1, f"Expected 1 log line, got {len(lines)}"
    parsed = json.loads(lines[0])

    assert "inference_region" in parsed, "Missing 'inference_region' in log entry"
    assert parsed["inference_region"] == region

    assert "model_id" in parsed, "Missing 'model_id' in log entry"
    assert parsed["model_id"] == model

    assert "latency_ms" in parsed, "Missing 'latency_ms' in log entry"
    assert parsed["latency_ms"] == latency
