"""Property-based test for retry behavior.

Feature: nl-search-chatbot, Property 14: Transient error retry with exponential backoff

Validates: Requirements 9.3

For any cross-region inference request that fails with a transient error
(throttling, timeout, 5xx), the system must retry up to 3 times with
exponentially increasing delays, and must not retry on non-transient errors
(4xx client errors other than throttling).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError
from hypothesis import given, settings
from hypothesis import strategies as st

from core.config import Settings
from services.llm import LLMService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_retryable_error_st = st.sampled_from([
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
])

_non_retryable_error_st = st.sampled_from([
    "ValidationException",
    "AccessDeniedException",
    "ResourceNotFoundException",
])


def _make_client_error(error_code: str) -> ClientError:
    """Build a ``ClientError`` with the given error code."""
    return ClientError(
        {"Error": {"Code": error_code, "Message": "test"}},
        "InvokeModel",
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(error_code=_retryable_error_st)
def test_retryable_errors_are_retried_max_times(error_code: str) -> None:
    """**Validates: Requirements 9.3**

    For any retryable error, invoke_model must be called exactly
    max_retries + 1 times (1 initial attempt + max_retries retries).
    """
    app_settings = Settings(retry_max_attempts=3)
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = _make_client_error(error_code)

    service = LLMService(settings=app_settings, client=mock_client)
    body = '{"prompt": "test"}'

    with patch("services.llm.time.sleep"):
        try:
            service._invoke_with_retry(body)
        except ClientError:
            pass

    assert mock_client.invoke_model.call_count == 4, (
        f"Expected 4 calls (1 initial + 3 retries) for {error_code}, "
        f"got {mock_client.invoke_model.call_count}"
    )


@settings(max_examples=100)
@given(error_code=_non_retryable_error_st)
def test_non_retryable_errors_are_not_retried(error_code: str) -> None:
    """**Validates: Requirements 9.3**

    For any non-retryable error, invoke_model must be called exactly once
    (no retry).
    """
    app_settings = Settings(retry_max_attempts=3)
    mock_client = MagicMock()
    mock_client.invoke_model.side_effect = _make_client_error(error_code)

    service = LLMService(settings=app_settings, client=mock_client)
    body = '{"prompt": "test"}'

    with patch("services.llm.time.sleep"):
        try:
            service._invoke_with_retry(body)
        except ClientError:
            pass

    assert mock_client.invoke_model.call_count == 1, (
        f"Expected exactly 1 call (no retry) for {error_code}, "
        f"got {mock_client.invoke_model.call_count}"
    )
