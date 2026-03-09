"""Property-based test for input validation rejecting malformed requests.

Feature: nl-search-chatbot, Property 8: Input validation rejects malformed requests

Validates: Requirements 3.4, 6.1, 6.5

For any chat request payload missing a required field (query or session_id),
or containing an invalid search_mode value (not one of "semantic", "text",
"hybrid"), or containing a query that is empty or exceeds 1000 characters,
the API must return HTTP 400 with a descriptive error message.
"""

import uuid

from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pydantic import ValidationError

from api.models import ChatRequest


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_valid_session_id = st.uuids().map(str)

_valid_search_modes = ("semantic", "text", "hybrid")

# Generates search_mode values that are NOT in the allowed set.
_invalid_search_mode = st.text(min_size=1, max_size=30).filter(
    lambda s: s not in _valid_search_modes
)

# Queries that are empty (length 0).
_empty_query = st.just("")

# Queries that exceed the 1000-character limit.
_oversized_query = st.integers(min_value=1001, max_value=2000).flatmap(
    lambda n: st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=n,
        max_size=n,
    )
)

# Session IDs that do NOT match the UUID hex-with-dashes pattern ^[0-9a-f-]{36}$.
_invalid_session_id = st.text(min_size=1, max_size=50).filter(
    lambda s: not _is_valid_session_id(s)
)

# top_k values outside the valid 1-20 range.
_top_k_too_low = st.integers(max_value=0)
_top_k_too_high = st.integers(min_value=21, max_value=10000)


def _is_valid_session_id(s: str) -> bool:
    """Check if a string matches the session_id pattern ^[0-9a-f-]{36}$."""
    import re
    return bool(re.match(r"^[0-9a-f-]{36}$", s))


# ---------------------------------------------------------------------------
# Property Tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(session_id=_valid_session_id)
def test_missing_query_rejected(session_id: str) -> None:
    """**Validates: Requirements 3.4, 6.1, 6.5**

    A ChatRequest missing the required ``query`` field must raise
    ValidationError.
    """
    try:
        ChatRequest(**{"session_id": session_id})  # type: ignore[arg-type]
        raise AssertionError("Expected ValidationError for missing query")
    except ValidationError:
        pass


@settings(max_examples=100)
@given(query=st.text(min_size=1, max_size=100).filter(lambda s: s.strip() != ""))
def test_missing_session_id_rejected(query: str) -> None:
    """**Validates: Requirements 3.4, 6.1, 6.5**

    A ChatRequest missing the required ``session_id`` field must raise
    ValidationError.
    """
    try:
        ChatRequest(**{"query": query})  # type: ignore[arg-type]
        raise AssertionError("Expected ValidationError for missing session_id")
    except ValidationError:
        pass


@settings(max_examples=100)
@given(
    invalid_mode=_invalid_search_mode,
    session_id=_valid_session_id,
)
def test_invalid_search_mode_rejected(invalid_mode: str, session_id: str) -> None:
    """**Validates: Requirements 3.4, 6.1, 6.5**

    A ChatRequest with a search_mode not in {"semantic", "text", "hybrid"}
    must raise ValidationError.
    """
    try:
        ChatRequest(query="valid query", session_id=session_id, search_mode=invalid_mode)  # type: ignore[arg-type]
        raise AssertionError(
            f"Expected ValidationError for invalid search_mode '{invalid_mode}'"
        )
    except ValidationError:
        pass


@settings(max_examples=100)
@given(session_id=_valid_session_id)
def test_empty_query_rejected(session_id: str) -> None:
    """**Validates: Requirements 3.4, 6.1, 6.5**

    A ChatRequest with an empty query string must raise ValidationError.
    """
    try:
        ChatRequest(query="", session_id=session_id)
        raise AssertionError("Expected ValidationError for empty query")
    except ValidationError:
        pass


@settings(max_examples=100)
@given(
    oversized=_oversized_query,
    session_id=_valid_session_id,
)
def test_oversized_query_rejected(oversized: str, session_id: str) -> None:
    """**Validates: Requirements 3.4, 6.1, 6.5**

    A ChatRequest with a query exceeding 1000 characters must raise
    ValidationError.
    """
    assume(len(oversized) > 1000)
    try:
        ChatRequest(query=oversized, session_id=session_id)
        raise AssertionError(
            f"Expected ValidationError for query of length {len(oversized)}"
        )
    except ValidationError:
        pass


@settings(max_examples=100)
@given(invalid_sid=_invalid_session_id)
def test_invalid_session_id_rejected(invalid_sid: str) -> None:
    """**Validates: Requirements 3.4, 6.1, 6.5**

    A ChatRequest with a session_id that does not match the UUID pattern
    ``^[0-9a-f-]{36}$`` must raise ValidationError.
    """
    try:
        ChatRequest(query="valid query", session_id=invalid_sid)
        raise AssertionError(
            f"Expected ValidationError for invalid session_id '{invalid_sid}'"
        )
    except ValidationError:
        pass


@settings(max_examples=100)
@given(
    bad_top_k=st.one_of(_top_k_too_low, _top_k_too_high),
    session_id=_valid_session_id,
)
def test_top_k_out_of_range_rejected(bad_top_k: int, session_id: str) -> None:
    """**Validates: Requirements 3.4, 6.1, 6.5**

    A ChatRequest with top_k outside the valid range [1, 20] must raise
    ValidationError.
    """
    try:
        ChatRequest(query="valid query", session_id=session_id, top_k=bad_top_k)
        raise AssertionError(
            f"Expected ValidationError for top_k={bad_top_k}"
        )
    except ValidationError:
        pass
