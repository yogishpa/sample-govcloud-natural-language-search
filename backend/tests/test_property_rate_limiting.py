"""Property-based test for rate limiting enforcement.

Feature: nl-search-chatbot, Property 12: Rate limiting enforces per-session limits

Validates: Requirements 6.7

For any session, if more than the configured rate limit of requests are made
within the rate window, subsequent requests must be rejected with HTTP 429
until the window resets.
"""

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from core.security import RateLimiter


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Random max_requests values (1-50 as specified in task).
_max_requests_st = st.integers(min_value=1, max_value=50)

# Random session IDs — simple alphanumeric strings.
_session_id_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=36,
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(max_requests=_max_requests_st, session_id=_session_id_st)
def test_rate_limit_allows_exactly_max_then_rejects(
    max_requests: int,
    session_id: str,
) -> None:
    """**Validates: Requirements 6.7**

    For any max_requests value and any session, the first max_requests calls
    must all return True (allowed), and subsequent calls must return False
    (rejected).
    """
    # Use a large window so no expiry occurs during the test.
    limiter = RateLimiter(max_requests=max_requests, window_seconds=3600)

    # First max_requests calls must all be allowed.
    for i in range(max_requests):
        assert limiter.check_rate_limit(session_id) is True, (
            f"Request {i + 1} of {max_requests} should be allowed "
            f"for session {session_id!r}"
        )

    # Additional calls must be rejected.
    for i in range(3):
        assert limiter.check_rate_limit(session_id) is False, (
            f"Request {max_requests + i + 1} should be rejected "
            f"(limit={max_requests}) for session {session_id!r}"
        )


@settings(max_examples=100)
@given(
    max_requests=_max_requests_st,
    session_a=_session_id_st,
    session_b=_session_id_st,
)
def test_rate_limit_sessions_are_independent(
    max_requests: int,
    session_a: str,
    session_b: str,
) -> None:
    """**Validates: Requirements 6.7**

    For any two distinct sessions sharing the same RateLimiter, exhausting
    the limit on one session must not affect the other session.
    """
    assume(session_a != session_b)

    limiter = RateLimiter(max_requests=max_requests, window_seconds=3600)

    # Exhaust session_a's limit.
    for _ in range(max_requests):
        limiter.check_rate_limit(session_a)

    # session_a should now be rejected.
    assert limiter.check_rate_limit(session_a) is False, (
        f"session_a ({session_a!r}) should be rate-limited"
    )

    # session_b must still be fully allowed.
    for i in range(max_requests):
        assert limiter.check_rate_limit(session_b) is True, (
            f"Request {i + 1} for session_b ({session_b!r}) should be allowed "
            f"even though session_a is exhausted"
        )
