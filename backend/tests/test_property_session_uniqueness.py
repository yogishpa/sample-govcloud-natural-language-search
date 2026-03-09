"""Property-based test for session ID uniqueness.

Feature: nl-search-chatbot, Property 9: Session ID uniqueness

Validates: Requirements 6.3

For any sequence of N session creation requests, all N returned session IDs
must be distinct.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from services.session import SessionStore


# Strategy: number of sessions to create (2–200)
_num_sessions_st = st.integers(min_value=2, max_value=200)


@settings(max_examples=100)
@given(n=_num_sessions_st)
def test_all_session_ids_are_distinct(n: int) -> None:
    """**Validates: Requirements 6.3**

    For any sequence of N session creation requests, all N returned session
    IDs must be distinct.
    """
    store = SessionStore()
    session_ids = [store.create().session_id for _ in range(n)]

    assert len(set(session_ids)) == n, (
        f"Expected {n} unique session IDs but got {len(set(session_ids))} unique "
        f"out of {n} created sessions"
    )
