"""Property-based test for new session initialization.

Feature: nl-search-chatbot, Property 7: New session initialization produces empty history

Validates: Requirements 5.4

For any newly created session, the conversation history must be empty
(zero messages) and the session_id must be a valid UUID.
"""

import uuid

from hypothesis import given, settings
from hypothesis import strategies as st

from services.session import SessionStore


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Number of sessions to create in a single test run: 1–50
_num_sessions_st = st.integers(min_value=1, max_value=50)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(num_sessions=_num_sessions_st)
def test_new_sessions_have_empty_history_and_valid_uuid(num_sessions: int) -> None:
    """**Validates: Requirements 5.4**

    For any number of newly created sessions, each session must have an
    empty conversation history (zero messages) and a session_id that is
    a valid UUID.
    """
    store = SessionStore()

    for _ in range(num_sessions):
        session = store.create()

        # session_id must be a valid UUID
        parsed = uuid.UUID(session.session_id)
        assert str(parsed) == session.session_id, (
            f"session_id {session.session_id!r} is not a canonical UUID string"
        )

        # Conversation history must be empty
        assert session.messages == [], (
            f"New session {session.session_id} has {len(session.messages)} messages, "
            "expected 0"
        )
