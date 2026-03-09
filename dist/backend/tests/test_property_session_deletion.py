"""Property-based test for session deletion making sessions inaccessible.

Feature: nl-search-chatbot, Property 10: Session deletion makes session inaccessible

Validates: Requirements 6.4

For any existing session, after deletion via DELETE /sessions/{session_id},
subsequent requests using that session_id must fail with an appropriate error
(404 or equivalent).
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from services.session import SessionStore


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Number of sessions to create: 1–20
_num_sessions_st = st.integers(min_value=1, max_value=20)

# Short random content for messages
_content_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=40,
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(num_sessions=_num_sessions_st)
def test_deleted_session_get_returns_none(num_sessions: int) -> None:
    """**Validates: Requirements 6.4**

    After creating N sessions and deleting each one, get() must return None
    for every deleted session_id.
    """
    store = SessionStore()

    # Create sessions
    sessions = [store.create() for _ in range(num_sessions)]
    session_ids = [s.session_id for s in sessions]

    # Verify all sessions exist before deletion
    for sid in session_ids:
        assert store.get(sid) is not None, f"Session {sid} should exist before deletion"

    # Delete all sessions
    for sid in session_ids:
        result = store.delete(sid)
        assert result is True, f"delete() should return True for existing session {sid}"

    # Verify get() returns None for all deleted sessions
    for sid in session_ids:
        assert store.get(sid) is None, (
            f"get() should return None for deleted session {sid}"
        )


@settings(max_examples=100)
@given(num_sessions=_num_sessions_st)
def test_deleted_session_add_message_returns_false(num_sessions: int) -> None:
    """**Validates: Requirements 6.4**

    After creating N sessions and deleting each one, add_message() must
    return False for every deleted session_id.
    """
    store = SessionStore()

    # Create sessions
    sessions = [store.create() for _ in range(num_sessions)]
    session_ids = [s.session_id for s in sessions]

    # Verify add_message works before deletion
    for sid in session_ids:
        assert store.add_message(sid, "user", "test") is True, (
            f"add_message() should succeed for existing session {sid}"
        )

    # Delete all sessions
    for sid in session_ids:
        store.delete(sid)

    # Verify add_message() returns False for all deleted sessions
    for sid in session_ids:
        assert store.add_message(sid, "user", "post-delete msg") is False, (
            f"add_message() should return False for deleted session {sid}"
        )


@settings(max_examples=100)
@given(num_sessions=_num_sessions_st)
def test_delete_is_idempotent_returns_false_on_second_call(num_sessions: int) -> None:
    """**Validates: Requirements 6.4**

    Deleting an already-deleted session must return False, confirming the
    session is no longer present.
    """
    store = SessionStore()

    sessions = [store.create() for _ in range(num_sessions)]
    session_ids = [s.session_id for s in sessions]

    # First deletion succeeds
    for sid in session_ids:
        assert store.delete(sid) is True

    # Second deletion returns False (session already gone)
    for sid in session_ids:
        assert store.delete(sid) is False, (
            f"Second delete() should return False for already-deleted session {sid}"
        )


@settings(max_examples=100)
@given(
    num_sessions=_num_sessions_st,
    data=st.data(),
)
def test_deletion_does_not_affect_other_sessions(
    num_sessions: int,
    data: st.DataObject,
) -> None:
    """**Validates: Requirements 6.4**

    Deleting a subset of sessions must not affect the remaining sessions.
    """
    store = SessionStore()

    sessions = [store.create() for _ in range(num_sessions)]

    # Pick a random subset to delete (at least 1, at most all)
    delete_count = data.draw(st.integers(min_value=1, max_value=num_sessions))
    to_delete = [s.session_id for s in sessions[:delete_count]]
    to_keep = [s.session_id for s in sessions[delete_count:]]

    # Delete the chosen subset
    for sid in to_delete:
        store.delete(sid)

    # Deleted sessions are inaccessible
    for sid in to_delete:
        assert store.get(sid) is None, f"Deleted session {sid} should be inaccessible"

    # Remaining sessions are still accessible
    for sid in to_keep:
        assert store.get(sid) is not None, (
            f"Non-deleted session {sid} should still be accessible"
        )
