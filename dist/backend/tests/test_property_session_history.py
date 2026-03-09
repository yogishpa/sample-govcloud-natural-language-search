"""Property-based test for session history limit and eviction order.

Feature: nl-search-chatbot, Property 6: Session history respects size limit and eviction order

Validates: Requirements 5.1, 5.3

For any session with a configured history limit of N message pairs, after adding
any number of messages: (a) the stored history never exceeds N user-assistant
pairs, (b) when the limit is exceeded, the oldest messages are evicted first,
and (c) the system prompt is always preserved regardless of eviction.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from services.session import ConversationMessage, Session


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# History limit: 1–20 pairs
_max_history_st = st.integers(min_value=1, max_value=20)

# Number of user-assistant pairs to add: 1–50
_num_pairs_st = st.integers(min_value=1, max_value=50)

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
@given(
    max_history=_max_history_st,
    num_pairs=_num_pairs_st,
    contents=st.data(),
)
def test_history_never_exceeds_limit(
    max_history: int,
    num_pairs: int,
    contents: st.DataObject,
) -> None:
    """**Validates: Requirements 5.1, 5.3**

    After adding any number of user-assistant pairs, the stored non-system
    message pairs never exceed the configured max_history limit.
    """
    session = Session(session_id="prop-test", max_history=max_history)

    for _ in range(num_pairs):
        user_content = contents.draw(_content_st)
        asst_content = contents.draw(_content_st)
        session.add_message("user", user_content)
        session.add_message("assistant", asst_content)

    non_system = [m for m in session.messages if m.role != "system"]
    user_count = sum(1 for m in non_system if m.role == "user")
    asst_count = sum(1 for m in non_system if m.role == "assistant")
    pair_count = min(user_count, asst_count)

    assert pair_count <= max_history, (
        f"History has {pair_count} pairs but limit is {max_history}"
    )


@settings(max_examples=100)
@given(
    max_history=_max_history_st,
    num_pairs=_num_pairs_st,
)
def test_eviction_retains_most_recent_messages(
    max_history: int,
    num_pairs: int,
) -> None:
    """**Validates: Requirements 5.1, 5.3**

    When the limit is exceeded, the oldest messages are evicted first,
    so the retained messages are the most recently added ones.
    """
    session = Session(session_id="prop-test", max_history=max_history)

    # Add numbered pairs so we can verify ordering
    for i in range(num_pairs):
        session.add_message("user", f"q{i}")
        session.add_message("assistant", f"a{i}")

    non_system = [m for m in session.messages if m.role != "system"]
    retained_pairs = min(num_pairs, max_history)

    # We expect exactly retained_pairs pairs (2 messages each)
    assert len(non_system) == retained_pairs * 2, (
        f"Expected {retained_pairs * 2} non-system messages, got {len(non_system)}"
    )

    # The retained messages should be the last `retained_pairs` pairs
    expected_start = num_pairs - retained_pairs
    for idx in range(retained_pairs):
        pair_idx = expected_start + idx
        user_msg = non_system[idx * 2]
        asst_msg = non_system[idx * 2 + 1]
        assert user_msg.content == f"q{pair_idx}", (
            f"Expected user msg 'q{pair_idx}' at position {idx * 2}, "
            f"got '{user_msg.content}'"
        )
        assert asst_msg.content == f"a{pair_idx}", (
            f"Expected assistant msg 'a{pair_idx}' at position {idx * 2 + 1}, "
            f"got '{asst_msg.content}'"
        )


@settings(max_examples=100)
@given(
    max_history=_max_history_st,
    num_pairs=_num_pairs_st,
    system_content=_content_st,
)
def test_system_prompt_preserved_after_eviction(
    max_history: int,
    num_pairs: int,
    system_content: str,
) -> None:
    """**Validates: Requirements 5.1, 5.3**

    The system prompt is always preserved regardless of eviction. System
    messages are never removed and do not count toward the pair limit.
    """
    session = Session(session_id="prop-test", max_history=max_history)

    # Add a system prompt before any user messages
    session.messages.append(
        ConversationMessage(role="system", content=system_content)
    )

    for i in range(num_pairs):
        session.add_message("user", f"q{i}")
        session.add_message("assistant", f"a{i}")

    # System message must still be present
    system_msgs = [m for m in session.messages if m.role == "system"]
    assert len(system_msgs) == 1, (
        f"Expected 1 system message, found {len(system_msgs)}"
    )
    assert system_msgs[0].content == system_content, (
        f"System prompt content changed: expected {system_content!r}, "
        f"got {system_msgs[0].content!r}"
    )

    # Non-system pairs must still respect the limit
    non_system = [m for m in session.messages if m.role != "system"]
    pair_count = min(
        sum(1 for m in non_system if m.role == "user"),
        sum(1 for m in non_system if m.role == "assistant"),
    )
    assert pair_count <= max_history, (
        f"History has {pair_count} pairs but limit is {max_history}"
    )
