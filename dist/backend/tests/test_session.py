"""Unit tests for the in-memory session store."""

import uuid
from datetime import datetime, timezone

from services.session import ConversationMessage, Session, SessionStore


class TestConversationMessage:
    def test_default_timestamp(self):
        msg = ConversationMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert isinstance(msg.timestamp, datetime)

    def test_custom_timestamp(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        msg = ConversationMessage(role="assistant", content="hi", timestamp=ts)
        assert msg.timestamp == ts


class TestSession:
    def test_new_session_has_empty_messages(self):
        session = Session(session_id="test-id")
        assert session.messages == []
        assert session.max_history == 10

    def test_add_message_appends(self):
        session = Session(session_id="test-id")
        session.add_message("user", "hello")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "hello"

    def test_add_message_updates_last_active(self):
        session = Session(session_id="test-id")
        before = session.last_active
        session.add_message("user", "hello")
        assert session.last_active >= before

    def test_eviction_preserves_system_prompt(self):
        session = Session(session_id="test-id", max_history=2)
        session.messages.append(ConversationMessage(role="system", content="You are a helpful assistant."))
        # Add 3 pairs (exceeds limit of 2)
        for i in range(3):
            session.add_message("user", f"question {i}")
            session.add_message("assistant", f"answer {i}")

        system_msgs = [m for m in session.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0].content == "You are a helpful assistant."

    def test_eviction_removes_oldest_pairs_first(self):
        session = Session(session_id="test-id", max_history=2)
        for i in range(4):
            session.add_message("user", f"q{i}")
            session.add_message("assistant", f"a{i}")

        non_system = [m for m in session.messages if m.role != "system"]
        # Should have 2 pairs = 4 messages, the newest ones
        assert len(non_system) == 4
        assert non_system[0].content == "q2"
        assert non_system[1].content == "a2"
        assert non_system[2].content == "q3"
        assert non_system[3].content == "a3"

    def test_eviction_at_exact_limit(self):
        session = Session(session_id="test-id", max_history=2)
        session.add_message("user", "q0")
        session.add_message("assistant", "a0")
        session.add_message("user", "q1")
        session.add_message("assistant", "a1")
        # Exactly at limit, no eviction
        assert len(session.messages) == 4

    def test_get_context_messages_returns_copy(self):
        session = Session(session_id="test-id")
        session.add_message("user", "hello")
        ctx = session.get_context_messages()
        assert len(ctx) == 1
        # Modifying the returned list shouldn't affect the session
        ctx.clear()
        assert len(session.messages) == 1

    def test_clear_preserves_system_messages(self):
        session = Session(session_id="test-id")
        session.messages.append(ConversationMessage(role="system", content="system prompt"))
        session.add_message("user", "hello")
        session.add_message("assistant", "hi")
        session.clear()
        assert len(session.messages) == 1
        assert session.messages[0].role == "system"

    def test_clear_removes_all_when_no_system(self):
        session = Session(session_id="test-id")
        session.add_message("user", "hello")
        session.add_message("assistant", "hi")
        session.clear()
        assert len(session.messages) == 0

    def test_max_history_configurable(self):
        session = Session(session_id="test-id", max_history=1)
        session.add_message("user", "q0")
        session.add_message("assistant", "a0")
        session.add_message("user", "q1")
        session.add_message("assistant", "a1")
        non_system = [m for m in session.messages if m.role != "system"]
        assert len(non_system) == 2  # 1 pair


class TestSessionStore:
    def test_create_returns_session_with_uuid(self):
        store = SessionStore()
        session = store.create()
        # Validate it's a proper UUID
        uuid.UUID(session.session_id)
        assert session.messages == []

    def test_create_uses_default_max_history(self):
        store = SessionStore(default_max_history=5)
        session = store.create()
        assert session.max_history == 5

    def test_create_with_custom_max_history(self):
        store = SessionStore(default_max_history=5)
        session = store.create(max_history=3)
        assert session.max_history == 3

    def test_get_existing_session(self):
        store = SessionStore()
        session = store.create()
        retrieved = store.get(session.session_id)
        assert retrieved is session

    def test_get_nonexistent_returns_none(self):
        store = SessionStore()
        assert store.get("nonexistent-id") is None

    def test_delete_existing_session(self):
        store = SessionStore()
        session = store.create()
        assert store.delete(session.session_id) is True
        assert store.get(session.session_id) is None

    def test_delete_nonexistent_returns_false(self):
        store = SessionStore()
        assert store.delete("nonexistent-id") is False

    def test_add_message_to_existing_session(self):
        store = SessionStore()
        session = store.create()
        result = store.add_message(session.session_id, "user", "hello")
        assert result is True
        assert len(session.messages) == 1

    def test_add_message_to_nonexistent_session(self):
        store = SessionStore()
        result = store.add_message("nonexistent-id", "user", "hello")
        assert result is False

    def test_multiple_sessions_independent(self):
        store = SessionStore()
        s1 = store.create()
        s2 = store.create()
        store.add_message(s1.session_id, "user", "hello from s1")
        assert len(s1.messages) == 1
        assert len(s2.messages) == 0

    def test_session_ids_are_unique(self):
        store = SessionStore()
        ids = {store.create().session_id for _ in range(100)}
        assert len(ids) == 100
