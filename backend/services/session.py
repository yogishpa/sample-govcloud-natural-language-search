"""In-memory session store for conversation history management.

Provides session creation, retrieval, deletion, and message management
with configurable history limits and oldest-first eviction that preserves
system prompts.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ConversationMessage:
    """A single message in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Session:
    """A conversation session with message history and eviction logic."""

    session_id: str
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_history: int = 10  # Max user-assistant pairs to retain

    def add_message(self, role: str, content: str) -> None:
        """Append a message and evict oldest pairs if over the limit.

        The history limit counts user-assistant pairs. System messages are
        never evicted and do not count toward the pair limit.
        """
        self.messages.append(
            ConversationMessage(role=role, content=content, timestamp=datetime.now(timezone.utc))
        )
        self.last_active = datetime.now(timezone.utc)
        self._evict_if_needed()

    def get_context_messages(self) -> list[ConversationMessage]:
        """Return messages suitable for LLM context, preserving system prompt."""
        return list(self.messages)

    def clear(self) -> None:
        """Reset conversation history, preserving system messages."""
        self.messages = [m for m in self.messages if m.role == "system"]
        self.last_active = datetime.now(timezone.utc)

    def _evict_if_needed(self) -> None:
        """Remove oldest user-assistant pairs if count exceeds max_history."""
        system_msgs = [m for m in self.messages if m.role == "system"]
        non_system_msgs = [m for m in self.messages if m.role != "system"]

        # Count pairs: each pair is one user + one assistant message
        pair_count = min(
            sum(1 for m in non_system_msgs if m.role == "user"),
            sum(1 for m in non_system_msgs if m.role == "assistant"),
        )

        while pair_count > self.max_history:
            # Find and remove the oldest user-assistant pair
            first_user_idx = None
            first_assistant_idx = None

            for i, m in enumerate(non_system_msgs):
                if m.role == "user" and first_user_idx is None:
                    first_user_idx = i
                elif m.role == "assistant" and first_assistant_idx is None:
                    first_assistant_idx = i
                if first_user_idx is not None and first_assistant_idx is not None:
                    break

            if first_user_idx is None or first_assistant_idx is None:
                break

            # Remove the later index first to avoid shifting
            indices = sorted([first_user_idx, first_assistant_idx], reverse=True)
            for idx in indices:
                non_system_msgs.pop(idx)

            pair_count -= 1

        # Reconstruct: system messages first, then non-system in order
        self.messages = system_msgs + non_system_msgs


class SessionStore:
    """In-memory session store backed by a dictionary."""

    def __init__(self, default_max_history: int = 10) -> None:
        self._sessions: dict[str, Session] = {}
        self._default_max_history = default_max_history

    def create(self, max_history: int | None = None) -> Session:
        """Create a new session with a unique ID and empty history."""
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            max_history=max_history if max_history is not None else self._default_max_history,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed, False otherwise."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to an existing session. Returns False if session not found."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.add_message(role, content)
        return True
