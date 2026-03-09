"""Input sanitization and rate limiting module.

Provides input validation, prompt injection detection/neutralization,
and per-session sliding window rate limiting.

Requirements: 6.7, 11.5
"""

import re
import time
import unicodedata
from collections import defaultdict
from threading import Lock
from typing import Optional

from core.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 1000

# Prompt injection patterns to detect and neutralize.
# Each tuple is (compiled regex, human-readable label).
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE), "instruction_override"),
    (re.compile(r"ignore\s+(all\s+)?prior\s+instructions", re.IGNORECASE), "instruction_override"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE), "instruction_override"),
    (re.compile(r"forget\s+everything", re.IGNORECASE), "instruction_override"),
    (re.compile(r"forget\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE), "instruction_override"),
    (re.compile(r"system\s*:", re.IGNORECASE), "system_prompt_override"),
    (re.compile(r"<\s*system\s*>", re.IGNORECASE), "system_prompt_override"),
    (re.compile(r"you\s+are\s+now", re.IGNORECASE), "role_override"),
    (re.compile(r"new\s+instructions\s*:", re.IGNORECASE), "instruction_override"),
    (re.compile(r"override\s+(previous\s+)?instructions", re.IGNORECASE), "instruction_override"),
    (re.compile(r"act\s+as\s+if", re.IGNORECASE), "role_override"),
    (re.compile(r"pretend\s+you\s+are", re.IGNORECASE), "role_override"),
]

# Delimiter injection patterns (standalone delimiter lines).
_DELIMITER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^#{3,}\s*$", re.MULTILINE),   # ### on its own line
    re.compile(r"^-{3,}\s*$", re.MULTILINE),   # --- on its own line
]

# Control character regex: matches C0/C1 control chars except common whitespace.
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


def sanitize_input(text: str) -> str:
    """Sanitize user input before it reaches the LLM or search engine.

    Steps applied in order:
    1. Length validation — raise ``ValueError`` if > MAX_QUERY_LENGTH.
    2. Strip control characters and null bytes.
    3. Normalize Unicode to NFC form (prevents homoglyph attacks).
    4. Detect and neutralize prompt injection patterns.

    Returns the sanitized string.
    """
    # 1. Length validation
    if len(text) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
        )

    # 2. Strip control characters (keep \n, \r, \t as they are normal whitespace)
    result = _CONTROL_CHAR_RE.sub("", text)

    # 3. Unicode NFC normalization
    result = unicodedata.normalize("NFC", result)

    # 4. Neutralize injection patterns
    result = _neutralize_injections(result)

    return result


def _neutralize_injections(text: str) -> str:
    """Detect and neutralize known prompt injection patterns.

    Matched patterns are replaced with a sanitized marker that does not
    contain the original text, ensuring the raw injection pattern never
    reaches the LLM.  For example ``ignore previous instructions`` becomes
    ``[BLOCKED:instruction_override]``.
    """
    result = text

    # Neutralize phrase-based injection patterns.
    for pattern, label in _INJECTION_PATTERNS:
        result = pattern.sub(f"[BLOCKED:{label}]", result)

    # Neutralize standalone delimiter lines.
    for pattern in _DELIMITER_PATTERNS:
        result = pattern.sub("[BLOCKED:delimiter]", result)

    return result


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Per-session sliding window rate limiter.

    Tracks request timestamps per session and rejects requests that exceed
    the configured limit within the configured window.

    Parameters
    ----------
    max_requests:
        Maximum number of requests allowed per session within the window.
    window_seconds:
        Duration of the sliding window in seconds.
    """

    def __init__(
        self,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        _settings = settings or get_settings()
        self.max_requests = max_requests if max_requests is not None else _settings.rate_limit_max_requests
        self.window_seconds = window_seconds if window_seconds is not None else _settings.rate_limit_window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def check_rate_limit(self, session_id: str) -> bool:
        """Check whether a request from *session_id* is allowed.

        Returns ``True`` if the request is within the rate limit,
        ``False`` if the limit has been exceeded.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        with self._lock:
            # Prune timestamps outside the current window.
            timestamps = self._requests[session_id]
            self._requests[session_id] = [
                ts for ts in timestamps if ts > window_start
            ]

            if len(self._requests[session_id]) >= self.max_requests:
                return False

            self._requests[session_id].append(now)
            return True

    def reset(self, session_id: Optional[str] = None) -> None:
        """Reset rate limit state.

        If *session_id* is provided, only that session is reset.
        Otherwise all sessions are cleared.
        """
        with self._lock:
            if session_id is not None:
                self._requests.pop(session_id, None)
            else:
                self._requests.clear()
