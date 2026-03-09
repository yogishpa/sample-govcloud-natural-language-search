"""Property-based test for input sanitization neutralizing injection patterns.

Feature: nl-search-chatbot, Property 17: Input sanitization neutralizes injection patterns

Validates: Requirements 11.5

For any user input string containing known prompt injection patterns (e.g.,
"ignore previous instructions", system prompt override attempts, delimiter
injection), the sanitization function must transform the input such that the
output does not contain the raw injection pattern when passed to the LLM.
"""

import re

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from core.security import sanitize_input, MAX_QUERY_LENGTH


# ---------------------------------------------------------------------------
# Known injection patterns (mirrors core/security.py definitions)
# ---------------------------------------------------------------------------

_PHRASE_INJECTIONS: list[tuple[str, str]] = [
    ("ignore previous instructions", "instruction_override"),
    ("ignore all previous instructions", "instruction_override"),
    ("ignore prior instructions", "instruction_override"),
    ("ignore all prior instructions", "instruction_override"),
    ("disregard previous instructions", "instruction_override"),
    ("disregard all previous instructions", "instruction_override"),
    ("disregard above instructions", "instruction_override"),
    ("disregard all above instructions", "instruction_override"),
    ("forget everything", "instruction_override"),
    ("forget all previous instructions", "instruction_override"),
    ("forget previous instructions", "instruction_override"),
    ("new instructions:", "instruction_override"),
    ("override instructions", "instruction_override"),
    ("override previous instructions", "instruction_override"),
    ("system:", "system_prompt_override"),
    ("<system>", "system_prompt_override"),
    ("you are now", "role_override"),
    ("act as if", "role_override"),
    ("pretend you are", "role_override"),
]

_DELIMITER_INJECTIONS: list[str] = [
    "###",
    "---",
    "####",
    "----",
]

# Strategy: pick one phrase injection pattern at random.
_phrase_injection_st = st.sampled_from(_PHRASE_INJECTIONS)

# Strategy: pick one delimiter injection pattern at random.
_delimiter_injection_st = st.sampled_from(_DELIMITER_INJECTIONS)

# Strategy: short safe text for padding around injections.
_safe_padding = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Zs"),
        blacklist_characters="\x00",
    ),
    min_size=0,
    max_size=80,
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    injection=_phrase_injection_st,
    prefix=_safe_padding,
    suffix=_safe_padding,
)
def test_phrase_injection_neutralized(
    injection: tuple[str, str],
    prefix: str,
    suffix: str,
) -> None:
    """**Validates: Requirements 11.5**

    For any known phrase injection pattern embedded in arbitrary text,
    sanitize_input must remove the raw pattern and insert a [BLOCKED:...]
    marker.
    """
    pattern_text, category = injection
    text = f"{prefix} {pattern_text} {suffix}"

    # Skip if assembled text exceeds max length.
    assume(len(text) <= MAX_QUERY_LENGTH)

    result = sanitize_input(text)

    # The raw injection phrase must not survive sanitization (case-insensitive).
    assert pattern_text.lower() not in result.lower(), (
        f"Raw injection pattern '{pattern_text}' survived sanitization. "
        f"Input: {text!r}, Output: {result!r}"
    )

    # A BLOCKED marker for the correct category must be present.
    assert f"[BLOCKED:{category}]" in result, (
        f"Expected [BLOCKED:{category}] marker in output. "
        f"Input: {text!r}, Output: {result!r}"
    )


@settings(max_examples=100)
@given(
    delimiter=_delimiter_injection_st,
    prefix=_safe_padding,
    suffix=_safe_padding,
)
def test_delimiter_injection_neutralized(
    delimiter: str,
    prefix: str,
    suffix: str,
) -> None:
    """**Validates: Requirements 11.5**

    For any known delimiter injection pattern placed on its own line,
    sanitize_input must remove the raw delimiter line and insert a
    [BLOCKED:delimiter] marker.
    """
    # Delimiters must appear on their own line to be detected.
    text = f"{prefix}\n{delimiter}\n{suffix}"

    assume(len(text) <= MAX_QUERY_LENGTH)

    result = sanitize_input(text)

    # The raw standalone delimiter line must not survive.
    standalone_re = re.compile(
        r"^" + re.escape(delimiter) + r"\s*$", re.MULTILINE
    )
    assert not standalone_re.search(result), (
        f"Raw delimiter '{delimiter}' survived sanitization on its own line. "
        f"Input: {text!r}, Output: {result!r}"
    )

    # A BLOCKED:delimiter marker must be present.
    assert "[BLOCKED:delimiter]" in result, (
        f"Expected [BLOCKED:delimiter] marker in output. "
        f"Input: {text!r}, Output: {result!r}"
    )


@settings(max_examples=100)
@given(
    injection=_phrase_injection_st,
    prefix=_safe_padding,
    suffix=_safe_padding,
)
def test_sanitized_output_preserves_safe_content(
    injection: tuple[str, str],
    prefix: str,
    suffix: str,
) -> None:
    """**Validates: Requirements 11.5**

    Sanitization must neutralize injection patterns while preserving
    the surrounding safe text content (prefix and suffix).
    """
    import unicodedata

    pattern_text, _ = injection
    text = f"{prefix} {pattern_text} {suffix}"

    assume(len(text) <= MAX_QUERY_LENGTH)

    result = sanitize_input(text)

    # sanitize_input applies NFC normalization, so compare against
    # the NFC-normalized form of the prefix/suffix.
    stripped_prefix = unicodedata.normalize("NFC", prefix).strip()
    stripped_suffix = unicodedata.normalize("NFC", suffix).strip()
    if stripped_prefix:
        assert stripped_prefix in result, (
            f"Safe prefix '{stripped_prefix}' was lost during sanitization. "
            f"Output: {result!r}"
        )
    if stripped_suffix:
        assert stripped_suffix in result, (
            f"Safe suffix '{stripped_suffix}' was lost during sanitization. "
            f"Output: {result!r}"
        )
