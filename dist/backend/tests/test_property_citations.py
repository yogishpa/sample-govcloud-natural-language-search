"""Property-based test for citation extraction from LLM response.

Feature: nl-search-chatbot, Property 4: Citation extraction from LLM response

Validates: Requirements 4.3

For any LLM response containing citation markers in the expected format,
the citation parser must extract all citations with their corresponding
document references, and the count of extracted citations must equal the
count of unique citation markers in the response.
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from services.llm import LLMService
from services.search import SearchResultItem


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_text_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
    min_size=1,
    max_size=80,
)

_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

_chunk_st = st.builds(
    SearchResultItem,
    relevance_score=_score_st,
    document_id=_text_st,
    document_name=_text_st,
    chunk_text=_text_st,
    s3_uri=_text_st,
)

_CITATION_RE = re.compile(r"\[(\d+)\]")


@st.composite
def _response_with_markers(draw: st.DrawFn) -> tuple[str, list[SearchResultItem]]:
    """Generate a list of chunks and a response text with random citation markers.

    Markers are in the range [1..N] where N = len(chunks).  The response may
    contain duplicate markers and arbitrary surrounding text.
    """
    chunks = draw(st.lists(_chunk_st, min_size=1, max_size=10))
    n = len(chunks)

    # Pick a random subset of valid marker indices (at least one)
    marker_indices = draw(
        st.lists(st.integers(min_value=1, max_value=n), min_size=1, max_size=n * 3)
    )

    # Build response text with markers interspersed with filler
    parts: list[str] = []
    for idx in marker_indices:
        filler = draw(_text_st)
        parts.append(f"{filler} [{idx}]")
    parts.append(draw(_text_st))  # trailing text

    response_text = " ".join(parts)
    return response_text, chunks


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=_response_with_markers())
def test_citation_count_equals_unique_valid_markers(
    data: tuple[str, list[SearchResultItem]],
) -> None:
    """**Validates: Requirements 4.3**

    The count of extracted citations must equal the count of unique valid
    citation markers in the response text.
    """
    response_text, chunks = data
    citations = LLMService.extract_citations(response_text, chunks)

    # Compute expected unique valid markers
    all_markers = _CITATION_RE.findall(response_text)
    unique_valid = {int(m) for m in all_markers if 1 <= int(m) <= len(chunks)}

    assert len(citations) == len(unique_valid), (
        f"Expected {len(unique_valid)} citations but got {len(citations)}. "
        f"Markers found: {all_markers}, unique valid: {unique_valid}"
    )


@settings(max_examples=100)
@given(data=_response_with_markers())
def test_citation_document_references_match_chunks(
    data: tuple[str, list[SearchResultItem]],
) -> None:
    """**Validates: Requirements 4.3**

    Each extracted citation must carry the correct document_id and s3_uri
    from the corresponding chunk (1-indexed).  Citations are returned in
    order of first appearance in the response text.
    """
    response_text, chunks = data
    citations = LLMService.extract_citations(response_text, chunks)

    # Determine expected order: first-appearance of each unique valid marker
    all_markers = _CITATION_RE.findall(response_text)
    seen: set[int] = set()
    expected_order: list[int] = []
    for m in all_markers:
        idx = int(m)
        if 1 <= idx <= len(chunks) and idx not in seen:
            seen.add(idx)
            expected_order.append(idx)

    for citation, expected_idx in zip(citations, expected_order):
        expected_chunk = chunks[expected_idx - 1]
        assert citation["document_id"] == expected_chunk.document_id, (
            f"document_id mismatch for marker [{expected_idx}]"
        )
        assert citation["s3_uri"] == expected_chunk.s3_uri, (
            f"s3_uri mismatch for marker [{expected_idx}]"
        )
