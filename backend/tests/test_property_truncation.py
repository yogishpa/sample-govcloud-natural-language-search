"""Property-based test for context window truncation.

Feature: nl-search-chatbot, Property 5: Context window truncation preserves most relevant chunks

Validates: Requirements 4.5

For any set of retrieved chunks whose total token count exceeds the configured
maximum context window size, the truncation function must: (a) produce a result
whose total token count is within the limit, and (b) retain chunks with higher
relevance scores over chunks with lower relevance scores (i.e., removed chunks
all have scores <= the minimum score of retained chunks).
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from services.llm import LLMService, _estimate_tokens
from services.search import SearchResultItem


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _chunks_exceeding_limit(draw: st.DrawFn) -> tuple[list[SearchResultItem], int]:
    """Generate chunks with distinct scores and a max_tokens forcing truncation.

    All chunks use the same text length so the greedy truncation algorithm
    (which processes chunks in descending score order) deterministically
    preserves the highest-scored chunks.  This isolates the relevance-
    ordering property from token-size variance.
    """
    n_chunks = draw(st.integers(min_value=2, max_value=20))

    # Fixed-length text so every chunk has the same token cost
    text_len = draw(st.integers(min_value=4, max_value=200))
    text_char = draw(st.characters(whitelist_categories=("L", "N")))
    chunk_text = text_char * text_len

    # Unique scores so ordering is deterministic
    scores = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=n_chunks,
            max_size=n_chunks,
            unique=True,
        )
    )

    chunks = [
        SearchResultItem(
            relevance_score=score,
            document_id="doc-id",
            document_name="doc.pdf",
            chunk_text=chunk_text,
            s3_uri="s3://bucket/doc.pdf",
        )
        for score in scores
    ]

    tokens_per_chunk = _estimate_tokens(chunk_text)
    total_tokens = tokens_per_chunk * n_chunks

    # max_tokens must be < total (force truncation) and >= one chunk
    assume(total_tokens > tokens_per_chunk)
    max_tokens = draw(
        st.integers(min_value=tokens_per_chunk, max_value=total_tokens - 1)
    )

    return chunks, max_tokens


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=_chunks_exceeding_limit())
def test_truncation_respects_token_limit(
    data: tuple[list[SearchResultItem], int],
) -> None:
    """**Validates: Requirements 4.5**

    The total estimated tokens of the truncated result must not exceed
    max_tokens.
    """
    chunks, max_tokens = data

    result = LLMService.truncate_context(chunks, max_tokens)

    result_tokens = sum(_estimate_tokens(c.chunk_text) for c in result)
    assert result_tokens <= max_tokens, (
        f"Truncated result has {result_tokens} tokens, exceeds limit of {max_tokens}"
    )


@settings(max_examples=100)
@given(data=_chunks_exceeding_limit())
def test_truncation_preserves_highest_relevance(
    data: tuple[list[SearchResultItem], int],
) -> None:
    """**Validates: Requirements 4.5**

    All removed chunks must have relevance scores <= the minimum score of
    retained chunks.  This ensures the most relevant chunks are kept.
    """
    chunks, max_tokens = data

    result = LLMService.truncate_context(chunks, max_tokens)
    assume(len(result) > 0)
    assume(len(result) < len(chunks))  # some chunks must have been removed

    retained_score_set = {c.relevance_score for c in result}
    min_retained_score = min(retained_score_set)

    removed_scores = [
        c.relevance_score for c in chunks
        if c.relevance_score not in retained_score_set
    ]

    for removed_score in removed_scores:
        assert removed_score <= min_retained_score, (
            f"Removed chunk with score {removed_score} > min retained score "
            f"{min_retained_score}. Truncation did not preserve most relevant chunks."
        )
