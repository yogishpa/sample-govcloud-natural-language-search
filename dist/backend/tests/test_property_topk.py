"""Property-based test for top-k filtering.

Feature: nl-search-chatbot, Property 2: Top-k filtering returns correct count and ordering

Validates: Requirements 2.3

For any list of scored search results and any valid value of k (1 ≤ k ≤ 20),
the filtered result set must contain exactly min(k, total_results) items, and
the items must be sorted in descending order by relevance score.
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from core.config import Settings
from services.search import SearchService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_s3_uri_st = st.from_regex(
    r"s3://[a-z][a-z0-9\-]{2,20}/[a-z0-9/\-_]{1,40}\.[a-z]{2,4}",
    fullmatch=True,
)

_score_st = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

_chunk_text_st = st.text(min_size=1, max_size=200)

_k_st = st.integers(min_value=1, max_value=20)


def _bedrock_result_st():
    """Strategy that builds a single Bedrock retrievalResults entry."""
    return st.fixed_dictionaries(
        {
            "score": _score_st,
            "content": st.fixed_dictionaries({"text": _chunk_text_st}),
            "location": st.fixed_dictionaries(
                {
                    "s3Location": st.fixed_dictionaries({"uri": _s3_uri_st}),
                }
            ),
            "metadata": _s3_uri_st.map(
                lambda uri: {"x-amz-bedrock-kb-source-uri": uri}
            ),
        }
    )


_bedrock_response_st = st.lists(
    _bedrock_result_st(),
    min_size=1,
    max_size=30,
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(retrieval_results=_bedrock_response_st, k=_k_st)
def test_topk_filtering_returns_correct_count_and_ordering(
    retrieval_results: list[dict],
    k: int,
) -> None:
    """**Validates: Requirements 2.3**

    For any list of scored search results and any valid value of k (1 ≤ k ≤ 20),
    the filtered result set must contain exactly min(k, total_results) items,
    and the items must be sorted in descending order by relevance score.
    """
    mock_client = MagicMock()
    mock_client.retrieve.return_value = {
        "retrievalResults": retrieval_results,
    }

    test_settings = Settings(bedrock_kb_id="test-kb-id")
    service = SearchService(test_settings, client=mock_client)

    result = service.search("test query", top_k=k)

    total_results = len(retrieval_results)
    expected_count = min(k, total_results)

    # Assert correct count
    assert len(result.results) == expected_count, (
        f"Expected {expected_count} results (k={k}, total={total_results}), "
        f"got {len(result.results)}"
    )

    # Assert descending order by relevance_score
    scores = [item.relevance_score for item in result.results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Results not sorted in descending order: "
            f"score[{i}]={scores[i]} < score[{i+1}]={scores[i+1]}"
        )
