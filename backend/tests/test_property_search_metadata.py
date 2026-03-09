"""Property-based test for search result metadata.

Feature: nl-search-chatbot, Property 1: Search results contain required metadata

Validates: Requirements 2.2

For any search result returned by the search service, the result object must
contain a non-null relevance score (float between 0 and 1) and a non-empty
source document reference (document_id and s3_uri).
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from core.config import Settings
from services.search import SearchService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# S3 URI: always non-empty, realistic shape
_s3_uri_st = st.from_regex(r"s3://[a-z][a-z0-9\-]{2,20}/[a-z0-9/\-_]{1,40}\.[a-z]{2,4}", fullmatch=True)

# Relevance score as returned by Bedrock (float 0-1)
_score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Chunk text
_chunk_text_st = st.text(min_size=1, max_size=200)

# Number of results in a single Bedrock response (1-15)
_num_results_st = st.integers(min_value=1, max_value=15)


def _bedrock_result_st():
    """Strategy that builds a single Bedrock retrievalResults entry."""
    return st.fixed_dictionaries({
        "score": _score_st,
        "content": st.fixed_dictionaries({"text": _chunk_text_st}),
        "location": st.fixed_dictionaries({
            "s3Location": st.fixed_dictionaries({"uri": _s3_uri_st}),
        }),
        "metadata": _s3_uri_st.map(
            lambda uri: {"x-amz-bedrock-kb-source-uri": uri}
        ),
    })


_bedrock_response_st = st.lists(
    _bedrock_result_st(),
    min_size=1,
    max_size=15,
)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(retrieval_results=_bedrock_response_st)
def test_search_results_contain_required_metadata(
    retrieval_results: list[dict],
) -> None:
    """**Validates: Requirements 2.2**

    For any search result returned by the search service, each result item
    must have: a relevance_score that is a float between 0 and 1, a non-empty
    document_id, and a non-empty s3_uri.
    """
    # Build a mock boto3 client that returns the generated response
    mock_client = MagicMock()
    mock_client.retrieve.return_value = {
        "retrievalResults": retrieval_results,
    }

    test_settings = Settings(bedrock_kb_id="test-kb-id")
    service = SearchService(test_settings, client=mock_client)

    result = service.search("test query")

    # Every returned item must satisfy the metadata contract
    assert len(result.results) > 0, "Expected at least one result"

    for item in result.results:
        # relevance_score must be a float in [0, 1]
        assert isinstance(item.relevance_score, float), (
            f"relevance_score should be float, got {type(item.relevance_score)}"
        )
        assert 0.0 <= item.relevance_score <= 1.0, (
            f"relevance_score {item.relevance_score} not in [0, 1]"
        )

        # document_id must be non-empty
        assert isinstance(item.document_id, str) and len(item.document_id) > 0, (
            f"document_id must be a non-empty string, got {item.document_id!r}"
        )

        # s3_uri must be non-empty
        assert isinstance(item.s3_uri, str) and len(item.s3_uri) > 0, (
            f"s3_uri must be a non-empty string, got {item.s3_uri!r}"
        )
