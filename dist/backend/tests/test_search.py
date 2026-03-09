"""Unit tests for search service edge cases.

Tests cover:
- Empty result set returns message (Req 2.4)
- k=1 returns single result; k > total returns all
- Hybrid mode parameter is passed correctly
- Semantic mode uses default (no overrideSearchType)

Requirements: 2.4, 3.4
"""

from unittest.mock import MagicMock

from core.config import Settings
from services.search import SearchService


def _make_settings() -> Settings:
    """Create a Settings instance with a test KB ID."""
    return Settings(bedrock_kb_id="test-kb")


def _mock_retrieval_result(score: float, index: int) -> dict:
    """Build a single Bedrock retrieval result dict."""
    return {
        "score": score,
        "content": {"text": f"chunk text {index}"},
        "location": {
            "s3Location": {"uri": f"s3://bucket/doc{index}.pdf"},
        },
        "metadata": {
            "x-amz-bedrock-kb-source-uri": f"s3://bucket/doc{index}.pdf",
        },
    }


class TestEmptyResultSet:
    """Req 2.4: empty result set returns appropriate message."""

    def test_empty_results_returns_message(self):
        client = MagicMock()
        client.retrieve.return_value = {"retrievalResults": []}

        service = SearchService(_make_settings(), client=client)
        result = service.search("find nothing")

        assert result.results == []
        assert result.total_found == 0
        assert result.message == "No relevant documents were found for your query."

    def test_empty_results_missing_key(self):
        """retrievalResults key absent should also yield empty."""
        client = MagicMock()
        client.retrieve.return_value = {}

        service = SearchService(_make_settings(), client=client)
        result = service.search("find nothing")

        assert result.results == []
        assert result.message == "No relevant documents were found for your query."


class TestTopKFiltering:
    """Req 2.3: top-k filtering returns correct count."""

    def test_k1_returns_single_result(self):
        client = MagicMock()
        client.retrieve.return_value = {
            "retrievalResults": [_mock_retrieval_result(0.9 - i * 0.1, i) for i in range(5)]
        }

        service = SearchService(_make_settings(), client=client)
        result = service.search("query", top_k=1)

        assert len(result.results) == 1
        assert result.results[0].relevance_score == 0.9

    def test_k_greater_than_total_returns_all(self):
        client = MagicMock()
        client.retrieve.return_value = {
            "retrievalResults": [_mock_retrieval_result(0.8 - i * 0.1, i) for i in range(3)]
        }

        service = SearchService(_make_settings(), client=client)
        result = service.search("query", top_k=10)

        assert len(result.results) == 3
        assert result.total_found == 3


class TestSearchModeParameter:
    """Req 3.4: search mode parameter is passed correctly."""

    def test_hybrid_mode_sets_override_search_type(self):
        client = MagicMock()
        client.retrieve.return_value = {"retrievalResults": [_mock_retrieval_result(0.9, 0)]}

        service = SearchService(_make_settings(), client=client)
        service.search("query", search_mode="hybrid")

        call_kwargs = client.retrieve.call_args[1]
        vector_config = call_kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]
        assert vector_config["overrideSearchType"] == "HYBRID"

    def test_semantic_mode_no_override(self):
        client = MagicMock()
        client.retrieve.return_value = {"retrievalResults": [_mock_retrieval_result(0.9, 0)]}

        service = SearchService(_make_settings(), client=client)
        service.search("query", search_mode="semantic")

        call_kwargs = client.retrieve.call_args[1]
        vector_config = call_kwargs["retrievalConfiguration"]["vectorSearchConfiguration"]
        assert "overrideSearchType" not in vector_config
