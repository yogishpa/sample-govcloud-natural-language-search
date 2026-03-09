"""Unit tests for LLM service.

Tests cover:
- Insufficient context returns fallback message (Req 4.4)
- Streaming SSE event format (Req 4.6)
- Non-transient errors are not retried (Req 9.3)

Requirements: 4.4, 4.6, 9.3
"""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from core.config import Settings
from services.llm import FALLBACK_MESSAGE, LLMService
from services.search import SearchResultItem
from services.session import ConversationMessage


def _make_settings() -> Settings:
    return Settings(bedrock_kb_id="test-kb")


def _make_chunk(text: str = "chunk text", score: float = 0.9) -> SearchResultItem:
    return SearchResultItem(
        relevance_score=score,
        document_id="doc-1",
        document_name="doc.pdf",
        chunk_text=text,
        s3_uri="s3://bucket/doc.pdf",
    )


def _mock_invoke_response(text: str) -> dict:
    body_content = json.dumps({"content": [{"type": "text", "text": text}]})
    return {"body": io.BytesIO(body_content.encode())}


def _client_error(code: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": f"{code} error"}},
        "InvokeModel",
    )


# -------------------------------------------------------------------
# Req 4.4: Insufficient context returns fallback message
# -------------------------------------------------------------------


class TestGenerateFallback:
    """generate() returns FALLBACK_MESSAGE when context is insufficient."""

    def test_empty_context_chunks(self):
        client = MagicMock()
        service = LLMService(_make_settings(), client=client)

        result = service.generate("what is X?", [], [])

        assert result["answer"] == FALLBACK_MESSAGE
        assert result["citations"] == []
        client.invoke_model.assert_not_called()

    def test_empty_llm_response(self):
        client = MagicMock()
        client.invoke_model.return_value = _mock_invoke_response("")
        service = LLMService(_make_settings(), client=client)

        result = service.generate("what is X?", [_make_chunk()], [])

        assert result["answer"] == FALLBACK_MESSAGE
        assert result["citations"] == []

    def test_blank_llm_response(self):
        client = MagicMock()
        client.invoke_model.return_value = _mock_invoke_response("   ")
        service = LLMService(_make_settings(), client=client)

        result = service.generate("what is X?", [_make_chunk()], [])

        assert result["answer"] == FALLBACK_MESSAGE


# -------------------------------------------------------------------
# Req 4.6: Streaming SSE event format
# -------------------------------------------------------------------


class TestGenerateStream:
    """generate_stream() yields tokens from the model stream."""

    def test_empty_context_yields_fallback(self):
        client = MagicMock()
        service = LLMService(_make_settings(), client=client)

        tokens = list(service.generate_stream("query", [], []))

        assert tokens == [FALLBACK_MESSAGE]
        client.invoke_model_with_response_stream.assert_not_called()

    @patch("services.llm.time.monotonic", return_value=0.0)
    @patch("services.llm.time.sleep")
    def test_valid_chunks_yield_tokens(self, _mock_sleep, _mock_monotonic):
        stream_events = [
            {"chunk": {"bytes": json.dumps({"type": "content_block_delta", "delta": {"text": "Hello"}}).encode()}},
            {"chunk": {"bytes": json.dumps({"type": "content_block_delta", "delta": {"text": " world"}}).encode()}},
            {"chunk": {"bytes": json.dumps({"type": "ping"}).encode()}},
        ]

        client = MagicMock()
        client.invoke_model_with_response_stream.return_value = {"body": stream_events}
        service = LLMService(_make_settings(), client=client)

        tokens = list(service.generate_stream("query", [_make_chunk()], []))

        assert tokens == ["Hello", " world"]


# -------------------------------------------------------------------
# Req 9.3: Non-transient errors are not retried
# -------------------------------------------------------------------


class TestNonTransientErrorNoRetry:
    """_invoke_with_retry raises immediately on non-transient errors."""

    @patch("services.llm.time.sleep")
    def test_validation_exception_no_retry(self, mock_sleep):
        client = MagicMock()
        client.invoke_model.side_effect = _client_error("ValidationException")
        service = LLMService(_make_settings(), client=client)

        with pytest.raises(ClientError) as exc_info:
            service._invoke_with_retry('{"test": true}')

        assert exc_info.value.response["Error"]["Code"] == "ValidationException"
        assert client.invoke_model.call_count == 1
        mock_sleep.assert_not_called()

    @patch("services.llm.time.sleep")
    def test_access_denied_no_retry(self, mock_sleep):
        client = MagicMock()
        client.invoke_model.side_effect = _client_error("AccessDeniedException")
        service = LLMService(_make_settings(), client=client)

        with pytest.raises(ClientError):
            service._invoke_with_retry('{"test": true}')

        assert client.invoke_model.call_count == 1
        mock_sleep.assert_not_called()

    @patch("services.llm.time.sleep")
    def test_resource_not_found_no_retry(self, mock_sleep):
        client = MagicMock()
        client.invoke_model.side_effect = _client_error("ResourceNotFoundException")
        service = LLMService(_make_settings(), client=client)

        with pytest.raises(ClientError):
            service._invoke_with_retry('{"test": true}')

        assert client.invoke_model.call_count == 1
        mock_sleep.assert_not_called()
