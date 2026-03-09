"""Unit tests for Pydantic API models."""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from api.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    HealthResponse,
    SessionResponse,
)


class TestChatRequest:
    """Tests for ChatRequest validation."""

    def test_valid_request(self):
        req = ChatRequest(query="What is RAG?", session_id=str(uuid.uuid4()))
        assert req.query == "What is RAG?"
        assert req.search_mode == "semantic"
        assert req.top_k == 5

    def test_search_mode_options(self):
        sid = str(uuid.uuid4())
        for mode in ("semantic", "text", "hybrid"):
            req = ChatRequest(query="test", session_id=sid, search_mode=mode)
            assert req.search_mode == mode

    def test_invalid_search_mode(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="test", session_id=str(uuid.uuid4()), search_mode="invalid")

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="", session_id=str(uuid.uuid4()))

    def test_query_exceeds_max_length(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="a" * 1001, session_id=str(uuid.uuid4()))

    def test_query_at_max_length(self):
        req = ChatRequest(query="a" * 1000, session_id=str(uuid.uuid4()))
        assert len(req.query) == 1000

    def test_query_min_length(self):
        req = ChatRequest(query="x", session_id=str(uuid.uuid4()))
        assert req.query == "x"

    def test_invalid_session_id_pattern(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="test", session_id="not-a-uuid")

    def test_missing_query(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id=str(uuid.uuid4()))

    def test_missing_session_id(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="test")

    def test_top_k_below_min(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="test", session_id=str(uuid.uuid4()), top_k=0)

    def test_top_k_above_max(self):
        with pytest.raises(ValidationError):
            ChatRequest(query="test", session_id=str(uuid.uuid4()), top_k=21)

    def test_top_k_boundaries(self):
        sid = str(uuid.uuid4())
        assert ChatRequest(query="t", session_id=sid, top_k=1).top_k == 1
        assert ChatRequest(query="t", session_id=sid, top_k=20).top_k == 20


class TestCitation:
    """Tests for Citation model."""

    def test_valid_citation(self):
        c = Citation(
            document_id="doc-1",
            document_name="guide.pdf",
            chunk_text="Some relevant text",
            relevance_score=0.95,
            s3_uri="s3://bucket/guide.pdf",
        )
        assert c.relevance_score == 0.95


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_valid_response(self):
        resp = ChatResponse(
            answer="The answer is 42.",
            citations=[],
            search_mode_used="semantic",
            session_id=str(uuid.uuid4()),
        )
        assert resp.answer == "The answer is 42."
        assert resp.citations == []


class TestSessionResponse:
    """Tests for SessionResponse model."""

    def test_valid_session_response(self):
        now = datetime.now(timezone.utc)
        resp = SessionResponse(session_id=str(uuid.uuid4()), created_at=now)
        assert resp.created_at == now


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_valid_health_response(self):
        now = datetime.now(timezone.utc)
        resp = HealthResponse(
            status="healthy",
            components={"opensearch": "ok", "bedrock": "ok"},
            timestamp=now,
        )
        assert resp.status == "healthy"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            HealthResponse(
                status="unknown",
                components={},
                timestamp=datetime.now(timezone.utc),
            )
