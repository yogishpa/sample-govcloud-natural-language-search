"""Pydantic request/response models for the NL Search Chatbot API.

Defines validated schemas for chat requests, responses, citations,
session management, and health check endpoints.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for the POST /chat endpoint."""

    query: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., pattern=r"^[0-9a-f-]{36}$")
    search_mode: Literal["semantic", "text", "hybrid"] = "semantic"
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    """A single citation referencing a source document chunk."""

    document_id: str
    document_name: str
    chunk_text: str
    relevance_score: float
    s3_uri: str


class ChatResponse(BaseModel):
    """Response payload for the POST /chat endpoint."""

    answer: str
    citations: list[Citation]
    search_mode_used: str
    session_id: str


class SessionResponse(BaseModel):
    """Response payload for the POST /sessions endpoint."""

    session_id: str
    created_at: datetime


class HealthResponse(BaseModel):
    """Response payload for the GET /health endpoint."""

    status: Literal["healthy", "degraded", "unhealthy"]
    components: dict[str, str]
    timestamp: datetime
