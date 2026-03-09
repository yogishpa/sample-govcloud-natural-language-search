"""Unit tests for API routes.

Tests health endpoint status logic, session lifecycle, and input validation.

Requirements: 6.2, 6.3, 6.4, 6.5
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from api.middleware import register_error_handlers
from api.routes import ServiceContainer, router
from core.security import RateLimiter
from services.session import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_app(
    search_service: Any = None,
    llm_service: Any = None,
    session_store: SessionStore | None = None,
) -> FastAPI:
    """Build a FastAPI app with the given services wired in."""
    app = FastAPI()
    register_error_handlers(app)

    container = ServiceContainer(
        search_service=search_service,
        llm_service=llm_service,
        session_store=session_store or SessionStore(),
        rate_limiter=RateLimiter(max_requests=1000, window_seconds=60),
    )

    import api.routes as routes_mod

    original = routes_mod._services
    routes_mod._services = container
    app.include_router(router)
    app._original_services = original  # type: ignore[attr-defined]
    return app


def _restore(app: FastAPI) -> None:
    import api.routes as routes_mod

    routes_mod._services = app._original_services  # type: ignore[attr-defined]


def _make_search_mock(*, raises: Exception | None = None) -> MagicMock:
    mock = MagicMock()
    if raises is not None:
        mock.search.side_effect = raises
    else:
        result = MagicMock()
        result.results = []
        result.message = ""
        result.total_found = 0
        mock.search.return_value = result
    return mock


def _make_llm_mock(*, raises: Exception | None = None) -> MagicMock:
    mock = MagicMock()
    if raises is not None:
        mock.generate.side_effect = raises
    else:
        mock.generate.return_value = {"answer": "test answer", "citations": []}
    return mock


# ---------------------------------------------------------------------------
# 1. Health endpoint tests (Req 6.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_all_healthy() -> None:
    """GET /health with all services healthy → status 'healthy', both 'ok'."""
    search = _make_search_mock()
    llm = _make_llm_mock()
    app = _build_app(search_service=search, llm_service=llm)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["components"]["opensearch"] == "ok"
        assert data["components"]["bedrock"] == "ok"
    finally:
        _restore(app)


@pytest.mark.asyncio
async def test_health_search_failing_is_degraded() -> None:
    """GET /health with search service failing → 'degraded', opensearch 'error'."""
    search = _make_search_mock(raises=RuntimeError("down"))
    llm = _make_llm_mock()
    app = _build_app(search_service=search, llm_service=llm)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["components"]["opensearch"] == "error"
        assert data["components"]["bedrock"] == "ok"
    finally:
        _restore(app)


@pytest.mark.asyncio
async def test_health_all_failing_is_unhealthy() -> None:
    """GET /health with all services failing → 'unhealthy', both 'error'."""
    search = _make_search_mock(raises=RuntimeError("down"))
    llm = _make_llm_mock(raises=RuntimeError("down"))
    app = _build_app(search_service=search, llm_service=llm)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/health")
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["opensearch"] == "error"
        assert data["components"]["bedrock"] == "error"
    finally:
        _restore(app)


# ---------------------------------------------------------------------------
# 2. Session lifecycle tests (Req 6.3, 6.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_session_returns_201() -> None:
    """POST /sessions → 201, returns session_id and created_at."""
    app = _build_app()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/sessions")
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert "created_at" in data
        assert len(data["session_id"]) == 36  # UUID format
    finally:
        _restore(app)


@pytest.mark.asyncio
async def test_session_create_use_delete_verify_gone() -> None:
    """Full lifecycle: create → chat → delete → verify 404."""
    session_store = SessionStore()
    search = _make_search_mock()
    llm = _make_llm_mock()
    app = _build_app(search_service=search, llm_service=llm, session_store=session_store)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # Create session
            create_resp = await c.post("/sessions")
            assert create_resp.status_code == 201
            sid = create_resp.json()["session_id"]

            # Use session via /chat
            chat_resp = await c.post(
                "/chat",
                json={"query": "hello", "session_id": sid, "search_mode": "semantic", "top_k": 5},
            )
            assert chat_resp.status_code == 200

            # Delete session
            del_resp = await c.delete(f"/sessions/{sid}")
            assert del_resp.status_code == 204

            # Verify gone — chat with deleted session returns 404
            gone_resp = await c.post(
                "/chat",
                json={"query": "hello", "session_id": sid, "search_mode": "semantic", "top_k": 5},
            )
            assert gone_resp.status_code == 404
    finally:
        _restore(app)


@pytest.mark.asyncio
async def test_delete_nonexistent_session_returns_404() -> None:
    """DELETE /sessions/{nonexistent} → 404."""
    app = _build_app()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.delete("/sessions/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404
    finally:
        _restore(app)


# ---------------------------------------------------------------------------
# 3. Input validation tests (Req 6.5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_missing_query_returns_400() -> None:
    """POST /chat with missing query → 400."""
    app = _build_app()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post(
                "/chat",
                json={"session_id": "00000000-0000-0000-0000-000000000000"},
            )
        assert resp.status_code == 400
    finally:
        _restore(app)


@pytest.mark.asyncio
async def test_chat_missing_session_id_returns_400() -> None:
    """POST /chat with missing session_id → 400."""
    app = _build_app()
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/chat", json={"query": "hello"})
        assert resp.status_code == 400
    finally:
        _restore(app)
