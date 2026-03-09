"""Unit tests for main.py application wiring.

Verifies that the FastAPI app is configured correctly with CORS, error handlers,
router, and that the lifespan initializes all services properly.

Requirements: 6.1, 8.2
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

import api.routes as routes_mod
from main import app


# ---------------------------------------------------------------------------
# 1. App configuration tests
# ---------------------------------------------------------------------------


def test_app_title_and_version() -> None:
    """App metadata is set correctly."""
    assert app.title == "NL Search Chatbot"
    assert app.version == "0.1.0"


def test_cors_middleware_registered() -> None:
    """CORS middleware is present in the middleware stack."""
    middleware_classes = [m.cls.__name__ for m in app.user_middleware if hasattr(m, "cls")]
    assert "CORSMiddleware" in middleware_classes


def test_router_included() -> None:
    """API routes from api.routes are registered on the app."""
    paths = [route.path for route in app.routes]
    assert "/chat" in paths
    assert "/sessions" in paths
    assert "/health" in paths


# ---------------------------------------------------------------------------
# 2. Lifespan / service initialization tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_populates_services() -> None:
    """The lifespan context manager populates the module-level _services container."""
    from main import lifespan

    # Save originals so we can restore after the test
    original = routes_mod._services.__dict__.copy()
    try:
        async with lifespan(app):
            services = routes_mod._services
            assert services.search_service is not None
            assert services.llm_service is not None
            assert services.session_store is not None
            assert services.rate_limiter is not None
    finally:
        # Restore original state
        for k, v in original.items():
            setattr(routes_mod._services, k, v)


@pytest.mark.asyncio
async def test_health_endpoint_accessible() -> None:
    """GET /health is reachable through the wired-up app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/health")
    # The endpoint should respond (status depends on whether real AWS services are reachable,
    # but the route itself must not 404).
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "components" in data


@pytest.mark.asyncio
async def test_sessions_endpoint_accessible() -> None:
    """POST /sessions is reachable and creates a session."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.post("/sessions")
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_old_placeholder_health_removed() -> None:
    """The old placeholder /health (defined directly on app) is replaced by the router version.

    The router's /health returns a HealthResponse with 'components' key,
    not the old simple {"status": "healthy"} dict.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        resp = await c.get("/health")
    data = resp.json()
    # Router version includes 'components' and 'timestamp'; old placeholder did not.
    assert "components" in data
    assert "timestamp" in data
