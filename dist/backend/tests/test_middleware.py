"""Tests for error handling middleware.

Validates that the middleware maps exceptions to the correct HTTP status
codes and response bodies as specified in the design doc error table.

Requirements: 6.5, 6.6, 6.7
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field

from api.middleware import register_error_handlers


# ---------------------------------------------------------------------------
# Helpers – tiny FastAPI app wired with the middleware for testing
# ---------------------------------------------------------------------------


class _DummyBody(BaseModel):
    name: str = Field(..., min_length=1)


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with error handlers registered."""
    app = FastAPI()
    register_error_handlers(app)

    @app.post("/validate")
    async def validate_endpoint(body: _DummyBody):
        return {"ok": True}

    @app.get("/http-400")
    async def raise_400():
        raise HTTPException(status_code=400, detail="Bad request from route")

    @app.get("/http-404")
    async def raise_404():
        raise HTTPException(status_code=404, detail="Session abc-123 not found")

    @app.get("/http-429")
    async def raise_429():
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )

    @app.get("/http-503")
    async def raise_503():
        raise HTTPException(
            status_code=503,
            detail="Service unavailable",
            headers={"Retry-After": "30"},
        )

    @app.get("/unexpected")
    async def raise_unexpected():
        raise RuntimeError("something broke")

    @app.get("/botocore-error")
    async def raise_botocore():
        from botocore.exceptions import ClientError

        raise ClientError(
            error_response={
                "Error": {"Code": "ServiceUnavailableException", "Message": "nope"}
            },
            operation_name="InvokeModel",
        )

    return app


@pytest.fixture()
def app():
    return _make_app()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_error_returns_400(app: FastAPI):
    """RequestValidationError → 400 with descriptive message."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/validate", json={})
    assert resp.status_code == 400
    body = resp.json()
    assert "detail" in body
    # Should mention the missing field
    assert "name" in body["detail"].lower() or "required" in body["detail"].lower()


@pytest.mark.asyncio
async def test_validation_error_bad_value_returns_400(app: FastAPI):
    """RequestValidationError for constraint violation → 400."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/validate", json={"name": ""})
    assert resp.status_code == 400
    body = resp.json()
    assert "detail" in body


@pytest.mark.asyncio
async def test_http_exception_400_passthrough(app: FastAPI):
    """HTTPException 400 raised by route passes through."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/http-400")
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Bad request from route"


@pytest.mark.asyncio
async def test_http_exception_404_passthrough(app: FastAPI):
    """HTTPException 404 raised by route passes through."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/http-404")
    assert resp.status_code == 404
    assert "abc-123" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_http_exception_429_with_retry_after(app: FastAPI):
    """HTTPException 429 includes Retry-After header."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/http-429")
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "60"
    assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_http_exception_503_with_retry_after(app: FastAPI):
    """HTTPException 503 includes Retry-After header."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/http-503")
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "30"
    assert "detail" in resp.json()


@pytest.mark.asyncio
async def test_unexpected_error_returns_500(app: FastAPI):
    """Unhandled RuntimeError → 500 with generic message."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/unexpected")
    assert resp.status_code == 500
    body = resp.json()
    assert "detail" in body
    # Must NOT leak internal error details
    assert "something broke" not in body["detail"]
    assert "internal error" in body["detail"].lower()


@pytest.mark.asyncio
async def test_botocore_client_error_returns_503(app: FastAPI):
    """botocore ClientError → 503 with Retry-After header."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/botocore-error")
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "30"
    body = resp.json()
    assert "detail" in body
    assert "unavailable" in body["detail"].lower()
