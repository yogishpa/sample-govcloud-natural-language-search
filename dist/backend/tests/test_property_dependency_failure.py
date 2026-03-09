"""Property-based test for dependency failure returning 503 with Retry-After.

Feature: nl-search-chatbot, Property 11: Dependency failure returns 503 with retry-after

Validates: Requirements 6.6

For any chat request where the LLM client or Vector Store client raises a
connection/availability error, the API must return HTTP 503 with a
Retry-After header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from hypothesis import given, settings
from hypothesis import strategies as st

from api.middleware import register_error_handlers
from api.routes import ServiceContainer, _services, router
from core.security import RateLimiter
from services.session import SessionStore


# ---------------------------------------------------------------------------
# Error strategies
# ---------------------------------------------------------------------------

# Botocore error codes that indicate service unavailability
_BOTOCORE_ERROR_CODES = [
    "ServiceUnavailableException",
    "InternalServerException",
    "ThrottlingException",
    "RequestLimitExceeded",
    "ProvisionedThroughputExceededException",
]

_botocore_error_code_st = st.sampled_from(_BOTOCORE_ERROR_CODES)

# Python-level connectivity errors
_PYTHON_ERRORS = [
    ConnectionError("Connection refused"),
    ConnectionResetError("Connection reset by peer"),
    TimeoutError("Request timed out"),
    OSError("Network is unreachable"),
    RuntimeError("Service unavailable"),
]

_python_error_st = st.sampled_from(_PYTHON_ERRORS)

# Which service fails: search or llm
_failing_service_st = st.sampled_from(["search", "llm"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_botocore_error(error_code: str) -> ClientError:
    """Create a botocore ClientError with the given error code."""
    return ClientError(
        error_response={"Error": {"Code": error_code, "Message": "test error"}},
        operation_name="TestOperation",
    )


def _build_app(
    search_service: Any,
    llm_service: Any,
    session_store: SessionStore,
) -> FastAPI:
    """Build a FastAPI app with the given services wired in."""
    app = FastAPI()
    register_error_handlers(app)

    container = ServiceContainer(
        search_service=search_service,
        llm_service=llm_service,
        session_store=session_store,
        rate_limiter=RateLimiter(max_requests=1000, window_seconds=60),
    )

    # Patch the module-level _services so get_services() returns our container
    import api.routes as routes_mod
    original = routes_mod._services
    routes_mod._services = container

    app.include_router(router)

    # Store original so caller can restore if needed
    app._original_services = original  # type: ignore[attr-defined]
    return app


def _make_search_mock(*, raises: Exception | None = None) -> MagicMock:
    """Create a mock search service that optionally raises on search()."""
    mock = MagicMock()
    if raises is not None:
        mock.search.side_effect = raises
    else:
        # Return a valid search result
        result = MagicMock()
        result.results = []
        result.message = ""
        result.total_found = 0
        mock.search.return_value = result
    return mock


def _make_llm_mock(*, raises: Exception | None = None) -> MagicMock:
    """Create a mock LLM service that optionally raises on generate()."""
    mock = MagicMock()
    if raises is not None:
        mock.generate.side_effect = raises
    else:
        mock.generate.return_value = {"answer": "test", "citations": []}
    return mock


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    error_code=_botocore_error_code_st,
    failing_service=_failing_service_st,
)
@pytest.mark.asyncio
async def test_botocore_error_returns_503_with_retry_after(
    error_code: str,
    failing_service: str,
) -> None:
    """**Validates: Requirements 6.6**

    For any chat request where the search or LLM service raises a botocore
    ClientError with a service-level error code, the API must return HTTP 503
    with a Retry-After header.
    """
    error = _make_botocore_error(error_code)

    if failing_service == "search":
        search_mock = _make_search_mock(raises=error)
        llm_mock = _make_llm_mock()
    else:
        search_mock = _make_search_mock()
        llm_mock = _make_llm_mock(raises=error)

    session_store = SessionStore()
    session = session_store.create()

    app = _build_app(search_mock, llm_mock, session_store)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={
                    "query": "test question",
                    "session_id": session.session_id,
                    "search_mode": "semantic",
                    "top_k": 5,
                },
            )

        assert resp.status_code == 503, (
            f"Expected 503 for {failing_service} raising ClientError({error_code}), "
            f"got {resp.status_code}"
        )
        assert resp.headers.get("retry-after") is not None, (
            f"Expected Retry-After header for {failing_service} raising "
            f"ClientError({error_code})"
        )
    finally:
        import api.routes as routes_mod
        routes_mod._services = app._original_services  # type: ignore[attr-defined]


@settings(max_examples=100)
@given(
    error=_python_error_st,
    failing_service=_failing_service_st,
)
@pytest.mark.asyncio
async def test_python_connection_error_returns_503_with_retry_after(
    error: Exception,
    failing_service: str,
) -> None:
    """**Validates: Requirements 6.6**

    For any chat request where the search or LLM service raises a Python
    connection/availability error (ConnectionError, TimeoutError, OSError),
    the API must return HTTP 503 with a Retry-After header.
    """
    if failing_service == "search":
        search_mock = _make_search_mock(raises=error)
        llm_mock = _make_llm_mock()
    else:
        search_mock = _make_search_mock()
        llm_mock = _make_llm_mock(raises=error)

    session_store = SessionStore()
    session = session_store.create()

    app = _build_app(search_mock, llm_mock, session_store)
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={
                    "query": "test question",
                    "session_id": session.session_id,
                    "search_mode": "semantic",
                    "top_k": 5,
                },
            )

        assert resp.status_code == 503, (
            f"Expected 503 for {failing_service} raising {type(error).__name__}, "
            f"got {resp.status_code}"
        )
        assert resp.headers.get("retry-after") is not None, (
            f"Expected Retry-After header for {failing_service} raising "
            f"{type(error).__name__}"
        )
    finally:
        import api.routes as routes_mod
        routes_mod._services = app._original_services  # type: ignore[attr-defined]
