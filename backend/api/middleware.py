"""Error handling middleware for the NL Search Chatbot API.

Provides centralized exception handling that maps errors to appropriate
HTTP responses with consistent JSON format.

Error Categories (from design doc):
| Error Category              | HTTP Status | Behavior                                    |
|-----------------------------|-------------|---------------------------------------------|
| Invalid input               | 400         | Return descriptive error message             |
| Session not found           | 404         | Return error with session_id                 |
| Rate limit exceeded         | 429         | Return error with Retry-After header         |
| Bedrock/OpenSearch unavail  | 503         | Return service unavailable with Retry-After  |
| Unexpected internal error   | 500         | Log full stack trace, return generic message |

Requirements: 6.5, 6.6, 6.7
"""

from __future__ import annotations

import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import get_logger

logger = get_logger(__name__)

# Retry-After value (seconds) for 503 responses caused by AWS service errors.
_SERVICE_RETRY_AFTER = "30"


async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic / FastAPI request validation errors → HTTP 400."""
    errors = exc.errors()
    messages = []
    for err in errors:
        loc = " -> ".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        messages.append(f"{loc}: {msg}")
    detail = "; ".join(messages) if messages else "Validation error"
    return JSONResponse(status_code=400, content={"detail": detail})


async def _http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Pass through HTTPException with its status code and headers.

    Routes already raise HTTPException for 400, 404, 429, 503 cases;
    this handler ensures they are rendered as JSON with any extra headers
    (e.g. Retry-After).
    """
    headers = getattr(exc, "headers", None) or {}
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers,
    )


class _CatchAllMiddleware(BaseHTTPMiddleware):
    """Middleware that catches unhandled exceptions and botocore ClientError.

    Starlette's built-in ServerErrorMiddleware re-raises unhandled exceptions
    before ``add_exception_handler(Exception, ...)`` can fire.  This middleware
    sits inside the stack and converts those exceptions to proper JSON responses.
    """

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            # Check for botocore ClientError first (more specific).
            try:
                from botocore.exceptions import ClientError

                if isinstance(exc, ClientError):
                    logger.error(
                        "AWS service error",
                        extra={"error": str(exc)},
                        exc_info=True,
                    )
                    return JSONResponse(
                        status_code=503,
                        content={
                            "detail": "Service temporarily unavailable. Please retry later."
                        },
                        headers={"Retry-After": _SERVICE_RETRY_AFTER},
                    )
            except ImportError:
                pass

            # Generic catch-all — log full stack trace, return generic message.
            logger.error(
                "Unexpected internal error",
                extra={"error_type": type(exc).__name__},
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An internal error occurred. Please try again later."
                },
            )


def register_error_handlers(app: FastAPI) -> None:
    """Register all exception handlers on the FastAPI application.

    Call this from ``main.py`` after creating the app instance::

        app = FastAPI(...)
        register_error_handlers(app)
    """
    # Structured handlers for FastAPI-level exceptions.
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPException, _http_exception_handler)

    # Catch-all middleware for unhandled exceptions and botocore errors.
    app.add_middleware(_CatchAllMiddleware)
