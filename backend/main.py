"""FastAPI application entrypoint for the NL Search Chatbot backend.

Creates the FastAPI app with CORS middleware, registers error handlers and
API routes, and initializes all services during the lifespan startup phase.

Requirements: 6.1, 8.2
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import register_error_handlers
from api.routes import ServiceContainer, _services, router
from core.config import Settings
from core.logging import get_logger, setup_logging
from core.security import RateLimiter
from services.llm import LLMService
from services.search import SearchService
from services.session import SessionStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize services on startup, cleanup on shutdown."""
    # --- Startup ---
    setup_logging()
    logger.info("Starting NL Search Chatbot backend")

    settings = Settings()

    search_service = SearchService(settings=settings)
    llm_service = LLMService(settings=settings)
    session_store = SessionStore(default_max_history=settings.session_history_limit)
    rate_limiter = RateLimiter(settings=settings)

    # Populate the module-level service container used by routes
    _services.search_service = search_service
    _services.llm_service = llm_service
    _services.session_store = session_store
    _services.rate_limiter = rate_limiter

    logger.info("All services initialized successfully")

    yield

    # --- Shutdown ---
    logger.info("Shutting down NL Search Chatbot backend")


app = FastAPI(
    title="NL Search Chatbot",
    description="RAG-based natural language search chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware — allow all origins for demo; tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling middleware (validation errors, catch-all, botocore errors)
register_error_handlers(app)

# API routes
app.include_router(router)
