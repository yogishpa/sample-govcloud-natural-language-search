"""LLM service wrapping Bedrock invoke_model and invoke_model_with_response_stream.

Uses cross-region inference profile ARN for Claude and direct model ID for
Titan Embeddings.  Builds prompts with retrieved context chunks and conversation
history, implements context window truncation, citation extraction, retry with
exponential backoff, and streaming response support.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.1, 9.2, 9.3, 9.4
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Generator

import boto3
from botocore.exceptions import ClientError

from core.config import Settings
from core.logging import get_logger, log_bedrock_call
from services.search import SearchResultItem
from services.session import ConversationMessage

logger = get_logger(__name__)

FALLBACK_MESSAGE = (
    "I couldn't find enough information in the available documents "
    "to answer your question."
)

# Retryable Bedrock error codes
_RETRYABLE_ERRORS = frozenset({
    "ThrottlingException",
    "ServiceUnavailableException",
    "InternalServerException",
})

# Non-retryable error codes — fail immediately
_NON_RETRYABLE_ERRORS = frozenset({
    "ValidationException",
    "AccessDeniedException",
    "ResourceNotFoundException",
})

# Regex for citation markers like [1], [2], etc.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


class LLMService:
    """Wraps Bedrock ``invoke_model`` / ``invoke_model_with_response_stream``.

    Parameters
    ----------
    settings:
        Application settings providing model IDs, ARNs, and retry config.
    client:
        Optional pre-built boto3 bedrock-runtime client (useful for testing).
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._model_id = settings.effective_inference_profile_arn
        self._embedding_model_id = settings.embedding_model_id
        self._max_context_tokens = settings.max_context_tokens
        self._max_retries = settings.retry_max_attempts
        self._base_delay = settings.retry_base_delay_seconds
        self._max_delay = settings.retry_max_delay_seconds
        self._region = settings.bedrock_region
        self._client = client or boto3.client(
            "bedrock-runtime",
            region_name=settings.bedrock_region,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        context_chunks: list[SearchResultItem],
        history: list[ConversationMessage],
    ) -> dict[str, Any]:
        """Generate an answer using Claude with retrieved context.

        Returns a dict with ``answer`` (str) and ``citations`` (list[dict]).
        If context is insufficient the answer is the fallback message.
        """
        truncated = self.truncate_context(context_chunks, self._max_context_tokens)

        if not truncated:
            return {"answer": FALLBACK_MESSAGE, "citations": []}

        prompt = self.build_prompt(query, truncated, history)
        body = self._build_request_body(prompt)

        response = self._invoke_with_retry(body)
        answer = self._parse_response(response)

        if not answer or not answer.strip():
            return {"answer": FALLBACK_MESSAGE, "citations": []}

        citations = self.extract_citations(answer, truncated)
        return {"answer": answer, "citations": citations}

    def generate_stream(
        self,
        query: str,
        context_chunks: list[SearchResultItem],
        history: list[ConversationMessage],
    ) -> Generator[str, None, None]:
        """Stream response tokens from Claude via SSE.

        Yields individual token strings as they arrive from the model.
        """
        truncated = self.truncate_context(context_chunks, self._max_context_tokens)

        if not truncated:
            yield FALLBACK_MESSAGE
            return

        prompt = self.build_prompt(query, truncated, history)
        body = self._build_request_body(prompt)

        start = time.monotonic()
        try:
            response = self._invoke_stream_with_retry(body)
            stream = response.get("body", [])
            for event in stream:
                chunk = event.get("chunk")
                if chunk:
                    payload = json.loads(chunk["bytes"].decode("utf-8"))
                    if payload.get("type") == "content_block_delta":
                        delta = payload.get("delta", {})
                        text = delta.get("text", "")
                        if text:
                            yield text
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            log_bedrock_call(
                logger,
                inference_region=self._region,
                model_id=self._model_id,
                latency_ms=round(latency_ms, 2),
                streaming=True,
            )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt(
        query: str,
        context_chunks: list[SearchResultItem],
        history: list[ConversationMessage],
    ) -> str:
        """Build the LLM prompt with context chunks and conversation history.

        Context chunks are numbered [1], [2], … so the model can cite them.
        Conversation history is included in chronological order.
        """
        parts: list[str] = []

        # System instruction
        parts.append(
            "You are a helpful assistant that answers questions based on the "
            "provided document context. Use citation markers like [1], [2] to "
            "reference the source documents. If the context does not contain "
            "enough information to answer, say so."
        )

        # Context section
        if context_chunks:
            parts.append("\n--- Retrieved Context ---")
            for idx, chunk in enumerate(context_chunks, start=1):
                parts.append(
                    f"[{idx}] (source: {chunk.document_name}, "
                    f"score: {chunk.relevance_score:.2f})\n{chunk.chunk_text}"
                )
            parts.append("--- End Context ---\n")

        # Conversation history in chronological order
        if history:
            parts.append("--- Conversation History ---")
            for msg in history:
                parts.append(f"{msg.role}: {msg.content}")
            parts.append("--- End History ---\n")

        # Current query
        parts.append(f"User question: {query}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Context truncation
    # ------------------------------------------------------------------

    @staticmethod
    def truncate_context(
        chunks: list[SearchResultItem],
        max_tokens: int,
    ) -> list[SearchResultItem]:
        """Truncate context chunks to fit within the token limit.

        Removes lowest-score chunks first until the total estimated token
        count is within *max_tokens*.  Returns a new list (does not mutate
        the input).
        """
        if not chunks:
            return []

        # Sort by relevance descending so we keep the best chunks
        sorted_chunks = sorted(chunks, key=lambda c: c.relevance_score, reverse=True)

        total_tokens = sum(_estimate_tokens(c.chunk_text) for c in sorted_chunks)

        if total_tokens <= max_tokens:
            return list(sorted_chunks)

        # Greedily keep highest-score chunks that fit
        kept: list[SearchResultItem] = []
        running_tokens = 0
        for chunk in sorted_chunks:
            chunk_tokens = _estimate_tokens(chunk.chunk_text)
            if running_tokens + chunk_tokens <= max_tokens:
                kept.append(chunk)
                running_tokens += chunk_tokens
            # Once we exceed the budget, skip remaining (lower-score) chunks

        return kept

    # ------------------------------------------------------------------
    # Citation extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_citations(
        response_text: str,
        chunks: list[SearchResultItem],
    ) -> list[dict[str, Any]]:
        """Parse citation markers [1], [2], … from the response text.

        Maps each marker to the corresponding chunk (1-indexed) and returns
        a list of citation dicts with document metadata.  Duplicate markers
        produce a single citation entry.
        """
        markers = _CITATION_RE.findall(response_text)
        seen: set[int] = set()
        citations: list[dict[str, Any]] = []

        for marker in markers:
            idx = int(marker)
            if idx < 1 or idx > len(chunks) or idx in seen:
                continue
            seen.add(idx)
            chunk = chunks[idx - 1]
            citations.append({
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "chunk_text": chunk.chunk_text,
                "relevance_score": chunk.relevance_score,
                "s3_uri": chunk.s3_uri,
            })

        return citations


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_request_body(self, prompt: str) -> str:
        """Build the JSON request body for Claude via Bedrock Messages API."""
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        return json.dumps(payload)

    def _invoke_with_retry(self, body: str) -> dict[str, Any]:
        """Invoke Bedrock with retry logic for transient errors.

        Retries up to ``max_retries`` times with exponential backoff and
        jitter for retryable errors only.  Non-retryable errors are raised
        immediately.
        """
        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            start = time.monotonic()
            try:
                response = self._client.invoke_model(
                    modelId=self._model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                )
                latency_ms = (time.monotonic() - start) * 1000
                log_bedrock_call(
                    logger,
                    inference_region=self._region,
                    model_id=self._model_id,
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                )
                return response
            except ClientError as exc:
                latency_ms = (time.monotonic() - start) * 1000
                error_code = exc.response.get("Error", {}).get("Code", "")

                log_bedrock_call(
                    logger,
                    inference_region=self._region,
                    model_id=self._model_id,
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    error_code=error_code,
                )

                if error_code in _NON_RETRYABLE_ERRORS:
                    raise

                if error_code not in _RETRYABLE_ERRORS:
                    raise

                last_exception = exc

                if attempt < self._max_retries:
                    delay = min(
                        self._base_delay * (2 ** attempt) + random.uniform(0, 0.5),
                        self._max_delay,
                    )
                    logger.warning(
                        "Retrying Bedrock call after transient error",
                        extra={
                            "error_code": error_code,
                            "attempt": attempt,
                            "delay_seconds": round(delay, 3),
                        },
                    )
                    time.sleep(delay)
            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000
                log_bedrock_call(
                    logger,
                    inference_region=self._region,
                    model_id=self._model_id,
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    error=str(exc),
                )
                raise

        # All retries exhausted
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry loop exited without result or exception")

    def _invoke_stream_with_retry(self, body: str) -> dict[str, Any]:
        """Invoke Bedrock streaming with retry logic for transient errors."""
        last_exception: Exception | None = None

        for attempt in range(self._max_retries + 1):
            start = time.monotonic()
            try:
                response = self._client.invoke_model_with_response_stream(
                    modelId=self._model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=body,
                )
                latency_ms = (time.monotonic() - start) * 1000
                log_bedrock_call(
                    logger,
                    inference_region=self._region,
                    model_id=self._model_id,
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    streaming=True,
                )
                return response
            except ClientError as exc:
                latency_ms = (time.monotonic() - start) * 1000
                error_code = exc.response.get("Error", {}).get("Code", "")

                log_bedrock_call(
                    logger,
                    inference_region=self._region,
                    model_id=self._model_id,
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    error_code=error_code,
                    streaming=True,
                )

                if error_code in _NON_RETRYABLE_ERRORS:
                    raise

                if error_code not in _RETRYABLE_ERRORS:
                    raise

                last_exception = exc

                if attempt < self._max_retries:
                    delay = min(
                        self._base_delay * (2 ** attempt) + random.uniform(0, 0.5),
                        self._max_delay,
                    )
                    logger.warning(
                        "Retrying Bedrock streaming call after transient error",
                        extra={
                            "error_code": error_code,
                            "attempt": attempt,
                            "delay_seconds": round(delay, 3),
                        },
                    )
                    time.sleep(delay)
            except Exception as exc:
                latency_ms = (time.monotonic() - start) * 1000
                log_bedrock_call(
                    logger,
                    inference_region=self._region,
                    model_id=self._model_id,
                    latency_ms=round(latency_ms, 2),
                    attempt=attempt,
                    error=str(exc),
                    streaming=True,
                )
                raise

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry loop exited without result or exception")

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> str:
        """Extract the text content from a Bedrock Claude response."""
        body_bytes = response.get("body")
        if body_bytes is None:
            return ""

        if hasattr(body_bytes, "read"):
            raw = body_bytes.read()
        else:
            raw = body_bytes

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        payload = json.loads(raw)
        content_blocks = payload.get("content", [])
        texts = [
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ]
        return "".join(texts)
