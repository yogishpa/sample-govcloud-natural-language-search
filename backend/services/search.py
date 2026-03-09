"""Search service wrapping Bedrock Agent Runtime retrieve API.

Provides semantic, text, and hybrid search against a Bedrock Knowledge Base.
Returns ranked results with relevance scores and source document metadata.

Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4
"""

from dataclasses import dataclass, field
from typing import Literal

import boto3

from core.config import Settings
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResultItem:
    """A single search result with metadata."""

    relevance_score: float
    document_id: str
    document_name: str
    chunk_text: str
    s3_uri: str


@dataclass
class SearchResult:
    """Aggregated search response."""

    results: list[SearchResultItem] = field(default_factory=list)
    message: str = ""
    total_found: int = 0


class SearchService:
    """Wraps the Bedrock Agent Runtime ``retrieve`` API.

    Parameters
    ----------
    settings:
        Application settings providing the Knowledge Base ID and region.
    client:
        Optional pre-built boto3 client (useful for testing).
    """

    def __init__(self, settings: Settings, *, client=None) -> None:
        self._kb_id = settings.bedrock_kb_id
        self._client = client or boto3.client(
            "bedrock-agent-runtime",
            region_name=settings.bedrock_region,
        )

    def search(
        self,
        query: str,
        search_mode: Literal["semantic", "text", "hybrid"] = "semantic",
        top_k: int = 5,
    ) -> SearchResult:
        """Execute a search against the Bedrock Knowledge Base.

        Parameters
        ----------
        query:
            Natural language search query.
        search_mode:
            One of ``"semantic"``, ``"text"``, or ``"hybrid"``.
        top_k:
            Maximum number of results to return (1-20).

        Returns
        -------
        SearchResult
            Ranked results with metadata, or an empty result with a message
            when no documents match.
        """
        search_type_map = {
            "semantic": "SEMANTIC",
            "text": "HYBRID",  # Bedrock uses HYBRID for text-based; see note below
            "hybrid": "HYBRID",
        }
        # Bedrock Agent Runtime retrieve API accepts SEMANTIC or HYBRID.
        # For pure text mode we still use HYBRID but the KB handles it.
        # Semantic is the default vector-only mode.
        bedrock_search_type = search_type_map.get(search_mode, "SEMANTIC")

        retrieval_config: dict = {
            "vectorSearchConfiguration": {
                "numberOfResults": top_k,
            }
        }
        # Only set searchType when not using the default SEMANTIC
        if bedrock_search_type != "SEMANTIC":
            retrieval_config["vectorSearchConfiguration"]["overrideSearchType"] = bedrock_search_type

        try:
            response = self._client.retrieve(
                knowledgeBaseId=self._kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=retrieval_config,
            )
        except Exception:
            logger.error("Bedrock retrieve call failed", extra={"search_mode": search_mode})
            raise

        raw_results = response.get("retrievalResults", [])

        if not raw_results:
            return SearchResult(
                results=[],
                message="No relevant documents were found for your query.",
                total_found=0,
            )

        items: list[SearchResultItem] = []
        for r in raw_results:
            score = r.get("score", 0.0)
            content_body = r.get("content", {}).get("text", "")
            location = r.get("location", {})
            s3_location = location.get("s3Location", {})
            s3_uri = s3_location.get("uri", "")

            metadata = r.get("metadata", {})
            doc_id = metadata.get("x-amz-bedrock-kb-source-uri", s3_uri)
            doc_name = s3_uri.rsplit("/", 1)[-1] if s3_uri else ""

            items.append(
                SearchResultItem(
                    relevance_score=score,
                    document_id=doc_id,
                    document_name=doc_name,
                    chunk_text=content_body,
                    s3_uri=s3_uri,
                )
            )

        # Sort by descending relevance score and apply top-k
        items.sort(key=lambda item: item.relevance_score, reverse=True)
        filtered = items[: min(top_k, len(items))]

        return SearchResult(
            results=filtered,
            message="",
            total_found=len(raw_results),
        )
