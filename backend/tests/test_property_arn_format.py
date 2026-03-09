"""Property-based test for cross-region inference ARN format.

Feature: nl-search-chatbot, Property 13: Cross-region inference uses correct ARN format

Validates: Requirements 9.2

For any Bedrock API call to the LLM, the model identifier must use the
cross-region inference profile ARN format (containing "inference-profile").
For any Bedrock API call to the embedding model, the model identifier must
be the direct model ID (not an inference profile ARN).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from core.config import Settings
from services.llm import LLMService


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_environment_st = st.sampled_from(["commercial", "govcloud"])

_region_st = st.sampled_from([
    "us-east-1",
    "us-west-2",
    "eu-west-1",
    "us-gov-east-1",
    "us-gov-west-1",
])

_account_id_st = st.from_regex(r"[0-9]{12}", fullmatch=True)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    environment=_environment_st,
    bedrock_region=_region_st,
    aws_account_id=_account_id_st,
)
def test_llm_model_id_uses_inference_profile_arn(
    environment: str,
    bedrock_region: str,
    aws_account_id: str,
) -> None:
    """**Validates: Requirements 9.2**

    The LLM model_id stored on LLMService must contain "inference-profile",
    confirming it uses the cross-region inference profile ARN format.
    """
    cfg = Settings(
        environment=environment,
        bedrock_region=bedrock_region,
        aws_account_id=aws_account_id,
    )
    svc = LLMService(settings=cfg, client=MagicMock())

    assert "inference-profile" in svc._model_id, (
        f"LLM model_id should contain 'inference-profile' but got: {svc._model_id}"
    )


@settings(max_examples=100)
@given(
    environment=_environment_st,
    bedrock_region=_region_st,
    aws_account_id=_account_id_st,
)
def test_embedding_model_id_is_not_inference_profile(
    environment: str,
    bedrock_region: str,
    aws_account_id: str,
) -> None:
    """**Validates: Requirements 9.2**

    The embedding model_id stored on LLMService must NOT contain
    "inference-profile" — it should be a direct model ID.
    """
    cfg = Settings(
        environment=environment,
        bedrock_region=bedrock_region,
        aws_account_id=aws_account_id,
    )
    svc = LLMService(settings=cfg, client=MagicMock())

    assert "inference-profile" not in svc._embedding_model_id, (
        f"Embedding model_id should NOT contain 'inference-profile' "
        f"but got: {svc._embedding_model_id}"
    )
