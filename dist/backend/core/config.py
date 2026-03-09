"""Core configuration module.

Loads application settings from environment variables using Pydantic BaseSettings.
Supports both commercial and GovCloud environments with appropriate ARN prefixes.
"""

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_prefix": "", "case_sensitive": False}

    # Environment: "commercial" or "govcloud"
    environment: str = Field(default="commercial", description="Deployment environment: commercial or govcloud")

    # OpenSearch configuration
    opensearch_endpoint: str = Field(default="", description="OpenSearch Serverless collection endpoint URL")
    opensearch_collection_name: str = Field(default="nl-search-vectors", description="OpenSearch collection name")

    # Bedrock configuration
    bedrock_region: str = Field(default="us-east-1", description="AWS region for Bedrock API calls")
    bedrock_kb_id: str = Field(default="", description="Bedrock Knowledge Base ID")
    inference_profile_arn: str = Field(
        default="",
        description="Cross-region inference profile ARN for Claude LLM. Auto-generated if not set.",
    )
    embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Bedrock embedding model ID (invoked directly, no cross-region needed)",
    )
    aws_account_id: str = Field(default="", description="AWS account ID for ARN construction")

    # Rate limiting
    rate_limit_window_seconds: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    rate_limit_max_requests: int = Field(default=20, ge=1, description="Max requests per session per window")

    # Session configuration
    session_history_limit: int = Field(default=10, ge=1, description="Max conversation pairs to retain per session")

    # Context / token limits
    max_context_tokens: int = Field(default=4096, ge=1, description="Max tokens for LLM context window")

    # Retry configuration
    retry_max_attempts: int = Field(default=3, ge=0, description="Max retry attempts for transient errors")
    retry_base_delay_seconds: float = Field(default=1.0, ge=0, description="Base delay in seconds for exponential backoff")
    retry_max_delay_seconds: float = Field(default=8.0, ge=0, description="Max delay in seconds for exponential backoff")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def arn_prefix(self) -> str:
        """Return the ARN prefix based on the deployment environment."""
        if self.environment == "govcloud":
            return "arn:aws-us-gov"
        return "arn:aws"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_inference_profile_arn(self) -> str:
        """Return the inference profile ARN, auto-generating if not explicitly set."""
        if self.inference_profile_arn:
            return self.inference_profile_arn

        region = "us-gov-east-1" if self.environment == "govcloud" else self.bedrock_region
        account = self.aws_account_id or "{account}"
        return (
            f"{self.arn_prefix}:bedrock:{region}:{account}"
            f":inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        )


def get_settings() -> Settings:
    """Factory function returning a Settings instance (useful for FastAPI dependency injection)."""
    return Settings()
